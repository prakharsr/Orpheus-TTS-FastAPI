import torch
import asyncio
import uuid
import os
import wave
import logging
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Literal
from concurrent.futures import ThreadPoolExecutor

from orpheus_tts import OrpheusModel
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from audio_decoder import (
    tokens_decoder, 
    AudioDecodingError, 
    TokenRepetitionError, 
    AudioDurationOutlierError,
    TokenCountOutlierError,
    check_token_repetition, 
    check_token_count_ratio, 
    check_token_variance,
    normalize_and_count_words
)
from audio_analysis import quick_audio_check
from dotenv import load_dotenv

load_dotenv()

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY = 0.1

# Debug configuration
DEBUG_AUDIO_DIR = "debug_audio_errors"
DEBUG_SUCCESS_DIR = "debug_audio_success"

# Debug Configuration
ENABLE_DEBUG_SAVING = os.getenv("ENABLE_DEBUG_SAVING", "False").lower() == "true"
ENABLE_SUCCESS_LOGGING = os.getenv("ENABLE_SUCCESS_LOGGING", "False").lower() == "true"

def is_audio_duration_outlier(text: str, duration_seconds: float) -> Tuple[bool, dict]:
    """
    Check if audio duration is an outlier for given text.
    Uses multi-tier detection to catch both obvious outliers and moderate slowdowns.
    Accounts for emotion/sound effect tags that add non-speech audio duration.
    
    Args:
        text: The input text
        duration_seconds: The generated audio duration in seconds
        
    Returns:
        Tuple of (is_outlier: bool, metrics: dict)
    """
    # Count words using normalized counting (handles hyphens, underscores, etc.)
    word_count = normalize_and_count_words(text)
    
    # Detect emotion/sound effect tags that add extra audio duration
    # These tags cause the model to generate vocal sounds/effects beyond just speech
    import re
    emotion_tags = re.findall(r'<(\w+)>', text)
    
    # Orpheus-supported emotion/sound tags that add audio duration
    # Each tag causes the model to generate vocal sound effects
    ORPHEUS_EMOTION_TAGS = [
        'laugh',      # Laughter or laughing sounds
        'chuckle',    # Light laughter or chuckling
        'sigh',       # Sighing sounds (resignation/relief)
        'cough',      # Coughing or throat clearing
        'sniffle',    # Sniffling or nasal sounds
        'groan',      # Groaning (discomfort/frustration)
        'yawn',       # Yawning (tiredness)
        'gasp',       # Gasping (surprise/shock)
    ]
    
    # Count emotion tags in the text
    tag_count = sum(1 for tag in emotion_tags if tag.lower() in ORPHEUS_EMOTION_TAGS)
    
    # Add time allowance for emotion tags
    # Conservative estimate: 1.2 seconds per tag (covers most vocal effects without being too lenient)
    # This prevents false positives while still catching real repetition issues
    SECONDS_PER_EMOTION_TAG = 1.2
    emotion_tag_allowance = tag_count * SECONDS_PER_EMOTION_TAG
    
    # === TIER 1: Absolute threshold check (catches extreme outliers) ===
    # Tightened thresholds based on typical TTS behavior
    MIN_WORDS_PER_SECOND = 2.0  # Slow TTS (120 wpm) - was 1.5 (too lenient)
    MAX_WORDS_PER_SECOND = 4.5  # Very fast TTS (270 wpm) - was 4.0
    
    # Expected duration range
    min_expected_duration = word_count / MAX_WORDS_PER_SECOND
    max_expected_duration = word_count / MIN_WORDS_PER_SECOND
    
    # Add buffer for punctuation/pauses (20% buffer, reduced from 30%)
    max_expected_duration_absolute = max_expected_duration * 1
    
    # # For very short texts, be more lenient
    # # Single words with punctuation can have natural pauses and emphasis
    # if word_count == 1:
    #     max_expected_duration_absolute *= 5.0  # Very lenient for single words
    # elif word_count <= 5:
    #     max_expected_duration_absolute *= 3.0  # Lenient for short texts (1-5 words)
    # elif word_count <= 10:
    #     max_expected_duration_absolute *= 2.2  # Moderately lenient for short phrases (6-10 words)
    
    # === TIER 2: Expected speech rate check (catches moderate slowdowns) ===
    # Use typical TTS speech rate as baseline
    # Based on observed data, this model tends to generate faster speech (3+ wps)
    EXPECTED_WORDS_PER_SECOND = 3.0  # Adjusted for this model's typical output (180 wpm)
    expected_duration = word_count / EXPECTED_WORDS_PER_SECOND
    
    # Graduated deviation thresholds based on text length
    # Shorter texts have more natural variation in pacing
    if word_count <= 20:
        # Very short texts: use absolute threshold only (most lenient)
        max_expected_duration_final = max_expected_duration_absolute
        detection_method = "absolute"
    elif word_count <= 60:
        # Medium texts (21-60 words): allow more deviation due to higher natural pacing variation
        MAX_DEVIATION_PERCENT = 35  # More lenient for medium texts
        max_expected_duration_relative = expected_duration * (1 + MAX_DEVIATION_PERCENT / 100)
        max_expected_duration_final = min(max_expected_duration_absolute, max_expected_duration_relative)
        detection_method = "absolute" if max_expected_duration_absolute < max_expected_duration_relative else "relative"
    elif word_count <= 100:
        # Longer texts (61-100 words): moderate deviation
        MAX_DEVIATION_PERCENT = 22  # Moderate
        max_expected_duration_relative = expected_duration * (1 + MAX_DEVIATION_PERCENT / 100)
        max_expected_duration_final = min(max_expected_duration_absolute, max_expected_duration_relative)
        detection_method = "absolute" if max_expected_duration_absolute < max_expected_duration_relative else "relative"
    else:
        # Very long texts (>100 words): strict detection
        MAX_DEVIATION_PERCENT = 17  # Strict for catching edge cases like the 103-word, 40s sample
        max_expected_duration_relative = expected_duration * (1 + MAX_DEVIATION_PERCENT / 100)
        max_expected_duration_final = min(max_expected_duration_absolute, max_expected_duration_relative)
        detection_method = "absolute" if max_expected_duration_absolute < max_expected_duration_relative else "relative"
    
    # Calculate metrics
    words_per_second = word_count / duration_seconds if duration_seconds > 0 else 0
    deviation_from_expected = ((duration_seconds - expected_duration) / expected_duration * 100) if expected_duration > 0 else 0
    
    # === TIER 3: Grace buffer (absolute seconds, not percentage) ===
    # Add a flat grace buffer to account for natural TTS variation
    # This prevents flagging samples that are only marginally over the percentage threshold
    # Graduated by text length to be appropriate for each range
    if word_count <= 20:
        GRACE_BUFFER_SECONDS = 1.0  # Conservative for short texts (prevents missing real slowdowns)
    elif word_count <= 60:
        GRACE_BUFFER_SECONDS = 2.5  # Medium buffer for medium texts (21-60 words) - increased to match 61-100 range
    elif word_count <= 100:
        GRACE_BUFFER_SECONDS = 2.5  # Need more buffer for 61-100 word range due to natural variation
    else:
        GRACE_BUFFER_SECONDS = 0.0  # No grace buffer for >100 words (strict edge case detection)
    
    # Apply grace buffer and emotion tag allowance
    adjusted_threshold = max_expected_duration_final + emotion_tag_allowance + GRACE_BUFFER_SECONDS
    adjusted_threshold = 1.15 * adjusted_threshold

    # Determine if this is an outlier (after accounting for emotion tags and grace buffer)
    is_outlier = duration_seconds > adjusted_threshold
    
    metrics = {
        "word_count": word_count,
        "duration_seconds": round(duration_seconds, 2),
        "words_per_second": round(words_per_second, 2),
        "min_expected_duration": round(min_expected_duration, 2),
        "max_expected_duration": round(adjusted_threshold, 2),
        "max_expected_duration_base": round(max_expected_duration_final, 2),
        "expected_duration_baseline": round(expected_duration, 2),
        "max_expected_absolute": round(max_expected_duration_absolute, 2),
        "max_expected_relative": round(max_expected_duration_relative, 2) if word_count > 20 else None,
        "detection_method": detection_method,
        "grace_buffer_seconds": GRACE_BUFFER_SECONDS,
        "emotion_tag_count": tag_count,
        "emotion_tag_allowance": round(emotion_tag_allowance, 2),
        "emotion_tags_detected": emotion_tags if tag_count > 0 else [],
        "is_duration_outlier": is_outlier,
        "duration_deviation_percent": round(((duration_seconds - adjusted_threshold) / adjusted_threshold * 100), 2) if adjusted_threshold > 0 else 0,
        "deviation_from_expected_percent": round(deviation_from_expected, 2)
    }
    
    if is_outlier:
        emotion_tag_info = f", +{emotion_tag_allowance:.1f}s emotion" if tag_count > 0 else ""
        logger.warning(
            f"⚠️ Audio duration outlier detected ({detection_method}): "
            f"{duration_seconds:.2f}s for {word_count} words "
            f"(expected: ~{expected_duration:.2f}s, max: {max_expected_duration_final:.2f}s +{GRACE_BUFFER_SECONDS:.1f}s grace{emotion_tag_info}, "
            f"actual: {words_per_second:.2f} words/sec, deviation: {deviation_from_expected:+.1f}%)"
        )
    
    return is_outlier, metrics

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

async def save_debug_audio_with_metadata(audio_chunks: list[bytes], text: str, error_type: str, 
                                       request_id: str, metadata: dict, executor: ThreadPoolExecutor) -> str:
    """
    Save problematic audio file and detailed JSON metadata for debugging.
    
    Args:
        audio_chunks: The audio data chunks
        text: The input text
        error_type: Type of error/edge case detected
        request_id: Unique request identifier
        metadata: Additional metadata to save (should include 'attempts' field)
        executor: Thread pool executor for file I/O
        
    Returns:
        Path to the saved debug directory
    """
    if not ENABLE_DEBUG_SAVING:
        logger.debug("Debug saving is disabled")
        return ""
    
    try:
        # Get attempt number from metadata
        attempt_num = metadata.get("attempts", 1)
        
        # Create debug directory with error type, attempt, timestamp, and request ID for easy browsing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join(DEBUG_AUDIO_DIR, f"{error_type}_attempt{attempt_num}_{timestamp}_{request_id}")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save audio file
        audio_path = os.path.join(debug_dir, "audio.wav")
        
        def write_debug_audio():
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(1)  # mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # 24kHz
                
                for chunk in audio_chunks:
                    if chunk:
                        wf.writeframes(chunk)
        
        # Write audio in executor
        await asyncio.get_running_loop().run_in_executor(executor, write_debug_audio)
        
        # Prepare comprehensive metadata with retry info at top level for easy viewing
        # Convert numpy types to native Python types for JSON serialization
        debug_metadata = {
            "timestamp": timestamp,
            "request_id": request_id,
            "error_type": error_type,
            "attempt_number": attempt_num,
            "retry_number": attempt_num - 1,  # 0-indexed retry count
            "input_text": text,
            "text_length": len(text),
            "word_count": normalize_and_count_words(text),
            "audio_file": "audio.wav",
            "full_metadata": convert_numpy_types(metadata)
        }
        
        # Save metadata as JSON
        metadata_path = os.path.join(debug_dir, "metadata.json")
        
        def write_metadata():
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(debug_metadata, f, indent=2, ensure_ascii=False)
        
        # Write metadata in executor
        await asyncio.get_running_loop().run_in_executor(executor, write_metadata)
        
        logger.info(f"🔍 Debug files saved to: {debug_dir}")
        return debug_dir
        
    except Exception as e:
        logger.error(f"Failed to save debug files: {e}")
        return ""

async def save_successful_audio_with_metadata(audio_chunks: list[bytes], text: str, 
                                             request_id: str, metadata: dict, 
                                             tokens: list[str], executor: ThreadPoolExecutor) -> str:
    """
    Save successful audio generation with complete metadata for debugging and fine-tuning.
    
    This function saves all successful requests (not just errors) with comprehensive metadata
    including tokens, detection metrics, and parameters. This allows for later debugging and
    fine-tuning of detection parameters (repetition, duration, stretching, etc.).
    
    Args:
        audio_chunks: The audio data chunks
        text: The input text
        request_id: Unique request identifier
        metadata: Complete metadata from generation (includes all checks and metrics)
        tokens: The generated tokens (for reproducibility)
        executor: Thread pool executor for file I/O
        
    Returns:
        Path to the saved debug directory
    """
    if not ENABLE_SUCCESS_LOGGING:
        logger.debug("Success logging is disabled")
        return ""
    
    try:
        # Calculate audio duration
        total_audio_frames = sum(len(chunk) // 2 for chunk in audio_chunks if chunk)  # 16-bit audio
        duration = total_audio_frames / 24000  # 24kHz sample rate
        
        # Create directory with duration and request_id as specified by user
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        duration_str = f"{duration:.2f}s".replace(".", "_")
        debug_dir = os.path.join(DEBUG_SUCCESS_DIR, f"{duration_str}_{request_id}")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save audio file with duration in filename as specified
        duration_str = f"{duration:.2f}s".replace(".", "_")
        audio_filename = f"{duration_str}_{request_id}.wav"
        audio_path = os.path.join(debug_dir, audio_filename)
        
        def write_debug_audio():
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(1)  # mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # 24kHz
                
                for chunk in audio_chunks:
                    if chunk:
                        wf.writeframes(chunk)
        
        # Write audio in executor
        await asyncio.get_running_loop().run_in_executor(executor, write_debug_audio)
        
        # Prepare comprehensive metadata with ALL information needed for debugging
        # Convert numpy types to native Python types for JSON serialization
        debug_metadata = {
            "timestamp": timestamp,
            "request_id": request_id,
            "status": "success",
            "duration_seconds": round(duration, 2),
            "input_text": text,
            "text_length": len(text),
            "word_count": normalize_and_count_words(text),
            "audio_file": audio_filename,
            "token_count": len(tokens),
            "tokens": tokens,  # Save tokens for reproducibility
            "attempts": metadata.get("attempts", 1),
            "retries": metadata.get("retries", 0),
            
            # Detection results - critical for fine-tuning parameters
            "detection_results": {
                "token_repetition": {
                    "passed": True,  # Since this is a successful request
                    "metrics": metadata.get("checks", {}).get("token_count", {})
                },
                "token_variance": metadata.get("checks", {}).get("token_variance", {}),
                "audio_duration": {
                    "passed": not metadata.get("checks", {}).get("audio_duration", {}).get("is_duration_outlier", False),
                    "metrics": metadata.get("checks", {}).get("audio_duration", {})
                },
                "audio_quality": {
                    "passed": True,  # Since this is a successful request
                    "metrics": metadata.get("checks", {}).get("audio_quality", {})
                }
            },
            
            # Complete metadata for full context
            "full_metadata": convert_numpy_types(metadata)
        }
        
        # Save metadata as JSON
        metadata_path = os.path.join(debug_dir, "metadata.json")
        
        def write_metadata():
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(debug_metadata, f, indent=2, ensure_ascii=False)
        
        # Write metadata in executor
        await asyncio.get_running_loop().run_in_executor(executor, write_metadata)
        
        # Also save tokens as a separate text file for easy inspection
        tokens_path = os.path.join(debug_dir, "tokens.txt")
        
        def write_tokens():
            with open(tokens_path, "w", encoding="utf-8") as f:
                for i, token in enumerate(tokens):
                    f.write(f"{i}: {repr(token)}\n")
        
        # Write tokens in executor
        await asyncio.get_running_loop().run_in_executor(executor, write_tokens)
        
        logger.info(f"✅ Success log saved to: {debug_dir}")
        return debug_dir
        
    except Exception as e:
        logger.error(f"Failed to save success log: {e}")
        return ""

class OrpheusModelExtended(OrpheusModel):
    """Extended OrpheusModel with additional vLLM parameters"""
    
    def __init__(self, model_name, dtype=torch.bfloat16, max_model_len=2048, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_num_seqs=64, enable_chunked_prefill=True, enable_prefix_caching=True):
        # Store additional parameters
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_chunked_prefill = enable_chunked_prefill
        self.enable_prefix_caching = enable_prefix_caching
        self.max_num_seqs = max_num_seqs
        # Call parent constructor with original parameters
        super().__init__(model_name, dtype)
    
    def _setup_engine(self):
        """Override to include additional vLLM parameters"""
        # Map torch dtype to vLLM ModelDType literals
        vllm_dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
        if self.dtype == torch.bfloat16:
            vllm_dtype = "bfloat16"
        elif self.dtype == torch.float16:
            vllm_dtype = "float16"
        elif self.dtype == torch.float32:
            vllm_dtype = "float32"
        else:
            vllm_dtype = "bfloat16"  # default fallback
        
        engine_args = AsyncEngineArgs(
            enforce_eager=False,
            model=self.model_name,
            dtype=vllm_dtype,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_chunked_prefill=self.enable_chunked_prefill,
            enable_prefix_caching=self.enable_prefix_caching,
            max_num_seqs=self.max_num_seqs
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

async def generate_tokens_async(engine, prompt: str, voice: str, request_id: str, 
                               temperature: Optional[float] = None, top_p: Optional[float] = None, 
                               repetition_penalty: Optional[float] = None, max_tokens: Optional[int] = None,
                               default_temperature: float = 0.2, default_top_p: float = 0.9,
                               default_repetition_penalty: float = 1.1, default_max_tokens: int = 4096) -> list[str]:
    """Generate tokens using the async vLLM engine directly"""
    
    # Format prompt using the same logic as OrpheusModel
    adapted_prompt = f"{voice}: {prompt}"
    prompt_tokens = engine.tokeniser(adapted_prompt, return_tensors="pt")
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
    prompt_string = engine.tokeniser.decode(all_input_ids[0])
    
    # Use provided parameters or fall back to defaults
    temperature = temperature if temperature is not None else default_temperature
    top_p = top_p if top_p is not None else default_top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else default_repetition_penalty
    max_tokens = max_tokens if max_tokens is not None else default_max_tokens
    
    # Set up sampling parameters with configurable values
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=[128258],
        repetition_penalty=repetition_penalty,
    )
    
    logger.debug(f"Using sampling params - temp: {temperature}, top_p: {top_p}, "
                f"max_tokens: {max_tokens}, rep_penalty: {repetition_penalty}")
    
    # Generate tokens using async engine
    tokens = []
    async for result in engine.engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
        tokens.append(result.outputs[0].text)
    
    return tokens

async def generate_speech_tokens_with_retry(engine, prompt: str, voice: str, request_id: str, 
                                          temperature: Optional[float] = None, top_p: Optional[float] = None,
                                          repetition_penalty: Optional[float] = None, max_tokens: Optional[int] = None,
                                          default_temperature: float = 0.2, default_top_p: float = 0.9,
                                          default_repetition_penalty: float = 1.1, default_max_tokens: int = 4096,
                                          executor: Optional[ThreadPoolExecutor] = None) -> tuple[list[bytes], dict]:
    """Generate speech tokens with retry logic for audio decoding errors"""
    
    last_error = None
    current_repetition_penalty = repetition_penalty  # Track current penalty for adjustments
    current_temperature = temperature if temperature is not None else default_temperature
    effective_max_tokens = max_tokens if max_tokens is not None else default_max_tokens
    
    for attempt in range(MAX_RETRIES):
        token_repetition_error = None
        token_count_error = None
        audio_chunks = []
        tokens = []
        
        try:
            # Generate tokens using async engine
            tokens = await generate_tokens_async(engine, prompt, voice, request_id, 
                                               current_temperature, top_p, current_repetition_penalty, max_tokens,
                                               default_temperature, default_top_p, default_repetition_penalty, default_max_tokens)
            
            # === STAGE 1: Token-level checks (BEFORE audio generation - lightweight) ===
            
            # Check 1: Token repetition patterns (catch but don't raise yet)
            try:
                check_token_repetition(tokens, effective_max_tokens)
            except TokenRepetitionError as e:
                token_repetition_error = e
                logger.warning(f"⚠️ Token repetition detected, will generate audio for debugging: {e}")
            
            # Check 2: Token count vs text length ratio (catch but don't raise yet)
            try:
                token_count_metrics = check_token_count_ratio(tokens, prompt, effective_max_tokens)
            except TokenCountOutlierError as e:
                token_count_error = e
                token_count_metrics = {
                    "token_count": len(tokens),
                    "word_count": normalize_and_count_words(prompt),
                    "is_ratio_outlier": True,
                    "error": str(e)
                }
                logger.warning(f"⚠️ Token count outlier detected, will generate audio for debugging: {e}")
            
            # Check 3: Token variance analysis (always runs, returns metrics)
            token_variance_metrics = check_token_variance(tokens, window_size=100)
            
            # === STAGE 2: Audio generation ===
            
            # Try to decode tokens to audio chunks (even if token checks failed, for debugging)
            async for audio_chunk in tokens_decoder(tokens):
                audio_chunks.append(audio_chunk)
            
            # If we get here, decoding was successful
            chunk_count = len(audio_chunks)
            total_audio_frames = sum(len(chunk) // 2 for chunk in audio_chunks if chunk)  # 16-bit audio
            duration = total_audio_frames / 24000  # 24kHz sample rate
            
            # === STAGE 3: Audio-level checks (AFTER audio generation) ===
            
            # Check 4: Audio duration outlier detection
            is_outlier, duration_metrics = is_audio_duration_outlier(prompt, duration)
            
            # Check 5: Advanced audio quality analysis (if available and enabled)
            audio_quality_issues = False
            audio_quality_metrics = {}
            
            # Skip audio quality checks for extremely short texts (≤3 words)
            # These naturally have high silence ratios due to lead-in/out
            word_count_for_check = normalize_and_count_words(prompt)
            skip_quality_check = word_count_for_check <= 3
            
            if not skip_quality_check:
                try:
                    # Count emotion tags for audio quality analysis
                    import re
                    emotion_tags_found = re.findall(r'<(\w+)>', prompt)
                    ORPHEUS_EMOTION_TAGS = ['laugh', 'chuckle', 'sigh', 'cough', 'sniffle', 'groan', 'yawn', 'gasp']
                    emotion_tag_count = sum(1 for tag in emotion_tags_found if tag.lower() in ORPHEUS_EMOTION_TAGS)
                    
                    audio_quality_issues, audio_quality_metrics = quick_audio_check(audio_chunks, sample_rate=24000, emotion_tag_count=emotion_tag_count)
                    if audio_quality_issues:
                        logger.warning(
                            f"⚠️ Audio quality issues detected for request_id: {request_id} - "
                            f"Silence: {audio_quality_metrics.get('silence_percent', 0):.1f}%, "
                            f"Repetition: {audio_quality_metrics.get('has_repetition', False)}, "
                            f"Stretched: {audio_quality_metrics.get('is_stretched', False)}"
                        )
                except Exception as e:
                    logger.debug(f"Audio quality analysis failed: {e}")
                    audio_quality_metrics = {"error": str(e)}
            elif skip_quality_check:
                audio_quality_metrics = {"skipped": "text too short (≤3 words)", "word_count": word_count_for_check}
            
            # Prepare comprehensive metadata
            metadata = {
                "request_id": request_id,
                "chunk_count": chunk_count,
                "total_audio_frames": total_audio_frames,
                "duration_seconds": round(duration, 2),
                "voice": voice,
                "prompt_length": len(prompt),
                "attempts": attempt + 1,
                "retries": attempt,
                "checks": {
                    "token_count": token_count_metrics,
                    "token_variance": token_variance_metrics,
                    "audio_duration": duration_metrics,
                    "audio_quality": audio_quality_metrics if audio_quality_metrics else None
                }
            }
            
            # Now handle token-level errors (after we have audio for debugging)
            if token_repetition_error is not None:
                logger.error(f"🚨 Token repetition error for request_id: {request_id}")
                
                # Add error details to metadata
                metadata["error"] = str(token_repetition_error)
                metadata["token_count"] = len(tokens)
                
                # Save debug files if executor is available
                if executor is not None:
                    await save_debug_audio_with_metadata(
                        audio_chunks, prompt, "token_repetition", request_id, metadata, executor
                    )
                
                # Raise the error to trigger retry
                raise token_repetition_error
            
            if token_count_error is not None:
                logger.error(f"🚨 Token count error for request_id: {request_id}")
                
                # Add error details to metadata
                metadata["error"] = str(token_count_error)
                metadata["token_count"] = len(tokens)
                
                # Save debug files if executor is available
                if executor is not None:
                    await save_debug_audio_with_metadata(
                        audio_chunks, prompt, "token_count_outlier", request_id, metadata, executor
                    )
                
                # Raise the error to trigger retry
                raise token_count_error
            
            # If duration is an outlier OR audio quality issues detected, save debug files and retry
            if is_outlier or audio_quality_issues:
                # Determine error type and message
                has_low_variance = token_variance_metrics.get("is_low_variance", False)
                
                if is_outlier and audio_quality_issues:
                    error_type = "duration_outlier_with_quality_issues"
                    error_msg = (
                        f"Audio duration outlier with quality issues: {duration:.2f}s for {duration_metrics['word_count']} words "
                        f"(expected max: {duration_metrics['max_expected_duration']:.2f}s, "
                        f"silence: {audio_quality_metrics.get('silence_percent', 0):.1f}%, "
                        f"repetition: {audio_quality_metrics.get('has_repetition', False)}, "
                        f"stretched: {audio_quality_metrics.get('is_stretched', False)})"
                    )
                    logger.error(f"🚨 {error_msg} for request_id: {request_id}")
                elif is_outlier and has_low_variance:
                    error_type = "duration_outlier_with_low_variance"
                    error_msg = (
                        f"Audio duration outlier with low token variance: {duration:.2f}s for {duration_metrics['word_count']} words "
                        f"(expected max: {duration_metrics['max_expected_duration']:.2f}s, "
                        f"CV: {token_variance_metrics.get('coefficient_of_variation', 'N/A')}, "
                        f"unique ratio: {token_variance_metrics.get('unique_tokens_ratio', 'N/A')})"
                    )
                    logger.error(f"🚨 {error_msg} for request_id: {request_id}")
                elif is_outlier:
                    error_type = "duration_outlier"
                    error_msg = (
                        f"Audio duration outlier: {duration:.2f}s for {duration_metrics['word_count']} words "
                        f"(expected max: {duration_metrics['max_expected_duration']:.2f}s, "
                        f"detection method: {duration_metrics.get('detection_method', 'unknown')})"
                    )
                    logger.error(f"🚨 Audio duration outlier detected for request_id: {request_id}")
                else:  # audio_quality_issues only
                    error_type = "audio_quality_issues"
                    error_msg = (
                        f"Audio quality issues detected: {duration:.2f}s for {duration_metrics['word_count']} words "
                        f"(silence: {audio_quality_metrics.get('silence_percent', 0):.1f}%, "
                        f"repetition: {audio_quality_metrics.get('has_repetition', False)}, "
                        f"repeated_segments: {audio_quality_metrics.get('repeated_segment_count', 0)}, "
                        f"stretched: {audio_quality_metrics.get('is_stretched', False)})"
                    )
                    logger.error(f"🚨 Audio quality issues for request_id: {request_id}")
                
                # Add error details to metadata
                metadata["token_count"] = len(tokens)
                metadata["error"] = error_msg
                metadata["has_low_variance"] = has_low_variance
                metadata["has_audio_quality_issues"] = audio_quality_issues
                
                # Save debug files if executor is available
                if executor is not None:
                    await save_debug_audio_with_metadata(
                        audio_chunks, prompt, error_type, request_id, metadata, executor
                    )
                
                # Raise appropriate error type to trigger retry
                if audio_quality_issues and not is_outlier:
                    # Only quality issues, treat as duration outlier for retry logic
                    raise AudioDurationOutlierError(error_msg)
                else:
                    # Duration outlier (with or without quality issues)
                    raise AudioDurationOutlierError(error_msg)
            
            # Check for warnings in variance or token count (log but don't retry)
            has_warnings = (
                token_variance_metrics.get("is_low_variance", False) or
                token_count_metrics.get("is_near_limit", False)
            )
            
            if has_warnings and executor is not None:
                logger.warning(f"⚠️ Edge case warnings detected for request_id: {request_id}")
                # Save debug files for analysis even if not retrying
                await save_debug_audio_with_metadata(
                    audio_chunks, prompt, "edge_case_warning", request_id, metadata, executor
                )
            
            # Save successful requests for debugging and fine-tuning detection parameters
            if executor is not None:
                await save_successful_audio_with_metadata(
                    audio_chunks, prompt, request_id, metadata, tokens, executor
                )
            
            if attempt > 0:
                logger.info(f"✅ Token generation successful after {attempt + 1} attempts for request_id: {request_id}")
            
            return audio_chunks, metadata
            
        except TokenRepetitionError as e:
            last_error = e
            logger.error(f"🔄 Token repetition detected on attempt {attempt + 1}/{MAX_RETRIES} for request_id: {request_id}: {e}")
            
            if attempt < MAX_RETRIES - 1:  # Not the last attempt
                # For repetition errors, adjust sampling parameters
                logger.info(f"⏳ Retrying with adjusted parameters in {RETRY_DELAY} seconds...")
                
                # Increase repetition penalty for retry attempts
                if current_repetition_penalty is None:
                    current_repetition_penalty = default_repetition_penalty * (1.0 + 0.15 * (attempt + 1))  # Increase by 10% per retry
                    current_temperature = default_temperature * (1.0 + 0.15 * (attempt + 1))
                else:
                    current_repetition_penalty = current_repetition_penalty * (1.0 + 0.15 * (attempt + 1))
                    current_temperature = current_temperature * (1.0 + 0.15 * (attempt + 1))
                logger.info(f"🔧 Adjusted repetition penalty to {current_repetition_penalty:.2f} and temperature to {current_temperature:.2f} for retry")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"❌ All {MAX_RETRIES} attempts failed due to token repetition for request_id: {request_id}")
        
        except TokenCountOutlierError as e:
            last_error = e
            logger.error(f"🔄 Token count outlier on attempt {attempt + 1}/{MAX_RETRIES} for request_id: {request_id}: {e}")
            
            if attempt < MAX_RETRIES - 1:  # Not the last attempt
                # For token count outliers, adjust sampling parameters
                logger.info(f"⏳ Retrying with adjusted parameters in {RETRY_DELAY} seconds...")
                
                # Increase repetition penalty for retry attempts
                if current_repetition_penalty is None:
                    current_repetition_penalty = default_repetition_penalty * (1.0 + 0.15 * (attempt + 1))  # Increase by 15% per retry
                    current_temperature = default_temperature * (1.0 + 0.15 * (attempt + 1))
                else:
                    current_repetition_penalty = current_repetition_penalty * (1.0 + 0.15 * (attempt + 1))
                    current_temperature = current_temperature * (1.0 + 0.15 * (attempt + 1))
                
                logger.info(f"🔧 Adjusted repetition penalty to {current_repetition_penalty:.2f} and temperature to {current_temperature:.2f} for retry")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"❌ All {MAX_RETRIES} attempts failed due to token count outlier for request_id: {request_id}")
        
        except AudioDurationOutlierError as e:
            last_error = e
            logger.error(f"🔄 Audio duration outlier on attempt {attempt + 1}/{MAX_RETRIES} for request_id: {request_id}: {e}")
            
            if attempt < MAX_RETRIES - 1:  # Not the last attempt
                logger.info(f"⏳ Retrying in {RETRY_DELAY} seconds...")
                
                # Increase repetition penalty slightly for retry attempts
                if current_repetition_penalty is None:
                    current_repetition_penalty = default_repetition_penalty * (1.0 + 0.15 * (attempt + 1))
                    current_temperature = default_temperature * (1.0 + 0.15 * (attempt + 1))
                else:
                    current_repetition_penalty = current_repetition_penalty * (1.0 + 0.15 * (attempt + 1))
                    current_temperature = current_temperature * (1.0 + 0.15 * (attempt + 1))
                logger.info(f"🔧 Adjusted repetition penalty to {current_repetition_penalty:.2f} and temperature to {current_temperature:.2f} for retry")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"❌ All {MAX_RETRIES} attempts failed due to audio duration outlier for request_id: {request_id}")
                
        except AudioDecodingError as e:
            last_error = e
            logger.info(f"🔄 Audio decoding failed on attempt {attempt + 1}/{MAX_RETRIES} for request_id: {request_id}: {e}")
            
            # No need to save debug audio - audio decoding errors are always true positives
            
            if attempt < MAX_RETRIES - 1:  # Not the last attempt
                logger.info(f"⏳ Retrying in {RETRY_DELAY} seconds...")

                # Increase repetition penalty slightly for retry attempts
                if current_repetition_penalty is None:
                    current_repetition_penalty = default_repetition_penalty * (1.0 + 0.15 * (attempt + 1))
                    current_temperature = default_temperature * (1.0 + 0.15 * (attempt + 1))
                else:
                    current_repetition_penalty = current_repetition_penalty * (1.0 + 0.15 * (attempt + 1))
                    current_temperature = current_temperature * (1.0 + 0.15 * (attempt + 1))
                logger.info(f"🔧 Adjusted repetition penalty to {current_repetition_penalty:.2f} and temperature to {current_temperature:.2f} for retry")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"❌ All {MAX_RETRIES} attempts failed for request_id: {request_id}")
                
        except Exception as e:
            # For other exceptions, don't retry - they're likely not transient
            logger.error(f"❌ Non-retryable error in token generation for request_id: {request_id}: {e}")
            raise
    
    # If we get here, all retries failed - raise the specific error type
    if isinstance(last_error, TokenRepetitionError):
        raise TokenRepetitionError(f"Token repetition persisted after {MAX_RETRIES} attempts. Last error: {last_error}")
    elif isinstance(last_error, TokenCountOutlierError):
        raise TokenCountOutlierError(f"Token count outlier persisted after {MAX_RETRIES} attempts. Last error: {last_error}")
    elif isinstance(last_error, AudioDurationOutlierError):
        raise AudioDurationOutlierError(f"Audio duration outlier persisted after {MAX_RETRIES} attempts. Last error: {last_error}")
    else:
        raise AudioDecodingError(f"Token generation failed after {MAX_RETRIES} attempts. Last error: {last_error}")

async def generate_speech_tokens_direct(engine, prompt: str, voice: str, 
                                      temperature: Optional[float] = None, top_p: Optional[float] = None,
                                      repetition_penalty: Optional[float] = None, max_tokens: Optional[int] = None,
                                      default_temperature: float = 0.2, default_top_p: float = 0.9,
                                      default_repetition_penalty: float = 1.1, default_max_tokens: int = 4096,
                                      executor: Optional[ThreadPoolExecutor] = None) -> tuple[list[bytes], dict]:
    """Generate speech tokens using direct async vLLM engine access with retry logic"""
    try:
        # Generate unique request ID
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        
        # logger.info(f"Starting speech token generation with retry logic for voice: {voice}, request_id: {request_id}")
        
        # Use retry logic
        audio_chunks, metadata = await generate_speech_tokens_with_retry(
            engine, prompt, voice, request_id,
            temperature, top_p, repetition_penalty, max_tokens,
            default_temperature, default_top_p, default_repetition_penalty, default_max_tokens,
            executor
        )
        
        # logger.info(f"Speech token generation complete: {metadata['chunk_count']} chunks, "
        #            f"{metadata['duration_seconds']:.2f}s audio, {metadata['attempts']} attempts")
        
        return audio_chunks, metadata
        
    except Exception as e:
        logger.error(f"Error in generate_speech_tokens_direct: {e}")
        raise

def combine_token_chunks(token_chunks_list: list[list[bytes]]) -> list[bytes]:
    """Combine multiple token chunk lists into a single list"""
    combined_chunks = []
    total_chunks = 0
    
    for chunk_list in token_chunks_list:
        combined_chunks.extend(chunk_list)
        total_chunks += len(chunk_list)
    
    # logger.info(f"Combined {len(token_chunks_list)} batches into {total_chunks} total chunks")
    return combined_chunks

async def generate_speech_chunks(engine, text_chunks: list[str], voice: str, 
                               temperature: Optional[float] = None, top_p: Optional[float] = None,
                               repetition_penalty: Optional[float] = None, max_tokens: Optional[int] = None,
                               default_temperature: float = 0.2, default_top_p: float = 0.9,
                               default_repetition_penalty: float = 1.1, default_max_tokens: int = 4096,
                               executor: Optional[ThreadPoolExecutor] = None) -> tuple[list[bytes], dict]:
    """
    Generate speech tokens for multiple text chunks in parallel with retry logic.
    Returns combined tokens and detailed metadata.
    """
    total_metadata = {
        "total_chunks": len(text_chunks),
        "chunk_details": [],
        "combined_stats": {},
        "retry_stats": {
            "total_attempts": 0,
            "total_retries": 0,
            "failed_chunks": 0
        }
    }
    
    # logger.info(f"Starting parallel processing of {len(text_chunks)} text chunks with retry logic")
    
    # Generate tokens for all chunks in parallel with retry logic
    chunk_tasks = []
    for i, chunk in enumerate(text_chunks):
        # logger.info(f"Queuing chunk {i+1}/{len(text_chunks)}: {len(chunk)} characters")
        request_id = f"chunk-{i+1}-{uuid.uuid4().hex[:8]}"
        
        # Each chunk gets its own retry logic
        task = generate_speech_tokens_with_retry(
            engine, chunk, voice, request_id,
            temperature, top_p, repetition_penalty, max_tokens,
            default_temperature, default_top_p, default_repetition_penalty, default_max_tokens,
            executor
        )
        chunk_tasks.append(task)
    
    # Wait for all chunks to complete
    chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
    
    # Process results and collect metadata
    all_chunk_results = []
    failed_chunks = 0
    
    for i, result in enumerate(chunk_results):
        if isinstance(result, BaseException):
            # Chunk failed completely
            failed_chunks += 1
            logger.error(f"Chunk {i+1} failed after all retries: {result}")
            total_metadata["retry_stats"]["failed_chunks"] += 1
            # Add empty result to maintain order
            all_chunk_results.append([])
            
            chunk_info = {
                "chunk_index": i,
                "text_length": len(text_chunks[i]),
                "audio_chunks": 0,
                "duration_seconds": 0,
                "audio_frames": 0,
                "request_id": f"chunk-{i+1}-failed",
                "attempts": MAX_RETRIES,
                "retries": MAX_RETRIES - 1,
                "failed": True,
                "error": str(result)
            }
            total_metadata["chunk_details"].append(chunk_info)
        else:
            # Chunk succeeded - result is a tuple
            chunk_tokens, chunk_metadata = result
            all_chunk_results.append(chunk_tokens)
            
            # Add chunk-specific metadata
            chunk_info = {
                "chunk_index": i,
                "text_length": len(text_chunks[i]),
                "audio_chunks": chunk_metadata["chunk_count"],
                "duration_seconds": chunk_metadata["duration_seconds"],
                "audio_frames": chunk_metadata["total_audio_frames"],
                "request_id": chunk_metadata["request_id"],
                "attempts": chunk_metadata.get("attempts", 1),
                "retries": chunk_metadata.get("retries", 0),
                "failed": False
            }
            total_metadata["chunk_details"].append(chunk_info)
            
            # Update retry stats
            total_metadata["retry_stats"]["total_attempts"] += chunk_metadata.get("attempts", 1)
            total_metadata["retry_stats"]["total_retries"] += chunk_metadata.get("retries", 0)
            
            # logger.info(f"Chunk {i+1} complete: {chunk_metadata['chunk_count']} audio chunks, "
            #            f"{chunk_metadata['duration_seconds']}s duration, "
            #            f"{chunk_metadata.get('attempts', 1)} attempts")
    
    # Log retry statistics
    retry_stats = total_metadata["retry_stats"]
    # logger.info(f"Retry statistics: {retry_stats['total_attempts']} total attempts, "
    #            f"{retry_stats['total_retries']} total retries, "
    #            f"{retry_stats['failed_chunks']} failed chunks")
    
    # Combine all token chunks (excluding failed ones)
    combined_tokens = combine_token_chunks(all_chunk_results)
    
    # Calculate combined statistics
    successful_chunks = [chunk for chunk in total_metadata["chunk_details"] if not chunk.get("failed", False)]
    total_duration = sum(chunk["duration_seconds"] for chunk in successful_chunks)
    total_frames = sum(chunk["audio_frames"] for chunk in successful_chunks)
    total_audio_chunks = sum(chunk["audio_chunks"] for chunk in successful_chunks)
    
    total_metadata["combined_stats"] = {
        "total_text_length": sum(len(chunk) for chunk in text_chunks),
        "total_audio_chunks": total_audio_chunks,
        "total_duration_seconds": round(total_duration, 2),
        "total_audio_frames": total_frames,
        "successful_chunks": len(successful_chunks),
        "failed_chunks": failed_chunks,
        "voice": voice
    }
    
    # logger.info(f"Generated and combined {len(text_chunks)} text chunks into {len(combined_tokens)} audio chunks, "
    #            f"total duration: {total_duration:.2f}s, {failed_chunks} failed chunks")
    
    # If any chunks failed, raise an error
    if failed_chunks > 0:
        raise AudioDecodingError(f"{failed_chunks} of {len(text_chunks)} chunks failed after retries")
    
    return combined_tokens, total_metadata

def _write_tokens_to_file(token_chunks: list[bytes], output_path: str) -> dict:
    """Helper function to write token chunks to WAV file in executor"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create WAV file
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(24000)  # 24kHz
        
        total_frames = 0
        
        for chunk in token_chunks:
            if chunk:
                frame_count = len(chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(chunk)
        
        duration = total_frames / wf.getframerate()
    
    file_stats = {
        "total_chunks": len(token_chunks),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
        "file_size_bytes": os.path.getsize(output_path),
        "output_path": output_path
    }
    
    return file_stats

async def tokens_to_audio_file(token_chunks: list[bytes], output_path: str, executor: ThreadPoolExecutor) -> dict:
    """Convert token chunks to WAV audio file"""
    try:
        # Run file I/O in executor
        file_stats = await asyncio.get_running_loop().run_in_executor(
            executor, _write_tokens_to_file, token_chunks, output_path
        )
        
        # logger.info(f"Audio file created: {len(token_chunks)} chunks, {file_stats['duration_seconds']:.2f}s, {file_stats['file_size_bytes']} bytes")
        
        return file_stats
        
    except Exception as e:
        logger.error(f"Error in tokens_to_audio_file: {e}")
        # Clean up partial file on error
        if os.path.exists(output_path):
            os.remove(output_path)
        raise 