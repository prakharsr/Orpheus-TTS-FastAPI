from typing import Optional, Literal
import time
import logging
import os
import uuid
import wave
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import uvicorn

# Force V0 engine (same as Flask) to avoid async issues
os.environ["VLLM_USE_V1"] = "0"

from orpheus_tts import OrpheusModel
from vllm import AsyncEngineArgs, AsyncLLMEngine

# ===============================
# CONFIGURATION FROM ENVIRONMENT
# ===============================

def get_dtype_from_string(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype"""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "bf16": torch.bfloat16,  # alias
        "fp16": torch.float16,  # alias
        "fp32": torch.float32,  # alias
    }
    dtype_str = dtype_str.lower()
    if dtype_str not in dtype_map:
        logger.warning(f"Unknown dtype '{dtype_str}', defaulting to bfloat16")
        return torch.bfloat16
    return dtype_map[dtype_str]

# Model Configuration
MODEL_NAME = os.getenv("TTS_MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod")
DTYPE = get_dtype_from_string(os.getenv("TTS_DTYPE", "bfloat16"))
MAX_MODEL_LEN = int(os.getenv("TTS_MAX_MODEL_LEN", "8192"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TTS_TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.getenv("TTS_GPU_MEMORY_UTILIZATION", "0.9"))

# Performance Configuration  
MAX_WORKERS = int(os.getenv("TTS_MAX_WORKERS", "8"))

# Generation Parameters (defaults)
DEFAULT_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("TTS_TOP_P", "0.9"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("TTS_REPETITION_PENALTY", "1.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("TTS_MAX_TOKENS", "4096"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Log configuration on startup
logger.info("🔧 Configuration loaded:")
logger.info(f"   Model: {MODEL_NAME}")
logger.info(f"   Dtype: {DTYPE}")
logger.info(f"   Max Model Length: {MAX_MODEL_LEN}")
logger.info(f"   Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
logger.info(f"   GPU Memory Utilization: {GPU_MEMORY_UTILIZATION}")
logger.info(f"   Max Workers: {MAX_WORKERS}")
logger.info(f"   Temperature: {DEFAULT_TEMPERATURE}")
logger.info(f"   Top P: {DEFAULT_TOP_P}")
logger.info(f"   Repetition Penalty: {DEFAULT_REPETITION_PENALTY}")
logger.info(f"   Max Tokens: {DEFAULT_MAX_TOKENS}")

class OrpheusModelExtended(OrpheusModel):
    """Extended OrpheusModel with additional vLLM parameters"""
    
    def __init__(self, model_name, dtype=torch.bfloat16, max_model_len=2048, tensor_parallel_size=1, gpu_memory_utilization=1):
        # Store additional parameters
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Call parent constructor with original parameters
        super().__init__(model_name, dtype)
    
    def _setup_engine(self):
        """Override to include additional vLLM parameters"""
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

# Global engine variable
engine = None
executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("🚀 Loading Orpheus TTS model...")
    global engine, executor

    engine = OrpheusModelExtended(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION
    )
    
    # Create thread pool executor for blocking operations (token decoding + file I/O only)
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    logger.info("✅ Model loaded successfully!")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    logger.info("📁 Created outputs directory")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Orpheus TTS model...")
    if executor:
        executor.shutdown(wait=True)
    engine = None
    executor = None
    logger.info("✅ Shutdown complete!")

app = FastAPI(
    title="Orpheus TTS API",
    description="OpenAI-compatible Text-to-Speech API using Orpheus TTS",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add timeout middleware for long-running requests
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Middleware to handle long-running requests with extended timeout"""
    start_time = time.time()
    
    # Set extended timeout for TTS endpoints
    if request.url.path.startswith("/v1/audio/"):
        # For TTS endpoints, set a much longer timeout (10 minutes)
        timeout = 3600.0
    else:
        # For other endpoints, use default timeout (30 seconds)
        timeout = 30.0
    
    try:
        # Process request with timeout
        response = await asyncio.wait_for(call_next(request), timeout=timeout)
        
        # Log request duration
        process_time = time.time() - start_time
        logger.info(f"Request {request.method} {request.url.path} completed in {process_time:.2f}s")
        
        return response
    
    except asyncio.TimeoutError:
        logger.error(f"Request {request.method} {request.url.path} timed out after {timeout}s")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "timeout_error",
                "message": f"Request timed out after {timeout} seconds",
                "type": "server_error",
            }
        )
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request.method} {request.url.path} failed after {process_time:.2f}s: {e}")
        raise

# OpenAI-compatible request models
class SpeechRequest(BaseModel):
    model: Literal["orpheus"] = Field(default="orpheus", description="The TTS model to use")
    input: str = Field(..., description="The text to generate audio for", max_length=4096)
    voice: Literal["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"] = Field(
        default="tara", 
        description="The voice to use for generation"
    )
    response_format: Literal["wav"] = Field(
        default="wav", 
        description="The format of the audio output"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="The speed of the generated audio")
    
    # Optional sampling parameters (will use environment defaults if not specified)
    temperature: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=2.0, 
        description="Temperature for sampling (0.0 to 2.0). Uses environment default if not specified."
    )
    top_p: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Top-p for nucleus sampling (0.0 to 1.0). Uses environment default if not specified."
    )
    repetition_penalty: Optional[float] = Field(
        default=None, 
        ge=0.5, 
        le=2.0, 
        description="Repetition penalty (0.5 to 2.0). Uses environment default if not specified."
    )
    max_tokens: Optional[int] = Field(
        default=None, 
        ge=100, 
        le=8192, 
        description="Maximum tokens to generate. Uses environment default if not specified."
    )

# Old functions removed - using direct async access now

async def generate_tokens_async(engine, prompt: str, voice: str, request_id: str, 
                               temperature: float = None, top_p: float = None, 
                               repetition_penalty: float = None, max_tokens: int = None) -> list[str]:
    """Generate tokens using the async vLLM engine directly"""
    from vllm import SamplingParams
    
    # Format prompt using the same logic as OrpheusModel
    adapted_prompt = f"{voice}: {prompt}"
    prompt_tokens = engine.tokeniser(adapted_prompt, return_tensors="pt")
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
    prompt_string = engine.tokeniser.decode(all_input_ids[0])
    
    # Use provided parameters or fall back to environment defaults
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    top_p = top_p if top_p is not None else DEFAULT_TOP_P
    repetition_penalty = repetition_penalty if repetition_penalty is not None else DEFAULT_REPETITION_PENALTY
    max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    
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

async def generate_speech_tokens_direct(prompt: str, voice: str, 
                                      temperature: float = None, top_p: float = None,
                                      repetition_penalty: float = None, max_tokens: int = None) -> tuple[list[bytes], dict]:
    """Generate speech tokens using direct async vLLM engine access - executor only for decoder"""
    try:
        # Import decoder here to avoid circular imports
        from orpheus_tts.decoder import tokens_decoder_sync
        
        # Generate unique request ID
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting direct async speech token generation with voice: {voice}, request_id: {request_id}")
        
        # Generate tokens using async engine (NO executor needed here)
        tokens = await generate_tokens_async(engine, prompt, voice, request_id, 
                                           temperature, top_p, repetition_penalty, max_tokens)
        
        # Run decoder in executor since it's CPU/GPU intensive
        def decode_tokens():
            return list(tokens_decoder_sync(tokens))
        
        # Decode tokens to audio chunks (executor needed here)
        audio_chunks = await asyncio.get_running_loop().run_in_executor(executor, decode_tokens)
        
        # Count chunks and calculate metadata
        chunk_count = len(audio_chunks)
        total_audio_frames = sum(len(chunk) // 2 for chunk in audio_chunks if chunk)  # 16-bit audio
        duration = total_audio_frames / 24000  # 24kHz sample rate
        
        metadata = {
            "request_id": request_id,
            "chunk_count": chunk_count,
            "total_audio_frames": total_audio_frames,
            "duration_seconds": round(duration, 2),
            "voice": voice,
            "prompt_length": len(prompt)
        }
        
        logger.info(f"Direct async token generation complete: {chunk_count} chunks, {duration:.2f}s audio, request_id: {request_id}")
        
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
    
    logger.info(f"Combined {len(token_chunks_list)} batches into {total_chunks} total chunks")
    return combined_chunks


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

async def tokens_to_audio_file(token_chunks: list[bytes], output_path: str) -> dict:
    """Convert token chunks to WAV audio file"""
    try:
        # Run file I/O in executor
        file_stats = await asyncio.get_running_loop().run_in_executor(
            executor, _write_tokens_to_file, token_chunks, output_path
        )
        
        logger.info(f"Audio file created: {len(token_chunks)} chunks, {file_stats['duration_seconds']:.2f}s, {file_stats['file_size_bytes']} bytes")
        
        return file_stats
        
    except Exception as e:
        logger.error(f"Error in tokens_to_audio_file: {e}")
        # Clean up partial file on error
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


# generate_speech_to_file removed - using direct async access now


def cleanup_file(file_path: str) -> None:
    """Clean up a single file after serving"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")

def split_text_into_sentences(text):
    """
    Split text into sentences while preserving dialogue integrity and narrative flow.
    
    This function handles:
    - Dialogue preservation (keeping quotes together)
    - Dialogue attribution (keeping "he said" with dialogue)
    - Paragraph boundaries
    - Complex punctuation within dialogue
    - Reasonable chunk sizes for TTS processing
    """
    import re
    
    # First, split by paragraphs to maintain document structure
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    all_segments = []
    
    for paragraph in paragraphs:
        segments = _split_paragraph_intelligently(paragraph)
        all_segments.extend(segments)
    
    # Post-process to ensure reasonable sizes and combine very short segments
    return _combine_short_segments(all_segments)


def _split_paragraph_intelligently(paragraph):
    """Split a single paragraph while preserving dialogue and narrative flow."""
    import re
    
    # If paragraph is short enough, return as-is
    if len(paragraph) <= 200:
        return [paragraph]
    
    # Find all dialogue blocks (text within quotes)
    dialogue_pattern = r'"([^"]*)"'
    dialogues = list(re.finditer(dialogue_pattern, paragraph))
    
    segments = []
    last_end = 0
    
    for dialogue_match in dialogues:
        # Get text before this dialogue
        before_dialogue = paragraph[last_end:dialogue_match.start()].strip()
        
        # Get the dialogue with quotes
        dialogue_text = dialogue_match.group(0)
        dialogue_start = dialogue_match.start()
        dialogue_end = dialogue_match.end()
        
        # Look for attribution after the dialogue
        # Common attribution patterns: "he said", "she cried", "Jack whispered", etc.
        attribution_pattern = r'^\s*([A-Z][a-zA-Z]*\.?\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*(?:\s+(?:said|cried|whispered|shouted|asked|replied|continued|added|muttered|declared|announced|exclaimed|sobbed|laughed|sighed|nodded|shook|clutched|tugged|looked|turned|moved|went|came|walked|ran|stood|sat|knelt|rose|smiled|frowned|gasped|breathed|swallowed|attempted|tried|began|finished)(?:\s+[a-zA-Z]+)*)?)'
        
        # Look ahead for attribution (up to 150 characters)
        text_after_dialogue = paragraph[dialogue_end:dialogue_end + 150]
        attribution_match = re.match(attribution_pattern, text_after_dialogue)
        
        # Look for attribution before the dialogue (within last 100 characters of before_dialogue)
        before_attribution_pattern = r'([A-Z][a-zA-Z]*\.?\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s*$'
        before_attribution_match = None
        if before_dialogue:
            before_attribution_match = re.search(before_attribution_pattern, before_dialogue[-100:])
        
        # Process text before dialogue
        if before_dialogue:
            if before_attribution_match:
                # Split the text, keeping attribution with dialogue
                attribution_start_in_before = before_dialogue.rfind(before_attribution_match.group(1))
                pre_attribution_text = before_dialogue[:attribution_start_in_before].strip()
                
                if pre_attribution_text:
                    # Split the pre-attribution text if it's long
                    pre_segments = _split_long_text(pre_attribution_text)
                    segments.extend(pre_segments)
            else:
                # No attribution before, split the before_dialogue text normally
                before_segments = _split_long_text(before_dialogue)
                segments.extend(before_segments)
        
        # Create the dialogue segment
        dialogue_segment = ""
        
        # Add preceding attribution if found
        if before_attribution_match:
            dialogue_segment += before_attribution_match.group(1) + " "
        
        # Add the dialogue
        dialogue_segment += dialogue_text
        
        # Add following attribution if found
        if attribution_match:
            dialogue_segment += " " + attribution_match.group(1).strip()
            last_end = dialogue_end + attribution_match.end()
        else:
            last_end = dialogue_end
        
        segments.append(dialogue_segment.strip())
    
    # Handle any remaining text after the last dialogue
    remaining_text = paragraph[last_end:].strip()
    if remaining_text:
        remaining_segments = _split_long_text(remaining_text)
        segments.extend(remaining_segments)
    
    # If no dialogues were found, just split the paragraph normally
    if not dialogues:
        return _split_long_text(paragraph)
    
    return segments


def _split_long_text(text, max_length=400):
    """Split long text on sentence boundaries, preserving meaning."""
    import re
    
    if len(text) <= max_length:
        return [text.strip()] if text.strip() else []
    
    # Split on sentence endings, but be careful with abbreviations
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences into reasonable chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence would exceed max_length
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def _combine_short_segments(segments, min_length=40, max_length=500):
    """Combine very short segments while keeping segments under max_length."""
    if not segments:
        return []
    
    combined = []
    current_segment = ""
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # If adding this segment would exceed max_length, finalize current segment
        if current_segment and len(current_segment) + len(segment) + 1 > max_length:
            combined.append(current_segment)
            current_segment = segment
        # If current segment is too short, try to combine with next
        elif len(current_segment) < min_length and len(current_segment) + len(segment) + 1 <= max_length:
            if current_segment:
                current_segment += " " + segment
            else:
                current_segment = segment
        # If the current segment is good length, add it and start new
        else:
            if current_segment:
                combined.append(current_segment)
            current_segment = segment
    
    # Add the last segment
    if current_segment:
        combined.append(current_segment)
    
    return combined


async def generate_speech_chunks(text_chunks: list[str], voice: str, 
                               temperature: float = None, top_p: float = None,
                               repetition_penalty: float = None, max_tokens: int = None) -> tuple[list[bytes], dict]:
    """
    Generate speech tokens for multiple text chunks in parallel using direct async vLLM access.
    Returns combined tokens and detailed metadata.
    """
    total_metadata = {
        "total_chunks": len(text_chunks),
        "chunk_details": [],
        "combined_stats": {}
    }
    
    logger.info(f"Starting parallel processing of {len(text_chunks)} text chunks using direct async vLLM access")
    
    # Generate tokens for all chunks in parallel using direct async access
    chunk_tasks = []
    for i, chunk in enumerate(text_chunks):
        logger.info(f"Queuing chunk {i+1}/{len(text_chunks)}: {len(chunk)} characters")
        task = generate_speech_tokens_direct(chunk, voice, temperature, top_p, repetition_penalty, max_tokens)
        chunk_tasks.append(task)
    
    # Wait for all chunks to complete
    chunk_results = await asyncio.gather(*chunk_tasks)
    
    # Process results and collect metadata
    all_chunk_results = []
    for i, (chunk_tokens, chunk_metadata) in enumerate(chunk_results):
        all_chunk_results.append(chunk_tokens)
        
        # Add chunk-specific metadata
        chunk_info = {
            "chunk_index": i,
            "text_length": len(text_chunks[i]),
            "audio_chunks": chunk_metadata["chunk_count"],
            "duration_seconds": chunk_metadata["duration_seconds"],
            "audio_frames": chunk_metadata["total_audio_frames"],
            "request_id": chunk_metadata["request_id"]
        }
        total_metadata["chunk_details"].append(chunk_info)
        
        logger.info(f"Chunk {i+1} complete: {chunk_metadata['chunk_count']} audio chunks, "
                   f"{chunk_metadata['duration_seconds']}s duration")
    
    # Combine all token chunks
    combined_tokens = combine_token_chunks(all_chunk_results)
    
    # Calculate combined statistics
    total_duration = sum(chunk["duration_seconds"] for chunk in total_metadata["chunk_details"])
    total_frames = sum(chunk["audio_frames"] for chunk in total_metadata["chunk_details"])
    total_audio_chunks = sum(chunk["audio_chunks"] for chunk in total_metadata["chunk_details"])
    
    total_metadata["combined_stats"] = {
        "total_text_length": sum(len(chunk) for chunk in text_chunks),
        "total_audio_chunks": total_audio_chunks,
        "total_duration_seconds": round(total_duration, 2),
        "total_audio_frames": total_frames,
        "voice": voice
    }
    
    logger.info(f"Generated and combined {len(text_chunks)} text chunks into {len(combined_tokens)} audio chunks in parallel, "
               f"total duration: {total_duration:.2f}s")
    
    return combined_tokens, total_metadata

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest, background_tasks: BackgroundTasks):
    """
    Create speech from text using OpenAI-compatible API.
    
    Generates audio from the input text using the Orpheus TTS model.
    Automatically chunks long text and combines audio at token level for seamless playback.
    Returns audio file in WAV format.
    
    Features:
    - Automatic text chunking for long inputs
    - Token-level audio combination for seamless playback
    - Actual vs estimated token counting
    - Detailed logging of chunk processing
    """
    try:
        print("Got request: ", request.input)
        # Validate input
        if not request.input.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "validation_error",
                    "message": "Input text cannot be empty",
                    "type": "invalid_request_error",
                }
            )
        

        logger.info(f"Processing TTS request: {len(request.input)} chars, voice: {request.voice}")

        # Generate unique filename
        unique_id = uuid.uuid4()
        output_path = f"outputs/{request.voice}_{unique_id}.wav"
        
        # Generate speech
        start_time = time.time()
        try:
            if len(request.input) > 500:
                # Split the text into sentences
                sentences = split_text_into_sentences(request.input)
                print(f"Split text into {len(sentences)} segments")
                
                # Create batches by combining sentences up to max_batch_chars
                batches = []
                current_batch = ""
                
                for sentence in sentences:
                    # If adding this sentence would exceed the batch size, start a new batch
                    if len(current_batch) + len(sentence) > 500 and current_batch:
                        batches.append(current_batch)
                        current_batch = sentence
                    else:
                        # Add separator space if needed
                        if current_batch:
                            current_batch += " "
                        current_batch += sentence
                
                # Add the last batch if it's not empty
                if current_batch:
                    batches.append(current_batch)
                
                print(f"Created {len(batches)} batches for processing")
                
                # Generate tokens for all chunks and combine them
                combined_tokens, metadata = await generate_speech_chunks(batches, request.voice, 
                                                                       request.temperature, request.top_p, 
                                                                       request.repetition_penalty, request.max_tokens)

                # Convert combined tokens to audio file
                file_stats = await tokens_to_audio_file(combined_tokens, output_path)

                # Log detailed results
                logger.info(f"Chunked processing complete:")
            else:
                # Single processing for shorter text using direct async access
                token_chunks, metadata = await generate_speech_tokens_direct(request.input, request.voice, 
                                                                            request.temperature, request.top_p, 
                                                                            request.repetition_penalty, request.max_tokens)
                
                # Convert tokens to audio file
                file_stats = await tokens_to_audio_file(token_chunks, output_path)
                
                # Combine metadata
                result = {**metadata, **file_stats}
                
                # Log single processing results
                logger.info(f"Single processing complete: {result}")
        except Exception as e:
            logger.error(f"Error during TTS generation: {e}")
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "processing_error",
                    "message": f"TTS generation failed: {str(e)}",
                    "type": "server_error",
                }
            )
        
        end_time = time.time()
        generation_time = round(end_time - start_time, 2)
        
        # Verify the output file was created
        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "processing_error", 
                    "message": "Audio file generation failed",
                    "type": "server_error",
                }
            )
        
        logger.info(f"TTS generation completed in {generation_time}s, returning file: {output_path}")
        
        # Schedule file cleanup after response is sent
        background_tasks.add_task(cleanup_file, output_path)
        
        # Return audio file
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=f"{request.voice}_{unique_id}.wav",
            headers={
                "Content-Disposition": f"attachment; filename={request.voice}_{unique_id}.wav"
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            }
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "type": "server_error",
            }
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": engine is not None,
        "timestamp": time.time()
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Orpheus TTS API",
        "version": "1.0.0",
        "description": "OpenAI-compatible Text-to-Speech API",
        "endpoints": {
            "speech": "/v1/audio/speech",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=1234,
        reload=False,
        log_level="info",
        timeout_keep_alive=3600,  # Keep connection alive for 5 minutes
    ) 