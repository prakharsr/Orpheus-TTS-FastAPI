import asyncio
import uuid
import os
import wave
import logging
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI
import torch
from audio_decoder import tokens_decoder, AudioDecodingError, TokenRepetitionError, check_token_repetition
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-tts-0.1-finetune-prod")

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY = 1.0

class OrpheusVLLMClient:
    """Client for communicating with vLLM server running Orpheus model"""
    
    def __init__(self, base_url: str, api_key: str, model_name: str = "orpheus"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        
    async def health_check(self) -> bool:
        """Check if the vLLM server is healthy"""
        try:
            # Try to list models as a health check
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.error(f"vLLM server health check failed: {e}")
            return False

async def generate_tokens_async(client: OrpheusVLLMClient, prompt: str, voice: str, request_id: str, 
                               temperature: Optional[float] = None, top_p: Optional[float] = None, 
                               repetition_penalty: Optional[float] = None, max_tokens: Optional[int] = None,
                               default_temperature: float = 0.2, default_top_p: float = 0.9,
                               default_repetition_penalty: float = 1.1, default_max_tokens: int = 4096) -> list[str]:
    """Generate tokens using the vLLM server via OpenAI API"""
    
    # Format prompt using the same logic as OrpheusModel
    # Note: The tokenization and special tokens will be handled by the vLLM server
    adapted_prompt = f"{voice}: {prompt}"
    prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
    prompt_string = tokenizer.decode(all_input_ids[0])
    
    # Use provided parameters or fall back to defaults
    temperature = temperature if temperature is not None else default_temperature
    top_p = top_p if top_p is not None else default_top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else default_repetition_penalty
    max_tokens = max_tokens if max_tokens is not None else default_max_tokens
    
    logger.debug(f"Using sampling params - temp: {temperature}, top_p: {top_p}, "
                f"max_tokens: {max_tokens}, rep_penalty: {repetition_penalty}")
    
    # Create completion request
    try:
        completion = await client.client.completions.create(
            model=client.model_name,
            prompt=prompt_string,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            extra_body={
                "repeat_penalty": repetition_penalty,
                "repetition_penalty": repetition_penalty,
                "stop_token_ids": [128258]
            }
        )
        
        # Extract the generated text
        generated_text = completion.choices[0].text
        split_tokens = generated_text.split(">")
        all_tokens = []
        for token in split_tokens:
            if token.strip():
                token_text = f'{token.strip()}>'
                all_tokens.append(token_text)
        return all_tokens  # Return as list to maintain compatibility
        
    except Exception as e:
        logger.error(f"Error calling vLLM server for request {request_id}: {e}")
        raise

async def generate_speech_tokens_with_retry(client: OrpheusVLLMClient, prompt: str, voice: str, request_id: str, 
                                          temperature: Optional[float] = None, top_p: Optional[float] = None,
                                          repetition_penalty: Optional[float] = None, max_tokens: Optional[int] = None,
                                          default_temperature: float = 0.2, default_top_p: float = 0.9,
                                          default_repetition_penalty: float = 1.1, default_max_tokens: int = 4096) -> tuple[list[bytes], dict]:
    """Generate speech tokens with retry logic for audio decoding errors"""
    
    last_error = None
    current_repetition_penalty = repetition_penalty  # Track current penalty for adjustments
    
    for attempt in range(MAX_RETRIES):
        try:
            # Generate tokens using vLLM server
            tokens = await generate_tokens_async(client, prompt, voice, request_id, 
                                               temperature, top_p, current_repetition_penalty, max_tokens,
                                               default_temperature, default_top_p, default_repetition_penalty, default_max_tokens)
            
            # Check for repetition patterns BEFORE audio generation (lightweight step)
            effective_max_tokens = max_tokens if max_tokens is not None else default_max_tokens
            check_token_repetition(tokens, effective_max_tokens)
            
            # Try to decode tokens to audio chunks
            audio_chunks = []
            async for audio_chunk in tokens_decoder(tokens):
                audio_chunks.append(audio_chunk)
            
            # If we get here, decoding was successful
            chunk_count = len(audio_chunks)
            total_audio_frames = sum(len(chunk) // 2 for chunk in audio_chunks if chunk)  # 16-bit audio
            duration = total_audio_frames / 24000  # 24kHz sample rate
            
            metadata = {
                "request_id": request_id,
                "chunk_count": chunk_count,
                "total_audio_frames": total_audio_frames,
                "duration_seconds": round(duration, 2),
                "voice": voice,
                "prompt_length": len(prompt),
                "attempts": attempt + 1,
                "retries": attempt
            }
            
            if attempt > 0:
                logger.info(f"✅ Token generation successful after {attempt + 1} attempts for request_id: {request_id}")
            
            return audio_chunks, metadata
            
        except TokenRepetitionError as e:
            last_error = e
            logger.error(f"🔄 Token repetition detected on attempt {attempt + 1}/{MAX_RETRIES} for request_id: {request_id}: {e}")
            
            if attempt < MAX_RETRIES - 1:  # Not the last attempt
                # For repetition errors, we might want to adjust sampling parameters
                logger.info(f"⏳ Retrying with adjusted parameters in {RETRY_DELAY} seconds...")
                
                # Increase repetition penalty for retry attempts
                if current_repetition_penalty is None:
                    current_repetition_penalty = default_repetition_penalty * (1.0 + 0.1 * (attempt + 1))  # Increase by 10% per retry
                else:
                    current_repetition_penalty = current_repetition_penalty * (1.0 + 0.1 * (attempt + 1))
                
                logger.info(f"🔧 Adjusted repetition penalty to {current_repetition_penalty:.2f} for retry")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"❌ All {MAX_RETRIES} attempts failed due to repetition for request_id: {request_id}")
                
        except AudioDecodingError as e:
            last_error = e
            logger.info(f"🔄 Audio decoding failed on attempt {attempt + 1}/{MAX_RETRIES} for request_id: {request_id}: {e}")
            
            if attempt < MAX_RETRIES - 1:  # Not the last attempt
                logger.info(f"⏳ Retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"❌ All {MAX_RETRIES} attempts failed for request_id: {request_id}")
                
        except Exception as e:
            # For other exceptions, don't retry - they're likely not transient
            logger.error(f"❌ Non-retryable error in token generation for request_id: {request_id}: {e}")
            raise
    
    # If we get here, all retries failed
    if isinstance(last_error, TokenRepetitionError):
        raise TokenRepetitionError(f"Token repetition persisted after {MAX_RETRIES} attempts. Last error: {last_error}")
    else:
        raise AudioDecodingError(f"Token generation failed after {MAX_RETRIES} attempts. Last error: {last_error}")

async def generate_speech_tokens_direct(client: OrpheusVLLMClient, prompt: str, voice: str, 
                                      temperature: Optional[float] = None, top_p: Optional[float] = None,
                                      repetition_penalty: Optional[float] = None, max_tokens: Optional[int] = None,
                                      default_temperature: float = 0.2, default_top_p: float = 0.9,
                                      default_repetition_penalty: float = 1.1, default_max_tokens: int = 4096) -> tuple[list[bytes], dict]:
    """Generate speech tokens using vLLM server with retry logic"""
    try:
        # Generate unique request ID
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        
        # logger.info(f"Starting speech token generation with retry logic for voice: {voice}, request_id: {request_id}")
        
        # Use retry logic
        audio_chunks, metadata = await generate_speech_tokens_with_retry(
            client, prompt, voice, request_id,
            temperature, top_p, repetition_penalty, max_tokens,
            default_temperature, default_top_p, default_repetition_penalty, default_max_tokens
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

async def generate_speech_chunks(client: OrpheusVLLMClient, text_chunks: list[str], voice: str, 
                               temperature: Optional[float] = None, top_p: Optional[float] = None,
                               repetition_penalty: Optional[float] = None, max_tokens: Optional[int] = None,
                               default_temperature: float = 0.2, default_top_p: float = 0.9,
                               default_repetition_penalty: float = 1.1, default_max_tokens: int = 4096) -> tuple[list[bytes], dict]:
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
            client, chunk, voice, request_id,
            temperature, top_p, repetition_penalty, max_tokens,
            default_temperature, default_top_p, default_repetition_penalty, default_max_tokens
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
    
    # If all chunks failed, raise an error
    if failed_chunks == len(text_chunks):
        raise AudioDecodingError(f"All {len(text_chunks)} chunks failed after retries")
    
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