from typing import Optional, Literal
import time
import logging
import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import uvicorn

# Force V0 engine to avoid async issues
os.environ["VLLM_USE_V1"] = "0"

from dotenv import load_dotenv

# Import our new modules
from audio_decoder import initialize_snac_model, shutdown_snac_model
from audio_generator import (
    OrpheusModelExtended, 
    generate_speech_tokens_direct, 
    generate_speech_chunks,
    tokens_to_audio_file
)
from text_processor import split_text_into_sentences, create_batches

load_dotenv()

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
        logger.info(f"Unknown dtype '{dtype_str}', defaulting to bfloat16")
        return torch.bfloat16
    return dtype_map[dtype_str]

# Model Configuration
MODEL_NAME = os.getenv("TTS_MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod")
DTYPE = get_dtype_from_string(os.getenv("TTS_DTYPE", "bfloat16"))
MAX_MODEL_LEN = int(os.getenv("TTS_MAX_MODEL_LEN", "8192"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TTS_TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.getenv("TTS_GPU_MEMORY_UTILIZATION", "0.9"))

# Performance Configuration  
MAX_WORKERS = int(os.getenv("TTS_MAX_WORKERS", "16"))

# SNAC Configuration
SNAC_DEVICE = os.getenv("SNAC_DEVICE", "cuda")

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
logger.info(f"   SNAC Device: {SNAC_DEVICE}")
logger.info(f"   Temperature: {DEFAULT_TEMPERATURE}")
logger.info(f"   Top P: {DEFAULT_TOP_P}")
logger.info(f"   Repetition Penalty: {DEFAULT_REPETITION_PENALTY}")
logger.info(f"   Max Tokens: {DEFAULT_MAX_TOKENS}")

# Global engine variable
engine = None
executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("🚀 Loading Orpheus TTS model...")
    global engine, executor

    # Load SNAC model for audio decoding
    initialize_snac_model(device=SNAC_DEVICE)

    # Load Orpheus TTS model
    engine = OrpheusModelExtended(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION
    )
    
    # Create thread pool executor for file I/O only (no more token decoding)
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
    shutdown_snac_model()
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
        timeout = 600.0
    else:
        # For other endpoints, use default timeout (30 seconds)
        timeout = 30.0
    
    try:
        # Process request with timeout
        response = await asyncio.wait_for(call_next(request), timeout=timeout)
        
        # Log request duration
        process_time = time.time() - start_time
        # logger.info(f"Request {request.method} {request.url.path} completed in {process_time:.2f}s")
        
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

def cleanup_file(file_path: str) -> None:
    """Clean up a single file after serving"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            # logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.info(f"Failed to cleanup file {file_path}: {e}")

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
    - Fully async processing with no threading overhead
    - Intelligent retry logic for audio decoding errors
    - Detailed logging of chunk processing and retry statistics
    - Configurable retry attempts and delays
    """
    try:
        # logger.info(f"Got request: {request.input}")
        
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

        # logger.info(f"Processing TTS request: {len(request.input)} chars, voice: {request.voice}")

        # Generate unique filename
        unique_id = uuid.uuid4()
        output_path = f"outputs/{request.voice}_{unique_id}.wav"
        
        # Generate speech
        start_time = time.time()
        try:
            if len(request.input) > 500:
                # Split the text into sentences
                sentences = split_text_into_sentences(request.input)
                # logger.info(f"Split text into {len(sentences)} segments")
                
                # Create batches by combining sentences up to max_batch_chars
                batches = create_batches(sentences, max_batch_chars=500)
                # logger.info(f"Created {len(batches)} batches for processing")
                
                # Generate tokens for all chunks and combine them
                combined_tokens, metadata = await generate_speech_chunks(
                    engine, batches, request.voice, 
                    request.temperature, request.top_p, 
                    request.repetition_penalty, request.max_tokens,
                    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY, DEFAULT_MAX_TOKENS
                )

                # Convert combined tokens to audio file
                if executor is None:
                    raise HTTPException(status_code=500, detail="Executor not initialized")
                file_stats = await tokens_to_audio_file(combined_tokens, output_path, executor)

                # Log detailed results including retry statistics
                retry_stats = metadata.get("retry_stats", {})
                # logger.info(f"Chunked processing complete - Total attempts: {retry_stats.get('total_attempts', 0)}, "
                #            f"Total retries: {retry_stats.get('total_retries', 0)}, "
                #            f"Failed chunks: {retry_stats.get('failed_chunks', 0)}")
            else:
                # Single processing for shorter text using direct async access
                token_chunks, metadata = await generate_speech_tokens_direct(
                    engine, request.input, request.voice, 
                    request.temperature, request.top_p, 
                    request.repetition_penalty, request.max_tokens,
                    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY, DEFAULT_MAX_TOKENS
                )
                
                # Convert tokens to audio file
                if executor is None:
                    raise HTTPException(status_code=500, detail="Executor not initialized")
                file_stats = await tokens_to_audio_file(token_chunks, output_path, executor)
                
                # Combine metadata
                result = {**metadata, **file_stats}
                
                # Log single processing results including retry information
                attempts = metadata.get("attempts", 1)
                retries = metadata.get("retries", 0)
                # logger.info(f"Single processing complete - Duration: {file_stats['duration_seconds']:.2f}s, "
                #            f"Attempts: {attempts}, Retries: {retries}")
                
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
        
        # logger.info(f"TTS generation completed in {generation_time}s, returning file: {output_path}")
        
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
        logger.info(f"Validation error: {str(e)}")
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
        "fastapi_app:app",
        host="0.0.0.0",
        port=8880,
        reload=False,
        log_level="info",
        timeout_keep_alive=600,  # Keep connection alive for 10 minutes
    ) 