# Orpheus TTS FastAPI Server (Async)

A high-performance, production-ready FastAPI-based server that provides OpenAI-compatible Text-to-Speech (TTS) endpoints using the [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) model with advanced error detection and async parallel processing. This project uses the original `orpheus-speech` python package with vLLM backend, loading the model in bfloat16 by default (with float16/float32 options). Using higher precision formats requires more VRAM but eliminates audio quality issues and artifacts commonly found in quantized models or alternative inference engines.

The server features sophisticated multi-stage error detection, adaptive retry logic, and comprehensive debugging tools to ensure reliable, high-quality audio generation. Supports async parallel chunk processing with intelligent text chunking that preserves dialogue and narrative flow.

## 🚀 Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's TTS API
- **Async Parallel Processing**: Process multiple text chunks simultaneously for faster generation
- **Direct vLLM Integration**: Uses vLLM's AsyncLLMEngine for optimal performance
- **Intelligent Text Chunking**: Preserves dialogue, attribution, and narrative flow
- **Complete Audio Files**: Returns complete WAV files optimized for quality
- **Advanced Error Detection**: Multi-stage analysis prevents audio artifacts and quality issues
- **Adaptive Retry Logic**: Up to 5 automatic retries with parameter adjustment
- **Token Repetition Detection**: Prevents infinite audio loops with pattern analysis
- **Audio Quality Analysis**: Detects silence, repetition, stretching, and monotonic audio
- **Duration Outlier Detection**: Identifies abnormally long audio generation
- **Debug & Success Logging**: Optional comprehensive logging for troubleshooting and tuning
- **SNAC Audio Decoding**: High-quality audio reconstruction from tokens

## 🔧 Architecture

### Modular Design
The server follows a modular architecture with specialized components:

- **`fastapi_app.py`**: Main FastAPI application with request handling, CORS, and timeout middleware
- **`audio_generator.py`**: Advanced audio generation with multi-stage error detection and retry logic
- **`audio_decoder.py`**: SNAC-based audio decoding with custom exception handling
- **`text_processor.py`**: Intelligent text chunking that preserves dialogue and narrative flow
- **`audio_analysis.py`**: Comprehensive audio quality analysis using spectrograms and cross-correlation

### Async Processing Pipeline
```
┌─────────────────────────────────────────────────┐
│ FastAPI Request Handler (Async)                 │
├─────────────────────────────────────────────────┤
│ Text Processing & Chunking (Intelligent)        │
├─────────────────────────────────────────────────┤
│ Parallel Token Generation (vLLM AsyncEngine)    │
├─────────────────────────────────────────────────┤
│ Multi-Stage Error Detection:                    │
│ • Token Repetition Detection                    │
│ • Token Count Ratio Analysis                    │
│ • Audio Duration Outlier Detection              │
│ • Audio Quality Analysis (Silence/Stretch/Mono) │
├─────────────────────────────────────────────────┤
│ Adaptive Retry Logic (5 attempts max)           │
├─────────────────────────────────────────────────┤
│ SNAC Audio Decoding                             │
├─────────────────────────────────────────────────┤
│ Audio File Generation (Async I/O)               │
├─────────────────────────────────────────────────┤
│ Debug/Success Logging (Optional)                │
└─────────────────────────────────────────────────┘
```

### Key Improvements Over Sync Version
- **4x faster** for long texts (parallel chunk processing)
- **Non-blocking operations** throughout the pipeline
- **Better resource utilization** with optimized thread pools
- **GPU memory monitoring** and automatic optimization
- **Advanced error detection** with multi-stage analysis to prevent audio artifacts
- **Intelligent retry logic** with adaptive parameter adjustment
- **Comprehensive debugging** with audio quality analysis and metadata logging

## 📋 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/v1/audio/speech` | POST | Generate speech from text (OpenAI-compatible) |
| `/health` | GET | Health check and model status |
| `/` | GET | API information and available endpoints |

### Interactive Documentation

- **Swagger UI**: `http://localhost:8880/docs`

## 🛠️ Installation

### Prerequisites

- CUDA-capable GPU (minimum 16GB VRAM recommended)
- Sufficient disk space for model downloads

### Install Dependencies

1. Install uv 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Create a virtual environment with Python 3.12:
```bash
uv venv --python 3.12
```
3. Activate the virtual environment:
```bash
source .venv/bin/activate
```
4. Install dependencies
```bash
# Install the required packages
uv pip install -r requirements.txt
```
5. Get access to the Orpheus model from [here](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)
6. Login to Huggingface using the CLI after gaining access to the repo
```bash
huggingface-cli login
```

## 🔧 Configuration

### Environment Variables

The server supports extensive configuration through environment variables. Copy `.env.sample` to `.env` and modify as needed:

```bash
cp .env.sample .env
# Edit .env with your preferred settings
```

## 🎯 Usage

### Starting the Server

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8880
```

The server will start on `http://localhost:8880` by default.

### Making Requests

#### Using cURL

```bash
# Generate speech
curl -X POST "http://localhost:8880/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Hello, this is a test of the Orpheus TTS API!",
    "voice": "tara",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

#### Using Python with requests

```python
import requests

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "model": "orpheus",
        "input": "Hello, this is a test!",
        "voice": "tara",
        "response_format": "wav",
        "speed": 1.0
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### Using OpenAI Client Library

```python
from openai import OpenAI

# Configure client to use local server
client = OpenAI(
    api_key="dummy-key",  # Not validated by local server
    base_url="http://localhost:8880"
)

response = client.audio.speech.create(
    model="orpheus",
    voice="tara",
    input="Hello, this is a test!"
)

with open("speech.wav", "wb") as f:
    f.write(response.content)
```

## 🎤 Voice Options

The API supports native Orpheus voice names:

### Available Voices
- `tara` (default) - neutral/balanced female
- `leah` - warm female voice
- `jess` - expressive female
- `leo` - deep male voice
- `dan` - male voice
- `mia` - young female voice
- `zac` - male voice
- `zoe` - female voice

## 🔧 API Reference

### POST /v1/audio/speech

Generate speech from text with automatic parallel processing and advanced error detection for long texts.

**Request Body:**
```json
{
  "model": "orpheus",         // Model to use
  "input": "Text to speak",   // Text to synthesize (auto-chunked if long)
  "voice": "tara",           // Voice name (see voice options above)
  "response_format": "wav",   // Audio format (currently only "wav")
  "speed": 1.0,              // Speech speed (0.25 to 4.0)

  // Optional: Override environment defaults for sampling
  "temperature": 0.2,         // Sampling temperature (0.0-2.0, optional)
  "top_p": 0.9,              // Top-p nucleus sampling (0.0-1.0, optional)
  "repetition_penalty": 1.1,  // Repetition penalty (0.5-2.0, optional)
  "max_tokens": 4096         // Maximum tokens to generate (100-8192, optional)
}
```

**Features:**
- **Automatic Text Chunking**: Long texts (>500 chars) are intelligently split preserving dialogue and narrative flow
- **Parallel Processing**: Multiple chunks processed simultaneously with individual retry logic
- **Multi-Stage Error Detection**: Advanced analysis prevents audio artifacts:
  - Token repetition detection (prevents infinite loops)
  - Token count ratio analysis (catches outlier generation)
  - Audio duration outlier detection (identifies abnormally slow generation)
  - Audio quality analysis (detects silence, repetition, stretching, monotonic audio)
- **Adaptive Retry Logic**: Up to 5 automatic retries with parameter adjustment for failed chunks
- **Seamless Audio**: Token-level combination creates single WAV file with natural flow
- **Debug Logging**: Optional comprehensive logging of generation process and error analysis
- **Configurable Parameters**: Override defaults per request with environment fallbacks

**Response:**
- Content-Type: `audio/wav`
- Body: Binary audio data

**Error Handling:**
The API implements sophisticated error detection and recovery:

- **Token-Level Errors**: Repetition patterns, count outliers, invalid ranges
- **Audio-Level Errors**: Duration outliers, quality issues (silence, repetition, stretching)
- **Adaptive Retries**: Automatic parameter adjustment (temperature, repetition_penalty) on retry attempts
- **Graceful Degradation**: Failed chunks are logged and skipped rather than failing entire request
- **Debug Output**: Failed generations saved to `debug_audio_errors/` for analysis

### GET /health

Health check endpoint with detailed system information.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1699896489.123
}
```

## 📄 License

This project follows the same license as the original Orpheus TTS repository.

## 🔗 Links

- [Original Orpheus TTS Repository](https://github.com/canopyai/Orpheus-TTS)
- [OpenAI TTS API Documentation](https://platform.openai.com/docs/guides/text-to-speech)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [vLLM Documentation](https://docs.vllm.ai/)

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Ensure sufficient GPU memory (min 14GB)
   - Check CUDA installation and compatibility
   - Verify internet connection for model downloads
   - Adjust `TTS_GPU_MEMORY_UTILIZATION` if OOM errors occur

2. **Out of Memory Errors**
   - Reduce `TTS_GPU_MEMORY_UTILIZATION` (try 0.8 or 0.7)
   - Decrease `TTS_MAX_MODEL_LEN` (try 6144 or 4096)
   - Switch to `TTS_DTYPE="float16"` for lower VRAM usage

3. **Token Repetition Errors**
   - Check debug logs in `debug_audio_errors/` for pattern analysis
   - Increase `TTS_REPETITION_PENALTY` (try 1.2-1.5)
   - Adjust `TTS_TEMPERATURE` (try 0.3-0.5 for more variety)
   - Reduce `TTS_MAX_TOKENS` if hitting generation limits
   - Enable debug logging to analyze token patterns

4. **Audio Duration Outliers**
   - Check `debug_audio_errors/` for duration analysis
   - Lower `TTS_TEMPERATURE` (try 0.1-0.2) for more consistent output
   - Increase `TTS_REPETITION_PENALTY` (try 1.3+) to prevent loops
   - Adjust `TTS_TOP_P` (try 0.7-0.8) for more focused generation
   - Review text for emotion tags that may extend duration

5. **Audio Quality Issues (Silence/Stretching/Repetition)**
   - Enable `ENABLE_DEBUG_SAVING=true` to save problematic audio
   - Check `debug_audio_errors/` for quality metrics
   - Adjust `TTS_TEMPERATURE` and `TTS_REPETITION_PENALTY`
   - Use audio analysis tool: `python audio_analysis.py <audio_file>`
   - Consider text preprocessing for emotion tags

6. **Slow Performance**
   - Increase `TTS_MAX_WORKERS` (try 16+)
   - Ensure GPU is being used (`nvidia-smi`)
   - Check if text is being chunked properly
   - Verify `TTS_GPU_MEMORY_UTILIZATION` isn't too low
   - Monitor chunk processing times in logs

7. **Engine Shutdown Errors**
   - Ensure using V0 engine: `VLLM_USE_V1="0"`
   - Don't overload with too many parallel requests
   - Monitor GPU memory usage
   - Check `TTS_TENSOR_PARALLEL_SIZE` matches available GPUs

8. **Poor Audio Quality**
   - Try `TTS_DTYPE="float32"` for highest quality
   - Adjust `TTS_TEMPERATURE` (lower = more consistent)
   - Tune `TTS_REPETITION_PENALTY` (higher = less repetitive)
   - Check `TTS_TOP_P` settings (0.8-0.95 range)
   - Enable success logging to analyze good generations

### Debug Tools

The server includes comprehensive debugging tools:

- **Debug Audio Saving**: Failed generations saved to `debug_audio_errors/` with metadata
- **Success Logging**: Successful generations saved to `debug_audio_success/` for tuning
- **Audio Analysis**: Use `python audio_analysis.py <file.wav>` for quality analysis
- **Token Inspection**: Check `debug_audio_success/` for reproducibility data
- **Metadata Logging**: Full generation statistics and error analysis

Enable debugging with environment variables:
```bash
ENABLE_DEBUG_SAVING=true
ENABLE_SUCCESS_LOGGING=true
LOG_LEVEL=DEBUG
```
