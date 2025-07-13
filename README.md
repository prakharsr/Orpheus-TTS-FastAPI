# Orpheus TTS FastAPI Server (Async)

A high-performance FastAPI-based server that provides OpenAI-compatible Text-to-Speech (TTS) endpoints using the [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) model with async parallel processing. This project uses the original `orpheus-speech` python package with vLLM backend, loading the model in bfloat16 by default (with float16/float32 options). Using higher precision formats requires more VRAM but eliminates audio quality issues and artifacts commonly found in quantized models or alternative inference engines. The server supports async parallel chunk processing for significantly faster audio generation.

## 🚀 Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's TTS API
- **Async Parallel Processing**: Process multiple text chunks simultaneously for faster generation
- **Direct vLLM Integration**: Uses vLLM's AsyncLLMEngine for optimal performance
- **Text Chunking**: Automatic intelligent text splitting for long content
- **Complete Audio Files**: Returns complete WAV files optimized for quality
- **Intelligent Retry Logic**: Automatic retry on audio decoding errors for improved reliability
- **Token Repetition Detection**: Prevents infinite audio loops with adaptive pattern detection and automatic retry with adjusted parameters

## 🔧 Architecture

### Async Processing Pipeline
```
┌─────────────────────────────────────────────────┐
│ FastAPI Request Handler (Async)                 │
├─────────────────────────────────────────────────┤
│ Text Chunking & Batching (Intelligent)          │
├─────────────────────────────────────────────────┤
│ Parallel Token Generation (vLLM AsyncEngine)    │
├─────────────────────────────────────────────────┤
│ Token Repetition Detection        │
├─────────────────────────────────────────────────┤
│ Token Decoding (Async)                          │
├─────────────────────────────────────────────────┤
│ Audio File Generation (Async I/O)               │
└─────────────────────────────────────────────────┘
```

### Key Improvements Over Sync Version
- **4x faster** for long texts (parallel chunk processing)
- **Non-blocking operations** throughout the pipeline
- **Better resource utilization** with optimized thread pools
- **GPU memory monitoring** and automatic optimization
- **Robust error handling** with token repetition detection to prevent infinite audio loops

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

Generate speech from text with automatic parallel processing for long texts.

**Request Body:**
```json
{
  "model": "orpheus",         // Model to use
  "input": "Text to speak",   // Text to synthesize (auto-chunked if long)
  "voice": "tara",           // Voice name (see voice options above)
  "response_format": "wav",   // Audio format (currently only "wav")
  "speed": 1.0,              // Speech speed (0.25 to 4.0)
  
  // Optional: Override environment defaults for sampling
  "temperature": 0.2,         // Sampling temperature (0.0-1.0, optional)
  "top_p": 0.9,              // Top-p nucleus sampling (0.0-1.0, optional)
  "repetition_penalty": 1.1,  // Repetition penalty (0.5-2.0, optional)
  "max_tokens": 4096         // Maximum tokens to generate (100-8192, optional)
}
```

**Features:**
- **Automatic Text Chunking**: Long texts (>500 chars) are intelligently split
- **Parallel Processing**: Multiple chunks processed simultaneously
- **Seamless Audio**: Combined into single WAV file
- **Smart Batching**: Preserves dialogue and narrative flow
- **Configurable Parameters**: Override defaults per request
- **Environment Fallbacks**: Uses `.env` values when parameters not specified

**Response:**
- Content-Type: `audio/wav`
- Body: Binary audio data

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

3. **Slow Performance**
   - Increase `TTS_MAX_WORKERS` (try 16+)
   - Ensure GPU is being used (`nvidia-smi`)
   - Check if text is being chunked properly
   - Verify `TTS_GPU_MEMORY_UTILIZATION` isn't too low

4. **Engine Shutdown Errors**
   - Ensure using V0 engine: `VLLM_USE_V1="0"`
   - Don't overload with too many parallel requests
   - Monitor GPU memory usage
   - Check `TTS_TENSOR_PARALLEL_SIZE` matches available GPUs

5. **Poor Audio Quality**
   - Try `TTS_DTYPE="float32"` for highest quality
   - Adjust `TTS_TEMPERATURE` (lower = more consistent)
   - Tune `TTS_REPETITION_PENALTY` (higher = less repetitive)
   - Check `TTS_TOP_P` settings (0.8-0.95 range)
