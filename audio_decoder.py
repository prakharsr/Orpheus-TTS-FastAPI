# This file is a modified version of the orpheus_tts.decoder.py file. Some improvements were made to the error handling of failed audio decoding.

import torch
import numpy as np
import logging
from typing import Optional, AsyncGenerator, Union
from snac import SNAC

logger = logging.getLogger(__name__)

# Custom exceptions for audio decoding errors
class AudioDecodingError(Exception):
    """Base exception for audio decoding errors"""
    pass

class InsufficientTokensError(AudioDecodingError):
    """Raised when there are not enough tokens to decode audio"""
    pass

class InvalidTokenRangeError(AudioDecodingError):
    """Raised when tokens are outside the valid range (0-4096)"""
    pass

class TokenParsingError(AudioDecodingError):
    """Raised when token parsing fails"""
    pass

class TokenFormatError(AudioDecodingError):
    """Raised when tokens are not in the expected format"""
    pass

class TokenRepetitionError(AudioDecodingError):
    """Raised when repetitive token patterns are detected that cause audio artifacts"""
    pass

# Global SNAC model variable
_snac_model = None
_snac_device = None

def initialize_snac_model(device: str = "cuda") -> None:
    """Initialize the SNAC model for audio decoding"""
    global _snac_model, _snac_device
    
    if _snac_model is None:
        logger.info(f"Loading SNAC model on device: {device}")
        _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        _snac_device = device
        _snac_model = _snac_model.to(_snac_device)
        logger.info("SNAC model loaded successfully")

def shutdown_snac_model() -> None:
    """Shutdown the SNAC model"""
    global _snac_model, _snac_device
    
    if _snac_model is not None:
        logger.info("Shutting down SNAC model")
        _snac_model = None
        _snac_device = None

def convert_to_audio(multiframe: list[int], count: int) -> bytes:
    """Convert multiframe tokens to audio bytes with proper error handling"""
    if _snac_model is None:
        raise RuntimeError("SNAC model not initialized. Call initialize_snac_model() first.")
    
    if len(multiframe) < 7:
        logger.error(f"DEBUG: Not enough tokens to decode! Got {len(multiframe)} tokens, need at least 7")
        raise InsufficientTokensError(f"Not enough tokens to decode audio: {len(multiframe)} < 7")
    
    codes_0 = torch.tensor([], device=_snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=_snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=_snac_device, dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]

    for j in range(num_frames):
        i = 7*j
        if codes_0.shape[0] == 0:
            codes_0 = torch.tensor([frame[i]], device=_snac_device, dtype=torch.int32)
        else:
            codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=_snac_device, dtype=torch.int32)])

        if codes_1.shape[0] == 0:
            codes_1 = torch.tensor([frame[i+1]], device=_snac_device, dtype=torch.int32)
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=_snac_device, dtype=torch.int32)])
        else:
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=_snac_device, dtype=torch.int32)])
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=_snac_device, dtype=torch.int32)])
        
        if codes_2.shape[0] == 0:
            codes_2 = torch.tensor([frame[i+2]], device=_snac_device, dtype=torch.int32)
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=_snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=_snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=_snac_device, dtype=torch.int32)])
        else:
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=_snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=_snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=_snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=_snac_device, dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    
    # Check that all tokens are between 0 and 4096
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        
        # Find specific out-of-range values for better error reporting
        invalid_tokens = []
        for i, code_tensor in enumerate(codes):
            invalid_mask = (code_tensor < 0) | (code_tensor > 4096)
            if torch.any(invalid_mask):
                invalid_values = code_tensor[invalid_mask].tolist()
                invalid_tokens.append(f"codes_{i}: {invalid_values}")
        
        error_msg = f"Some tokens are out of range (0-4096): {'; '.join(invalid_tokens)}"
        logger.error(f"DEBUG: {error_msg}")
        raise InvalidTokenRangeError(error_msg)

    with torch.inference_mode():
        audio_hat = _snac_model.decode(codes)
    
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    return audio_bytes

def turn_token_into_id(token_string: str, index: int) -> int:
    """Convert token string to ID with proper error handling"""
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        logger.error(f"DEBUG: No token found in the string: '{token_string}'")
        raise TokenParsingError(f"No custom token found in string: '{token_string}'")
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError as e:
            logger.error(f"DEBUG: Value error in token conversion: {e}, token: '{last_token}'")
            raise TokenParsingError(f"Failed to parse token number from '{last_token}': {e}")
    else:
        logger.error(f"DEBUG: Token not in expected format: '{last_token}'")
        raise TokenFormatError(f"Token not in expected format: '{last_token}'")

def check_token_repetition(tokens: list[str], max_tokens: int = 4096) -> None:
    """
    Check for repetitive token patterns that cause audio artifacts.
    Uses adaptive detection based on the actual max_tokens limit and generation progress.
    
    Args:
        tokens: List of raw token strings from the language model
        max_tokens: Maximum tokens configured for generation
        
    Raises:
        TokenRepetitionError: When repetitive patterns are detected
    """
    logger.debug(f"Checking {len(tokens)} tokens for repetition patterns (max_tokens: {max_tokens})")
    
    # Convert tokens to token IDs for pattern analysis
    token_id_sequence = []
    count = 0
    
    for token_string in tokens:
        try:
            token_id = turn_token_into_id(token_string, count)
            if token_id > 0:
                token_id_sequence.append(token_id)
                count += 1
        except (TokenParsingError, TokenFormatError):
            # Skip invalid tokens, just like in tokens_decoder
            continue
    
    logger.debug(f"Converted {len(token_id_sequence)} valid token IDs for repetition analysis")
    
    # Only check for repetition if we have a significant number of tokens
    # Based on observation: repetition started around token 150, so check when we have >100 tokens
    MIN_TOKENS_FOR_REPETITION_CHECK = max(100, max_tokens // 40)  # At least 100 or 2.5% of max_tokens
    
    if len(token_id_sequence) < MIN_TOKENS_FOR_REPETITION_CHECK:
        logger.debug(f"Skipping repetition check: {len(token_id_sequence)} tokens < {MIN_TOKENS_FOR_REPETITION_CHECK} minimum")
        return
    
    # Adaptive configuration based on max_tokens and current generation progress
    generation_progress = len(token_id_sequence) / max_tokens
    
    # Dynamic pattern detection configuration based on max_tokens
    # Scale pattern lengths with max_tokens to handle different environments
    MIN_PATTERN_LENGTH = max(10, max_tokens // 400)     # 10 for 4096 tokens, scales up for larger limits
    MAX_PATTERN_LENGTH = max(50, max_tokens // 80)      # 50 for 4096 tokens, scales up for larger limits
    
    # Ensure reasonable bounds
    MIN_PATTERN_LENGTH = min(MIN_PATTERN_LENGTH, 50)    # Cap at 50 to avoid excessive computation
    MAX_PATTERN_LENGTH = min(MAX_PATTERN_LENGTH, 200)   # Cap at 200 to avoid excessive computation
    
    # Look at a larger window for long generations (up to 50% of generation)
    REPETITION_WINDOW = min(len(token_id_sequence), max(500, int(len(token_id_sequence) * 0.5)))
    
    # Adaptive repetition threshold based on generation progress and max_tokens
    # Higher base thresholds for more conservative detection
    base_threshold = max(8, max_tokens // 500)  # 8 for 4096 tokens, scales up for larger limits
    
    if generation_progress < 0.3:  # Early in generation
        REPETITION_THRESHOLD = base_threshold + 5  # More conservative
    elif generation_progress < 0.7:  # Mid generation
        REPETITION_THRESHOLD = base_threshold + 2  # Moderate
    else:  # Late in generation (likely approaching limit)
        REPETITION_THRESHOLD = max(5, base_threshold - 2)  # More aggressive for late-stage
    
    logger.debug(f"Repetition detection config: window={REPETITION_WINDOW}, pattern_len={MIN_PATTERN_LENGTH}-{MAX_PATTERN_LENGTH}, threshold={REPETITION_THRESHOLD}, progress={generation_progress:.2%}, max_tokens={max_tokens}")
    
    # Check for repetition patterns
    recent_tokens = token_id_sequence[-REPETITION_WINDOW:]
    
    # Check for repeating patterns of different lengths
    for pattern_length in range(MIN_PATTERN_LENGTH, min(MAX_PATTERN_LENGTH + 1, REPETITION_WINDOW // 2 + 1)):
        if len(recent_tokens) >= pattern_length * REPETITION_THRESHOLD:
            # Extract the pattern from the end and check if it repeats
            pattern = recent_tokens[-pattern_length:]
            repetitions = 0
            
            # Count consecutive repetitions at the end
            for i in range(len(recent_tokens) - pattern_length, -1, -pattern_length):
                if i >= 0 and i + pattern_length <= len(recent_tokens):
                    if recent_tokens[i:i + pattern_length] == pattern:
                        repetitions += 1
                    else:
                        break
            
            if repetitions >= REPETITION_THRESHOLD:
                logger.error(f"🔄 REPETITION DETECTED: Pattern length {pattern_length} repeated {repetitions} times")
                logger.error(f"🔄 Pattern: {pattern}")
                logger.error(f"🔄 Generation progress: {generation_progress:.1%} ({len(token_id_sequence)}/{max_tokens} tokens)")
                logger.error(f"🔄 Recent token sequence: {token_id_sequence[-min(50, len(token_id_sequence)):]}")
                raise TokenRepetitionError(f"Repetitive pattern detected: {pattern_length}-token pattern repeated {repetitions} times at {generation_progress:.1%} progress")
    
    # Check if we're close to max_tokens limit (likely repetition issue)
    HIGH_TOKEN_THRESHOLD = 0.85  # 85% of max_tokens - catch earlier
    if generation_progress > HIGH_TOKEN_THRESHOLD:
        logger.error(f"🔄 APPROACHING TOKEN LIMIT: {len(token_id_sequence)} tokens ({generation_progress:.1%} of {max_tokens} max)")
        logger.error(f"🔄 This likely indicates repetition causing excessive generation")
        logger.error(f"🔄 Recent token IDs: {token_id_sequence[-30:]}")
        raise TokenRepetitionError(f"Approaching token limit: {len(token_id_sequence)} tokens ({generation_progress:.1%} of max), likely repetition issue")
    
    logger.debug(f"No repetition patterns detected in {len(token_id_sequence)} tokens ({generation_progress:.1%} progress)")

async def tokens_decoder(tokens: list[str]) -> AsyncGenerator[bytes, None]:
    """
    Decode tokens into audio bytes with proper error handling.
    
    Args:
        tokens: List of token strings from the language model
        
    Yields:
        Audio bytes chunks
        
    Raises:
        AudioDecodingError: When token decoding fails
    """
    buffer = []
    count = 0
    
    logger.debug(f"Starting token decoding for {len(tokens)} tokens")
    
    for token_string in tokens:
        try:
            token_id = turn_token_into_id(token_string, count)
            
            if token_id > 0:
                buffer.append(token_id)
                count += 1

                # Process buffer when we have enough tokens
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    try:
                        audio_samples = convert_to_audio(buffer_to_proc, count)
                        if audio_samples is not None:
                            yield audio_samples
                    except AudioDecodingError as e:
                        # Re-raise audio decoding errors to trigger retry
                        logger.error(f"Audio decoding failed at count {count}: {e}")
                        raise
                        
        except (TokenParsingError, TokenFormatError) as e:
            # Log token parsing errors but continue processing
            logger.info(f"Token parsing error at count {count}: {e}")
            continue
    
    logger.debug(f"Token decoding completed. Processed {count} valid tokens") 