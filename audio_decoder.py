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