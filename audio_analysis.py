"""
Advanced audio analysis for detecting TTS generation issues.
Uses cross-correlation for repetition detection and spectrogram analysis for stretched audio.
Detects: repeated segments, stretched/slowed audio, silence, clipping, and energy anomalies.
"""

import numpy as np
import logging
import os
import wave
from pathlib import Path
from typing import Tuple, Dict, List
from scipy.signal import spectrogram, correlate
from dotenv import load_dotenv

load_dotenv()

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global flag to enable/disable audio segment extraction for debugging
ENABLE_SEGMENT_EXTRACTION = os.getenv("EXTRACT_AUDIO_SEGMENTS", "False").lower() == "true"

def analyze_audio_quality(audio_bytes: bytes, sample_rate: int = 24000, 
                          emotion_tag_count: int = 0) -> Tuple[bool, Dict]:
    """
    Analyze audio quality using cross-correlation and spectrogram analysis.
    Detects: repeated segments, stretched audio, silence, clipping, and energy anomalies.
    
    Args:
        audio_bytes: Raw audio data (16-bit PCM)
        sample_rate: Sample rate in Hz (default 24000 for Orpheus)
        emotion_tag_count: Number of emotion tags in the text (for adjusting silence thresholds)
        
    Returns:
        Tuple of (has_issues: bool, metrics: dict)
    """
    # Convert bytes to numpy array (-1.0 to 1.0 range)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    if len(audio_array) == 0:
        return True, {"error": "Empty audio"}
    
    metrics = {}
    has_issues = False
    
    # === 1. BASIC STATS (silence, clipping, energy) ===
    stats = compute_basic_stats(audio_array)
    metrics["stats"] = stats
    
    # Check for silence issues (adjusted for emotion tags)
    silence_threshold = 75.0 + (emotion_tag_count * 5.0)  # More tolerance with emotion tags
    silence_threshold = min(silence_threshold, 75.0)  # Cap at 75%
    
    if stats["silence_percent"] > silence_threshold:
        has_issues = True
        logger.warning(f"⚠️ Excessive silence: {stats['silence_percent']:.1f}% of audio")
    
    # Check for clipping issues
    if stats["clipped_percent"] > 1.0:
        has_issues = True
        logger.warning(f"⚠️ Audio clipping detected: {stats['clipped_percent']:.1f}%")
    
    # Check for very low energy (volume problem)
    if stats["rms_energy"] < 0.01:
        has_issues = True
        logger.warning(f"⚠️ Very low audio energy: {stats['rms_energy']:.4f}")
    
    # === 2. REPEATED SEGMENTS DETECTION (cross-correlation) ===
    repeated_segments = detect_repeated_segments(audio_array, sample_rate)
    # Require at least 2 repeated segments to flag as issue (reduces false positives)
    # Single similar segment can occur naturally in speech (similar phonemes)
    has_repetition = len(repeated_segments) >= 2
    
    metrics["repeated_segments"] = {
        "segments": repeated_segments,
        "count": len(repeated_segments),
        "has_repetition": has_repetition
    }
    
    if has_repetition:
        has_issues = True
        logger.warning(f"⚠️ Repeated audio segments detected: {len(repeated_segments)} regions")
    
    # === 3. SUSTAINED STRETCHING DETECTION ===
    has_sustained_stretching, sustained_metrics = detect_sustained_stretching(audio_array, sample_rate)
    metrics["sustained_stretching"] = sustained_metrics
    
    if has_sustained_stretching:
        has_issues = True
        logger.warning(f"⚠️ Sustained audio stretching detected: {sustained_metrics['max_sustained_duration']:.2f}s")
    
    # === 4. MONOTONIC AUDIO DETECTION (pitch variance) ===
    is_monotonic, pitch_variance = detect_monotonic_audio(audio_array, sample_rate)
    metrics["monotonic"] = {
        "is_monotonic": is_monotonic,
        "pitch_variance": pitch_variance
    }
    
    if is_monotonic:
        has_issues = True
        logger.warning(f"⚠️ Monotonic audio detected (pitch variance: {pitch_variance:.1f})")
    
    # Overall assessment
    metrics["has_quality_issues"] = has_issues
    metrics["duration_seconds"] = len(audio_array) / sample_rate
    
    return has_issues, metrics


def compute_basic_stats(audio: np.ndarray) -> Dict:
    """
    Compute basic audio statistics: silence %, clipping %, energy, etc.
    
    Args:
        audio: Audio signal as numpy array (-1.0 to 1.0)
        
    Returns:
        Dictionary with basic audio statistics
    """
    return {
        "max": float(np.max(audio)),
        "min": float(np.min(audio)),
        "mean": float(np.mean(audio)),
        "rms_energy": float(np.sqrt(np.mean(audio**2))),
        "silence_percent": float(np.mean(np.abs(audio) < 0.01)) * 100,
        "clipped_percent": float(np.mean(np.abs(audio) > 0.98)) * 100,
    }


def detect_repeated_segments(audio: np.ndarray, sample_rate: int, 
                             window_ms: int = 300, 
                             similarity_threshold: float = 0.85,
                             energy_threshold: float = 0.02) -> List[Tuple[float, float]]:
    """
    Detect repeated or looped audio segments using normalized cross-correlation.
    Uses sliding-window similarity detection.
    
    Args:
        audio: Audio signal as numpy array (-1.0 to 1.0)
        sample_rate: Sample rate in Hz
        window_ms: Window size in milliseconds for segment comparison
        similarity_threshold: Correlation threshold (0.85 = 85% similar)
        energy_threshold: Minimum RMS energy to consider (skips silent segments)
        
    Returns:
        List of (time_start, time_end) tuples for suspicious repeated regions
    """
    window_samples = int(sample_rate * window_ms / 1000)
    repeated = []
    
    for i in range(0, len(audio) - 2 * window_samples, window_samples):
        seg1 = audio[i:i + window_samples]
        seg2 = audio[i + window_samples:i + 2 * window_samples]
        
        # Skip if either segment has no variation (pure silence)
        if np.std(seg1) == 0 or np.std(seg2) == 0:
            continue
        
        # Skip segments with very low energy (silence/near-silence)
        rms1 = np.sqrt(np.mean(seg1 ** 2))
        rms2 = np.sqrt(np.mean(seg2 ** 2))
        if rms1 < energy_threshold or rms2 < energy_threshold:
            continue
        
        # Normalize segments
        seg1_norm = (seg1 - np.mean(seg1)) / np.std(seg1)
        seg2_norm = (seg2 - np.mean(seg2)) / np.std(seg2)
        
        # Cross-correlation
        corr = correlate(seg1_norm, seg2_norm, mode='valid')
        corr /= len(seg1_norm)
        sim = np.max(corr)
        
        if sim > similarity_threshold:
            time_start = i / sample_rate
            time_end = (i + 2 * window_samples) / sample_rate
            repeated.append((round(time_start, 2), round(time_end, 2)))
    
    return repeated


def detect_sustained_stretching(audio: np.ndarray, sample_rate: int,
                                min_sustained_duration: float = 1.0,
                                energy_threshold: float = 0.02,
                                variation_threshold: float = 0.1977) -> Tuple[bool, Dict]:
    """
    Detect stretched/prolonged audio (e.g., "iiiiiii", "myyyyyyy", "buuuuut") by finding
    long periods of sustained high energy with low variation.
    
    Stretched phonemes maintain consistent amplitude for unnaturally long periods.
    This works for vowels, consonants, or any speech sound that gets abnormally prolonged.
    
    Args:
        audio: Audio signal as numpy array (-1.0 to 1.0)
        sample_rate: Sample rate in Hz
        min_sustained_duration: Minimum duration to flag as stretched (seconds)
        energy_threshold: Minimum RMS energy to consider as "active" speech
        variation_threshold: Maximum coefficient of variation for sustained regions
        
    Returns:
        Tuple of (has_stretching: bool, metrics: dict)
    """
    # Calculate RMS energy in small windows (50ms)
    window_size = int(0.05 * sample_rate)
    hop_size = window_size // 2
    
    rms_values = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)
    
    rms_values = np.array(rms_values)
    
    if len(rms_values) < 20:  # Too short to analyze
        return False, {"info": "Audio too short for sustained vowel analysis"}
    
    # Find regions with sustained high energy
    is_high_energy = rms_values > energy_threshold
    
    # Look for long stretches of high energy with low variation
    max_sustained_duration = 0.0
    max_sustained_variation = 0.0
    current_stretch = []
    current_stretch_start_idx = 0
    sustained_regions = []
    
    for i, high_energy in enumerate(is_high_energy):
        if high_energy:
            if len(current_stretch) == 0:
                current_stretch_start_idx = i  # Mark the start
            current_stretch.append(rms_values[i])
        else:
            # End of a stretch
            if len(current_stretch) > 0:
                duration_seconds = (len(current_stretch) * hop_size) / sample_rate
                
                # Check if this stretch has low variation (sustained)
                if len(current_stretch) >= 3:  # Need at least 3 samples
                    mean_energy = np.mean(current_stretch)
                    std_energy = np.std(current_stretch)
                    cv = std_energy / mean_energy if mean_energy > 0 else 0
                    
                    if duration_seconds > min_sustained_duration and cv < variation_threshold:
                        # Calculate actual time positions in the audio
                        start_time = (current_stretch_start_idx * hop_size) / sample_rate
                        end_time = start_time + duration_seconds
                        
                        sustained_regions.append({
                            'start_time': round(float(start_time), 2),
                            'end_time': round(float(end_time), 2),
                            'duration': round(float(duration_seconds), 2),
                            'cv': round(float(cv), 3),
                            'mean_energy': round(float(mean_energy), 4)
                        })
                        
                        if duration_seconds > max_sustained_duration:
                            max_sustained_duration = duration_seconds
                            max_sustained_variation = cv
                
                current_stretch = []
    
    # Check final stretch
    if len(current_stretch) > 0:
        duration_seconds = (len(current_stretch) * hop_size) / sample_rate
        if len(current_stretch) >= 3:
            mean_energy = np.mean(current_stretch)
            std_energy = np.std(current_stretch)
            cv = std_energy / mean_energy if mean_energy > 0 else 0
            
            if duration_seconds > min_sustained_duration and cv < variation_threshold:
                # Calculate actual time positions in the audio
                start_time = (current_stretch_start_idx * hop_size) / sample_rate
                end_time = start_time + duration_seconds
                
                sustained_regions.append({
                    'start_time': round(float(start_time), 2),
                    'end_time': round(float(end_time), 2),
                    'duration': round(float(duration_seconds), 2),
                    'cv': round(float(cv), 3),
                    'mean_energy': round(float(mean_energy), 4)
                })
                
                if duration_seconds > max_sustained_duration:
                    max_sustained_duration = duration_seconds
                    max_sustained_variation = cv
    
    has_stretching = len(sustained_regions) > 0
    
    metrics = {
        "has_sustained_stretching": has_stretching,
        "sustained_regions_count": len(sustained_regions),
        "max_sustained_duration": round(float(max_sustained_duration), 2),
        "max_sustained_variation": round(float(max_sustained_variation), 3),
        "sustained_regions": sustained_regions  # Show all regions with timestamps
    }
    
    return has_stretching, metrics


def detect_monotonic_audio(audio: np.ndarray, sample_rate: int, 
                           min_pitch_variance: float = 30.0) -> Tuple[bool, float]:
    """
    Detect monotonic/flat TTS artifacts by analyzing pitch stability.
    
    Only checks for TOO LOW variance (monotonic/robot-like speech).
    
    Uses spectrogram to compute dominant frequency over time, then measures variance.
    
    Args:
        audio: Audio signal as numpy array (-1.0 to 1.0)
        sample_rate: Sample rate in Hz
        min_pitch_variance: Minimum acceptable pitch variance (default 30 Hz²)
        
    Returns:
        Tuple of (is_monotonic: bool, pitch_variance: float)
    """
    try:
        # Compute spectrogram
        f, t, Sxx = spectrogram(audio, sample_rate, nperseg=1024)
        
        # Find dominant frequency at each time step
        dominant_freq = f[np.argmax(Sxx, axis=0)]
        
        # Calculate pitch variance
        pitch_variance = float(np.var(dominant_freq))
        
        # Only check for low variance (monotonic/flat)
        is_monotonic = pitch_variance < min_pitch_variance
        
        return is_monotonic, round(pitch_variance, 2)
    
    except Exception as e:
        logger.warning(f"Error in monotonic audio detection: {e}")
        return False, 0.0


def quick_audio_check(audio_chunks: list[bytes], sample_rate: int = 24000, 
                      emotion_tag_count: int = 0) -> Tuple[bool, Dict]:
    """
    Quick check of audio quality using cross-correlation and spectrogram analysis.
    Detects: repeated segments, stretched audio, silence, clipping.
    
    Args:
        audio_chunks: List of audio chunk bytes
        sample_rate: Sample rate in Hz
        emotion_tag_count: Number of emotion tags in the text (affects silence thresholds)
        
    Returns:
        Tuple of (has_issues: bool, summary_metrics: dict)
    """
    # Combine chunks
    combined_audio = b''.join(audio_chunks)
    
    # Run full analysis
    has_issues, metrics = analyze_audio_quality(combined_audio, sample_rate, emotion_tag_count)
    
    # Summarize key findings
    stats = metrics.get("stats", {})
    repeated = metrics.get("repeated_segments", {})
    sustained = metrics.get("sustained_stretching", {})
    monotonic = metrics.get("monotonic", {})
    
    summary = {
        "has_quality_issues": has_issues,
        "duration_seconds": metrics.get("duration_seconds", 0),
        "silence_percent": stats.get("silence_percent", 0),
        "clipped_percent": stats.get("clipped_percent", 0),
        "rms_energy": stats.get("rms_energy", 0),
        "repeated_segment_count": repeated.get("count", 0),
        "has_repetition": repeated.get("has_repetition", False),
        "has_sustained_stretching": sustained.get("has_sustained_stretching", False),
        "max_sustained_duration": sustained.get("max_sustained_duration", 0),
        "is_monotonic": monotonic.get("is_monotonic", False),
        "pitch_variance": monotonic.get("pitch_variance", 0),
        "emotion_tag_count": emotion_tag_count,
    }
    
    return has_issues, summary


def extract_audio_segment(input_path: str, start_time: float, end_time: float, output_path: str):
    """
    Extract a segment from a WAV file based on timestamps.
    
    Args:
        input_path: Path to input WAV file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to output WAV file
    """
    try:
        # Open input file
        with wave.open(input_path, 'rb') as wf_in:
            # Get audio parameters
            channels = wf_in.getnchannels()
            sample_width = wf_in.getsampwidth()
            frame_rate = wf_in.getframerate()
            
            # Calculate frame positions
            start_frame = int(start_time * frame_rate)
            end_frame = int(end_time * frame_rate)
            num_frames = end_frame - start_frame
            
            # Validate
            total_frames = wf_in.getnframes()
            if start_frame < 0 or end_frame > total_frames:
                logger.warning(f"Invalid time range for extraction: {start_time:.2f}s to {end_time:.2f}s")
                return
            
            # Move to start position
            wf_in.setpos(start_frame)
            
            # Read the segment
            audio_data = wf_in.readframes(num_frames)
            
            # Write to output file
            with wave.open(output_path, 'wb') as wf_out:
                wf_out.setnchannels(channels)
                wf_out.setsampwidth(sample_width)
                wf_out.setframerate(frame_rate)
                wf_out.writeframes(audio_data)
            
            logger.info(f"✂️ Extracted segment: {start_time:.2f}s to {end_time:.2f}s → {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to extract segment: {e}")


def analyze_audio_file(file_path: str, emotion_tag_count: int = 0) -> Dict:
    """
    Analyze an audio file and return comprehensive quality metrics.
    
    Args:
        file_path: Path to WAV audio file
        emotion_tag_count: Number of emotion tags in the original text (optional)
        
    Returns:
        Dictionary with analysis results
    """
    # Read WAV file
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)
    
    # Run analysis
    has_issues, metrics = analyze_audio_quality(audio_bytes, sample_rate, emotion_tag_count)
    
    return metrics


def main():
    """
    Command-line interface for analyzing audio files.
    Usage: python audio_analysis.py <audio_file.wav> [emotion_tag_count] [--extract]
    """
    import sys
    import json
    
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python audio_analysis.py <audio_file.wav> [emotion_tag_count] [--extract]")
        print("\nArguments:")
        print("  audio_file.wav       Path to audio file to analyze")
        print("  emotion_tag_count    Number of emotion tags (optional, default: 0)")
        print("  --extract            Extract problematic segments to files (optional)")
        print("\nExamples:")
        print("  python audio_analysis.py debug_audio_errors/audio.wav")
        print("  python audio_analysis.py debug_audio_errors/audio.wav 2")
        print("  python audio_analysis.py debug_audio_errors/audio.wav 0 --extract")
        print("\nEnvironment Variables:")
        print("  EXTRACT_AUDIO_SEGMENTS=true    Enable automatic segment extraction")
        sys.exit(1)
    
    # Parse arguments
    file_path = sys.argv[1]
    emotion_tag_count = 0
    extract_flag = '--extract' in sys.argv
    
    # Parse emotion_tag_count if provided
    for arg in sys.argv[2:]:
        if arg != '--extract':
            try:
                emotion_tag_count = int(arg)
            except ValueError:
                pass
    
    # Enable extraction if flag is set OR environment variable is true
    global ENABLE_SEGMENT_EXTRACTION
    if extract_flag:
        ENABLE_SEGMENT_EXTRACTION = True
    
    print(f"\n{'='*70}")
    print(f"🎵 Audio Quality Analysis")
    print(f"{'='*70}")
    print(f"File: {file_path}")
    print(f"Emotion tags: {emotion_tag_count}")
    if ENABLE_SEGMENT_EXTRACTION:
        print(f"Segment extraction: ✂️ ENABLED")
    print(f"{'='*70}\n")
    
    try:
        # Analyze the file
        metrics = analyze_audio_file(file_path, emotion_tag_count)
        
        # Extract key sections
        stats = metrics.get("stats", {})
        repeated = metrics.get("repeated_segments", {})
        sustained = metrics.get("sustained_stretching", {})
        monotonic = metrics.get("monotonic", {})
        has_issues = metrics.get("has_quality_issues", False)
        
        # Display results
        print("📊 OVERALL ASSESSMENT")
        print(f"  Quality Issues: {'❌ YES' if has_issues else '✅ NO'}")
        print(f"  Duration: {metrics.get('duration_seconds', 0):.2f} seconds")
        print()
        
        print("📈 BASIC STATISTICS")
        print(f"  RMS Energy: {stats.get('rms_energy', 0):.4f}")
        print(f"  Silence: {stats.get('silence_percent', 0):.1f}%")
        print(f"  Clipping: {stats.get('clipped_percent', 0):.1f}%")
        print(f"  Peak (max): {stats.get('max', 0):.4f}")
        print(f"  Peak (min): {stats.get('min', 0):.4f}")
        print()
        
        print("🔄 REPETITION DETECTION")
        print(f"  Has Repetition: {'❌ YES' if repeated.get('has_repetition', False) else '✅ NO'}")
        print(f"  Repeated Segments: {repeated.get('count', 0)}")
        if repeated.get('segments'):
            print(f"  Time Ranges:")
            for start, end in repeated.get('segments', [])[:5]:  # Show first 5
                print(f"    - {start:.2f}s to {end:.2f}s")
            if len(repeated.get('segments', [])) > 5:
                print(f"    ... and {len(repeated.get('segments', [])) - 5} more")
        print()
        
        print("🎤 SUSTAINED AUDIO STRETCHING DETECTION")
        has_sustained = sustained.get('has_sustained_stretching', False)
        max_duration = sustained.get('max_sustained_duration', 0)
        sustained_count = sustained.get('sustained_regions_count', 0)
        sustained_regions_list = sustained.get('sustained_regions', [])
        
        print(f"  Has Stretching: {'❌ YES' if has_sustained else '✅ NO'}")
        print(f"  Sustained Regions: {sustained_count}")
        
        if has_sustained:
            print(f"  Longest Sustained: {max_duration:.2f}s")
            print(f"  Max Variation: {sustained.get('max_sustained_variation', 0):.3f}")
            
            if sustained_regions_list:
                print(f"  Time Ranges:")
                for region in sustained_regions_list[:10]:  # Show first 10
                    start = region.get('start_time', 0)
                    end = region.get('end_time', 0)
                    dur = region.get('duration', 0)
                    cv = region.get('cv', 0)
                    print(f"    - {start:.2f}s to {end:.2f}s ({dur:.2f}s, CV: {cv:.3f})")
                
                if len(sustained_regions_list) > 10:
                    print(f"    ... and {len(sustained_regions_list) - 10} more regions")
        
        print()
        
        print("🎵 MONOTONIC AUDIO DETECTION")
        is_monotonic_flag = monotonic.get('is_monotonic', False)
        pitch_var = monotonic.get('pitch_variance', 0)
        print(f"  Is Monotonic: {'❌ YES' if is_monotonic_flag else '✅ NO'}")
        print(f"  Pitch Variance: {pitch_var:.2f} Hz²")
        
        if is_monotonic_flag:
            print(f"  Type: Flat/Robot-like (variance too low < 30 Hz²)")
        else:
            print(f"  Normal Range: > 30 Hz²")
        print()
        
        # Summary
        print(f"{'='*70}")
        if has_issues:
            print("⚠️  ISSUES DETECTED - Audio may need regeneration")
            issues = []
            if stats.get('silence_percent', 0) > 60:
                issues.append(f"Excessive silence ({stats.get('silence_percent', 0):.1f}%)")
            if stats.get('clipped_percent', 0) > 1:
                issues.append(f"Audio clipping ({stats.get('clipped_percent', 0):.1f}%)")
            if stats.get('rms_energy', 0) < 0.01:
                issues.append(f"Low energy ({stats.get('rms_energy', 0):.4f})")
            if repeated.get('has_repetition', False):
                issues.append(f"Repeated segments ({repeated.get('count', 0)})")
            if sustained.get('has_sustained_stretching', False):
                max_dur = sustained.get('max_sustained_duration', 0)
                issues.append(f"Sustained audio stretching ({max_dur:.2f}s)")
            if monotonic.get('is_monotonic', False):
                pitch_var = monotonic.get('pitch_variance', 0)
                issues.append(f"Monotonic audio (variance: {pitch_var:.1f} Hz²)")
            
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✅ Audio quality is good")
        print(f"{'='*70}\n")
        
        # Print full JSON for debugging
        print("📋 FULL METRICS (JSON):")
        print(json.dumps(metrics, indent=2))
        
        # Extract problematic segments if enabled
        if ENABLE_SEGMENT_EXTRACTION and has_issues:
            print(f"\n{'='*70}")
            print("✂️ EXTRACTING PROBLEMATIC SEGMENTS")
            print(f"{'='*70}")
            
            # Get the directory where the input file is located
            input_dir = Path(file_path).parent
            input_stem = Path(file_path).stem  # filename without extension
            
            extracted_count = 0
            
            # Extract sustained stretching segments
            if sustained.get('has_sustained_stretching', False):
                sustained_regions_list = sustained.get('sustained_regions', [])
                for i, region in enumerate(sustained_regions_list, 1):
                    start = region.get('start_time', 0)
                    end = region.get('end_time', 0)
                    output_name = f"{input_stem}_sustained_stretch_{i}_{start:.2f}s-{end:.2f}s.wav"
                    output_path = input_dir / output_name
                    
                    extract_audio_segment(file_path, start, end, str(output_path))
                    extracted_count += 1
                    print(f"  ✅ Sustained stretch #{i}: {output_path.name}")
            
            # Extract repeated segments
            if repeated.get('has_repetition', False):
                repeated_segments_list = repeated.get('segments', [])
                for i, (start, end) in enumerate(repeated_segments_list, 1):
                    output_name = f"{input_stem}_repetition_{i}_{start:.2f}s-{end:.2f}s.wav"
                    output_path = input_dir / output_name
                    
                    extract_audio_segment(file_path, start, end, str(output_path))
                    extracted_count += 1
                    print(f"  ✅ Repetition #{i}: {output_path.name}")
            
            if extracted_count > 0:
                print(f"\n✂️ Extracted {extracted_count} segment(s) to: {input_dir}")
            else:
                print(f"\n⚠️ Issues detected but no extractable segments")
            
            print(f"{'='*70}")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error analyzing audio: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

