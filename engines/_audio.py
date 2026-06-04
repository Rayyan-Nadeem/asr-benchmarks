"""
Shared audio loading for engine adapters.

The harness boundary is mono PCM S16LE @ 16 kHz (ffmpeg auto-converts upstream
in runner.py). Engines that need raw bytes (Speechmatics WebSocket stream) or
normalized float arrays (NeMo, ONNX Runtime, MLX) should both pull from here
so the validation logic doesn't drift across adapters.
"""
from __future__ import annotations

import wave
from pathlib import Path


SAMPLE_RATE = 16000


def audio_duration_s(path: Path) -> float:
    """Audio duration in seconds. Cheap — doesn't read samples."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def load_pcm_s16le_16k_bytes(path: Path) -> tuple[bytes, float]:
    """
    Load mono PCM S16LE @ 16 kHz WAV. Returns (raw_bytes, duration_s).
    Raises ValueError on format mismatch.
    """
    with wave.open(str(path), "rb") as w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        if ch != 1 or width != 2 or rate != SAMPLE_RATE:
            raise ValueError(
                f"{path}: expected mono PCM S16LE @ {SAMPLE_RATE} Hz, "
                f"got channels={ch} sampwidth={width} rate={rate}"
            )
        duration_s = w.getnframes() / rate
        return w.readframes(w.getnframes()), duration_s


def load_pcm_as_float32(path: Path):
    """
    Load WAV as a 1-D float32 numpy array in [-1.0, 1.0], plus duration_s.
    For NeMo, ONNX, MLX consumers that expect normalized samples.
    """
    import numpy as np
    raw, duration_s = load_pcm_s16le_16k_bytes(path)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr, duration_s
