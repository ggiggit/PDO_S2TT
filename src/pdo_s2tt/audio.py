"""Minimal WAV reader for the released streaming interface."""

from pathlib import Path
import wave


def read_wav(path: str | Path) -> bytes:
    try:
        with wave.open(str(path), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                    stream.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("not mono 16-bit 16 kHz PCM")
            pcm = stream.readframes(stream.getnframes())
    except (wave.Error, ValueError):
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if sample_rate != 16000:
            raise ValueError(f"audio must be 16 kHz, received {sample_rate} Hz")
        if getattr(audio, "ndim", 1) != 1:
            raise ValueError("audio must be mono")
        pcm = (audio.clip(-1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    if not pcm:
        raise ValueError("audio file is empty")
    return pcm


def packets(pcm: bytes, seconds: float = 0.1):
    size = int(round(16000 * seconds)) * 2
    if size <= 0:
        raise ValueError("packet duration must be positive")
    for offset in range(0, len(pcm), size):
        end = min(len(pcm), offset + size)
        yield pcm[offset:end], end == len(pcm)
