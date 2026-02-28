from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import io
import random
import wave

from protracker_mod_choral_generator import make_instrument_sample, make_drum_sample, bytes_to_float_sample

def _int16_wav_bytes_from_float(mono_floats: list[float], sr: int = 8287) -> bytes:
    # clamp and convert to 16-bit little endian mono
    pcm = bytearray()
    for x in mono_floats:
        if x > 1.0: x = 1.0
        if x < -1.0: x = -1.0
        v = int(x * 32767.0)
        pcm += int(v).to_bytes(2, "little", signed=True)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(pcm))
    return bio.getvalue()

def mod_sample_to_preview_wav(sample_bytes: bytes, sr: int = 8287) -> bytes:
    floats = bytes_to_float_sample(sample_bytes)
    return _int16_wav_bytes_from_float(floats, sr=sr)

@dataclass
class SamplePreviewSpec:
    instrument_kind: str
    disable_vibrato: bool = False
    seed: Optional[int] = None
    is_drum: bool = False
    drum_style: str = "Kick"

class SampleEngine:
    """
    Generates short WAV previews for instruments without needing a full song render.
    """
    def __init__(self):
        self._rng = random.Random()

    def preview_wav_for(self, spec: SamplePreviewSpec) -> bytes:
        # make deterministic if seed provided
        rng = random.Random(int(spec.seed)) if spec.seed is not None else self._rng

        if spec.is_drum:
            # One-shot-ish, smaller buffer
            sample_bytes = make_drum_sample(rng, kind=spec.drum_style, length=8192, sr=8287)
        else:
            sample_bytes = make_instrument_sample(
                kind=spec.instrument_kind,
                rng=rng,
                length=32768,
                sr=8287,
                disable_vibrato=bool(spec.disable_vibrato),
                ensemble_size=4,
            )
        return mod_sample_to_preview_wav(sample_bytes, sr=8287)

    def preview_wav_from_custom_path(self, wav_path: str) -> Optional[bytes]:
        p = Path(wav_path)
        if not p.exists():
            return None
        try:
            return p.read_bytes()
        except Exception:
            return None
