from __future__ import annotations

from pathlib import Path
from typing import Sequence

import protracker_mod_choral_generator as backend

MOD_PATTERN_OFFSET = 1084  # classic MOD header size

def patch_mod_patterns_in_file(mod_path: Path, patterns) -> None:
    """Rewrite pattern data in-place (preserves header + sample data)."""
    mod_path = Path(mod_path)
    if not mod_path.exists():
        return
    raw = bytearray(mod_path.read_bytes())
    pat_bytes = backend.patterns_to_bytes(patterns)
    n = len(pat_bytes)
    start = MOD_PATTERN_OFFSET
    end = start + n
    if end > len(raw):
        # Don't attempt if file seems shorter than expected
        return
    raw[start:end] = pat_bytes
    mod_path.write_bytes(bytes(raw))

MOD_SIGNATURE_OFFSET = 1080


def patch_mod_signature(mod_path: Path, signature: str) -> None:
    """Force-write 4-byte MOD signature at offset 1080 (classic magic)."""
    try:
        sig = (signature or "").encode("ascii", errors="ignore")[:4]
        sig = sig.ljust(4, b" ")
        mod_path = Path(mod_path)
        if not mod_path.exists():
            return
        raw = bytearray(mod_path.read_bytes())
        if len(raw) < MOD_SIGNATURE_OFFSET + 4:
            return
        raw[MOD_SIGNATURE_OFFSET:MOD_SIGNATURE_OFFSET+4] = sig
        mod_path.write_bytes(bytes(raw))
    except Exception:
        return
