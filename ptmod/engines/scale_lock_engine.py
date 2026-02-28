from __future__ import annotations

"""Scale-lock post processing.

Goal: avoid accidental "Major vs Minor" clashes across channels by ensuring that all pitched
notes stay inside the selected scale/mode.

This is intentionally conservative:

- Only touches NOTE tokens (does not alter sample numbers/effects).
- Skips channels configured as drumsets.
- Uses the song's resolved key_root / scale_mode when available.
"""

from typing import Callable, Optional, Set

import protracker_mod_choral_generator as backend


LogCb = Optional[Callable[[str], None]]


def _cb(cb: LogCb, msg: str) -> None:
    try:
        if cb:
            cb(msg)
    except Exception:
        pass


def _midi_to_pt_token(midi: int) -> str:
    # Supported range in this project: C-1..B-3.
    midi = int(midi)
    lo = backend._parse_note_token_to_midi("C-1") or 36
    hi = backend._parse_note_token_to_midi("B-3") or 71
    if midi < lo:
        midi = lo
    if midi > hi:
        midi = hi

    semi = midi % 12
    # inverse of backend _parse_note_token_to_midi: midi = (octv+2)*12 + semi
    octv = (midi // 12) - 2
    octv = max(1, min(3, int(octv)))
    name = backend.CHROMA[int(semi)]  # e.g. "C-" / "C#" / ...
    tok = f"{name}{octv}"
    return tok if tok in backend.CHROMATIC_SET else "C-2"


def _nearest_allowed_midi(midi: int, allowed_pcs: Set[int]) -> int:
    pc = int(midi) % 12
    if pc in allowed_pcs:
        return int(midi)

    # Search minimal semitone shift; tie-break prefers DOWN.
    for dist in range(1, 7):
        if ((pc - dist) % 12) in allowed_pcs:
            return int(midi - dist)
        if ((pc + dist) % 12) in allowed_pcs:
            return int(midi + dist)
    return int(midi)


def apply_scale_lock(song, cfg, log_cb: LogCb = None) -> int:
    """Quantize all pitched notes to the selected scale.

    Returns number of changed note cells.
    """

    try:
        key_root = str(getattr(song, "key_root", None) or getattr(cfg, "key_root_override", None) or "C-2")
        mode = str(getattr(song, "scale_mode", None) or getattr(cfg, "scale_mode", None) or "Major")
    except Exception:
        key_root, mode = "C-2", "Major"

    mode_clean = mode.strip().lower()
    if mode_clean == "mixed":
        # Mixed intentionally borrows.
        return 0

    # Resolve the actual scale.
    try:
        scale = backend.scale_from_mode(key_root, mode_clean)
        allowed_pcs: Set[int] = set()
        for n in scale:
            m = backend._parse_note_token_to_midi(n)
            if m is not None:
                allowed_pcs.add(int(m) % 12)
        if not allowed_pcs:
            return 0
    except Exception as e:
        _cb(log_cb, f"[scale_lock] failed to build scale: {e}")
        return 0

    # Skip drum channels
    drum_ch = set()
    try:
        for i, kind in enumerate(getattr(cfg, "instruments", []) or []):
            if backend.is_drumset_kind(str(kind)):
                drum_ch.add(int(i))
    except Exception:
        pass

    changed = 0
    try:
        patterns = song.patterns
        for pat in patterns:
            for row in pat:
                for ch in range(4):
                    if ch in drum_ch:
                        continue
                    note, samp, eff, par = row[ch]
                    if note is None:
                        continue
                    midi = backend._parse_note_token_to_midi(str(note))
                    if midi is None:
                        continue
                    nmidi = _nearest_allowed_midi(int(midi), allowed_pcs)
                    if nmidi != int(midi):
                        ntok = _midi_to_pt_token(nmidi)
                        if ntok != note:
                            row[ch] = (ntok, int(samp), int(eff), int(par))
                            changed += 1
    except Exception as e:
        _cb(log_cb, f"[scale_lock] failed while applying: {e}")
        return changed

    if changed:
        _cb(log_cb, f"[scale_lock] snapped {changed} notes to {mode} in {key_root}")
    return changed
