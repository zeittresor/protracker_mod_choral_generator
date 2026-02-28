from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

import protracker_mod_choral_generator as backend
from ptmod.config import SongConfig



@dataclass
class FxSummary:
    total: int = 0
    initial_ftempo: int = 0
    vibrato: int = 0
    portamento: int = 0
    arpeggio: int = 0
    volume: int = 0
    notecut: int = 0
    retrig: int = 0

    def bump(self, attr: str, n: int = 1):
        try:
            setattr(self, attr, int(getattr(self, attr)) + int(n))
            self.total += int(n)
        except Exception:
            self.total += int(n)
# Pattern cell: (note:str|None, sample:int, effect:int, param:int)

def _is_empty_eff(cell) -> bool:
    try:
        _n,_s,e,p = cell
        return int(e) == 0 and int(p) == 0
    except Exception:
        return False

def _set_eff(patterns, p: int, row: int, ch: int, eff: int, param: int) -> bool:
    try:
        n,s,_,_ = patterns[p][row][ch]
        patterns[p][row][ch] = (n, s, int(eff) & 0x0F, int(param) & 0xFF)
        return True
    except Exception:
        return False

def _iter_rows(pattern):
    for r in range(64):
        yield r, pattern[r]

def apply_fx_to_song(song, cfg: SongConfig, rng: Optional[random.Random] = None) -> FxSummary:
    """In-place FX injection. Only touches effect/param when currently empty."""
    if rng is None:
        try:
            rng = random.Random(int(getattr(song, 'seed', 0)) ^ 0xC001C0DE)
        except Exception:
            rng = random.Random()

    patterns = getattr(song, 'patterns', None)
    if not patterns:
        return FxSummary()

    summary = FxSummary()

    # Determine drum channels to avoid FX there
    drum_ch = set()
    try:
        drum_ch = set(int(k) for k in getattr(song, 'drum_channel_styles', {}).keys())
    except Exception:
        drum_ch = set()

    intensity = max(0, min(100, int(getattr(cfg, 'fx_intensity', 50) or 50)))
    prob = 0.10 + 0.30 * (intensity / 100.0)  # 0.10..0.40

    # --- initial speed/tempo (Fxx) on first played pattern (improves tracker compatibility) ---
    if bool(getattr(cfg, 'fx_insert_initial_speed_tempo', True)):
        try:
            order = list(getattr(song, 'order', []) or getattr(song, 'order_original', []) or [])
            first_pat = int(order[0]) if order else 0
            first_pat = max(0, min(first_pat, len(patterns)-1))
            spd = max(1, min(31, int(getattr(song, 'speed', cfg.speed))))
            tpo = max(32, min(255, int(getattr(song, 'tempo', cfg.tempo))))

            # CH1: speed, CH2: tempo (if cells are free)
            if 0 not in drum_ch and _is_empty_eff(patterns[first_pat][0][0]):
                summary.bump('initial_ftempo') if _set_eff(patterns, first_pat, 0, 0, 0x0F, spd) else None
            if 1 not in drum_ch and _is_empty_eff(patterns[first_pat][0][1]):
                summary.bump('initial_ftempo') if _set_eff(patterns, first_pat, 0, 1, 0x0F, tpo) else None
        except Exception:
            pass

    # --- vibrato (4xy) on melody channel (CH1) ---
    if bool(getattr(cfg, 'fx_vibrato_melody', False)):
        ch = 0
        if ch not in drum_ch:
            speed = 3 + int(5 * (intensity / 100.0))   # 3..8
            depth = 2 + int(6 * (intensity / 100.0))   # 2..8
            param = ((speed & 0x0F) << 4) | (depth & 0x0F)
            for p_idx, pat in enumerate(patterns):
                for row in (0, 16, 32, 48):
                    try:
                        n,s,e,p = pat[row][ch]
                        if n is None:
                            continue
                        if _is_empty_eff(pat[row][ch]) and rng.random() < (0.35 + 0.35*(intensity/100.0)):
                            summary.bump('vibrato') if _set_eff(patterns, p_idx, row, ch, 0x04, param) else None
                    except Exception:
                        pass

    # --- portamento (3xx) on melody transitions ---
    if bool(getattr(cfg, 'fx_portamento_melody', False)):
        ch = 0
        if ch not in drum_ch:
            porta = 0x12 + int(0x30 * (intensity / 100.0))  # 0x12..0x42
            for p_idx, pat in enumerate(patterns):
                prev_note = None
                for row in range(64):
                    try:
                        n,s,e,p = pat[row][ch]
                        if n is None:
                            continue
                        # only on transitions
                        if prev_note is not None and prev_note != n and _is_empty_eff(pat[row][ch]) and rng.random() < prob:
                            _set_eff(patterns, p_idx, row, ch, 0x03, porta)
                        prev_note = n
                    except Exception:
                        pass

    # --- arpeggio ornaments (0xy) on CH2 (often harmony) ---
    if bool(getattr(cfg, 'fx_arpeggio_ornaments', False)):
        ch = 1
        if ch not in drum_ch:
            # Choose semitone offsets: 0-4 for subtlety
            for p_idx, pat in enumerate(patterns):
                for row in range(0, 64, 4):
                    try:
                        n,s,e,p = pat[row][ch]
                        if n is None:
                            continue
                        if _is_empty_eff(pat[row][ch]) and rng.random() < (prob * 0.8):
                            x = rng.choice([3,4,5]) if intensity > 60 else rng.choice([2,3,4])
                            y = rng.choice([7,8]) if intensity > 70 else rng.choice([5,7])
                            param = ((x & 0x0F) << 4) | (y & 0x0F)
                            summary.bump('arpeggio') if _set_eff(patterns, p_idx, row, ch, 0x00, param) else None
                    except Exception:
                        pass

    # --- volume motion (Cxx / Axy) ---
    if bool(getattr(cfg, 'fx_volume_motion', False)):
        # Prefer Cxx at bar starts (stable across players)
        for p_idx, pat in enumerate(patterns):
            for ch in range(4):
                if ch in drum_ch:
                    continue
                for row in (0, 16, 32, 48):
                    try:
                        n,s,e,p = pat[row][ch]
                        if n is None:
                            continue
                        if _is_empty_eff(pat[row][ch]) and rng.random() < (0.20 + 0.30*(intensity/100.0)):
                            vol = int(40 + rng.random() * (24 * (intensity/100.0)))  # 40..64
                            vol = max(0, min(64, vol))
                            summary.bump('volume') if _set_eff(patterns, p_idx, row, ch, 0x0C, vol) else None
                    except Exception:
                        pass

    # --- note cut (E C x) for staccato ---
    if bool(getattr(cfg, 'fx_note_cut', False)):
        # Apply on melody channel end-of-bar rows (12..15)
        ch = 0
        if ch not in drum_ch:
            for p_idx, pat in enumerate(patterns):
                for row in (12, 28, 44, 60):
                    try:
                        n,s,e,p = pat[row][ch]
                        if n is None:
                            continue
                        if _is_empty_eff(pat[row][ch]) and rng.random() < (0.18 + 0.25*(intensity/100.0)):
                            x = 2 + int(6*(intensity/100.0))  # 2..8 ticks
                            x = max(0, min(0x0F, x))
                            summary.bump('notecut') if _set_eff(patterns, p_idx, row, ch, 0x0E, 0xC0 | x) else None
                    except Exception:
                        pass

    # --- retrig (E 9 x) on hats channel-ish (CH4) if not drums ---
    if bool(getattr(cfg, 'fx_retrig', False)):
        ch = 3
        if ch not in drum_ch:
            for p_idx, pat in enumerate(patterns):
                for row in range(0, 64, 8):
                    try:
                        n,s,e,p = pat[row][ch]
                        if n is None:
                            continue
                        if _is_empty_eff(pat[row][ch]) and rng.random() < (0.12 + 0.20*(intensity/100.0)):
                            x = 2 + int(6*(intensity/100.0))  # 2..8 retrigs per row
                            x = max(0, min(0x0F, x))
                            summary.bump('retrig') if _set_eff(patterns, p_idx, row, ch, 0x0E, 0x90 | x) else None
                    except Exception:
                        pass

    return summary
