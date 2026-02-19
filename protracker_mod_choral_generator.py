#!/usr/bin/env python3
# ProTracker MOD Choral Generator (v1.6.4)
# Source: https://github.com/zeittresor/protracker_mod_choral_generator

from __future__ import annotations

import argparse
import io
import math
import os
import random
import re
import struct
import subprocess
import sys
import threading
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

# -----------------------------
# ProTracker note period table (C-1 .. B-3) for standard Amiga / ProTracker tuning
# -----------------------------
PERIODS: dict[str, int] = {
    "C-1": 1712, "C#1": 1616, "D-1": 1524, "D#1": 1440, "E-1": 1356, "F-1": 1280, "F#1": 1208, "G-1": 1140, "G#1": 1076, "A-1": 1016, "A#1": 960, "B-1": 906,
    "C-2": 856, "C#2": 808, "D-2": 762, "D#2": 720, "E-2": 678, "F-2": 640, "F#2": 604, "G-2": 570, "G#2": 538, "A-2": 508, "A#2": 480, "B-2": 453,
    "C-3": 428, "C#3": 404, "D-3": 381, "D#3": 360, "E-3": 339, "F-3": 320, "F#3": 302, "G-3": 285, "G#3": 269, "A-3": 254, "A#3": 240, "B-3": 226,
}

CHROMA = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"]
OCTAVES = [1, 2, 3]
CHROMATIC = [f"{n}{o}" for o in OCTAVES for n in CHROMA]
CHROMATIC_SET = set(CHROMATIC)

DEFAULT_SPEED = 6
DEFAULT_TEMPO = 125

DEFAULT_ORDER_STR = "0, 1, 6, 2, 7, 3, 8, 4, 9, 5"
ORDER_PRESETS = [
    # legacy / compact
    "0, 1, 2, 3, 2, 4, 5",
    "0, 1, 2, 3, 2, 4, 1, 4, 2, 5",
    "5, 5, 1, 5, 0, 2, 3, 4, 2, 5, 0",
    "5, 0, 1, 5, 2, 3, 1, 4, 2, 5, 0",

    # new presets for patterns 0..9
    "0, 1, 6, 2, 7, 3, 8, 4, 9, 5",
    "0, 6, 1, 7, 2, 8, 4, 9, 5",
    "6, 0, 1, 7, 2, 4, 8, 9, 5",
    "0, 2, 6, 2, 7, 4, 8, 4, 9, 5",
    "0, 1, 2, 6, 7, 2, 8, 9, 5",
    "0, 3, 6, 1, 7, 2, 8, 4, 9, 5",
    "0, 6, 6, 1, 7, 7, 2, 8, 4, 9, 5",
    "0, 1, 2, 3, 6, 7, 8, 9, 5",
    "6, 1, 7, 2, 8, 4, 9, 5",
]


# -----------------------------
# Base melody presets (public domain carols + a few original hymn/folk motifs)
# Notes are expressed as scale degrees in the current major key (0..6), with an octave offset.
# Each preset contains exactly 4 bars (each bar = 16 rows). Durations are in rows.
#
# IMPORTANT: We intentionally do NOT ship exact melodies of modern copyrighted songs.
# If you want a "modern folk" vibe, use the included original presets.
# -----------------------------

MELODY_LIBRARY: dict[str, list[list[tuple[int | None, int, int]]]] = {
    # Public domain / traditional carols (approximate phrasing, tracker-friendly rhythm)
    "O Tannenbaum (trad.)": [
        [(0, 0, 4), (0, 0, 4), (0, 0, 4), (1, 0, 2), (2, 0, 2)],
        [(2, 0, 4), (2, 0, 4), (2, 0, 4), (3, 0, 2), (2, 0, 2)],
        [(1, 0, 4), (0, 0, 4), (2, 0, 4), (1, 0, 2), (0, 0, 2)],
        [(0, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],
    "Stille Nacht (trad.)": [
        [(4, 0, 4), (5, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(4, 0, 4), (5, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(6, 0, 4), (6, 0, 4), (5, 0, 4), (3, 0, 4)],
        [(4, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],
    "Es ist ein Ros entsprungen (trad.)": [
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(3, 0, 4), (2, 0, 4), (1, 0, 4), (0, 0, 4)],
        [(2, 0, 4), (4, 0, 4), (5, 0, 4), (4, 0, 4)],
        [(2, 0, 8), (1, 0, 4), (0, 0, 4)],
    ],
    "Alle Jahre wieder (trad.)": [
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (5, 0, 4)],
        [(4, 0, 4), (2, 0, 4), (1, 0, 4), (0, 0, 4)],
        [(2, 0, 4), (4, 0, 4), (5, 0, 4), (4, 0, 4)],
        [(2, 0, 8), (1, 0, 4), (0, 0, 4)],
    ],
    "O du fröhliche (trad.)": [
        [(0, 0, 4), (0, 0, 4), (2, 0, 4), (4, 0, 4)],
        [(5, 0, 4), (4, 0, 4), (2, 0, 4), (0, 0, 4)],
        [(2, 0, 4), (2, 0, 4), (4, 0, 4), (5, 0, 4)],
        [(4, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],

    # Original motifs (safe for shipping; designed to feel hymn/gospel/folk-ish)
    "Hymn Stepwise (original)": [
        [(0, 0, 4), (1, 0, 4), (2, 0, 4), (3, 0, 4)],
        [(4, 0, 4), (3, 0, 4), (2, 0, 4), (1, 0, 4)],
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(1, 0, 4), (0, 0, 8), (None, 0, 4)],
    ],
    "Gospel Turnaround (original)": [
        [(0, 0, 4), (2, 0, 2), (3, 0, 2), (4, 0, 4), (2, 0, 4)],
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (5, 0, 4)],
        [(4, 0, 4), (3, 0, 2), (2, 0, 2), (1, 0, 4), (0, 0, 4)],
        [(0, 0, 8), (4, 0, 4), (0, 0, 4)],
    ],
    "Modern Folk Ballad (original)": [
        [(0, 0, 4), (2, 0, 4), (1, 0, 2), (0, 0, 2), (4, 0, 4)],
        [(4, 0, 4), (2, 0, 4), (5, 0, 4), (4, 0, 4)],
        [(2, 0, 4), (1, 0, 4), (0, 0, 4), (2, 0, 4)],
        [(4, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],

    # More public domain / traditional-ish hymn & spiritual motifs (approx. phrasing)
    "Amazing Grace (trad. approx.)": [
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (4, 0, 4)],
        [(2, 0, 4), (4, 0, 4), (5, 0, 4), (4, 0, 4)],
        [(2, 0, 4), (0, 0, 4), (2, 0, 4), (4, 0, 4)],
        [(5, 0, 8), (4, 0, 4), (2, 0, 4)],
    ],
    "When the Saints (trad. approx.)": [
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (5, 0, 4)],
        [(4, 0, 4), (2, 0, 4), (0, 0, 4), (2, 0, 4)],
        [(4, 0, 4), (5, 0, 4), (6, 0, 4), (5, 0, 4)],
        [(4, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],
    "Swing Low (trad. approx.)": [
        [(4, 0, 4), (2, 0, 4), (0, 0, 4), (2, 0, 4)],
        [(4, 0, 4), (5, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(1, 0, 4), (0, 0, 8), (None, 0, 4)],
    ],

    # Additional original motifs for variety
    "Gospel Walk (original)": [
        [(0, 0, 4), (2, 0, 2), (3, 0, 2), (4, 0, 4), (5, 0, 4)],
        [(5, 0, 2), (4, 0, 2), (3, 0, 4), (2, 0, 4), (0, 0, 4)],
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (6, 0, 4)],
        [(5, 0, 8), (4, 0, 4), (2, 0, 4)],
    ],
    "Choral Rise (original)": [
        [(0, 0, 2), (1, 0, 2), (2, 0, 4), (4, 0, 4), (3, 0, 4)],
        [(2, 0, 4), (4, 0, 4), (5, 0, 4), (4, 0, 4)],
        [(3, 0, 4), (2, 0, 4), (1, 0, 4), (0, 0, 4)],
        [(2, 0, 8), (1, 0, 4), (0, 0, 4)],
    ],
    "Folk Wanderer (original)": [
        [(0, 0, 4), (2, 0, 4), (1, 0, 4), (0, 0, 4)],
        [(4, 0, 4), (5, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(1, 0, 4), (0, 0, 4), (2, 0, 4), (4, 0, 4)],
        [(5, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],

}

# -----------------------------
# Melody plugins (txt / midi)
# -----------------------------

PLUGIN_DIR_NAME = "melody_plugins"

NOTE_NAME_TO_SEMITONE = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4,
    "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11,
}
C_MAJOR_PCS = [0, 2, 4, 5, 7, 9, 11]


def _default_plugin_root() -> Path:
    try:
        here = Path(__file__).resolve().parent
    except Exception:
        here = Path.cwd()
    return here / PLUGIN_DIR_NAME


def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "melody"


def _bars_to_plugin_text(display_name: str, bars: list[list[tuple[int | None, int, int]]]) -> str:
    lines = []
    lines.append(f"name: {display_name}")
    lines.append("# format: DEG OCT DUR  (DEG=0..6, OCT=-2..2, DUR=rows; use R for rest)")
    for bi, bar in enumerate(bars, start=1):
        lines.append(f"# bar {bi}")
        for deg, octv, dur in bar:
            if deg is None:
                lines.append(f"R 0 {int(dur)}")
            else:
                lines.append(f"{int(deg)} {int(octv)} {int(dur)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class MelodyPlugin:
    name: str
    bars: list[list[tuple[int | None, int, int]]]
    meta: dict[str, str]
    folder: Path
    source: Path


def _parse_kv_metadata_line(s: str) -> tuple[str, str] | None:
    """Parse simple 'key: value' metadata lines."""
    if not s or ":" not in s:
        return None
    k, v = s.split(":", 1)
    k = k.strip().lower().replace(" ", "_")
    v = v.strip()
    if not k or not v:
        return None
    # ignore 'name:' here (handled elsewhere)
    if k == "name":
        return None
    return k, v


def _read_plugin_metadata(folder: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for fn in ("info.txt", "meta.txt", "metadata.txt"):
        p = folder / fn
        if not p.exists():
            continue
        try:
            for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = ln.strip()
                if not s or s.startswith("#") or s.startswith(";"):
                    continue
                kv = _parse_kv_metadata_line(s)
                if kv:
                    meta[kv[0]] = kv[1]
        except Exception:
            pass
    return meta


def _default_plugin_info_text(display_name: str) -> str:
    n = (display_name or "").lower()
    mode = "minor" if ("minor" in n or "moll" in n) else "major"
    tempo_hint = "90-140" if mode == "minor" else "100-150"
    preferred_key_range = "C-2..G-2"
    return (
        f"mode: {mode}\n"
        f"tempo_hint: {tempo_hint}\n"
        f"preferred_key_range: {preferred_key_range}\n"
    )


def ensure_default_melody_plugins(plugin_root: Path) -> None:
    """Create default plugin folders/files from the built-in melody library.

    This runs only if the plugin dir (or individual melody files) are missing.
    """
    plugin_root.mkdir(parents=True, exist_ok=True)

    for display_name, bars in MELODY_LIBRARY.items():
        sub = plugin_root / _slugify(display_name)
        sub.mkdir(parents=True, exist_ok=True)
        p = sub / "melody.txt"
        if not p.exists():
            try:
                p.write_text(_bars_to_plugin_text(display_name, bars), encoding="utf-8")
            except Exception:
                # best-effort; never crash the generator for plugin IO
                pass

        info = sub / "info.txt"
        if not info.exists():
            try:
                info.write_text(_default_plugin_info_text(display_name), encoding="utf-8")
            except Exception:
                pass


def _parse_note_token_to_midi(tok: str) -> int | None:
    """Parse either note tokens like C4, D#5, Bb3, or ProTracker-like C-3 / D#2."""
    t = (tok or "").strip()
    if not t:
        return None
    if t.upper() in ("R", "REST", "---"):
        return None

    # ProTracker: C-3, D#2, etc.
    m = re.fullmatch(r"([A-Ga-g])([#bB]?)[-]?([0-9])", t)
    if not m:
        return None
    letter = m.group(1).upper()
    acc = m.group(2).upper()
    octv = int(m.group(3))

    key = letter + acc
    if key not in NOTE_NAME_TO_SEMITONE:
        key = letter
    semi = NOTE_NAME_TO_SEMITONE.get(key)
    if semi is None:
        return None

    # Map: ProTracker octave N corresponds roughly to MIDI octave (N+1) for our purpose
    midi_oct = octv + 1
    midi = (midi_oct + 1) * 12 + semi  # MIDI octave numbering: C-1=0
    return int(midi)


def _midi_note_to_degree_octv(midi_note: int) -> tuple[int, int]:
    """Map MIDI note to (degree in C major, octave offset relative to C4).

    We snap to nearest C-major pitch class for robustness.
    """
    pc = int(midi_note) % 12
    best = None
    best_dist = 999
    for d, pc2 in enumerate(C_MAJOR_PCS):
        dist = min((pc - pc2) % 12, (pc2 - pc) % 12)
        if dist < best_dist:
            best_dist = dist
            best = (d, pc2)
    deg = int(best[0]) if best else 0

    # Choose snapped pitch in the closest octave to original
    base = 60  # C4
    # candidates around base +- 3 octaves
    candidates = []
    for o in range(-3, 4):
        candidates.append(base + o * 12 + C_MAJOR_PCS[deg])
    snapped = min(candidates, key=lambda x: abs(x - midi_note))

    octv_off = int(round((snapped - base) / 12.0))
    octv_off = max(-2, min(2, octv_off))
    return deg, octv_off


def _events_to_4bars_degree_template(events: list[tuple[int | None, int]]) -> list[list[tuple[int | None, int, int]]]:
    """Convert (midi_note|None, dur_rows) to 4 bars of (deg|None, octv, dur)."""
    # Ensure we cover 64 rows (4 bars) by looping if needed.
    total = sum(max(1, int(d)) for _, d in events) if events else 0
    if total <= 0:
        events = [(60, 4), (62, 4), (64, 4), (65, 4), (67, 4), (69, 4), (71, 4), (72, 4)]

    # Expand/loop into exactly 64 rows worth of events
    out_ev: list[tuple[int | None, int]] = []
    rows_left = 64
    idx = 0
    while rows_left > 0:
        n, d = events[idx % len(events)]
        d = max(1, int(d))
        if d > rows_left:
            d = rows_left
        out_ev.append((n, d))
        rows_left -= d
        idx += 1

    # Split into bars of 16 rows
    bars: list[list[tuple[int | None, int, int]]] = []
    cur: list[tuple[int | None, int, int]] = []
    cur_rows = 0
    for n, d in out_ev:
        if n is None:
            cur.append((None, 0, int(d)))
        else:
            deg, octv = _midi_note_to_degree_octv(int(n))
            cur.append((deg, octv, int(d)))
        cur_rows += int(d)
        if cur_rows >= 16:
            bars.append(cur)
            cur = []
            cur_rows = 0
            if len(bars) == 4:
                break

    while len(bars) < 4:
        bars.append([(0, 0, 16)])

    return bars


def _parse_plugin_txt(path: Path) -> tuple[str, list[list[tuple[int | None, int, int]]], dict[str, str]]:
    """Parse plugin text.

    Supports:
    - metadata lines: key: value
    - degree form: DEG OCT DUR (DEG 0..6, OCT int, DUR int)
    - note form: NOTE DUR (NOTE like C4, D#4, Bb3, C-3)
    """
    name = path.parent.name
    meta: dict[str, str] = {}
    events: list[tuple[int | None, int]] = []

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        if s.lower().startswith("name:"):
            name = s.split(":", 1)[1].strip() or name
            continue

        kv = _parse_kv_metadata_line(s)
        if kv:
            meta[kv[0]] = kv[1]
            continue

        parts = re.split(r"\s+", s)
        if len(parts) >= 3 and re.fullmatch(r"-?\d+|R", parts[0], re.I):
            # DEG OCT DUR
            deg_tok = parts[0]
            try:
                dur = int(parts[2])
            except Exception:
                dur = 4

            if deg_tok.upper() == "R":
                events.append((None, max(1, dur)))
            else:
                deg = int(deg_tok)
                deg = max(0, min(6, deg))
                # convert degree+octv to midi note in C major near C4
                try:
                    octv = int(parts[1])
                except Exception:
                    octv = 0
                base = 60
                midi = base + octv * 12 + C_MAJOR_PCS[deg]
                events.append((midi, max(1, dur)))
            continue

        if len(parts) >= 2:
            # NOTE DUR
            midi = _parse_note_token_to_midi(parts[0])
            try:
                dur = int(parts[1])
            except Exception:
                dur = 4
            events.append((midi, max(1, dur)))

    bars = _events_to_4bars_degree_template(events)
    return name, bars, meta


def _read_vlq(data: bytes, i: int) -> tuple[int, int]:
    v = 0
    while True:
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if (b & 0x80) == 0:
            break
    return v, i


def _parse_plugin_midi(path: Path) -> tuple[str, list[list[tuple[int | None, int, int]]]]:
    """Very small MIDI parser for monophonic melodies.

    Converts note durations to 'rows' assuming 4 rows per quarter note.
    """
    data = path.read_bytes()
    name = path.parent.name
    if not data.startswith(b"MThd"):
        return name, _events_to_4bars_degree_template([(60, 4)])

    hdr_len = int.from_bytes(data[4:8], "big")
    fmt = int.from_bytes(data[8:10], "big")
    ntr = int.from_bytes(data[10:12], "big")
    tpq = int.from_bytes(data[12:14], "big")
    off = 8 + hdr_len

    # Collect note on/off with absolute tick times.
    notes: list[tuple[int, int]] = []  # (midi, dur_ticks)

    for _ in range(ntr):
        if off + 8 > len(data) or data[off:off+4] != b"MTrk":
            break
        trk_len = int.from_bytes(data[off+4:off+8], "big")
        trk = data[off+8:off+8+trk_len]
        off += 8 + trk_len

        t = 0
        i = 0
        running = None
        on_map: dict[int, int] = {}

        while i < len(trk):
            dt, i = _read_vlq(trk, i)
            t += dt
            status = trk[i]
            if status < 0x80:
                if running is None:
                    break
                status = running
            else:
                i += 1
                running = status

            if status == 0xFF:
                # meta
                if i >= len(trk):
                    break
                meta_type = trk[i]
                i += 1
                ln, i = _read_vlq(trk, i)
                payload = trk[i:i+ln]
                i += ln
                if meta_type == 0x03:
                    # track name
                    try:
                        nm = payload.decode('utf-8', 'ignore').strip()
                        if nm:
                            name = nm
                    except Exception:
                        pass
                continue

            if status in (0xF0, 0xF7):
                ln, i = _read_vlq(trk, i)
                i += ln
                continue

            typ = status & 0xF0
            if typ in (0x80, 0x90):
                if i + 2 > len(trk):
                    break
                note = trk[i]
                vel = trk[i+1]
                i += 2
                is_on = (typ == 0x90 and vel > 0)
                if is_on:
                    # If multiple, keep the highest (melody-ish)
                    on_map[note] = t
                else:
                    t0 = on_map.pop(note, None)
                    if t0 is not None and t > t0:
                        notes.append((int(note), int(t - t0)))
                continue

            # Other MIDI events: skip params
            if typ in (0xA0, 0xB0, 0xE0):
                i += 2
            elif typ in (0xC0, 0xD0):
                i += 1
            else:
                # Unknown
                break

    # Convert to rows
    events: list[tuple[int | None, int]] = []
    for midi, dur_ticks in notes:
        # quarter note = tpq ticks -> 4 rows
        rows = int(round((dur_ticks / max(1, tpq)) * 4.0))
        rows = max(1, min(16, rows))
        events.append((midi, rows))

    if not events:
        events = [(60, 4), (62, 4), (64, 4), (65, 4), (67, 4), (69, 4), (71, 4), (72, 4)]

    bars = _events_to_4bars_degree_template(events)
    return name, bars


def load_melody_plugins(plugin_root: Path) -> dict[str, MelodyPlugin]:
    lib: dict[str, MelodyPlugin] = {}
    if not plugin_root.exists():
        return lib

    for sub in sorted([p for p in plugin_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        try:
            # Prefer conventional filenames so README files don't get mistaken as melodies.
            preferred_midi = None
            for cand in ("melody.mid", "melody.midi", "base.mid", "base.midi"):
                p = sub / cand
                if p.exists():
                    preferred_midi = p
                    break

            preferred_txt = None
            for cand in ("melody.txt", "base.txt"):
                p = sub / cand
                if p.exists():
                    preferred_txt = p
                    break

            midi_files: list[Path] = []
            txt_files: list[Path] = []
            if preferred_midi is None:
                midi_files = sorted([p for p in (list(sub.glob("*.mid")) + list(sub.glob("*.midi"))) if p.is_file()], key=lambda p: p.name.lower())
            if preferred_txt is None:
                txt_files = sorted([p for p in sub.glob("*.txt") if p.is_file() and p.name.lower() not in ("readme.txt", "info.txt", "meta.txt", "metadata.txt")], key=lambda p: p.name.lower())

            src: Path | None = None
            file_meta: dict[str, str] = {}
            if preferred_midi is not None:
                src = preferred_midi
                nm, bars = _parse_plugin_midi(preferred_midi)
            elif preferred_txt is not None:
                src = preferred_txt
                nm, bars, file_meta = _parse_plugin_txt(preferred_txt)
            elif midi_files:
                src = midi_files[0]
                nm, bars = _parse_plugin_midi(midi_files[0])
            elif txt_files:
                src = txt_files[0]
                nm, bars, file_meta = _parse_plugin_txt(txt_files[0])
            else:
                continue

            nm = (nm or sub.name).strip()
            if not nm or src is None:
                continue

            # Merge metadata:
            # - metadata embedded in melody.txt
            # - info/meta files next to the melody (also works for MIDI)
            info_meta = _read_plugin_metadata(sub)
            meta = dict(file_meta)
            meta.update(info_meta)

            # De-duplicate names (keep stable order)
            base_nm = nm
            if base_nm in lib:
                i = 2
                while f"{base_nm} ({i})" in lib:
                    i += 1
                nm = f"{base_nm} ({i})"

            lib[nm] = MelodyPlugin(name=nm, bars=bars, meta=meta, folder=sub, source=src)
        except Exception:
            continue

    return lib



try:
    _PLUGIN_ROOT = _default_plugin_root()
    ensure_default_melody_plugins(_PLUGIN_ROOT)
    PLUGIN_MELODIES = load_melody_plugins(_PLUGIN_ROOT)
except Exception:
    PLUGIN_MELODIES = {}


def get_melody_choices() -> list[str]:
    names = sorted(list(PLUGIN_MELODIES.keys()))
    return ["Random", "Pure Random"] + names

MELODY_CHOICES = get_melody_choices()

def reload_melody_plugins() -> list[str]:
    """Reload melody plugins from disk (used by the GUI Refresh button)."""
    global PLUGIN_MELODIES, MELODY_CHOICES, _PLUGIN_ROOT
    try:
        _PLUGIN_ROOT = _default_plugin_root()
        ensure_default_melody_plugins(_PLUGIN_ROOT)
        PLUGIN_MELODIES = load_melody_plugins(_PLUGIN_ROOT)
    except Exception:
        PLUGIN_MELODIES = {}
    MELODY_CHOICES = get_melody_choices()
    return MELODY_CHOICES


def get_plugin_metadata_display(name: str) -> str:
    try:
        pl = PLUGIN_MELODIES.get(name)
        if isinstance(pl, MelodyPlugin):
            meta = pl.meta or {}
            if not meta:
                return ""
            order = ["mode", "preferred_key_range", "tempo_hint"]
            parts = []
            for k in order:
                if k in meta:
                    parts.append(f"{k}={meta[k]}")
            # Add any extra keys (stable)
            for k in sorted([k for k in meta.keys() if k not in order]):
                parts.append(f"{k}={meta[k]}")
            return " | ".join(parts)
    except Exception:
        pass
    return ""


# Reference fundamental for all generated samples (Hz). Tuned so that C-3 plays consistently across instruments.
REF_F0 = 261.63

INSTRUMENT_CHOICES = [
    "Piano",
    "Clarinet",
    "Sax",
    "Synth Pad",
    "Violin",
    "Strings",
    "Choir Aah",
    "Tuba",
    "French Horn",
    "Trumpet",
    "Banjo",
    "Panflute",
    "Acoustic Guitar",
    "Flamenco Guitar",
    "Harp",
    "Organ",
    "Electric Piano",
    "Celesta",
    "Bell",
    "Flute",
    "Oboe",
]

DEFAULT_INSTRUMENTS = ["Piano", "Piano", "Piano", "Piano"]

AMIGA_PAL_CLOCK = 7093789.2


# -----------------------------
# MOD packing helpers
# -----------------------------

def note_shift(note: str, semitones: int) -> str:
    i = CHROMATIC.index(note)
    j = i + semitones
    j = max(0, min(len(CHROMATIC) - 1, j))
    return CHROMATIC[j]


def pack_cell(note_name: str | None = None, sample: int = 0, effect: int = 0, param: int = 0) -> bytes:
    period = 0 if note_name is None else PERIODS[note_name]
    samp = sample & 0x1F
    b0 = ((samp & 0x10) << 4) | ((period >> 8) & 0x0F)
    b1 = period & 0xFF
    b2 = ((samp & 0x0F) << 4) | (effect & 0x0F)
    b3 = param & 0xFF
    return bytes([b0, b1, b2, b3])


def inst_header(
    name: str,
    sample_bytes: bytes,
    finetune: int = 0,
    volume: int = 48,
    loop_start: int = 0,
    loop_len_words: int = 1,
) -> bytes:
    name_b = name.encode("ascii", "ignore")[:22].ljust(22, b"\x00")
    length_words = (len(sample_bytes) // 2) & 0xFFFF
    return (
        name_b
        + struct.pack(">H", length_words)
        + bytes([finetune & 0x0F])
        + bytes([max(0, min(64, volume))])
        + struct.pack(">H", loop_start & 0xFFFF)
        + struct.pack(">H", loop_len_words & 0xFFFF)
    )


# -----------------------------
# Sample synthesis (8-bit signed)
# -----------------------------

def make_pianoish_sample(rng: random.Random, length: int = 32768, sr: int = 8287, f0: float = REF_F0) -> bytes:
    attack = int(sr * rng.uniform(0.004, 0.008))
    decay = rng.uniform(0.9, 1.6)
    detune = rng.uniform(0.9990, 1.0025)

    h2 = rng.uniform(0.35, 0.50)
    h3 = rng.uniform(0.18, 0.28)
    h4 = rng.uniform(0.10, 0.20)
    d2 = rng.uniform(0.04, 0.10)

    data = bytearray()
    for n in range(length):
        t = n / sr

        x = (
            math.sin(2 * math.pi * f0 * t) * 1.00
            + math.sin(2 * math.pi * f0 * 2 * t) * h2
            + math.sin(2 * math.pi * f0 * 3 * t) * h3
            + math.sin(2 * math.pi * f0 * 4 * t) * h4
            + math.sin(2 * math.pi * (f0 * detune) * t) * d2
        )

        if t < 0.02:
            noise = (math.sin(2 * math.pi * 3200 * t) + math.sin(2 * math.pi * 1900 * t)) * 0.08
            x += noise * (1 - (t / 0.02))

        env = math.exp(-decay * t)
        if n < attack:
            env *= (n / max(1, attack))

        y = math.tanh(1.25 * x) * env
        v = int(max(-127, min(127, round(y * 120))))
        data.append(v & 0xFF)

    if len(data) % 2 == 1:
        data.append(0)
    return bytes(data)


def _one_pole_lowpass(x: float, state: float, alpha: float) -> float:
    return state + alpha * (x - state)


def make_instrument_sample(kind: str, rng: random.Random, length: int = 32768, sr: int = 8287, f0: float = REF_F0, disable_vibrato: bool = False) -> bytes:
    kind = (kind or "").strip()
    if kind not in INSTRUMENT_CHOICES:
        kind = "Piano"

    if kind == "Piano":
        return make_pianoish_sample(rng, length=length, sr=sr, f0=f0)

    detune = rng.uniform(0.9990, 1.0015)
    vib_rate = rng.uniform(4.5, 6.2)
    vib_amt = rng.uniform(0.0, 0.0030) if kind in ("Violin", "Strings", "Synth Pad", "Choir Aah", "Panflute", "Flute") else rng.uniform(0.0, 0.0015)

    if kind == "Organ":
        vib_amt = 0.0

    if disable_vibrato:
        vib_amt = 0.0

    # Envelope choices (kept conservative so pitch feels stable)
    if kind in ("Synth Pad", "Violin", "Strings", "Choir Aah", "Panflute", "Clarinet", "Sax", "Flute", "Oboe", "Organ", "French Horn", "Trumpet"):
        attack = int(sr * rng.uniform(0.012, 0.040))
        decay = rng.uniform(0.18, 0.55)
    elif kind == "Tuba":
        attack = int(sr * rng.uniform(0.015, 0.040))
        decay = rng.uniform(0.35, 0.70)
    else:  # Plucked (Banjo / Guitars)
        attack = int(sr * rng.uniform(0.002, 0.007))
        decay = rng.uniform(1.2, 2.9)

    if kind == "Organ":
        decay = rng.uniform(0.02, 0.08)
    if kind == "Flute":
        decay = rng.uniform(0.14, 0.32)

    noise_amt = 0.0
    drive = 1.1
    lp_alpha = 1.0
    partials: list[tuple[int, float]] = [(1, 1.0)]

    if kind == "Clarinet":
        partials = [(1, 1.0), (3, 0.55), (5, 0.35), (7, 0.22), (2, 0.08)]
        noise_amt = 0.020
        drive = 1.35
        lp_alpha = 0.22
    elif kind == "Sax":
        partials = [(1, 1.0), (2, 0.42), (3, 0.36), (4, 0.22), (5, 0.18), (6, 0.12)]
        noise_amt = 0.028
        drive = 1.55
        lp_alpha = 0.20
    elif kind == "Synth Pad":
        partials = [(1, 1.0), (2, 0.24), (3, 0.18), (4, 0.12)]
        noise_amt = 0.010
        drive = 1.10
        lp_alpha = 0.28
    elif kind == "Violin":
        partials = [(1, 1.0), (2, 0.60), (3, 0.45), (4, 0.30), (5, 0.22), (6, 0.16), (7, 0.12)]
        noise_amt = 0.012
        drive = 1.25
        lp_alpha = 0.18
    elif kind == "Tuba":
        partials = [(1, 1.0), (2, 0.40), (3, 0.25), (4, 0.12)]
        noise_amt = 0.006
        drive = 1.15
        lp_alpha = 0.12
    elif kind == "Banjo":
        partials = [(1, 1.0), (2, 0.52), (3, 0.42), (4, 0.32), (5, 0.24), (6, 0.18), (7, 0.12), (8, 0.10)]
        noise_amt = 0.018
        drive = 1.45
        lp_alpha = 0.32
    elif kind == "Panflute":
        partials = [(1, 1.0), (2, 0.18), (3, 0.08)]
        noise_amt = 0.030
        drive = 1.10
        lp_alpha = 0.16
    elif kind == "Acoustic Guitar":
        partials = [(1, 1.0), (2, 0.70), (3, 0.52), (4, 0.38), (5, 0.30), (6, 0.22), (7, 0.17), (8, 0.13), (9, 0.10)]
        noise_amt = 0.015
        drive = 1.38
        lp_alpha = 0.26
    elif kind == "Flamenco Guitar":
        partials = [(1, 1.0), (2, 0.74), (3, 0.56), (4, 0.42), (5, 0.34), (6, 0.26), (7, 0.19), (8, 0.15), (9, 0.11), (10, 0.08)]
        noise_amt = 0.020
        drive = 1.50
        lp_alpha = 0.36
    elif kind == "Organ":
        partials = [(1, 1.0), (2, 0.55), (3, 0.35), (4, 0.25), (5, 0.18), (6, 0.12)]
        noise_amt = 0.002
        drive = 1.10
        lp_alpha = 0.22
    elif kind == "Flute":
        partials = [(1, 1.0), (2, 0.08), (3, 0.03)]
        noise_amt = 0.018
        drive = 1.08
        lp_alpha = 0.14
    elif kind == "Oboe":
        partials = [(1, 1.0), (2, 0.30), (3, 0.45), (4, 0.20), (5, 0.18), (6, 0.10)]
        noise_amt = 0.020
        drive = 1.28
        lp_alpha = 0.18

    elif kind == "Strings":
        partials = [(1, 1.0), (2, 0.55), (3, 0.40), (4, 0.28), (5, 0.20), (6, 0.14)]
        noise_amt = 0.010
        drive = 1.22
        lp_alpha = 0.18
    elif kind == "Choir Aah":
        partials = [(1, 1.0), (2, 0.30), (3, 0.22), (4, 0.16), (5, 0.11), (6, 0.08)]
        noise_amt = 0.020
        drive = 1.12
        lp_alpha = 0.20
    elif kind == "French Horn":
        partials = [(1, 1.0), (2, 0.38), (3, 0.28), (4, 0.18), (5, 0.12)]
        noise_amt = 0.010
        drive = 1.22
        lp_alpha = 0.14
    elif kind == "Trumpet":
        partials = [(1, 1.0), (2, 0.32), (3, 0.30), (4, 0.22), (5, 0.16), (6, 0.10)]
        noise_amt = 0.018
        drive = 1.40
        lp_alpha = 0.20
    elif kind == "Harp":
        partials = [(1, 1.0), (2, 0.72), (3, 0.50), (4, 0.34), (5, 0.24), (6, 0.18), (7, 0.12)]
        noise_amt = 0.012
        drive = 1.28
        lp_alpha = 0.30
    elif kind == "Electric Piano":
        partials = [(1, 1.0), (2, 0.48), (3, 0.30), (4, 0.18), (5, 0.12)]
        noise_amt = 0.006
        drive = 1.18
        lp_alpha = 0.26
    elif kind == "Celesta":
        partials = [(1, 1.0), (2, 0.22), (3, 0.10), (4, 0.06)]
        noise_amt = 0.002
        drive = 1.06
        lp_alpha = 0.40
    elif kind == "Bell":
        partials = [(1, 1.0), (2, 0.28), (3, 0.12), (4, 0.07)]
        noise_amt = 0.001
        drive = 1.04
        lp_alpha = 0.45


    buf = [0.0] * length
    lp_state = 0.0

    for n in range(length):
        t = n / sr
        f = f0 * (1.0 + vib_amt * math.sin(2 * math.pi * vib_rate * t))

        x = 0.0
        for k, a in partials:
            x += a * math.sin(2 * math.pi * (f * k) * t)

        if kind in ("Synth Pad", "Violin", "Sax"):
            x += 0.12 * math.sin(2 * math.pi * (f * detune) * t)

        if noise_amt > 0.0:
            x += rng.uniform(-1.0, 1.0) * noise_amt

        # transient / pick noise for plucked instruments
        if kind in ("Banjo", "Acoustic Guitar", "Flamenco Guitar") and t < 0.020:
            amt = 0.10 if kind == "Banjo" else (0.12 if kind == "Acoustic Guitar" else 0.15)
            x += math.sin(2 * math.pi * 3100 * t) * (amt * (1.0 - (t / 0.020)))
            if kind == "Flamenco Guitar":
                x += math.sin(2 * math.pi * 4200 * t) * (0.06 * (1.0 - (t / 0.020)))

        if lp_alpha < 1.0:
            lp_state = _one_pole_lowpass(x, lp_state, lp_alpha)
            x = lp_state

        env = math.exp(-decay * t)
        if n < attack:
            env *= (n / max(1, attack))

        y = math.tanh(drive * x) * env
        buf[n] = y

    mx = max(1e-6, max(abs(v) for v in buf))
    scale = 120.0 / mx
    data = bytearray()
    for v in buf:
        s = int(max(-127, min(127, round(v * scale))))
        data.append(s & 0xFF)

    if len(data) % 2 == 1:
        data.append(0)
    return bytes(data)


def normalize_instrument_list(insts: list[str] | None) -> list[str]:
    if not insts or len(insts) != 4:
        return DEFAULT_INSTRUMENTS[:]
    out: list[str] = []
    for x in insts[:4]:
        x = (x or "").strip()
        out.append(x if x in INSTRUMENT_CHOICES else "Piano")
    return out


def bytes_to_float_sample(sample_bytes: bytes) -> list[float]:
    # MOD samples are 8-bit signed, stored as bytes 0..255.
    out = [0.0] * len(sample_bytes)
    for i, b in enumerate(sample_bytes):
        v = b - 256 if b > 127 else b
        out[i] = v / 128.0
    return out


# -----------------------------
# Music generation
# -----------------------------

def major_scale(root_note: str) -> list[str]:
    intervals = [0, 2, 4, 5, 7, 9, 11]
    return [note_shift(root_note, i) for i in intervals]


def triad_from_degree(scale: list[str], degree: int, octave_bias: int = 0) -> tuple[str, str, str]:
    r = scale[degree % 7]
    t = scale[(degree + 2) % 7]
    f = scale[(degree + 4) % 7]
    if octave_bias != 0:
        r = note_shift(r, 12 * octave_bias)
        t = note_shift(t, 12 * octave_bias)
        f = note_shift(f, 12 * octave_bias)
    return r, t, f


def pick_progression(rng: random.Random) -> list[int]:
    start = rng.choice([0, 5])
    mid_pool = [1, 3, 4, 5, 2]
    prog = [start]
    for _ in range(2):
        prog.append(rng.choice(mid_pool))
    prog.append(rng.choice([4, 3]))
    return prog


def build_bar_melody(
    rng: random.Random,
    scale: list[str],
    chord: tuple[str, str, str],
    base_note: str,
) -> list[tuple[str | None, int]]:
    chord_tones = list(chord)
    current = base_note if base_note in chord_tones else rng.choice(chord_tones)

    events: list[tuple[str | None, int]] = []
    remaining = 16

    n_events = rng.choice([3, 4, 5])
    durs: list[int] = []
    for i in range(n_events):
        if i == n_events - 1:
            durs.append(remaining)
        else:
            d = rng.choice([2, 4, 4, 6])
            d = min(d, remaining - (n_events - i - 1) * 2)
            durs.append(max(2, d))
            remaining -= durs[-1]

    for i, dur in enumerate(durs):
        if i == n_events - 1:
            note = rng.choice(chord_tones)
        else:
            if rng.random() < 0.18:
                note = None
            else:
                if rng.random() < 0.70:
                    step = rng.choice([-2, -1, 1, 2])
                    candidate = note_shift(current, step)
                    if candidate in scale:
                        note = candidate
                    else:
                        note = rng.choice(chord_tones)
                else:
                    note = rng.choice(chord_tones)
        if note is not None:
            current = note
        events.append((note, dur))

    if all(n is None for n, _ in events):
        events[0] = (rng.choice(chord_tones), events[0][1])
    return events



def _template_bar_to_events(
    scale_up: list[str],
    bar_tpl: list[tuple[int | None, int, int]],
) -> list[tuple[str | None, int]]:
    events: list[tuple[str | None, int]] = []
    for deg, octv, dur in bar_tpl:
        if dur <= 0:
            continue
        if deg is None:
            events.append((None, int(dur)))
            continue
        note = scale_up[int(deg) % 7]
        if int(octv) != 0:
            note = note_shift(note, 12 * int(octv))
        while note not in CHROMATIC_SET and note.endswith('3'):
            note = note_shift(note, -12)
        if note not in CHROMATIC_SET:
            note = note_shift(scale_up[0], 0)
        events.append((note, int(dur)))
    if all(n is None for n, _ in events):
        events[0] = (scale_up[0], events[0][1])
    return events


def _mutate_events(
    rng: random.Random,
    events: list[tuple[str | None, int]],
    scale_up: list[str],
    mode: str,
) -> list[tuple[str | None, int]]:
    """Mutate a bar's (note,dur) events.

    Near/far derivation uses these modes to add variation while staying in-key.
    """
    out: list[tuple[str | None, int]] = [(n, int(d)) for (n, d) in events]

    def _nearest_in_scale(note: str, step: int) -> str:
        cand = note_shift(note, step)
        if cand in scale_up:
            return cand
        for s in (step - 1, step + 1, -step, 0):
            cc = note_shift(note, s)
            if cc in scale_up:
                return cc
        return note

    m = (mode or 'base').strip().lower()

    if m == 'base':
        return out

    if m == 'transpose_up':
        out2: list[tuple[str | None, int]] = []
        for n, d in out:
            if n is None:
                out2.append((None, d))
            else:
                out2.append((_nearest_in_scale(n, rng.choice([1, 2])), d))
        return out2

    if m == 'lift':
        # octave lift where possible
        out2: list[tuple[str | None, int]] = []
        for n, d in out:
            if n is None:
                out2.append((None, d))
            else:
                nn = note_shift(n, 12)
                if nn not in CHROMATIC_SET:
                    nn = _nearest_in_scale(n, rng.choice([1, 2]))
                out2.append((nn if nn in CHROMATIC_SET else n, d))
        return out2

    if m == 'answer':
        out2 = []
        for i, (n, d) in enumerate(out):
            if n is None:
                out2.append((None, d))
                continue
            if i >= len(out) - 2 and rng.random() < 0.85:
                out2.append((_nearest_in_scale(n, rng.choice([-1, -2])), d))
            else:
                out2.append((n, d))
        return out2

    if m == 'cadence':
        out2 = []
        for i, (n, d) in enumerate(out):
            if i == len(out) - 1:
                out2.append((scale_up[0], d))
            elif i == len(out) - 2 and n is not None and d >= 2:
                out2.append((scale_up[1], d))
            else:
                out2.append((n, d))
        return out2

    if m == 'ornament':
        out2: list[tuple[str | None, int]] = []
        for n, d in out:
            if n is not None and d >= 4 and rng.random() < 0.55:
                d1 = 2
                d2 = max(1, d - 2)
                pn = _nearest_in_scale(n, rng.choice([-2, -1, 1, 2]))
                out2.append((n, d1))
                out2.append((pn, d2))
            else:
                out2.append((n, d))
        return out2

    if m == 'drive':
        # Add a little "push" by splitting longer notes and adding neighbor motion.
        out2: list[tuple[str | None, int]] = []
        for n, d in out:
            if n is None or d <= 2:
                out2.append((n, d))
                continue
            if d >= 4 and rng.random() < 0.75:
                # e.g. 2 + 2 (+ remainder)
                nn = _nearest_in_scale(n, rng.choice([-1, 1, 2, -2]))
                out2.append((n, 2))
                out2.append((nn, 2))
                rem = d - 4
                if rem > 0:
                    out2.append((n, rem))
            else:
                out2.append((n, d))
        return out2

    if m == 'arp':
        # Convert some sustained notes into small repeating figures.
        out2: list[tuple[str | None, int]] = []
        for n, d in out:
            if n is None or d <= 2:
                out2.append((n, d))
                continue
            if d >= 4 and rng.random() < 0.85:
                step = rng.choice([-2, -1, 1, 2])
                nn = _nearest_in_scale(n, step)
                seg = 2
                reps = max(1, d // seg)
                for i in range(reps):
                    out2.append((n if (i % 2 == 0) else nn, seg))
                tail = d - reps * seg
                if tail > 0:
                    out2.append((n, tail))
            else:
                out2.append((n, d))
        return out2

    if m == 'turn':
        # Add a short pickup into some notes.
        out2: list[tuple[str | None, int]] = []
        for n, d in out:
            if n is None or d <= 2:
                out2.append((n, d))
                continue
            if rng.random() < 0.65:
                pn = _nearest_in_scale(n, rng.choice([-1, -2]))
                out2.append((pn, 1))
                out2.append((n, d - 1))
            else:
                out2.append((n, d))
        return out2

    return out

def _pick_base_melody(
    rng: random.Random,
    melody_name: str | None,
) -> tuple[str, list[list[tuple[int | None, int, int]]] | None, dict[str, str]]:
    """Pick a base melody.

    - melody_name == "Random": choose a random plugin melody (if any)
    - melody_name == "Pure Random": return None => algorithmic base melody
    - otherwise: pick by exact name if found (plugins first, then built-in fallback)
    """
    if melody_name:
        mn = str(melody_name).strip()
        if mn.lower() == "pure random":
            return "Pure Random", None, {}

        # Plugin melodies
        if mn not in ("Random", "Pure Random") and mn in PLUGIN_MELODIES:
            pl = PLUGIN_MELODIES[mn]
            if isinstance(pl, MelodyPlugin):
                return mn, pl.bars, dict(pl.meta or {})
            # (old fallback)
            return mn, pl, {}

        # Built-in fallback (should rarely be used because we auto-seed plugins)
        if mn not in ("Random", "Pure Random") and mn in MELODY_LIBRARY:
            return mn, MELODY_LIBRARY[mn], {}

    # Random selection
    if PLUGIN_MELODIES:
        nm = rng.choice(list(PLUGIN_MELODIES.keys()))
        pl = PLUGIN_MELODIES[nm]
        if isinstance(pl, MelodyPlugin):
            return nm, pl.bars, dict(pl.meta or {})
        return nm, pl, {}

    if MELODY_LIBRARY:
        nm = rng.choice(list(MELODY_LIBRARY.keys()))
        return nm, MELODY_LIBRARY[nm], {}

    return "Pure Random", None, {}

def make_patterns(
    rng: random.Random,
    speed: int = DEFAULT_SPEED,
    tempo: int = DEFAULT_TEMPO,
    melody_name: str | None = None,
    derive_mode: str | None = None,
):
    NUM_CH = 4
    ROWS = 64
    patterns: list[list[list[tuple[str | None, int, int, int]]]] = []

    key_root = rng.choice(['C-2', 'G-2', 'F-2', 'D-2'])
    scale = major_scale(key_root)
    scale_up = [note_shift(n, 12) for n in scale]

    base_melody_name, base_tpl, base_meta = _pick_base_melody(rng, melody_name)

    if base_tpl is None:
        # Pure algorithmic base melody
        base_prog = [0, 3, 4, 0]
        base_bars = []
        for deg in base_prog:
            rt, th, fi = triad_from_degree(scale, deg, octave_bias=0)
            chord = [note_shift(rt, 12), note_shift(th, 12), note_shift(fi, 12)]
            chord = [n if n in CHROMATIC_SET else scale_up[0] for n in chord]
            base_bars.append(build_bar_melody(rng, scale=scale_up, chord=chord, base_note=chord[0]))
    else:
        base_bars = [_template_bar_to_events(scale_up, base_tpl[i]) for i in range(4)]

    N_PAT = 10
    for _ in range(N_PAT):
        pat = [[(None, 0, 0, 0) for _ in range(NUM_CH)] for _ in range(ROWS)]
        patterns.append(pat)

    def set_cell(p: int, row: int, ch: int, note: str | None = None, sample: int | None = None, effect: int = 0x00, param: int = 0x00):
        if note is None:
            samp = 0 if sample is None else sample
        else:
            samp = (ch + 1) if sample is None else sample
        if 0 <= row < 64:
            patterns[p][row][ch] = (note, samp, effect, param)

    # 10 progressions (degree in major scale) - designed to stay "choral" but offer more variety
    progs = [
        [0, 3, 4, 0],  # 0: base
        [0, 4, 5, 3],  # 1: ornament
        [3, 0, 4, 0],  # 2: answer
        [0, 3, 0, 4],  # 3: pad/hold
        [5, 3, 4, 0],  # 4: answer/sequence
        [0, 2, 3, 4],  # 5: cadence-ish
        [0, 3, 5, 4],  # 6: drive (pushes to V)
        [0, 2, 5, 3],  # 7: arp / motion
        [4, 3, 2, 0],  # 8: lift / descending tension-release
        [0, 5, 3, 4],  # 9: turnaround
    ]

    # Derivation style: Near = more recognizable, Far = motif-only (more variation)
    dm = (derive_mode or 'Random').strip().lower()
    if dm in ('random', 'auto'):
        dm = rng.choice(['near', 'far'])

    if dm.startswith('n') or dm.startswith('c'):
        mode_for_pattern = {
            0: 'base',
            1: 'ornament',
            2: 'base',
            3: 'pad',
            4: 'answer',
            5: 'cadence',
            6: 'drive',
            7: 'arp',
            8: 'lift',
            9: 'turn',
        }
    else:
        mode_for_pattern = {
            0: 'base',
            1: 'ornament',
            2: 'transpose_up',
            3: 'pad',
            4: 'answer',
            5: 'cadence',
            6: 'drive',
            7: 'arp',
            8: 'lift',
            9: 'turn',
        }

    derive_used = "Near" if (dm.startswith('n') or dm.startswith('c')) else "Far"

    for p_idx in range(N_PAT):
        prog = progs[p_idx]
        for bar, deg in enumerate(prog):
            r0 = bar * 16
            start_row = r0

            if p_idx == 3:
                bar_events = []
                strong_note = scale_up[0]
            else:
                mode = mode_for_pattern.get(p_idx, 'base')
                bar_events = _mutate_events(rng, base_bars[bar], scale_up, mode)
                strong_note = next((n for (n, _) in bar_events if n is not None), scale_up[0])

            def _chord_up_for_degree(d: int):
                rt, th, fi = triad_from_degree(scale, d, octave_bias=0)
                cu = (note_shift(rt, 12), note_shift(th, 12), note_shift(fi, 12))
                cu = tuple(n if n in CHROMATIC_SET else scale_up[0] for n in cu)
                return rt, th, fi, cu

            root, third, fifth, chord_up = _chord_up_for_degree(deg)

            if p_idx != 3 and strong_note is not None and strong_note not in chord_up:
                for cand in [0, 3, 4, 5, 2, 1]:
                    rrt, rth, rfi, cu = _chord_up_for_degree(cand)
                    if strong_note in cu:
                        root, third, fifth, chord_up = rrt, rth, rfi, cu
                        break

            bass = note_shift(root, -12)
            top = fifth

            # basic harmony bed
            set_cell(p_idx, start_row, 1, top)
            set_cell(p_idx, start_row, 2, bass)
            set_cell(p_idx, start_row, 3, third)

            if p_idx != 3 and rng.random() < 0.55:
                set_cell(p_idx, start_row + 8, 1, top)
                set_cell(p_idx, start_row + 8, 2, bass)

            # pad pattern
            if p_idx == 3:
                if bar == 0:
                    hold = rng.choice([third, fifth, note_shift(root, 12)])
                    hold = note_shift(hold, 12) if hold.endswith('2') else hold
                    hold = hold if hold in CHROMATIC_SET else scale_up[0]
                    set_cell(p_idx, start_row, 0, hold)
                elif bar == 1:
                    hold = rng.choice([fifth, third])
                    hold = note_shift(hold, 12) if hold.endswith('2') else hold
                    hold = hold if hold in CHROMATIC_SET else scale_up[0]
                    set_cell(p_idx, start_row, 0, hold)
                elif bar == 2:
                    hold = rng.choice([third, root])
                    hold = note_shift(hold, 12) if hold.endswith('2') else hold
                    hold = hold if hold in CHROMATIC_SET else scale_up[0]
                    set_cell(p_idx, start_row, 0, hold)
                else:
                    a = note_shift(root, 12)
                    b = note_shift(third, 12)
                    a = a if a in CHROMATIC_SET else scale_up[0]
                    b = b if b in CHROMATIC_SET else scale_up[0]
                    set_cell(p_idx, start_row, 0, a)
                    set_cell(p_idx, start_row + 8, 0, b)
            else:
                # melody
                if p_idx == 0 and bar == 0 and bar_events:
                    n_last, d_last = bar_events[-1]
                    if d_last > 1:
                        bar_events = list(bar_events)
                        bar_events[-1] = (n_last, d_last - 1)

                r = start_row
                for note, dur in bar_events:
                    if r >= 64:
                        break
                    if note is not None and note in CHROMATIC_SET:
                        set_cell(p_idx, r, 0, note)
                    r += max(1, int(dur))

            # extra motion layers for the new patterns
            if p_idx == 6:
                # rhythmic "walking" bass + chord stabs
                bass2 = note_shift(third, -12)
                bass5 = note_shift(fifth, -12)
                seq = [bass, bass5, bass2, bass5]
                for i in range(0, 16, 4):
                    set_cell(p_idx, start_row + i, 2, seq[(i // 4) % len(seq)])
                # stabs on offbeats
                stab = chord_up[1]  # third up
                for i in (2, 6, 10, 14):
                    set_cell(p_idx, start_row + i, 1, stab)

            if p_idx == 7:
                # arpeggio shimmer on CH4
                arp = [chord_up[0], chord_up[1], chord_up[2], chord_up[1]]
                for i in range(0, 16, 2):
                    set_cell(p_idx, start_row + i, 3, arp[(i // 2) % len(arp)])

            if p_idx == 8:
                # lift: add extra top octave support
                hi = note_shift(chord_up[2], 12)
                if hi in CHROMATIC_SET and rng.random() < 0.7:
                    set_cell(p_idx, start_row + 8, 1, hi)

            if p_idx == 9:
                # turnaround: a short answer in CH4
                tones = [chord_up[2], chord_up[1], chord_up[0], chord_up[1]]
                for i in range(8, 16, 2):
                    set_cell(p_idx, start_row + i, 3, tones[((i - 8) // 2) % len(tones)])

            # keep legacy arpeggio feel in some patterns
            if p_idx in (2, 4) and rng.random() < 0.75:
                tones = [third, root, fifth, root]
                tones = [note_shift(t, 12) if t.endswith('2') else t for t in tones]
                tones = [t if t in CHROMATIC_SET else scale_up[0] for t in tones]
                for i in range(0, 16, 2):
                    set_cell(p_idx, start_row + i, 3, tones[(i // 2) % len(tones)])

    # Ensure the selected speed/tempo is present in EVERY pattern.
    spd = max(1, min(31, int(speed)))
    bpm = max(32, min(255, int(tempo)))
    for pat in patterns:
        n, s, eff, par = pat[0][0]
        pat[0][0] = (n, s, 0x0F, spd)
        n, s, eff, par = pat[0][1]
        pat[0][1] = (n, s, 0x0F, bpm)

    return patterns, key_root, base_melody_name, base_meta, derive_used

def apply_end_slowdown_to_pattern(pattern, rng: random.Random):
    slow_tempo = rng.choice([0x64, 0x5A, 0x50])  # 100 / 90 / 80 BPM
    for row in range(64):
        for ch in range(4):
            note, samp, eff, param = pattern[row][ch]
            if eff == 0x00 and param == 0x00:
                pattern[row][ch] = (note, samp, 0x0F, slow_tempo)
                return
    # Fallback: overwrite very last cell's effect (rare)
    note, samp, eff, param = pattern[63][3]
    pattern[63][3] = (note, samp, 0x0F, slow_tempo)


def patterns_to_bytes(patterns) -> bytes:
    blob = bytearray()
    for pat in patterns:
        for r in range(64):
            for ch in range(4):
                note, samp, eff, par = pat[r][ch]
                blob += pack_cell(note, samp, eff, par)
        if len(blob) % 1024 != 0:
            raise RuntimeError("Pattern size mismatch")
    return bytes(blob)


def parse_order_string(order_str: str) -> list[int]:
    parts = [p.strip() for p in re.split(r"[,\s]+", order_str.strip()) if p.strip()]
    if not parts:
        raise ValueError("Order is empty.")
    order: list[int] = []
    for p in parts:
        if not re.fullmatch(r"-?\d+", p):
            raise ValueError(f"Invalid token '{p}'. Use numbers like: 0, 1, 2 ...")
        order.append(int(p))
    return order


def validate_order(order: list[int], n_patterns: int = 10) -> None:
    if len(order) > 128:
        raise ValueError("Order is too long (max 128 positions).")
    bad = [x for x in order if x < 0 or x >= n_patterns]
    if bad:
        raise ValueError(f"Order contains out-of-range pattern numbers {bad}. Allowed: 0..{n_patterns-1}")


@dataclass
class SongData:
    title_txt: str
    seed: int
    key_root: str
    base_melody: str
    base_melody_meta: dict[str, str]
    patterns: list
    order_original: list[int]
    order: list[int]
    samples_bytes: list[bytes]
    samples_float: list[list[float]]
    instrument_kinds: list[str]
    speed: int
    tempo: int
    slowdown_enabled: bool
    derive_mode: str
    vibrato_disabled: bool


def _cell_to_text(cell: tuple[str | None, int, int, int]) -> str:
    note, samp, eff, par = cell
    n = note if note is not None else "---"
    s = f"{samp:02d}" if samp else "--"
    e = f"{eff:X}{par:02X}" if (eff or par) else "---"
    return f"{n} {s} {e}"


def save_song_parameters_txt(mod_path: Path, song: SongData) -> Path:
    """Write a structured .txt sidecar next to the MOD.

    Skips writing if the file already exists.
    """
    txt_path = mod_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path

    lines: list[str] = []
    lines.append("ProTracker MOD Choral Generator - Song Parameters")
    lines.append("=")
    lines.append(f"mod_file: {mod_path.name}")
    lines.append(f"seed: {song.seed}")
    lines.append(f"title: {song.title_txt}")
    lines.append(f"key_root: {song.key_root}")
    lines.append(f"base_melody: {song.base_melody}")
    if getattr(song, "base_melody_meta", None):
        try:
            if song.base_melody_meta:
                lines.append("base_melody_meta:")
                for k in sorted(song.base_melody_meta.keys()):
                    lines.append(f"  {k}: {song.base_melody_meta[k]}")
                lines.append("")
        except Exception:
            pass
    lines.append(f"speed: {song.speed}")
    lines.append(f"tempo: {song.tempo}")
    lines.append(f"slowdown_enabled: {bool(song.slowdown_enabled)}")
    lines.append(f"derive_mode: {getattr(song, 'derive_mode', '')}")
    lines.append(f"vibrato_disabled: {bool(getattr(song, 'vibrato_disabled', False))}")
    lines.append("")
    lines.append("instruments:")
    for i, k in enumerate(song.instrument_kinds, start=1):
        lines.append(f"  ch{i}: {k} (sample {i})")
    lines.append("")

    lines.append("order_original:")
    lines.append("  " + ", ".join(str(x) for x in song.order_original))
    lines.append("order_final:")
    lines.append("  " + ", ".join(str(x) for x in song.order))
    lines.append("")

    lines.append("patterns:")
    # Keep it very readable and tracker-like.
    for p_idx, pat in enumerate(song.patterns):
        lines.append("")
        lines.append(f"PATTERN {p_idx}:")
        lines.append("row | CH1            | CH2            | CH3            | CH4")
        lines.append("----+----------------+----------------+----------------+----------------")
        for r in range(64):
            c0 = _cell_to_text(pat[r][0])
            c1 = _cell_to_text(pat[r][1])
            c2 = _cell_to_text(pat[r][2])
            c3 = _cell_to_text(pat[r][3])
            lines.append(f"{r:02d}  | {c0:<14} | {c1:<14} | {c2:<14} | {c3:<14}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def export_rendered_wav(wav_bytes: bytes, wav_path: Path) -> tuple[bool, str]:
    """Export already-rendered WAV bytes to disk (skip if it already exists)."""
    if wav_path.exists():
        return False, "WAV already exists (skipped)."
    try:
        wav_path.write_bytes(wav_bytes)
        return True, f"Exported WAV: {wav_path.name}"
    except Exception as e:
        return False, f"WAV export failed: {e}"


def generate_song(
    out_dir: str = "mods_out",
    seed: int | None = None,
    order: list[int] | None = None,
    enable_slowdown: bool = True,
    speed: int = DEFAULT_SPEED,
    tempo: int = DEFAULT_TEMPO,
    instruments: list[str] | None = None,
    melody_name: str | None = None,
    derive_mode: str | None = "Random",
    disable_vibrato: bool = False,
) -> tuple[Path, SongData]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = int(time.time() * 1000) ^ (os.getpid() << 8)
    rng = random.Random(seed)

    inst_kinds = normalize_instrument_list(instruments)

    # Generate sample bytes (4 slots). If the same instrument is selected, we still keep distinct sample numbers.
    sample_cache: dict[tuple[str, bool], bytes] = {}
    samples_bytes: list[bytes] = []
    for k in inst_kinds:
        ck = (k, bool(disable_vibrato))
        if ck not in sample_cache:
            sample_cache[ck] = make_instrument_sample(k, rng, f0=REF_F0, disable_vibrato=bool(disable_vibrato))
        samples_bytes.append(sample_cache[ck])

    samples_float = [bytes_to_float_sample(b) for b in samples_bytes]

    patterns, key_root, base_melody, base_melody_meta, derive_used = make_patterns(rng, speed=speed, tempo=tempo, melody_name=melody_name, derive_mode=derive_mode)

    if order is None:
        order = parse_order_string(DEFAULT_ORDER_STR)
    validate_order(order, n_patterns=len(patterns))

    order_original = list(order)

    order_for_write = list(order)
    if enable_slowdown and len(order_for_write) > 0:
        src_pat = order_for_write[-1]
        ending_pat = [list(row) for row in patterns[src_pat]]
        apply_end_slowdown_to_pattern(ending_pat, rng)
        patterns.append(ending_pat)
        order_for_write[-1] = len(patterns) - 1

    pat_data = patterns_to_bytes(patterns)

    # Title
    section1 = ["The", "A", "A_dirty", "a_holy", "Another", "The_wildest", "A_crazy", "A_funny"]
    section2 = ["banana", "DJ", "pianist", "stardestroyer", "dentist", "pope", "dictator", "dancingqueen", "jungleman", "toilet", "strawberry"]
    section3 = ["is_at", "move_to", "will_meet", "save_the", "want_see", "went_to", "dance_fame", "just_get", "have_meet", "move_on", "make_on_to", "get_on", "linked_by"]
    section4 = ["a_dancefloor__", "the_DJ__", "at_poolparty__", "a_busstation__", "in_heaven__", "ready_to_rock__", "disco__", "crazy__", "party__", "roll_around__", "fight__", "a_sausage__", "a_phonecall__"]
    title_txt = f"{rng.choice(section1)}_{rng.choice(section2)}_{rng.choice(section3)}_{rng.choice(section4)}_{rng.randint(1, 9999):04d}"
    title = title_txt.encode("ascii", "ignore")[:20].ljust(20, b"\x00")

    # Instrument headers
    insts: list[bytes] = []
    insts.append(inst_header(inst_kinds[0], samples_bytes[0], volume=48))
    insts.append(inst_header(inst_kinds[1], samples_bytes[1], volume=48))
    insts.append(inst_header(inst_kinds[2], samples_bytes[2], volume=48))
    insts.append(inst_header(inst_kinds[3], samples_bytes[3], volume=48))
    empty = b"\x00" * 22 + struct.pack(">H", 0) + bytes([0]) + bytes([0]) + struct.pack(">H", 0) + struct.pack(">H", 1)
    insts += [empty] * 27

    song_len = len(order_for_write)
    order_table = bytes(order_for_write + [0] * (128 - len(order_for_write)))

    mod = bytearray()
    mod += title
    for ih in insts:
        mod += ih
    mod += bytes([song_len])
    mod += bytes([0])  # restart byte
    mod += order_table
    mod += b"M.K."
    mod += pat_data
    for s in samples_bytes:
        mod += s

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{title_txt.replace(' ', '_')}_{ts}_key_{key_root.replace('-', '').replace('#','s')}.mod"
    path = out_dir_p / fname
    path.write_bytes(mod)

    song = SongData(
        title_txt=title_txt,
        seed=int(seed),
        key_root=key_root,
        base_melody=base_melody,
        base_melody_meta=dict(base_melody_meta or {}),
        patterns=patterns,
        order_original=order_original,
        order=order_for_write,
        samples_bytes=samples_bytes,
        samples_float=samples_float,
        instrument_kinds=inst_kinds,
        speed=int(speed),
        tempo=int(tempo),
        slowdown_enabled=bool(enable_slowdown),
        derive_mode=str(derive_used),
        vibrato_disabled=bool(disable_vibrato),
    )

    return path, song


# -----------------------------
# Preview renderer + playback
# -----------------------------

def _freq_from_period(period: int) -> float:
    return AMIGA_PAL_CLOCK / (2.0 * max(1, period))


def _tick_seconds(tempo: int) -> float:
    # ProTracker: tick duration ~= 2.5 / BPM
    return 2.5 / max(32, min(255, tempo))



def render_song_to_pcm16(song: SongData, out_rate: int = 44100, progress_cb=None, cancel_event: threading.Event | None = None) -> tuple[bytes, int, list[array]]:
    # Very small MOD subset renderer (enough for our generator):
    # - Note on, sample select, Fxx speed/tempo.
    # - No finetune, no loops, no other effects.
    # - Fixed panning (ch1+ch4 left, ch2+ch3 right).
    # Additionally returns per-channel mono int16 buffers for tracker-like scopes.

    # channel state
    chan_period = [0, 0, 0, 0]
    chan_samp = [0, 1, 2, 3]
    chan_pos = [0.0, 0.0, 0.0, 0.0]
    chan_vol = [48, 48, 48, 48]

    speed = int(song.speed)
    tempo = int(song.tempo)

    mix_l = array("h")
    mix_r = array("h")
    ch_out = [array("h"), array("h"), array("h"), array("h")]

    patterns = song.patterns

    total_rows = max(1, len(song.order) * 64)
    done_rows = 0

    def _clamp01(x: float) -> float:
        return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)

    for pat_id in song.order:
        pat = patterns[pat_id]
        for row in range(64):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError('Render cancelled')

            # Apply row events
            for ch in range(4):
                note, samp, eff, par = pat[row][ch]

                if eff == 0x0F and par != 0:
                    if par <= 0x1F:
                        speed = max(1, min(31, int(par)))
                    else:
                        tempo = max(32, min(255, int(par)))

                if note is not None:
                    if note not in PERIODS:
                        continue
                    chan_period[ch] = PERIODS[note]
                    if samp:
                        # MOD sample numbers 1..31; we only use 1..4
                        chan_samp[ch] = max(0, min(3, int(samp) - 1))
                    chan_pos[ch] = 0.0

            row_secs = max(0.001, speed * _tick_seconds(tempo))
            n = int(row_secs * out_rate)

            # localize for speed
            sp0, sp1, sp2, sp3 = song.samples_float
            pos0, pos1, pos2, pos3 = chan_pos
            per0, per1, per2, per3 = chan_period
            sidx0, sidx1, sidx2, sidx3 = chan_samp
            vol0, vol1, vol2, vol3 = chan_vol

            for _ in range(n):
                l = 0.0
                r = 0.0
                c0 = c1 = c2 = c3 = 0.0

                # channel 0 (L)
                if per0 > 0:
                    step = _freq_from_period(per0) / out_rate
                    samp_arr = (sp0, sp1, sp2, sp3)[sidx0]
                    i0 = int(pos0)
                    if i0 < len(samp_arr):
                        v = samp_arr[i0] * (vol0 / 64.0)
                        c0 = v
                        l += v
                    pos0 += step

                # channel 1 (R)
                if per1 > 0:
                    step = _freq_from_period(per1) / out_rate
                    samp_arr = (sp0, sp1, sp2, sp3)[sidx1]
                    i1 = int(pos1)
                    if i1 < len(samp_arr):
                        v = samp_arr[i1] * (vol1 / 64.0)
                        c1 = v
                        r += v
                    pos1 += step

                # channel 2 (R)
                if per2 > 0:
                    step = _freq_from_period(per2) / out_rate
                    samp_arr = (sp0, sp1, sp2, sp3)[sidx2]
                    i2 = int(pos2)
                    if i2 < len(samp_arr):
                        v = samp_arr[i2] * (vol2 / 64.0)
                        c2 = v
                        r += v
                    pos2 += step

                # channel 3 (L)
                if per3 > 0:
                    step = _freq_from_period(per3) / out_rate
                    samp_arr = (sp0, sp1, sp2, sp3)[sidx3]
                    i3 = int(pos3)
                    if i3 < len(samp_arr):
                        v = samp_arr[i3] * (vol3 / 64.0)
                        c3 = v
                        l += v
                    pos3 += step

                # mild master gain to avoid clipping
                l *= 0.25
                r *= 0.25
                c0 *= 0.25
                c1 *= 0.25
                c2 *= 0.25
                c3 *= 0.25

                l = _clamp01(l)
                r = _clamp01(r)
                c0 = _clamp01(c0)
                c1 = _clamp01(c1)
                c2 = _clamp01(c2)
                c3 = _clamp01(c3)

                mix_l.append(int(l * 32767))
                mix_r.append(int(r * 32767))
                ch_out[0].append(int(c0 * 32767))
                ch_out[1].append(int(c1 * 32767))
                ch_out[2].append(int(c2 * 32767))
                ch_out[3].append(int(c3 * 32767))

            chan_pos = [pos0, pos1, pos2, pos3]

            done_rows += 1
            if progress_cb is not None:
                try:
                    progress_cb(done_rows, total_rows)
                except BaseException:
                    pass

    interleaved = array("h")
    for i in range(len(mix_l)):
        interleaved.append(mix_l[i])
        interleaved.append(mix_r[i])

    return interleaved.tobytes(), out_rate, ch_out


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int, nch: int = 2) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return bio.getvalue()



class Player:
    """
    Minimal playback helper.

    - On Windows, prefers winsound (no extra dependencies).
    - Otherwise, uses simpleaudio if installed.

    We also keep our own start/duration timestamps so the GUI can animate the
    spectrum analyzer even if the backend doesn't expose playhead position.
    """

    def __init__(self):
        self._is_windows = sys.platform.startswith("win")
        self._backend: str | None = None  # "ps_soundplayer" / "winsound_file" / "simpleaudio"
        self._play_obj = None            # simpleaudio.PlayObject
        self._proc = None                # subprocess.Popen (Windows SoundPlayer backend)
        self._start_t = 0.0
        self._duration_s = 0.0
        self._sr = 44100
        self._lock = threading.Lock()
        self._tmp_wav_path: str | None = None

    @staticmethod
    def _wav_info(wav_bytes: bytes) -> tuple[int, int, int, int, bytes]:
        # returns: (nch, sampw, sr, frames, pcm_frames)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            nch = int(wf.getnchannels())
            sampw = int(wf.getsampwidth())
            sr = int(wf.getframerate())
            frames = int(wf.getnframes())
            pcm = wf.readframes(frames)
        return nch, sampw, sr, frames, pcm

    def play(self, wav_bytes: bytes):
        # Stop any previous playback first
        self.stop()

        nch, sampw, sr, frames, pcm = self._wav_info(wav_bytes)

        with self._lock:
            self._start_t = time.perf_counter()
            self._duration_s = frames / float(max(1, sr))
            self._sr = sr

        # Windows: prefer a separate SoundPlayer process.
        # This avoids a class of rare-but-nasty driver issues where stopping winsound playback
        # can terminate the Python process on some systems.
        if self._is_windows:
            try:
                import tempfile

                fd, tmp_path = tempfile.mkstemp(prefix="pt_preview_", suffix=".wav")
                try:
                    os.close(fd)
                except Exception:
                    pass
                with open(tmp_path, "wb") as f:
                    f.write(wav_bytes)

                # PowerShell SoundPlayer (async via separate process)
                # PlaySync keeps the process alive until audio ends; we can stop by terminating it.
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"$p=New-Object System.Media.SoundPlayer '{tmp_path}'; $p.PlaySync();",
                ]

                creationflags = 0
                if hasattr(subprocess, "CREATE_NO_WINDOW"):
                    creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )

                with self._lock:
                    self._backend = "ps_soundplayer"
                    self._proc = proc
                    self._play_obj = None
                    self._tmp_wav_path = tmp_path
                return
            except Exception:
                # fall back to winsound filename backend
                try:
                    import winsound  # type: ignore
                    import tempfile

                    fd, tmp_path = tempfile.mkstemp(prefix="pt_preview_", suffix=".wav")
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                    with open(tmp_path, "wb") as f:
                        f.write(wav_bytes)
                    winsound.PlaySound(tmp_path, winsound.SND_ASYNC | winsound.SND_FILENAME)
                    with self._lock:
                        self._backend = "winsound_file"
                        self._tmp_wav_path = tmp_path
                        self._proc = None
                        self._play_obj = None
                    return
                except Exception:
                    # fall through to simpleaudio
                    pass

        # Cross-platform: simpleaudio if available
        try:
            import simpleaudio  # type: ignore

            wave_obj = simpleaudio.WaveObject(pcm, nch, sampw, sr)
            play_obj = wave_obj.play()
            with self._lock:
                self._backend = "simpleaudio"
                self._play_obj = play_obj
            return
        except Exception as e:
            with self._lock:
                self._backend = None
                self._play_obj = None
                self._start_t = 0.0
                self._duration_s = 0.0
            raise RuntimeError(
                "Playback backend not available. On Windows this should work via winsound; otherwise install 'simpleaudio'."
            ) from e

    def stop(self):
        with self._lock:
            backend = self._backend
            play_obj = self._play_obj
            tmp_path = self._tmp_wav_path
            proc = self._proc
            self._backend = None
            self._play_obj = None
            self._tmp_wav_path = None
            self._proc = None
            self._start_t = 0.0
            self._duration_s = 0.0

        if backend == "ps_soundplayer":
            try:
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    # Give it a moment, then hard-kill if needed
                    try:
                        proc.wait(timeout=0.4)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            except BaseException:
                pass
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        elif backend == "winsound_file":
            try:
                import winsound  # type: ignore
                # Stop async playback. Using flags=0 is the most compatible way.
                winsound.PlaySound(None, 0)
            except BaseException:
                pass
            # Best-effort cleanup of temp file
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        elif backend == "simpleaudio":
            try:
                if play_obj is not None:
                    play_obj.stop()
            except Exception:
                pass

    def is_playing(self) -> bool:
        with self._lock:
            backend = self._backend
            play_obj = self._play_obj
            start_t = self._start_t
            duration = self._duration_s
            proc = self._proc

        if backend is None or start_t <= 0.0 or duration <= 0.0:
            return False

        if backend == "ps_soundplayer":
            try:
                if proc is not None and proc.poll() is None:
                    return True
            except Exception:
                pass
            # Fallback to time window
            return (time.perf_counter() - start_t) < duration

        if backend == "winsound_file":
            return (time.perf_counter() - start_t) < duration

        try:
            return bool(play_obj.is_playing())  # type: ignore[attr-defined]
        except Exception:
            # Fallback to time window
            return (time.perf_counter() - start_t) < duration

    def start_time(self) -> float:
        with self._lock:
            return self._start_t

    def duration_s(self) -> float:
        with self._lock:
            return self._duration_s

    def backend_name(self) -> str:
        with self._lock:
            return self._backend or "none"

    def playback_sample_index(self) -> int:
        with self._lock:
            start_t = self._start_t
            sr = self._sr
        if start_t <= 0.0:
            return 0
        t = max(0.0, time.perf_counter() - start_t)
        return int(t * sr)


# -----------------------------
# Spectrum analyzer
# -----------------------------

try:
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False



class SpectrumAnalyzer:
    def __init__(self, canvas, bars: int = 32, width: int = 560, height: int = 160, segments: int = 22):
        self.canvas = canvas
        self.bars = int(bars)
        self.width = int(width)
        self.height = int(height)
        self.segments = max(8, int(segments))

        self._levels = [0.0] * self.bars
        self._cleared = True

        self.canvas.configure(width=self.width, height=self.height, bg="#8f8f8f", highlightthickness=0)

        # precompute band edges (log-spaced)
        self._fmin = 60.0
        self._fmax = 5200.0
        self._edges = [self._fmin * ((self._fmax / self._fmin) ** (i / self.bars)) for i in range(self.bars + 1)]

        self._pad = 6
        self._full_h = (self.height - 2 * self._pad)
        self._slot_w = (self.width - 2 * self._pad) / self.bars

        # gradient colors (bottom->top: green -> yellow -> red)
        self._seg_colors = self._make_gradient(self.segments)

        # ids per bar: list of segment rect ids from bottom to top
        self._seg_ids: list[list[int]] = []

        for i in range(self.bars):
            x0 = self._pad + i * self._slot_w
            x1 = x0 + self._slot_w - 2
            y0 = self._pad
            y1 = self.height - self._pad

            # slot outline
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#6f6f6f", width=1)

            segs: list[int] = []
            # create collapsed segments (we'll place them on update)
            for c in self._seg_colors:
                rid = self.canvas.create_rectangle(x0 + 1, y1, x1 - 1, y1, outline="", fill=c)
                segs.append(rid)
            self._seg_ids.append(segs)

    @staticmethod
    def _lerp(a: int, b: int, t: float) -> int:
        return int(round(a + (b - a) * t))

    @classmethod
    def _hex_lerp(cls, c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> str:
        r = cls._lerp(c0[0], c1[0], t)
        g = cls._lerp(c0[1], c1[1], t)
        b = cls._lerp(c0[2], c1[2], t)
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def _make_gradient(cls, n: int) -> list[str]:
        green = (0x28, 0xff, 0x28)
        yellow = (0xff, 0xd4, 0x28)
        red = (0xff, 0x30, 0x30)

        out: list[str] = []
        if n <= 1:
            return [cls._hex_lerp(green, red, 0.0)]

        for i in range(n):
            t = i / float(max(1, n - 1))
            if t < 0.55:
                tt = t / 0.55
                out.append(cls._hex_lerp(green, yellow, tt))
            else:
                tt = (t - 0.55) / (1.0 - 0.55)
                out.append(cls._hex_lerp(yellow, red, tt))
        return out

    def reset(self):
        self._levels = [0.0] * self.bars
        self._cleared = True

        y_bottom = self.height - self._pad
        for i, segs in enumerate(self._seg_ids):
            x0 = self._pad + i * self._slot_w + 2
            x1 = x0 + self._slot_w - 6
            for rid in segs:
                self.canvas.coords(rid, x0, y_bottom, x1, y_bottom)

    def _compute_levels(self, mono: list[float], sr: int) -> list[float]:
        n = len(mono)
        if n < 64:
            return [0.0] * self.bars

        if _HAS_NUMPY:
            x = _np.array(mono, dtype=_np.float32)
            win = _np.hanning(n).astype(_np.float32)
            x = x * win
            spec = _np.fft.rfft(x)
            mag = _np.abs(spec)
            freqs = _np.fft.rfftfreq(n, 1.0 / sr)

            levels = []
            for i in range(self.bars):
                f0, f1 = self._edges[i], self._edges[i + 1]
                idx = _np.where((freqs >= f0) & (freqs < f1))[0]
                if idx.size == 0:
                    levels.append(0.0)
                else:
                    levels.append(float(_np.mean(mag[idx])))
            return levels

        # Fallback: lightweight Goertzel at band centers
        centers = [((self._edges[i] * self._edges[i + 1]) ** 0.5) for i in range(self.bars)]
        levels: list[float] = []
        for f in centers:
            w = 2.0 * 3.141592653589793 * f / sr
            cw = math.cos(w)
            coeff = 2.0 * cw
            s_prev = 0.0
            s_prev2 = 0.0
            for x in mono:
                s = x + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
            levels.append(math.sqrt(max(0.0, power)))
        return levels

    def update_from_pcm(self, pcm16: bytes, sr: int, sample_index: int, window: int = 1024):
        if not pcm16:
            return

        total_frames = len(pcm16) // 4
        if total_frames <= 0:
            return

        i0 = max(0, min(total_frames - 1, int(sample_index)))
        i1 = min(total_frames, i0 + int(window))
        if i1 - i0 < 64:
            return

        # extract mono window
        mono: list[float] = []
        off = i0 * 4
        end = i1 * 4
        for j in range(off, end, 4):
            l = int.from_bytes(pcm16[j : j + 2], byteorder="little", signed=True)
            r = int.from_bytes(pcm16[j + 2 : j + 4], byteorder="little", signed=True)
            mono.append(((l + r) * 0.5) / 32768.0)

        raw = self._compute_levels(mono, sr)

        # normalize + smooth
        mx = max(1e-9, max(raw))
        for i in range(self.bars):
            v = raw[i] / mx
            v = math.sqrt(v)  # mild compression
            self._levels[i] = self._levels[i] * 0.75 + v * 0.25

        self._cleared = False

        # draw with a smooth vertical gradient
        y_bottom = self.height - self._pad
        seg_h = self._full_h / float(self.segments)

        for i, segs in enumerate(self._seg_ids):
            x0 = self._pad + i * self._slot_w + 2
            x1 = x0 + self._slot_w - 6

            level = max(0.0, min(1.0, self._levels[i]))
            h = self._full_h * level
            full = int(h // seg_h)
            frac = (h - full * seg_h) / seg_h if seg_h > 0 else 0.0

            # segments are bottom->top
            for s_i, rid in enumerate(segs):
                if s_i < full:
                    y1 = y_bottom - s_i * seg_h
                    y0 = y1 - seg_h
                    self.canvas.coords(rid, x0, y0, x1, y1)
                elif s_i == full and frac > 1e-3:
                    y1 = y_bottom - s_i * seg_h
                    y0 = y1 - (seg_h * frac)
                    self.canvas.coords(rid, x0, y0, x1, y1)
                else:
                    self.canvas.coords(rid, x0, y_bottom, x1, y_bottom)


class OscilloscopeView:
    """Tracker-like 4-channel scopes (click visualizer to toggle from spectrum)."""

    def __init__(self, canvas, width: int = 560, height: int = 160):
        self.canvas = canvas
        self.width = int(width)
        self.height = int(height)
        self._pad = 6
        self._cleared = True

        self.canvas.configure(width=self.width, height=self.height, bg="#8f8f8f", highlightthickness=0)

        self._scope_ids: list[int] = []
        self._mid_ids: list[int] = []

        inner_h = self.height - 2 * self._pad
        self._slot_h = inner_h / 4.0

        for ch in range(4):
            x0 = self._pad
            x1 = self.width - self._pad
            y0 = self._pad + ch * self._slot_h
            y1 = y0 + self._slot_h

            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#6f6f6f", width=1)
            mid = (y0 + y1) * 0.5
            mid_id = self.canvas.create_line(x0 + 1, mid, x1 - 1, mid, fill="#6f6f6f")
            self._mid_ids.append(mid_id)

            # channel label
            self.canvas.create_text(x0 + 18, y0 + 10, text=f"CH{ch+1}", fill="#1a1a1a", font=("Courier New", 12, "bold"))

            # waveform polyline
            line_id = self.canvas.create_line(x0 + 1, mid, x1 - 1, mid, fill="#1a1a1a", width=1)
            self._scope_ids.append(line_id)

    def reset(self):
        self._cleared = True
        x0 = self._pad
        x1 = self.width - self._pad
        for ch, lid in enumerate(self._scope_ids):
            y0 = self._pad + ch * self._slot_h
            y1 = y0 + self._slot_h
            mid = (y0 + y1) * 0.5
            self.canvas.coords(lid, x0 + 1, mid, x1 - 1, mid)

    def update_from_channels(self, ch_bufs: list[array], sr: int, sample_index: int, window: int = 1024):
        if not ch_bufs or len(ch_bufs) < 4:
            return

        total_frames = min(len(ch_bufs[0]), len(ch_bufs[1]), len(ch_bufs[2]), len(ch_bufs[3]))
        if total_frames <= 0:
            return

        i0 = max(0, min(total_frames - 1, int(sample_index)))
        i1 = min(total_frames, i0 + int(window))
        if i1 - i0 < 16:
            return

        x0 = self._pad
        x1 = self.width - self._pad
        w = max(16, int(x1 - x0 - 2))

        # number of points drawn per scope
        pts = min(320, i1 - i0, w)

        for ch in range(4):
            y0 = self._pad + ch * self._slot_h
            y1 = y0 + self._slot_h
            mid = (y0 + y1) * 0.5
            amp = (self._slot_h * 0.42)

            buf = ch_bufs[ch]
            segment = buf[i0:i1]

            coords: list[float] = []
            # downsample by index mapping
            for p in range(pts):
                si = int(p * (len(segment) - 1) / max(1, pts - 1))
                v = segment[si] / 32768.0
                x = x0 + 1 + (p * (w - 1) / max(1, pts - 1))
                y = mid - (v * amp)
                coords.extend([x, y])

            self.canvas.coords(self._scope_ids[ch], *coords)

        self._cleared = False
# -----------------------------
# GUI (ProTracker-ish style)
# -----------------------------

def run_gui():
    import tkinter as tk
    from tkinter import messagebox
    from tkinter import ttk

    player = Player()

    last_song: SongData | None = None
    last_mod_path: Path | None = None

    # cached preview audio
    preview_pcm: bytes | None = None
    preview_wav: bytes | None = None
    preview_sr = 44100
    preview_frames = 0
    preview_ch = None  # per-channel mono int16 arrays for scope view
    
    # playback state (GUI-side)
    play_state = "idle"  # idle | playing
    play_started_t = 0.0
    play_duration_s = 0.0

    render_lock = threading.Lock()
    render_thread: threading.Thread | None = None

    root = tk.Tk()
    def _tk_exception_handler(exc, val, tb):
        try:
            import traceback as _tb
            msg = "".join(_tb.format_exception(exc, val, tb))
        except Exception:
            msg = f"{exc}: {val}"
        # Log to console and to the UI (best effort), but do not crash the app.
        try:
            print(msg, file=sys.stderr)
        except Exception:
            pass
        try:
            messagebox.showerror("Internal error", msg)
        except Exception:
            pass

    root.report_callback_exception = _tk_exception_handler
    root.title("ProTracker MOD Choral Generator (v1.6.4)")
    root.configure(bg="#8f8f8f")
    # Keep a stable window size (prevents width jitter from varying filename lengths)
    try:
        root.geometry("1040x680")
        root.minsize(1040, 680)
    except Exception:
        pass

    # Style (best-effort ProTracker vibe)
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    base_font = ("Courier New", 10, "bold")

    style.configure("PT.TButton", font=base_font, padding=(8, 3), relief="raised")
    style.configure("PT.TLabel", font=base_font, background="#8f8f8f", foreground="#1a1a1a")
    style.configure("PT.TFrame", background="#8f8f8f")
    style.configure("PT.TCheckbutton", font=base_font, background="#8f8f8f")
    style.configure("PT.TCombobox", font=base_font)

    # layout frames
    main = ttk.Frame(root, style="PT.TFrame", padding=10)
    main.grid(row=0, column=0, sticky="nsew")

    left = tk.Frame(main, bg="#8f8f8f", bd=2, relief="ridge")
    left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

    right = tk.Frame(main, bg="#8f8f8f", bd=2, relief="ridge")
    right.grid(row=0, column=1, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)

    # --- left controls ---
    def pt_label(parent, text_):
        return ttk.Label(parent, text=text_, style="PT.TLabel")

    def _open_folder(path: Path):
        p = Path(path).resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            try:
                messagebox.showerror("Open folder", str(e))
            except Exception:
                pass

    def _open_output_folder():
        _open_folder(Path("mods_out"))

    def _open_plugin_folder():
        try:
            _open_folder(_PLUGIN_ROOT)
        except Exception:
            _open_folder(_default_plugin_root())

    def _refresh_plugins():
        try:
            reload_melody_plugins()
        except Exception:
            pass
        try:
            melody_combo.configure(values=get_melody_choices())
        except Exception:
            pass
        # keep current selection if still available
        try:
            cur = melody_var.get()
            if cur not in get_melody_choices():
                melody_var.set("Pure Random")
        except Exception:
            pass
        try:
            log("Plugin list refreshed.")
        except Exception:
            pass

    pt_label(left, "PATTERN ORDER").grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 2))

    order_var = tk.StringVar(value=DEFAULT_ORDER_STR)
    order_combo = ttk.Combobox(left, textvariable=order_var, values=ORDER_PRESETS, width=32, style="PT.TCombobox", state="normal")
    order_combo.grid(row=1, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))

    pt_label(left, "BASE MELODY").grid(row=2, column=0, columnspan=2, sticky="w", padx=8)
    melody_var = tk.StringVar(value="Pure Random")
    melody_combo = ttk.Combobox(left, textvariable=melody_var, values=MELODY_CHOICES, width=32, style="PT.TCombobox", state="readonly")
    melody_combo.grid(row=3, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))

    pt_label(left, "MELODY DERIVATION").grid(row=4, column=0, columnspan=2, sticky="w", padx=8)
    derive_var = tk.StringVar(value="Random")
    derive_combo = ttk.Combobox(left, textvariable=derive_var, values=["Random", "Near", "Far"], width=32, style="PT.TCombobox", state="readonly")
    derive_combo.grid(row=5, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))


    pt_label(left, "SPEED").grid(row=6, column=0, sticky="w", padx=8)
    speed_var = tk.StringVar(value=str(DEFAULT_SPEED))
    speed_entry = tk.Entry(left, textvariable=speed_var, width=6, font=base_font, bg="#9b9b9b", fg="#000000", relief="sunken")
    speed_entry.grid(row=6, column=1, sticky="e", padx=8, pady=2)

    pt_label(left, "TEMPO").grid(row=7, column=0, sticky="w", padx=8)
    tempo_var = tk.StringVar(value=str(DEFAULT_TEMPO))
    tempo_entry = tk.Entry(left, textvariable=tempo_var, width=6, font=base_font, bg="#9b9b9b", fg="#000000", relief="sunken")
    tempo_entry.grid(row=7, column=1, sticky="e", padx=8, pady=2)

    slowdown_var = tk.BooleanVar(value=False)
    slowdown_cb = ttk.Checkbutton(left, text="Enable slowdown to the end of the song", variable=slowdown_var, style="PT.TCheckbutton")
    slowdown_cb.grid(row=8, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 10))

    # These two exports are useful defaults in practice.
    export_wav_var = tk.BooleanVar(value=True)
    export_wav_cb = ttk.Checkbutton(left, text="Export rendered songs as WAV", variable=export_wav_var, style="PT.TCheckbutton")
    export_wav_cb.grid(row=9, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 2))

    save_params_var = tk.BooleanVar(value=True)
    save_params_cb = ttk.Checkbutton(left, text="Save song parameters", variable=save_params_var, style="PT.TCheckbutton")
    save_params_cb.grid(row=10, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 2))
    vibrato_var = tk.BooleanVar(value=False)
    vibrato_cb = ttk.Checkbutton(left, text="Disable vibrato in samples", variable=vibrato_var, style="PT.TCheckbutton")
    vibrato_cb.grid(row=11, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 10))


    pt_label(left, "INSTRUMENTS (CH1..CH4)").grid(row=12, column=0, columnspan=2, sticky="w", padx=8)

    inst_vars = [tk.StringVar(value=DEFAULT_INSTRUMENTS[i]) for i in range(4)]

    def add_inst_row(r: int, label: str, var: tk.StringVar):
        pt_label(left, label).grid(row=r, column=0, sticky="w", padx=8, pady=2)
        cb = ttk.Combobox(left, textvariable=var, values=INSTRUMENT_CHOICES, width=18, style="PT.TCombobox", state="readonly")
        cb.grid(row=r, column=1, sticky="e", padx=8, pady=2)

    add_inst_row(13, "CH1", inst_vars[0])
    add_inst_row(14, "CH2", inst_vars[1])
    add_inst_row(15, "CH3", inst_vars[2])
    add_inst_row(16, "CH4", inst_vars[3])

    # Keep the left panel compact; song details are written to the log on the right.

    # buttons
    btn_frame = tk.Frame(left, bg="#8f8f8f")
    btn_frame.grid(row=18, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 10))

    gen_btn = ttk.Button(btn_frame, text="GENERATE", style="PT.TButton")
    play_btn = ttk.Button(btn_frame, text="PLAY", style="PT.TButton")
    stop_btn = ttk.Button(btn_frame, text="STOP", style="PT.TButton")

    gen_btn.grid(row=0, column=0, sticky="we", padx=(0, 6))
    play_btn.grid(row=0, column=1, sticky="we", padx=(0, 6))
    stop_btn.grid(row=0, column=2, sticky="we")

    open_out_btn = ttk.Button(btn_frame, text="OPEN OUTPUT", style="PT.TButton", command=_open_output_folder)
    open_plg_btn = ttk.Button(btn_frame, text="OPEN PLUGINS", style="PT.TButton", command=_open_plugin_folder)
    refresh_plg_btn = ttk.Button(btn_frame, text="REFRESH", style="PT.TButton", command=_refresh_plugins)

    open_out_btn.grid(row=1, column=0, sticky="we", padx=(0, 6), pady=(6, 0))
    open_plg_btn.grid(row=1, column=1, sticky="we", padx=(0, 6), pady=(6, 0))
    refresh_plg_btn.grid(row=1, column=2, sticky="we", pady=(6, 0))

    # initial states
    _dummy = None
    try:
        play_btn.state(["disabled"])
        stop_btn.state(["disabled"])
    except Exception:
        pass

    btn_frame.columnconfigure(0, weight=1)
    btn_frame.columnconfigure(1, weight=1)
    btn_frame.columnconfigure(2, weight=1)

    # --- right: visualizer panel (click to toggle Spectrum / Scopes) ---
    title_bar = tk.Frame(right, bg="#8f8f8f")
    title_bar.pack(fill="x", padx=10, pady=(10, 2))

    viz_title_var = tk.StringVar(value="SPECTRUM ANALYZER")
    viz_title_lbl = tk.Label(title_bar, textvariable=viz_title_var, bg="#8f8f8f", fg="#1a1a1a", font=("Courier New", 11, "bold"))
    viz_title_lbl.pack(anchor="w")

    hint_lbl = tk.Label(title_bar, text="Click visualizer to toggle Spectrum / Scopes", bg="#8f8f8f", fg="#2a2a2a", font=("Courier New", 12, "bold"))
    hint_lbl.pack(anchor="w")

    canvas = tk.Canvas(right)
    canvas.pack(fill="x", padx=10, pady=(0, 10))

    viz_mode = "spectrum"  # spectrum | scope
    viz_view = None

    def set_viz_mode(mode: str):
        nonlocal viz_mode, viz_view
        mode = (mode or "").strip().lower()
        if mode not in ("spectrum", "scope"):
            mode = "spectrum"
        viz_mode = mode
        try:
            canvas.delete("all")
        except Exception:
            pass
        if viz_mode == "spectrum":
            viz_title_var.set("SPECTRUM ANALYZER")
            viz_view = SpectrumAnalyzer(canvas, bars=32, width=560, height=160, segments=22)
        else:
            viz_title_var.set("CHANNEL SCOPES")
            viz_view = OscilloscopeView(canvas, width=560, height=160)
        try:
            viz_view.reset()
        except Exception:
            pass

    def _toggle_viz(_evt=None):
        set_viz_mode("scope" if viz_mode == "spectrum" else "spectrum")

    canvas.bind("<Button-1>", _toggle_viz)
    set_viz_mode("spectrum")

    info_bar = tk.Frame(right, bg="#8f8f8f")
    info_bar.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    info_bar.columnconfigure(0, weight=1)
    info_bar.rowconfigure(1, weight=2)
    info_bar.rowconfigure(3, weight=3)

    # Render / playback status belongs next to the log output (right side).
    render_var = tk.StringVar(value="")
    progress_lbl = tk.Label(
        info_bar,
        textvariable=render_var,
        bg="#8f8f8f",
        fg="#1a1a1a",
        font=("Courier New", 14, "bold"),
        anchor="w",
        justify="left",
    )
    progress_lbl.grid(row=0, column=0, sticky="we", pady=(0, 6))

    info_txt = tk.Text(info_bar, height=7, font=("Courier New", 9), bg="#9b9b9b", fg="#000000", relief="sunken", bd=2)
    info_txt.grid(row=1, column=0, sticky="nsew")
    info_txt.insert("end", "Generate a song, then hit PLAY.\n")
    # --- pattern preview (scrollable tracker grid) ---
    patt_header = tk.Frame(info_bar, bg="#8f8f8f")
    patt_header.grid(row=2, column=0, sticky="we", pady=(10, 2))
    patt_title = tk.Label(patt_header, text="PATTERN PREVIEW", bg="#8f8f8f", fg="#1a1a1a", font=("Courier New", 11, "bold"))
    patt_title.pack(side="left")

    patt_sel_var = tk.StringVar(value="0")
    patt_combo = ttk.Combobox(patt_header, textvariable=patt_sel_var, values=["0"], width=6, style="PT.TCombobox", state="readonly")
    patt_combo.pack(side="left", padx=(12, 0))

    patt_frame = tk.Frame(info_bar, bg="#8f8f8f")
    patt_frame.grid(row=3, column=0, sticky="nsew")
    patt_frame.columnconfigure(0, weight=1)
    patt_frame.rowconfigure(0, weight=1)

    patt_txt = tk.Text(patt_frame, height=10, font=("Courier New", 9), bg="#9b9b9b", fg="#000000", relief="sunken", bd=2, wrap="none")
    patt_txt.grid(row=0, column=0, sticky="nsew")
    patt_y = tk.Scrollbar(patt_frame, orient="vertical", command=patt_txt.yview)
    patt_y.grid(row=0, column=1, sticky="ns")
    patt_x = tk.Scrollbar(patt_frame, orient="horizontal", command=patt_txt.xview)
    patt_x.grid(row=1, column=0, sticky="we")
    patt_txt.configure(yscrollcommand=patt_y.set, xscrollcommand=patt_x.set)

    def _pattern_grid_text(song: SongData, p_idx: int) -> str:
        p_idx = max(0, min(int(p_idx), len(song.patterns) - 1))
        pat = song.patterns[p_idx]
        lines = []
        lines.append(f"PATTERN {p_idx}")
        lines.append("row | CH1            | CH2            | CH3            | CH4")
        lines.append("----+----------------+----------------+----------------+----------------")
        for r in range(64):
            c0 = _cell_to_text(pat[r][0])
            c1 = _cell_to_text(pat[r][1])
            c2 = _cell_to_text(pat[r][2])
            c3 = _cell_to_text(pat[r][3])
            lines.append(f"{r:02d}  | {c0:<14} | {c1:<14} | {c2:<14} | {c3:<14}")
        return "\n".join(lines) + "\n"

    def update_pattern_preview(_evt=None):
        try:
            patt_txt.config(state="normal")
            patt_txt.delete("1.0", "end")
            if last_song is None:
                patt_txt.insert("end", "(no song yet)\n")
            else:
                try:
                    idx = int(patt_sel_var.get().strip())
                except Exception:
                    idx = 0
                patt_txt.insert("end", _pattern_grid_text(last_song, idx))
            patt_txt.config(state="disabled")
        except Exception:
            pass

    patt_combo.bind("<<ComboboxSelected>>", update_pattern_preview)
    update_pattern_preview()

    info_txt.config(state="disabled")

    # analyzer update loop
    after_id = None

    def log(msg: str):
        info_txt.config(state="normal")
        info_txt.insert("end", msg.rstrip() + "\n")
        info_txt.see("end")
        info_txt.config(state="disabled")

    def post_log(msg: str):
        try:
            root.after(0, lambda: log(msg))
        except Exception:
            pass

    wav_state_lock = threading.Lock()
    wav_exporting = False

    def maybe_save_params():
        nonlocal last_song, last_mod_path
        if not save_params_var.get():
            return
        if last_song is None or last_mod_path is None:
            return
        try:
            txt_path = last_mod_path.with_suffix(".txt")
            existed = txt_path.exists()
            p = save_song_parameters_txt(last_mod_path, last_song)
            if existed:
                log("Song parameters already saved (skipped).")
            else:
                log(f"Saved song parameters: {p.name}")
        except Exception as e:
            log(f"Save parameters failed: {e}")

    def maybe_export_wav():
        nonlocal wav_exporting
        if not export_wav_var.get():
            return
        if last_mod_path is None:
            return
        with render_lock:
            wavb = preview_wav
        if wavb is None:
            return
        wav_path = last_mod_path.with_suffix(".wav")
        if wav_path.exists():
            return
        with wav_state_lock:
            if wav_exporting:
                return
            wav_exporting = True

        def _worker(wav_bytes: bytes, out_path: Path):
            nonlocal wav_exporting
            try:
                ok, msg = export_rendered_wav(wav_bytes, out_path)
                post_log(msg)
            finally:
                with wav_state_lock:
                    wav_exporting = False

        threading.Thread(target=_worker, args=(wavb, wav_path), daemon=True).start()

    def stop_analyzer():
        nonlocal after_id
        if after_id is not None:
            try:
                root.after_cancel(after_id)
            except Exception:
                pass
            after_id = None

    def analyzer_tick():
        nonlocal after_id
        try:
            if play_state == "playing":
                idx = int(max(0.0, time.perf_counter() - play_started_t) * preview_sr)
                if viz_mode == "spectrum":
                    if preview_pcm and viz_view is not None:
                        try:
                            viz_view.update_from_pcm(preview_pcm, preview_sr, idx, window=1024)
                        except Exception:
                            pass
                else:
                    if preview_ch and viz_view is not None:
                        try:
                            viz_view.update_from_channels(preview_ch, preview_sr, idx, window=1024)
                        except Exception:
                            pass
                after_id = root.after(50, analyzer_tick)
            else:
                # nothing playing -> snap back to 0
                if viz_view is not None and not getattr(viz_view, "_cleared", False):
                    try:
                        viz_view.reset()
                    except Exception:
                        pass
                after_id = root.after(200, analyzer_tick)
        except BaseException:
            # Never let the visualizer crash the app.
            try:
                after_id = root.after(200, analyzer_tick)
            except Exception:
                pass

    analyzer_tick()

    def parse_int_field(name: str, s: str, lo: int, hi: int) -> int:
        try:
            v = int(str(s).strip())
        except Exception:
            raise ValueError(f"{name} must be an integer.")
        if v < lo or v > hi:
            raise ValueError(f"{name} must be in range {lo}..{hi}.")
        return v

    def on_generate():
        nonlocal last_song, last_mod_path, preview_pcm, preview_wav, preview_frames, preview_sr, preview_ch
        nonlocal play_state, play_started_t, play_duration_s
        try:
            # If something is currently playing, stop it before generating a new song.
            if play_state == "playing":
                try:
                    player.stop()
                except Exception:
                    pass
                play_state = "idle"
                play_started_t = 0.0
                play_duration_s = 0.0
                try:
                    render_var.set("")
                except Exception:
                    pass
                try:
                    _set_btn_states(can_generate=True, can_play=True, can_stop=False)
                except Exception:
                    pass
            
            order_list = parse_order_string(order_var.get())
            validate_order(order_list, n_patterns=10)

            spd = parse_int_field("Speed", speed_var.get(), 1, 31)
            bpm = parse_int_field("Tempo", tempo_var.get(), 32, 255)

            instruments = [v.get() for v in inst_vars]

            path, song = generate_song(
                order=order_list,
                enable_slowdown=slowdown_var.get(),
                speed=spd,
                tempo=bpm,
                instruments=instruments,
                melody_name=melody_var.get(),
                derive_mode=derive_var.get(),
                disable_vibrato=vibrato_var.get(),
            )

            last_song = song
            last_mod_path = path

            # invalidate preview cache
            preview_pcm = None
            preview_wav = None
            preview_frames = 0
            preview_sr = 44100
            preview_ch = None

            derive_txt = getattr(song, "derive_mode", "")
            vib_txt = "OFF" if getattr(song, "vibrato_disabled", False) else "ON"
            log(f"Generated: {path}")
            log(f"Melody: {song.base_melody}")
            meta_disp = get_plugin_metadata_display(song.base_melody)
            if meta_disp:
                log(f"Melody meta: {meta_disp}")
            log(f"Derive: {derive_txt} | Vibrato: {vib_txt}")
            log(f"Instruments: {', '.join(song.instrument_kinds)}")

            try:
                patt_combo.configure(values=[str(i) for i in range(len(song.patterns))])
                patt_sel_var.set("0")
                update_pattern_preview()
            except Exception:
                pass

            # Optional: write sidecar parameters immediately after generation.
            try:
                maybe_save_params()
            except Exception:
                pass
            try:
                play_btn.state(["!disabled"])
                stop_btn.state(["disabled"])
            except Exception:
                pass
        except BaseException as e:
            try:
                messagebox.showerror("Error", str(e))
            except Exception:
                pass


    # --- render/playback state ---
    state_lock = threading.Lock()
    render_cancel = threading.Event()
    render_progress = 0.0
    render_error: str | None = None
    is_rendering = False
    auto_play_after_render = False

    # render_var is defined in the right-side log panel (next to the visualizer).

    def _set_btn_states(*, can_generate: bool, can_play: bool, can_stop: bool):
        gen_btn.state(["!disabled"] if can_generate else ["disabled"])
        play_btn.state(["!disabled"] if can_play else ["disabled"])
        stop_btn.state(["!disabled"] if can_stop else ["disabled"])

    def _render_preview_worker(song: SongData):
        nonlocal preview_pcm, preview_wav, preview_frames, preview_sr, preview_ch, render_error

        def _prog(done: int, total: int):
            nonlocal render_progress
            with state_lock:
                render_progress = 0.0 if total <= 0 else (done / float(total))

        try:
            pcm16, sr, chbufs = render_song_to_pcm16(song, out_rate=44100, progress_cb=_prog, cancel_event=render_cancel)
            wavb = pcm16_to_wav_bytes(pcm16, sr, nch=2)
            with render_lock:
                preview_pcm = pcm16
                preview_wav = wavb
                preview_sr = sr
                preview_frames = len(pcm16) // 4
                preview_ch = chbufs
            with state_lock:
                render_error = None
        except Exception as e:
            with state_lock:
                render_error = str(e)

    def _start_render(song: SongData, auto_play: bool):
        nonlocal render_thread, is_rendering, auto_play_after_render, render_progress, render_error

        if render_thread is not None and render_thread.is_alive():
            return

        render_cancel.clear()
        with state_lock:
            render_progress = 0.0
            render_error = None
        auto_play_after_render = auto_play
        is_rendering = True

        render_var.set("RENDER   0%")
        _set_btn_states(can_generate=False, can_play=False, can_stop=True)

        render_thread = threading.Thread(target=_render_preview_worker, args=(song,), daemon=True)
        render_thread.start()

    def on_play():
        nonlocal preview_wav, auto_play_after_render, is_rendering
        nonlocal play_state, play_started_t, play_duration_s
        try:
            if last_song is None:
                raise ValueError("No song generated yet.")

            # Already playing? Ignore.
            if play_state == "playing":
                return

            with render_lock:
                ready = (preview_wav is not None and preview_pcm is not None and preview_frames > 0)

            if ready:
                assert preview_wav is not None
                player.play(preview_wav)
                try:
                    log("Playback backend running..")
                except Exception:
                    pass

                # Optional exports
                try:
                    maybe_save_params()
                except Exception:
                    pass
                try:
                    maybe_export_wav()
                except Exception:
                    pass
                play_state = "playing"
                play_started_t = player.start_time()
                play_duration_s = player.duration_s()
                render_var.set("PLAYING")
                log("PLAY")
                _set_btn_states(can_generate=True, can_play=False, can_stop=True)
                return

            # Not ready -> render first, then auto-play.
            log("Rendering preview...")
            _start_render(last_song, auto_play=True)

        except BaseException as e:
            try:
                messagebox.showerror("Playback", str(e))
            except Exception:
                pass

    
    def _ui_tick():
        nonlocal is_rendering, auto_play_after_render
        nonlocal play_state, play_started_t, play_duration_s

        try:
            # Rendering progress / completion
            if is_rendering:
                with state_lock:
                    pct = int(max(0.0, min(1.0, render_progress)) * 100.0)
                    err = render_error

                render_var.set(f"RENDER {pct:3d}%")

                if render_thread is not None and not render_thread.is_alive():
                    is_rendering = False

                    if err:
                        if err == "Render cancelled":
                            render_var.set("RENDER CANCELLED")
                            log("Render cancelled.")
                        else:
                            render_var.set("RENDER FAILED")
                            log(f"Render failed: {err}")
                            try:
                                messagebox.showerror("Render", err)
                            except Exception:
                                pass
                        auto_play_after_render = False
                        _set_btn_states(can_generate=True, can_play=(last_song is not None), can_stop=False)
                    else:
                        # Render OK
                        render_var.set("")
                        if auto_play_after_render and last_song is not None:
                            auto_play_after_render = False
                            try:
                                with render_lock:
                                    wavb = preview_wav
                                    frames = preview_frames
                                if wavb is not None and frames > 0:
                                    player.play(wavb)
                                    try:
                                        log("Playback backend running..")
                                    except Exception:
                                        pass

                                    # Optional exports
                                    try:
                                        maybe_save_params()
                                    except Exception:
                                        pass
                                    try:
                                        maybe_export_wav()
                                    except Exception:
                                        pass
                                    play_state = "playing"
                                    play_started_t = player.start_time()
                                    play_duration_s = player.duration_s()
                                    render_var.set("PLAYING")
                                    log("PLAY")
                                    _set_btn_states(can_generate=True, can_play=False, can_stop=True)
                                else:
                                    _set_btn_states(can_generate=True, can_play=True, can_stop=False)
                            except Exception as e:
                                try:
                                    messagebox.showerror("Playback", str(e))
                                except Exception:
                                    pass
                                _set_btn_states(can_generate=True, can_play=True, can_stop=False)
                        else:
                            _set_btn_states(can_generate=True, can_play=(last_song is not None), can_stop=False)

            # Playback monitor (do NOT rely only on backend state; use wall clock too)
            if play_state == "playing":
                elapsed = max(0.0, time.perf_counter() - play_started_t)
                backend_says_playing = False
                try:
                    backend_says_playing = player.is_playing()
                except BaseException:
                    backend_says_playing = True  # be conservative

                if (play_duration_s > 0.0 and elapsed >= play_duration_s) or (not backend_says_playing and elapsed > 0.25):
                    # Finished
                    # Ensure we release any backend resources (e.g., temp WAV / helper process).
                    try:
                        player.stop()
                    except BaseException:
                        pass
                    play_state = "idle"
                    play_started_t = 0.0
                    play_duration_s = 0.0
                    if render_var.get() == "PLAYING":
                        render_var.set("")
                    _set_btn_states(can_generate=True, can_play=(last_song is not None), can_stop=False)
                    try:
                        viz_view.reset()
                    except Exception:
                        pass
                else:
                    # While playing, STOP should stay enabled
                    _set_btn_states(can_generate=True, can_play=False, can_stop=True)

        except BaseException as e:
            # Never let UI tick stop forever.
            try:
                log(f"UI tick error: {e}")
            except Exception:
                pass
        finally:
            try:
                root.after(120, _ui_tick)
            except Exception:
                pass

    _ui_tick()

    def on_stop():
        nonlocal auto_play_after_render, is_rendering
        nonlocal play_state, play_started_t, play_duration_s
        try:
            auto_play_after_render = False

            # Cancel render only if we are rendering
            if is_rendering:
                try:
                    render_cancel.set()
                except BaseException:
                    pass

            # Stop playback (safe even if not playing)
            try:
                player.stop()
            except BaseException:
                pass

            play_state = "idle"
            play_started_t = 0.0
            play_duration_s = 0.0

            try:
                render_var.set("")
            except BaseException:
                pass

            try:
                log("STOP")
            except BaseException:
                pass
            try:
                viz_view.reset()
            except Exception:
                pass

        except BaseException as e:
            # Never let STOP kill the whole process (e.g., SystemExit from an audio backend).
            try:
                log(f"STOP handler caught: {e}")
            except BaseException:
                pass

        # Always restore UI state
        try:
            _set_btn_states(can_generate=True, can_play=(last_song is not None), can_stop=False)
        except BaseException:
            pass

    def on_close():
        try:
            render_cancel.set()
        except BaseException:
            pass
        try:
            player.stop()
        except Exception:
            pass
        stop_analyzer()
        root.destroy()

    gen_btn.config(command=on_generate)
    play_btn.config(command=on_play)
    stop_btn.config(command=on_stop)

    root.protocol("WM_DELETE_WINDOW", on_close)

    left.columnconfigure(0, weight=1)
    left.columnconfigure(1, weight=1)

    root.mainloop()


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate churchy ProTracker .MOD files (GUI by default).")
    ap.add_argument("-nogui", action="store_true", help="Run in CLI mode (do not show GUI).")
    ap.add_argument("-speed", type=int, default=None, help="CLI: MOD speed (ticks/row, 1..31).")
    ap.add_argument("-tempo", type=int, default=None, help="CLI: MOD tempo (BPM, 32..255).")
    ap.add_argument("-inst1", type=str, default=None, help=f"CLI: instrument for channel 1. One of: {', '.join(INSTRUMENT_CHOICES)}")
    ap.add_argument("-inst2", type=str, default=None, help=f"CLI: instrument for channel 2. One of: {', '.join(INSTRUMENT_CHOICES)}")
    ap.add_argument("-inst3", type=str, default=None, help=f"CLI: instrument for channel 3. One of: {', '.join(INSTRUMENT_CHOICES)}")
    ap.add_argument("-inst4", type=str, default=None, help=f"CLI: instrument for channel 4. One of: {', '.join(INSTRUMENT_CHOICES)}")
    ap.add_argument("-noslowdown", action="store_true", help="Disable ending slowdown at the end of the song.")
    ap.add_argument("-order", type=str, default=None, help="CLI: override pattern order string.")
    ap.add_argument("-melody", type=str, default=None, help="CLI: base melody preset name (or omit for Random).")
    ap.add_argument("-derive", type=str, default="Random", choices=["Random", "Near", "Far"], help="CLI: melody derivation style (Random/Near/Far).")
    ap.add_argument("-novibrato", action="store_true", help="CLI: disable vibrato in generated samples.")

    args = ap.parse_args()

    if not args.nogui:
        run_gui()
        return

    speed = args.speed if args.speed is not None else DEFAULT_SPEED
    tempo = args.tempo if args.tempo is not None else DEFAULT_TEMPO

    if not (1 <= int(speed) <= 31):
        raise SystemExit("-speed must be 1..31")
    if not (32 <= int(tempo) <= 255):
        raise SystemExit("-tempo must be 32..255")

    insts = [args.inst1, args.inst2, args.inst3, args.inst4]
    instruments = [x if x is not None else DEFAULT_INSTRUMENTS[i] for i, x in enumerate(insts)]

    order_list = parse_order_string(args.order) if args.order else None

    path, _ = generate_song(
        enable_slowdown=not args.noslowdown,
        speed=int(speed),
        tempo=int(tempo),
        instruments=instruments,
        order=order_list,
        melody_name=(args.melody if args.melody else None),
        derive_mode=args.derive,
        disable_vibrato=bool(args.novibrato),
    )
    print(f"Generated: {path}")


if __name__ == "__main__":
    main()
