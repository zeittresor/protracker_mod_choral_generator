#!/usr/bin/env python3
# ProTracker MOD Choral Generator (v2.0)
# Source: https://github.com/zeittresor/protracker_mod_choral_generator

from __future__ import annotations

import argparse
import io
import math
import os
import random
import re
import struct
import shutil
import subprocess
import sys
import threading
import time
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path

# Import harmony analyzer for quality checking
try:
    from harmony_analyzer import HarmonyAnalyzer, MusicQualityChecker, analyze_and_improve_music
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    print("Warning: Harmony analyzer not available. Quality checking disabled.")

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


def normalize_key_root(s: str | None) -> str | None:
    """Normalize user-provided key root text to ProTracker note format (e.g. C-2, F#-2).

    Accepts: C-2, C2, C#, Bb, Bb-2, etc. Returns None for random/empty.
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t or t.lower() in ("random", "auto", "none"):
        return None
    # allow formats like C2, C-2, C#2, Bb-2, Bb2, etc
    m = re.match(r"^\s*([A-Ga-g])\s*([#bB]?)\s*(?:-?\s*(\d))?\s*$", t)
    if not m:
        return None
    note = m.group(1).upper()
    acc = m.group(2)
    octv = m.group(3) or "2"
    if acc in ("b", "B"):
        # convert flats to sharps
        flat_map = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
        key = note + "B"
        note = flat_map.get(key, note)
        # note may now include #
        if len(note) == 2 and note[1] == "#":
            pass
    elif acc == "#":
        note = note + "#"
    # ProTracker note format used in this script: e.g. C-2, F#-2
    norm = f"{note}-{octv}"
    return norm if norm in CHROMATIC_SET else None


def random_key_root() -> str:
    # Prefer common choral keys, octave 2
    pool = [
        "C-2","D-2","E-2","F-2","G-2","A-2","B-2",
        "C#2","D#2","F#2","G#2","A#2",
    ]
    rr = random.SystemRandom()
    return rr.choice(pool)


def random_seed_value() -> int:
    # 64-bit-ish random seed; not time-range clustered.
    return (int.from_bytes(os.urandom(8), "big") ^ (time.time_ns() & 0xFFFFFFFFFFFF)) & 0x7FFFFFFFFFFFFFFF

DEFAULT_SPEED = 6
DEFAULT_TEMPO = 125

# "Mixed" blends Major+Minor (modal mixture) while keeping everything harmonically aligned.
SCALE_MODE_CHOICES = ["Auto", "Major", "Minor", "Mixed", "Dorian", "Mixolydian"]

DEFAULT_ORDER_STR = "5, 15, 1, 5, 10, 12, 4, 2, 15, 0"

# How many base patterns are generated (0..PATTERN_COUNT-1).
PATTERN_COUNT = 20
ORDER_PRESETS = [
    # User's preferred order (default)
    "5, 15, 1, 5, 10, 12, 4, 2, 15, 0",
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
    "0, 1, 2, 13, 2, 4, 5, 17",
    # new presets for patterns 0..19
    "0, 10, 6, 11, 2, 12, 7, 13, 4, 14, 9, 15, 5",
    "0, 1, 10, 11, 6, 12, 2, 13, 8, 14, 4, 15, 9, 16, 5",
    "10, 0, 11, 1, 12, 6, 13, 2, 14, 4, 15, 9, 5",
    "0, 10, 12, 10, 6, 11, 13, 7, 14, 8, 15, 9, 5",
    "0, 6, 10, 7, 11, 2, 12, 8, 13, 4, 14, 9, 15, 5",
    "0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 9, 15, 5",
    "6, 10, 7, 11, 8, 12, 9, 13, 5",
    "0, 2, 10, 12, 6, 7, 13, 14, 4, 15, 9, 5",
    "0, 10, 16, 11, 17, 12, 18, 13, 19, 14, 15, 5",

    # extra long-form / high-variation presets
    "0, 3, 10, 6, 11, 7, 12, 8, 13, 9, 14, 18, 19, 5",
    "0, 10, 12, 6, 7, 13, 2, 14, 8, 15, 9, 16, 19, 5",
    "0, 6, 10, 7, 13, 8, 18, 9, 16, 17, 19, 5",
    "0, 1, 3, 6, 10, 15, 7, 13, 4, 14, 9, 16, 19, 5",
    "0, 2, 12, 6, 11, 7, 17, 8, 18, 9, 19, 5",
    "0, 10, 6, 7, 13, 8, 14, 18, 9, 15, 19, 5",

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

    # Extra built-ins (traditional / public domain-ish motifs, plus original variations)
    "Greensleeves (trad. approx., minor)": [
        [(0, 0, 4), (2, 0, 4), (3, 0, 4), (4, 0, 4)],
        [(5, 0, 4), (4, 0, 4), (3, 0, 4), (1, 0, 4)],
        [(0, 0, 4), (2, 0, 4), (3, 0, 4), (4, 0, 4)],
        [(3, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],
    
    # Renaissance / folk epoch motifs (approximate, diatonic-safe; intended as inspirational bases)
    "Scarborough Fair (trad. approx.)": [
        [(0, 0, 2), (2, 0, 2), (3, 0, 2), (4, 0, 2), (3, 0, 2), (2, 0, 2), (0, 0, 4)],
        [(0, 0, 2), (2, 0, 2), (3, 0, 4), (2, 0, 4), (1, 0, 2), (0, 0, 2)],
        [(4, 0, 2), (5, 0, 2), (4, 0, 2), (3, 0, 2), (2, 0, 2), (1, 0, 2), (0, 0, 4)],
        [(0, 0, 4), (1, 0, 2), (2, 0, 2), (1, 0, 4), (0, 0, 4)],
    ],
    "The Three Ravens (ballad approx.)": [
        [(0, 0, 4), (0, 0, 2), (2, 0, 2), (3, 0, 2), (2, 0, 2), (0, 0, 4)],
        [(4, 0, 4), (3, 0, 4), (2, 0, 4), (1, 0, 2), (0, 0, 2)],
        [(0, 0, 2), (2, 0, 2), (3, 0, 2), (4, 0, 2), (3, 0, 4), (2, 0, 4)],
        [(1, 0, 4), (0, 0, 8), (None, 0, 4)],
    ],
    "Pastime with Good Company (Henry VIII approx.)": [
        [(0, 0, 2), (2, 0, 2), (4, 0, 2), (5, 0, 2), (4, 0, 2), (2, 0, 2), (0, 0, 4)],
        [(0, 0, 2), (0, 0, 2), (2, 0, 4), (4, 0, 4), (5, 0, 2), (5, 0, 2)],
        [(4, 0, 2), (3, 0, 2), (2, 0, 4), (1, 0, 4), (0, 0, 2), (0, 0, 2)],
        [(5, 0, 4), (4, 0, 4), (2, 0, 4), (0, 0, 4)],
    ],
    "Flow My Tears (Dowland-ish approx.)": [
        [(0, 0, 2), (2, 0, 2), (3, 0, 4), (2, 0, 4), (1, 0, 2), (0, 0, 2)],
        [(4, 0, 4), (3, 0, 4), (2, 0, 2), (1, 0, 2), (0, 0, 2), (0, 0, 2)],
        [(0, 0, 2), (1, 0, 2), (2, 0, 2), (3, 0, 2), (4, 0, 4), (3, 0, 4)],
        [(2, 0, 4), (1, 0, 4), (0, 0, 8)],
    ],
    "Lachrimae Pavane (Dowland-ish approx.)": [
        [(0, 0, 2), (2, 0, 2), (3, 0, 2), (4, 0, 2), (3, 0, 2), (2, 0, 2), (0, 0, 4)],
        [(5, 0, 2), (4, 0, 2), (3, 0, 2), (2, 0, 2), (1, 0, 4), (0, 0, 4)],
        [(0, 1, 4), (6, 0, 4), (5, 0, 4), (4, 0, 4)],
        [(3, 0, 4), (2, 0, 4), (1, 0, 4), (0, 0, 4)],
    ],
    "Sellenger's Round (dance approx.)": [
        [(0, 0, 2), (0, 0, 2), (2, 0, 2), (4, 0, 2), (5, 0, 4), (4, 0, 4)],
        [(2, 0, 2), (3, 0, 2), (4, 0, 2), (5, 0, 2), (4, 0, 4), (2, 0, 4)],
        [(0, 0, 4), (2, 0, 2), (4, 0, 2), (2, 0, 4), (0, 0, 4)],
        [(5, 0, 4), (4, 0, 4), (2, 0, 4), (0, 0, 4)],
    ],
    "The Hunt Is Up (dance approx.)": [
        [(0, 0, 2), (2, 0, 2), (4, 0, 2), (2, 0, 2), (0, 0, 2), (2, 0, 2), (4, 0, 4)],
        [(5, 0, 4), (4, 0, 2), (3, 0, 2), (2, 0, 2), (1, 0, 2), (0, 0, 4)],
        [(0, 0, 2), (2, 0, 2), (3, 0, 2), (4, 0, 2), (5, 0, 4), (4, 0, 4)],
        [(2, 0, 4), (1, 0, 4), (0, 0, 8)],
    ],
    "John Barleycorn (folk approx.)": [
        [(0, 0, 4), (2, 0, 2), (3, 0, 2), (2, 0, 4), (0, 0, 4)],
        [(4, 0, 4), (3, 0, 4), (2, 0, 4), (1, 0, 2), (0, 0, 2)],
        [(0, 0, 4), (0, 0, 2), (2, 0, 2), (3, 0, 4), (4, 0, 4)],
        [(2, 0, 4), (1, 0, 4), (0, 0, 8)],
    ],
    "Barbara Allen (ballad approx.)": [
        [(0, 0, 2), (1, 0, 2), (2, 0, 2), (3, 0, 2), (2, 0, 2), (1, 0, 2), (0, 0, 4)],
        [(0, 0, 2), (2, 0, 2), (4, 0, 4), (3, 0, 2), (2, 0, 2), (1, 0, 4)],
        [(4, 0, 4), (3, 0, 4), (2, 0, 4), (1, 0, 2), (0, 0, 2)],
        [(1, 0, 4), (0, 0, 8), (None, 0, 4)],
    ],
    "Fortune My Foe (trad. approx.)": [
        [(0, 0, 2), (2, 0, 2), (3, 0, 2), (5, 0, 2), (4, 0, 2), (3, 0, 2), (2, 0, 4)],
        [(1, 0, 4), (0, 0, 4), (6, 0, 2), (5, 0, 2), (4, 0, 4)],
        [(3, 0, 4), (2, 0, 4), (1, 0, 4), (0, 0, 4)],
        [(0, 0, 4), (2, 0, 4), (1, 0, 4), (0, 0, 4)],
    ],
    "Watkins Ale (dance approx.)": [
        [(0, 0, 2), (2, 0, 2), (4, 0, 2), (5, 0, 2), (4, 0, 2), (2, 0, 2), (0, 0, 4)],
        [(0, 0, 2), (2, 0, 2), (3, 0, 2), (4, 0, 2), (5, 0, 4), (4, 0, 4)],
        [(2, 0, 4), (3, 0, 2), (4, 0, 2), (2, 0, 4), (0, 0, 4)],
        [(5, 0, 4), (4, 0, 2), (3, 0, 2), (2, 0, 4), (0, 0, 4)],
    ],
    "The Carman's Whistle (dance approx.)": [
        [(0, 0, 2), (2, 0, 2), (1, 0, 2), (0, 0, 2), (2, 0, 2), (3, 0, 2), (4, 0, 4)],
        [(4, 0, 4), (3, 0, 2), (2, 0, 2), (1, 0, 4), (0, 0, 4)],
        [(0, 0, 4), (2, 0, 2), (4, 0, 2), (2, 0, 4), (0, 0, 4)],
        [(1, 0, 4), (0, 0, 8), (None, 0, 4)],
    ],
    "Gaudete (carol approx.)": [
        [(0, 0, 2), (0, 0, 2), (2, 0, 2), (3, 0, 2), (2, 0, 4), (0, 0, 4)],
        [(4, 0, 2), (4, 0, 2), (5, 0, 2), (4, 0, 2), (3, 0, 4), (2, 0, 4)],
        [(2, 0, 2), (3, 0, 2), (4, 0, 2), (5, 0, 2), (4, 0, 4), (3, 0, 4)],
        [(2, 0, 4), (1, 0, 4), (0, 0, 8)],
    ],
    "My Robin is to the Greenwood Gone (folk approx.)": [
        [(0, 0, 2), (2, 0, 2), (4, 0, 2), (5, 0, 2), (4, 0, 2), (2, 0, 2), (0, 0, 4)],
        [(0, 0, 2), (1, 0, 2), (2, 0, 2), (3, 0, 2), (4, 0, 4), (5, 0, 4)],
        [(5, 0, 2), (4, 0, 2), (3, 0, 2), (2, 0, 2), (1, 0, 4), (0, 0, 4)],
        [(2, 0, 4), (1, 0, 4), (0, 0, 8)],
    ],
"Dona Nobis Pacem (trad. approx.)": [
        [(0, 0, 4), (1, 0, 4), (2, 0, 4), (3, 0, 4)],
        [(2, 0, 4), (1, 0, 4), (0, 0, 8)],
        [(0, 0, 4), (1, 0, 4), (2, 0, 4), (3, 0, 4)],
        [(4, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],
    "Kumbaya (trad. approx.)": [
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (4, 0, 4)],
        [(5, 0, 4), (4, 0, 4), (2, 0, 4), (0, 0, 4)],
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (5, 0, 4)],
        [(4, 0, 8), (2, 0, 4), (0, 0, 4)],
    ],
    "Adeste Fideles (trad. approx.)": [
        [(0, 0, 4), (0, 0, 4), (2, 0, 4), (4, 0, 4)],
        [(5, 0, 4), (4, 0, 4), (2, 0, 4), (0, 0, 4)],
        [(2, 0, 4), (4, 0, 4), (5, 0, 4), (6, 0, 4)],
        [(5, 0, 8), (4, 0, 4), (2, 0, 4)],
    ],
    "Chorale Motif II (original)": [
        [(0, 0, 4), (2, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(3, 0, 4), (1, 0, 4), (2, 0, 4), (0, 0, 4)],
        [(4, 0, 4), (5, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(1, 0, 4), (0, 0, 8), (None, 0, 4)],
    ],
    "Gospel Shuffle (original)": [
        [(0, 0, 2), (2, 0, 2), (3, 0, 4), (4, 0, 4), (2, 0, 4)],
        [(0, 0, 4), (2, 0, 2), (3, 0, 2), (4, 0, 4), (5, 0, 4)],
        [(4, 0, 4), (3, 0, 2), (2, 0, 2), (1, 0, 4), (0, 0, 4)],
        [(0, 0, 8), (4, 0, 4), (0, 0, 4)],
    ],

}


# -----------------------------
# Renaissance / folk epoch helpers
# -----------------------------

REN_FOLK_MELODY_NAMES: list[str] = [
    "Scarborough Fair (trad. approx.)",
    "The Three Ravens (ballad approx.)",
    "Pastime with Good Company (Henry VIII approx.)",
    "Flow My Tears (Dowland-ish approx.)",
    "Lachrimae Pavane (Dowland-ish approx.)",
    "Sellenger's Round (dance approx.)",
    "The Hunt Is Up (dance approx.)",
    "John Barleycorn (folk approx.)",
    "Barbara Allen (ballad approx.)",
    "Fortune My Foe (trad. approx.)",
    "Watkins Ale (dance approx.)",
    "The Carman's Whistle (dance approx.)",
    "Gaudete (carol approx.)",
    "My Robin is to the Greenwood Gone (folk approx.)",
    "Greensleeves (trad. approx., minor)",
]

def _collect_markov_sources(names: list[str]) -> tuple[dict[int, list[int]], list[list[int]]]:
    """Build a small Markov-ish transition map (scale degrees) and harvest duration templates."""
    trans: dict[int, list[int]] = {i: [] for i in range(7)}
    dur_templates: list[list[int]] = []

    for nm in names:
        bars = MELODY_LIBRARY.get(nm)
        if not bars:
            continue
        for bar in bars:
            degs = [int(d) for d, _o, _dur in bar if d is not None]
            if len(degs) >= 2:
                for a, b in zip(degs, degs[1:]):
                    trans[int(a) % 7].append(int(b) % 7)
            durs = [int(_dur) for _d, _o, _dur in bar if int(_dur) > 0]
            # Most shipped templates sum to 16; keep those as-is for stable phrasing.
            if sum(durs) == 16 and len(durs) >= 3:
                dur_templates.append(durs)

    # Fallback: if a state has no outgoing edges, give it a diatonic-safe escape list
    fallback = [0, 2, 4, 5, 3, 1, 6]
    for k in range(7):
        if not trans.get(k):
            trans[k] = fallback[:]
    if not dur_templates:
        dur_templates = [
            [4, 4, 4, 4],
            [2, 2, 4, 4, 4],
            [2, 2, 2, 2, 4, 4],
            [8, 4, 4],
        ]
    return trans, dur_templates

def _gen_markov_bar_tpl(rng: random.Random, trans: dict[int, list[int]], durs: list[int], bar_index: int) -> list[tuple[int | None, int, int]]:
    """Generate one bar template as (deg, oct, dur)."""
    tpl: list[tuple[int | None, int, int]] = []
    cur = 0 if bar_index == 0 else int(rng.choice([0, 2, 4, 5]))
    for i, dur in enumerate(durs):
        # sparse rests
        if dur <= 0:
            continue
        if rng.random() < 0.07:
            tpl.append((None, 0, int(dur)))
            continue
        nxt = int(rng.choice(trans.get(cur, [0, 2, 4, 5, 3, 1, 6])))
        cur = nxt
        octv = 0
        # occasional flourish octave up on strong beats
        if i == 0 and rng.random() < 0.12:
            octv = 1
        tpl.append((int(cur) % 7, int(octv), int(dur)))

    # Cadences
    if tpl:
        if bar_index >= 3:
            # final cadence to tonic
            # find last pitched note
            for j in range(len(tpl) - 1, -1, -1):
                if tpl[j][0] is not None:
                    tpl[j] = (0, tpl[j][1], tpl[j][2])
                    break
        elif bar_index == 1 and rng.random() < 0.55:
            # mid-cadence often to dominant
            for j in range(len(tpl) - 1, -1, -1):
                if tpl[j][0] is not None:
                    tpl[j] = (4, tpl[j][1], tpl[j][2])
                    break
    return tpl

def build_markov_folk_bars(
    rng: random.Random,
    scale_up: list[str],
    source_names: list[str] | None = None,
    n_bars: int = 4,
) -> list[list[tuple[str | None, int]]]:
    """Generate 4 bars of a 'folk-ish' base melody from Renaissance/folk templates.

    This is used for the "Pure Random" path to keep it harmonically coherent while still varied.
    """
    names = source_names or REN_FOLK_MELODY_NAMES
    trans, dur_templates = _collect_markov_sources([n for n in names if n in MELODY_LIBRARY])
    bar_tpls: list[list[tuple[int | None, int, int]]] = []
    for bi in range(max(1, int(n_bars))):
        durs = list(rng.choice(dur_templates))
        bar_tpls.append(_gen_markov_bar_tpl(rng, trans, durs, bi))
    # Ensure at least 4 bars for compatibility with downstream logic
    while len(bar_tpls) < 4:
        bar_tpls.append(list(bar_tpls[-1]))
    return [_template_bar_to_events(scale_up, bt) for bt in bar_tpls[:4]]



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
    try:
        a, b = tempo_hint.split("-", 1)
        tempo_min = int(a)
        tempo_max = int(b)
    except Exception:
        tempo_min = 90
        tempo_max = 140
    preferred_key_range = "C-2..G-2"
    return (
        f"mode: {mode}\n"
        f"tempo_hint: {tempo_hint}\n"
        f"tempo_min: {tempo_min}\n"
        f"tempo_max: {tempo_max}\n"
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

        # DEG OCT DUR  (strict; avoids mis-parsing tracker grids / parameter dumps)
        if len(parts) >= 3 and re.fullmatch(r"[0-6]|R", parts[0], re.I):
            deg_tok = parts[0].upper()
            try:
                octv = int(parts[1])
                dur = int(parts[2])
            except Exception:
                # not an actual DEG OCT DUR line
                pass
            else:
                if deg_tok == "R":
                    events.append((None, max(1, dur)))
                else:
                    deg = int(deg_tok)
                    base = 60
                    midi = base + octv * 12 + C_MAJOR_PCS[deg]
                    events.append((midi, max(1, dur)))
                continue

        # NOTE DUR  (only if it looks like a note token)
        if len(parts) >= 2 and re.match(r"^[A-Ga-g]", parts[0]):
            midi = _parse_note_token_to_midi(parts[0])
            if midi is None:
                continue
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

# ProTracker-ish UI background color (used by a few Tk frames)
PT_BG = "#8f8f8f"


MOD_SIGNATURE_CHOICES = ["M.K.", "M!K!"]
DEFAULT_MOD_SIGNATURE = "M!K!"  # tends to be accepted by more players

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
    "Bassoon",
    "Choir Ooh",
    "Synth Lead",
    "Square Lead",
    "Dubstep (Drumset)",
    "Techno Music (Drumset)",
    "Pop (Drumset)",
    "Folk (Drumset)",
    "Rock (Drumset)",
    "Hip-Hop (Drumset)",
]

DEFAULT_INSTRUMENTS = ["Piano", "Piano", "Piano", "Piano"]

# Per-instrument default volumes (0..64). This helps keep bright instruments from overpowering softer ones.
# You can still override volumes later in a tracker.
INSTRUMENT_VOL: dict[str, int] = {
    "Piano": 48,
    "Electric Piano": 46,
    "Organ": 44,
    "Strings": 46,
    "Violin": 44,
    "Choir Aah": 46,
    "Choir Ooh": 46,
    "Clarinet": 44,
    "Sax": 42,
    "Flute": 44,
    "Oboe": 44,
    "Bassoon": 46,
    "French Horn": 44,
    "Trumpet": 40,
    "Tuba": 50,
    "Banjo": 42,
    "Acoustic Guitar": 44,
    "Flamenco Guitar": 42,
    "Harp": 44,
    "Celesta": 40,
    "Bell": 34,
    "Synth Pad": 42,
    "Synth Lead": 40,
    "Square Lead": 38,
}


# Drumset "instruments" (style presets). Selecting one of these for a channel turns that channel into a drum track.
DRUMSET_STYLE_MAP: dict[str, str] = {
    "Dubstep (Drumset)": "dubstep",
    "Techno Music (Drumset)": "techno",
    "Pop (Drumset)": "pop",
    "Folk (Drumset)": "folk",
    "Rock (Drumset)": "rock",
    "Hip-Hop (Drumset)": "hiphop",
}

DRUM_STYLE_PREFIX: dict[str, str] = {
    "dubstep": "DUB",
    "techno": "TEC",
    "pop": "POP",
    "folk": "FOL",
    "rock": "ROC",
    "hiphop": "HIP",
}

DRUMKIT_ORDER = ["Kick", "Snare", "Clap", "CHat", "OHat", "Tom", "Crash", "Perc"]

DRUM_VOL = {
    "Kick": 64,
    "Snare": 56,
    "Clap": 52,
    "CHat": 40,
    "OHat": 44,
    "Tom": 50,
    "Crash": 40,
    "Perc": 46,
}


def is_drumset_kind(kind: str) -> bool:
    return (kind or "").strip() in DRUMSET_STYLE_MAP


def drumset_style_from_kind(kind: str) -> str | None:
    return DRUMSET_STYLE_MAP.get((kind or "").strip())

AMIGA_PAL_CLOCK = 7093789.2


# -----------------------------
# MOD packing helpers
# -----------------------------

def note_shift(note: str, semitones: int) -> str:
    i = CHROMATIC.index(note)
    j = i + semitones
    j = max(0, min(len(CHROMATIC) - 1, j))
    return CHROMATIC[j]


def note_shift_safe(note: str, semitones: int) -> str:
    """Shift by semitones, but NEVER clamp into a wrong note.

    If the target would fall outside the supported ProTracker note range (C-1..B-3),
    this returns the original note unchanged. This prevents octave-up operations from
    collapsing into B-3 when notes already sit at the top of the table.
    """
    try:
        i = CHROMATIC.index(note)
    except ValueError:
        return note
    j = i + semitones
    if 0 <= j < len(CHROMATIC):
        return CHROMATIC[j]
    return note



def pack_cell(note_name: str | None = None, sample: int = 0, effect: int = 0, param: int = 0) -> bytes:
    period = 0 if note_name is None else PERIODS[note_name]
    samp = sample & 0x1F
    # NOTE: sample numbers 16..31 store their high bit in byte0 as 0x10 (no shift).
    b0 = (samp & 0x10) | ((period >> 8) & 0x0F)
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
    detune = rng.uniform(0.99985, 1.00015)

    h2 = rng.uniform(0.35, 0.50)
    h3 = rng.uniform(0.18, 0.28)
    h4 = rng.uniform(0.10, 0.20)
    d2 = rng.uniform(0.02, 0.06)

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


def make_instrument_sample(kind: str, rng: random.Random, length: int = 32768, sr: int = 8287, f0: float = REF_F0, disable_vibrato: bool = False, ensemble_size: int = 4) -> bytes:
    kind = (kind or "").strip()
    if kind not in INSTRUMENT_CHOICES:
        kind = "Piano"

    if kind == "Piano":
        return make_pianoish_sample(rng, length=length, sr=sr, f0=f0)

    # Keep tuning cohesive across instruments (avoid random per-instrument detune that can sound "schräg" in ensembles).
    detune = 1.0

    # Use a mostly-shared vibrato rate; scale vibrato depth down when multiple voices play together.
    vib_rate = 5.3 + rng.uniform(-0.25, 0.25)
    vib_amt = (rng.uniform(0.0, 0.0020) if kind in ("Violin", "Strings", "Synth Pad", "Choir Aah", "Choir Ooh", "Panflute", "Flute") else rng.uniform(0.0, 0.0010))
    ens = max(1, int(ensemble_size))
    vib_amt *= (0.35 if ens >= 2 else 0.65)

    if kind == "Organ":
        vib_amt = 0.0

    if disable_vibrato:
        vib_amt = 0.0


    # Envelope choices (kept conservative so pitch feels stable)
    if kind in ("Synth Pad", "Synth Lead", "Square Lead", "Violin", "Strings", "Choir Aah", "Choir Ooh", "Panflute", "Clarinet", "Sax", "Flute", "Oboe", "Organ", "French Horn", "Trumpet", "Bassoon"):
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
    elif kind == "Choir Ooh":
        partials = [(1, 1.0), (2, 0.26), (3, 0.18), (4, 0.14), (5, 0.10), (6, 0.07)]
        noise_amt = 0.018
        drive = 1.10
        lp_alpha = 0.18
    elif kind == "Bassoon":
        partials = [(1, 1.0), (2, 0.34), (3, 0.26), (4, 0.16), (5, 0.10)]
        noise_amt = 0.010
        drive = 1.18
        lp_alpha = 0.14
    elif kind == "Synth Lead":
        partials = [(1, 1.0), (2, 0.55), (3, 0.36), (4, 0.26), (5, 0.20), (6, 0.16), (7, 0.13)]
        noise_amt = 0.004
        drive = 1.30
        lp_alpha = 0.34
    elif kind == "Square Lead":
        partials = [(1, 1.0), (3, 0.34), (5, 0.20), (7, 0.14), (9, 0.11)]
        noise_amt = 0.002
        drive = 1.25
        lp_alpha = 0.42
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

        if kind in ("Synth Pad", "Synth Lead", "Violin", "Sax"):
            chorus_detune = 1.0018
            x += 0.08 * math.sin(2 * math.pi * (f * chorus_detune) * t)

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

# -----------------------------
# Drumset sample synthesis (8-bit signed, no loops)
# -----------------------------

def _floatbuf_to_mod8(buf: list[float], peak: float = 120.0) -> bytes:
    if not buf:
        return b"\x00\x00"
    mx = max(1e-6, max(abs(v) for v in buf))
    scale = float(peak) / mx
    data = bytearray()
    for v in buf:
        s = int(max(-127, min(127, round(v * scale))))
        data.append(s & 0xFF)
    if len(data) % 2 == 1:
        data.append(0)
    return bytes(data)


def make_drum_sample(style: str, drum: str, rng: random.Random, sr: int = 8287) -> bytes:
    """Very small procedural drum synth. Result is intended to be played around C-3."""
    style = (style or "techno").strip().lower()
    drum = (drum or "Kick").strip()

    # base params by style
    if style in ("dubstep",):
        drive = 1.65
        tight = 0.85
    elif style in ("hiphop",):
        drive = 1.45
        tight = 0.95
    elif style in ("folk",):
        drive = 1.10
        tight = 1.10
    elif style in ("rock",):
        drive = 1.35
        tight = 0.95
    else:  # techno/pop default
        drive = 1.30
        tight = 1.00

    def tanh(x: float) -> float:
        return math.tanh(drive * x)

    if drum == "Kick":
        length = int(sr * (0.42 * tight))
        length = max(2048, min(8192, length))
        f_start = 130.0 if style in ("pop","rock","folk") else 110.0
        f_end = 48.0 if style in ("pop","rock","folk") else 38.0
        # dubstep/hiphop tend to go deeper
        if style in ("dubstep","hiphop"):
            f_start, f_end = 120.0, 32.0
        click_amt = 0.35 if style in ("techno","dubstep") else 0.22
        decay = 9.0 / tight
        phi = 0.0
        buf: list[float] = [0.0] * length
        for n in range(length):
            t = n / sr
            f = f_end + (f_start - f_end) * math.exp(-t * 18.0)
            phi += (2.0 * math.pi * f) / sr
            env = math.exp(-decay * t)
            x = math.sin(phi) * env
            # click
            if t < 0.012:
                x += (rng.uniform(-1.0, 1.0) * click_amt) * (1.0 - (t / 0.012))
            buf[n] = tanh(x)
        return _floatbuf_to_mod8(buf)

    if drum == "Snare":
        length = int(sr * (0.28 * tight))
        length = max(1536, min(6144, length))
        tone_f = 190.0 if style in ("folk","pop") else 210.0
        noise_amt = 0.95 if style in ("dubstep","techno") else 0.75
        decay = 18.0 / tight
        buf = [0.0] * length
        for n in range(length):
            t = n / sr
            env = math.exp(-decay * t)
            noise = rng.uniform(-1.0, 1.0) * noise_amt
            tone = math.sin(2 * math.pi * tone_f * t) * (0.25 if style in ("folk",) else 0.18)
            x = (noise + tone) * env
            # a short transient
            if t < 0.008:
                x += math.sin(2 * math.pi * 3200 * t) * (0.20 * (1.0 - t / 0.008))
            buf[n] = tanh(x)
        return _floatbuf_to_mod8(buf)

    if drum == "Clap":
        length = int(sr * (0.22 * tight))
        length = max(1024, min(4096, length))
        decay = 22.0 / tight
        buf = [0.0] * length
        # 3–4 bursts
        bursts = [0.0, 0.012, 0.024, 0.036]
        if style in ("dubstep","techno"):
            bursts = [0.0, 0.010, 0.020, 0.032]
        for n in range(length):
            t = n / sr
            env = 0.0
            for bt in bursts:
                dt = t - bt
                if 0.0 <= dt < 0.030:
                    env += math.exp(-decay * dt)
            noise = rng.uniform(-1.0, 1.0)
            x = noise * env * (0.65 if style in ("folk",) else 0.90)
            buf[n] = tanh(x)
        return _floatbuf_to_mod8(buf)

    if drum in ("CHat","OHat","Crash"):
        # noise-based metallic
        if drum == "CHat":
            length = int(sr * (0.06 * tight))
            decay = 65.0 / tight
            hp = 0.85
            level = 0.80
        elif drum == "OHat":
            length = int(sr * (0.18 * tight))
            decay = 35.0 / tight
            hp = 0.80
            level = 0.75
        else:  # Crash
            length = int(sr * (0.55 * tight))
            decay = 10.5 / tight
            hp = 0.72
            level = 0.70

        length = max(512, min(16384, length))
        buf = [0.0] * length
        last = 0.0
        # add a few inharmonic partials for "metal"
        p1 = 1900.0 + rng.uniform(-120, 120)
        p2 = 2500.0 + rng.uniform(-140, 140)
        p3 = 3100.0 + rng.uniform(-180, 180)
        for n in range(length):
            t = n / sr
            env = math.exp(-decay * t)
            noise = rng.uniform(-1.0, 1.0)
            # crude highpass: y = x - lp(x)
            last = last + (noise - last) * (1.0 - hp)
            hpv = noise - last
            metal = (math.sin(2*math.pi*p1*t) + 0.7*math.sin(2*math.pi*p2*t) + 0.5*math.sin(2*math.pi*p3*t)) / 2.2
            x = (hpv * level + metal * (0.20 if drum == "Crash" else 0.08)) * env
            buf[n] = tanh(x)
        return _floatbuf_to_mod8(buf)

    if drum == "Tom":
        length = int(sr * (0.30 * tight))
        length = max(1536, min(8192, length))
        f = 140.0 if style in ("folk","pop") else 165.0
        if style in ("dubstep",):
            f = 120.0
        decay = 11.5 / tight
        phi = 0.0
        buf = [0.0] * length
        for n in range(length):
            t = n / sr
            phi += (2.0 * math.pi * f) / sr
            env = math.exp(-decay * t)
            x = math.sin(phi) * env
            if t < 0.006:
                x += rng.uniform(-1.0, 1.0) * 0.12 * (1.0 - t / 0.006)
            buf[n] = tanh(x)
        return _floatbuf_to_mod8(buf)

    # Perc: short blip/noise
    length = int(sr * (0.10 * tight))
    length = max(512, min(4096, length))
    decay = 40.0 / tight
    f = 520.0 + rng.uniform(-60, 60)
    phi = 0.0
    buf = [0.0] * length
    for n in range(length):
        t = n / sr
        phi += (2.0 * math.pi * f) / sr
        env = math.exp(-decay * t)
        x = (0.55 * math.sin(phi) + 0.45 * rng.uniform(-1.0, 1.0)) * env
        buf[n] = tanh(x)
    return _floatbuf_to_mod8(buf)


def make_drumkit_samples(style: str, rng: random.Random) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for d in DRUMKIT_ORDER:
        out[d] = make_drum_sample(style, d, rng)
    return out


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


def scale_from_mode(root_note: str, mode: str | None) -> list[str]:
    """Return a 7-note diatonic scale for a given mode.

    Supported: major, minor (natural), dorian, mixolydian. Unknown -> major.
    """
    m = (mode or "major").strip().lower()
    if m in ("auto", "random"):
        m = "major"
    if m.startswith("maj"):
        intervals = [0, 2, 4, 5, 7, 9, 11]
    elif m.startswith("min") or "moll" in m or "aeolian" in m:
        intervals = [0, 2, 3, 5, 7, 8, 10]
    elif m.startswith("dor"):
        intervals = [0, 2, 3, 5, 7, 9, 10]
    elif m.startswith("mixo"):
        intervals = [0, 2, 4, 5, 7, 9, 10]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]
    return [note_shift(root_note, i) for i in intervals]


def parse_preferred_key_range(s: str | None) -> list[str]:
    """Parse ranges like 'C-2..G-2' or 'C-2 - G-2' into a list of PT notes."""
    if not s:
        return []
    t = str(s).strip()
    if not t:
        return []
    # normalize separators
    t = t.replace("—", "-").replace("–", "-")
    if ".." in t:
        a, b = t.split("..", 1)
    elif "-" in t and t.count("-") >= 2 and "#" not in t:
        # avoid splitting note like C#-2 incorrectly; fall through to regex
        a, b = t.split("-", 1)
    elif " - " in t:
        a, b = t.split(" - ", 1)
    else:
        # single note
        one = normalize_key_root(t)
        return [one] if one else []

    a = normalize_key_root(a)
    b = normalize_key_root(b)
    if not a or not b:
        return []
    try:
        ia = CHROMATIC.index(a)
        ib = CHROMATIC.index(b)
    except Exception:
        return []
    if ia > ib:
        ia, ib = ib, ia
    out = []
    for i in range(ia, ib + 1):
        out.append(CHROMATIC[i])
    return out


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
            note = note_shift_safe(note, -12)
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
    strength: float = 1.0,
) -> list[tuple[str | None, int]]:
    """Mutate a bar's (note,dur) events.

    Near/far derivation uses these modes to add variation while staying in-key.
    """
    out: list[tuple[str | None, int]] = [(n, int(d)) for (n, d) in events]

    st = float(strength)
    if not (st == st):
        st = 1.0
    st = max(0.0, min(1.5, st))

    def p(x: float) -> float:
        # scale probability by strength, but keep sane bounds
        return max(0.0, min(0.98, x * st))

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
                nn = note_shift_safe(n, 12)
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
            if i >= len(out) - 2 and rng.random() < p(0.85):
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
            if n is not None and d >= 4 and rng.random() < p(0.55):
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
            if d >= 4 and rng.random() < p(0.75):
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
            if d >= 4 and rng.random() < p(0.85):
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
            # Use a 2-row pickup (instead of 1-row) to keep the rhythm "in the grid".
            if d >= 4 and rng.random() < p(0.65):
                pn = _nearest_in_scale(n, rng.choice([-1, -2]))
                out2.append((pn, 2))
                out2.append((n, d - 2))
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
    # IMPORTANT: Do NOT lock randomness to plugins only.
    # If the shipped ZIP does not include a large plugin set, PLUGIN_MELODIES may contain only a
    # tiny default subset (sometimes a single melody). In that case, always picking from plugins
    # would make "Random" effectively non-random and songs start to sound identical.
    plugin_names = list(PLUGIN_MELODIES.keys()) if PLUGIN_MELODIES else []
    builtin_names = list(MELODY_LIBRARY.keys()) if MELODY_LIBRARY else []

    if plugin_names or builtin_names:
        # Prefer Renaissance/folk motifs for "Random" to get coherent, period-ish songs,
        # while still allowing plugins and other built-ins to appear.
        folk_names = [n for n in REN_FOLK_MELODY_NAMES if n in MELODY_LIBRARY]
        other_builtin = [n for n in builtin_names if n not in folk_names]
        pool = (plugin_names * 2) + (folk_names * 5) + other_builtin
        if not pool:
            pool = plugin_names + builtin_names
        nm = rng.choice(pool)

        if nm in PLUGIN_MELODIES:
            pl = PLUGIN_MELODIES[nm]
            if isinstance(pl, MelodyPlugin):
                return nm, pl.bars, dict(pl.meta or {})
            return nm, pl, {}

        if nm in MELODY_LIBRARY:
            return nm, MELODY_LIBRARY[nm], {}

    return "Pure Random", None, {}

def make_patterns(
    rng: random.Random,
    speed: int = DEFAULT_SPEED,
    tempo: int = DEFAULT_TEMPO,
    melody_name: str | None = None,
    derive_mode: str | None = None,
    key_root_override: str | None = None,
    scale_mode: str | None = None,
    variation: float = 1.0,
    octave_spans: list[int] | None = None,
    drum_channels: set[int] | None = None,
):
    NUM_CH = 4
    ROWS = 64
    patterns: list[list[list[tuple[str | None, int, int, int]]]] = []
    drum_ch = set(drum_channels or [])

    base_melody_name, base_tpl, base_meta = _pick_base_melody(rng, melody_name)

    # Key root selection: user override > plugin hint > default pool
    key_root = normalize_key_root(key_root_override)
    if key_root is None:
        pref = None
        try:
            pref = (base_meta or {}).get('preferred_key_range') or (base_meta or {}).get('key_range')
        except Exception:
            pref = None
        cand = parse_preferred_key_range(pref)
        key_root = rng.choice(cand) if cand else rng.choice(['C-2', 'G-2', 'F-2', 'D-2'])

    # Scale/mode: explicit GUI/CLI override, otherwise plugin meta 'mode'.
    sm = (scale_mode or 'auto').strip().lower()
    if sm in ('auto', 'random'):
        pm = str((base_meta or {}).get('mode', '') or '').strip().lower()
        sm = pm if pm else 'major'

    # "Mixed" = Major/Minor mixture. Keeps a stable home key but allows parallel borrowing.
    use_mixed = (sm or '').strip().lower() == 'mixed'
    if use_mixed:
        scale = scale_from_mode(key_root, 'major')
        scale_alt = scale_from_mode(key_root, 'minor')
        scale_up = [note_shift_safe(n, 12) for n in scale]
        scale_alt_up = [note_shift_safe(n, 12) for n in scale_alt]
        sm = 'mixed'
    else:
        scale = scale_from_mode(key_root, sm)
        scale_up = [note_shift_safe(n, 12) for n in scale]
        scale_alt = scale
        scale_alt_up = scale_up

    if base_tpl is None:
        # "Pure Random": generate a coherent folk-ish base melody from Renaissance/folk motifs.
        try:
            base_bars = build_markov_folk_bars(
                rng,
                scale_up=scale_up,
                source_names=REN_FOLK_MELODY_NAMES,
                n_bars=4,
            )
        except Exception:
            base_prog = [0, 3, 4, 0]
            base_bars = []
            for deg in base_prog:
                rt, th, fi = triad_from_degree(scale, deg, octave_bias=0)
                chord = [note_shift_safe(rt, 12), note_shift_safe(th, 12), note_shift_safe(fi, 12)]
                chord = [n if n in CHROMATIC_SET else scale_up[0] for n in chord]
                base_bars.append(build_bar_melody(rng, scale=scale_up, chord=chord, base_note=chord[0]))
    else:
        base_bars = [_template_bar_to_events(scale_up, base_tpl[i]) for i in range(4)]

    N_PAT = PATTERN_COUNT
    for _ in range(N_PAT):
        pat = [[(None, 0, 0, 0) for _ in range(NUM_CH)] for _ in range(ROWS)]
        patterns.append(pat)

    # For "Mixed" tonalities we decide (per pattern/bar) whether to borrow from the parallel mode.
    # This keeps the song coherent while adding color.
    bar_use_alt: list[list[bool]] = [[False, False, False, False] for _ in range(N_PAT)]
    if use_mixed:
        v = float(variation)
        if not (v == v):
            v = 1.0
        v = max(0.0, min(1.5, v))
        base_prob = 0.18 + 0.22 * min(1.0, v)  # 0.18 .. 0.40
        for p_idx in range(N_PAT):
            for bar in range(4):
                # Keep first and last bar "home" for stability.
                if bar in (0, 3):
                    continue
                prob = base_prob
                # Cadence-heavy patterns borrow less.
                if p_idx in (5, 11, 19):
                    prob *= 0.55
                bar_use_alt[p_idx][bar] = (rng.random() < prob)

    
    # Per-channel octave policy: how many octaves around the base key octave are allowed.
    def _parse_oct(note_: str) -> int | None:
        try:
            return int(note_[-1])
        except Exception:
            return None

    base_oct = _parse_oct(key_root) or 2
    spans = list(octave_spans) if (octave_spans and len(octave_spans) == 4) else [3, 3, 3, 3]
    spans = [max(1, min(3, int(x))) for x in spans]

    oct_limits = []
    for s in spans:
        if s % 2 == 1:
            lo = base_oct - (s // 2)
            hi = base_oct + (s // 2)
        else:
            lo = base_oct - (s // 2 - 1)
            hi = base_oct + (s // 2)
        lo = max(1, min(3, lo))
        hi = max(1, min(3, hi))
        if lo > hi:
            lo, hi = hi, lo
        oct_limits.append((lo, hi))

    def apply_octave_policy(note_: str, ch_: int) -> str:
        o = _parse_oct(note_)
        if o is None:
            return note_
        lo, hi = oct_limits[ch_]
        n = note_
        while o < lo:
            nn = note_shift_safe(n, 12)
            if nn == n:
                break
            n = nn
            o = _parse_oct(n) or o
        while o > hi:
            nn = note_shift_safe(n, -12)
            if nn == n:
                break
            n = nn
            o = _parse_oct(n) or o
        return n

    def set_cell(p: int, row: int, ch: int, note: str | None = None, sample: int | None = None, effect: int = 0x00, param: int = 0x00):
        if ch in drum_ch:
            # keep drum channels empty here; drum patterns are injected later
            return
        if note is None:
            samp = 0 if sample is None else sample
        else:
            samp = (ch + 1) if sample is None else sample
        if 0 <= row < 64:
            patterns[p][row][ch] = (apply_octave_policy(note, ch) if note is not None else note, samp, effect, param)


    # --- harmony helpers: voice-leading + non-crossing voicings ---
    def _note_idx(n: str | None) -> int | None:
        if n is None:
            return None
        try:
            return CHROMATIC.index(n)
        except ValueError:
            return None

    def _pc(n: str) -> str:
        # pitch class incl. accidental marker (e.g. "C-", "C#", "A#")
        return n[:-1]

    def _tones_for_channel(rt: str, th: str, fi: str, ch: int) -> list[str]:
        """Chord-tone candidates for a channel, across nearby octaves, respecting octave policy."""
        out: list[str] = []
        for t in (rt, th, fi):
            for sh in (-12, 0, 12):
                nn = note_shift_safe(t, sh)
                nn = apply_octave_policy(nn, ch)
                if nn in CHROMATIC_SET and nn not in out:
                    out.append(nn)
        # also allow the bass to reach further down if policy allows
        if ch == 2:
            for sh in (-24,):
                nn = note_shift_safe(rt, sh)
                nn = apply_octave_policy(nn, ch)
                if nn in CHROMATIC_SET and nn not in out:
                    out.append(nn)
        return out if out else [apply_octave_policy(rt, ch)]

    def _choose_voicing(rt: str, th: str, fi: str, melody_ub: str | None, prev: dict[int, str | None]) -> tuple[str, str, str]:
        """Pick (top, bass, inner) chord voicing for CH2..CH4.

        Goals:
        - avoid voice crossing: bass < inner < top
        - keep accompaniment top (CH2) below the melody on CH1 when possible
        - minimize motion (voice-leading) vs previous bar
        """
        cand_top = _tones_for_channel(rt, th, fi, 1)
        cand_bass = _tones_for_channel(rt, th, fi, 2)
        cand_inner = _tones_for_channel(rt, th, fi, 3)

        ub_i = _note_idx(melody_ub) if (melody_ub in CHROMATIC_SET) else None

        def _cost(top_n: str, bass_n: str, inner_n: str) -> float:
            ti = _note_idx(top_n) or 0
            bi = _note_idx(bass_n) or 0
            ii = _note_idx(inner_n) or 0

            cost = 0.0
            # hard constraints via large penalties
            if not (bi < ii < ti):
                cost += 500.0
            # keep some spacing (avoid clumped unisons)
            if (ti - ii) < 2:
                cost += 25.0
            if (ii - bi) < 2:
                cost += 25.0
            # keep CH2 (channel 1 in patterns) below CH1 melody if possible
            if ub_i is not None and ti >= ub_i:
                cost += 60.0 + (ti - ub_i) * 2.0

            # preferences: bass=root, inner=third, top=fifth (typical choral triad)
            if _pc(bass_n) != _pc(rt):
                cost += 8.0
            if _pc(inner_n) != _pc(th):
                cost += 2.5
            if _pc(top_n) != _pc(fi):
                cost += 2.0

            # voice-leading vs previous bar
            for ch, n in ((1, top_n), (2, bass_n), (3, inner_n)):
                pn = prev.get(ch)
                if pn is None:
                    continue
                pi = _note_idx(pn)
                ni = _note_idx(n)
                if pi is None or ni is None:
                    continue
                d = abs(ni - pi)
                cost += d * (1.0 if ch != 2 else 0.8)  # bass can move a bit more
                if d >= 10:
                    cost += 3.0  # discourage big leaps

            return cost

        best: tuple[str, str, str] | None = None
        best_cost = 1e18
        # small combinatorial search (<= 8^3 typical)
        for bass_n in cand_bass:
            for inner_n in cand_inner:
                for top_n in cand_top:
                    c = _cost(top_n, bass_n, inner_n)
                    if c < best_cost:
                        best_cost = c
                        best = (top_n, bass_n, inner_n)

        if best is None:
            return cand_top[0], cand_bass[0], cand_inner[0]
        return best

    # 20 progressions (degree in major scale) - designed to stay "choral" but offer more variety
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

        [0, 5, 3, 4],  # 10: pop-gospel (I-vi-IV-V)
        [3, 4, 0, 0],  # 11: amen cadence (IV-V-I-I)
        [0, 4, 5, 3],  # 12: pop lift (I-V-vi-IV)
        [0, 1, 3, 4],  # 13: step to cadence (I-ii-IV-V)
        [5, 3, 0, 4],  # 14: warm return (vi-IV-I-V)
        [2, 5, 1, 4],  # 15: circle-ish (iii-vi-ii-V)
        [0, 3, 1, 4],  # 16: gospel turn (I-IV-ii-V)
        [0, 0, 3, 4],  # 17: breakdown (I-I-IV-V)
        [4, 5, 3, 0],  # 18: tension-release (V-vi-IV-I)
        [5, 4, 3, 0],  # 19: descending close (vi-V-IV-I)
    ]

    # Derivation style: Near = more recognizable, Far = motif-only (more variation)
    dm = (derive_mode or 'Random').strip().lower()
    if dm in ('random', 'auto'):
        dm = rng.choice(['near', 'far'])

    if dm.startswith('n') or dm.startswith('c'):
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
            10: 'drive',
            11: 'cadence',
            12: 'transpose_up',
            13: 'arp',
            14: 'answer',
            15: 'drive',
            16: 'turn',
            17: 'pad',
            18: 'lift',
            19: 'cadence',
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
        prev_voice: dict[int, str | None] = {1: None, 2: None, 3: None}
        for bar, deg in enumerate(prog):
            r0 = bar * 16
            start_row = r0

            use_alt = bool(use_mixed and bar_use_alt[p_idx][bar])
            bar_scale = scale_alt if use_alt else scale
            bar_scale_up = scale_alt_up if use_alt else scale_up

            if p_idx == 3:
                bar_events = []
                strong_note = bar_scale_up[0]
            else:
                mode = mode_for_pattern.get(p_idx, 'base')
                # Variation strength: Near -> gentler; Far -> stronger
                v = float(variation)
                if not (v == v):
                    v = 1.0
                v = max(0.0, min(1.5, v))
                st = (0.70 + 0.55 * v) if dm.startswith('n') or dm.startswith('c') else (0.95 + 0.70 * v)
                bar_events = _mutate_events(rng, base_bars[bar], bar_scale_up, mode, strength=st)
                strong_note = next((n for (n, _) in bar_events if n is not None), bar_scale_up[0])

            def _chord_up_for_degree(d: int, sc: list[str], sc_up0: str):
                rt, th, fi = triad_from_degree(sc, d, octave_bias=0)
                cu = (note_shift_safe(rt, 12), note_shift_safe(th, 12), note_shift_safe(fi, 12))
                cu = tuple(n if n in CHROMATIC_SET else sc_up0 for n in cu)
                return rt, th, fi, cu

            # Choose a chord degree that best fits the melody on the strong beats of this bar.
            # This reduces occasional "schräg" moments when the melody uses passing tones.
            def _active_notes_16(ev: list[tuple[str | None, int]]) -> list[str | None]:
                cur: str | None = None
                outn: list[str | None] = []
                r = 0
                for nn, dd in ev:
                    dd = max(1, int(dd))
                    if nn is not None:
                        cur = nn
                    for _ in range(dd):
                        if r >= 16:
                            break
                        outn.append(cur)
                        r += 1
                    if r >= 16:
                        break
                if len(outn) < 16:
                    outn += [cur] * (16 - len(outn))
                return outn[:16]

            def _best_degree(base_deg: int) -> int:
                cands: list[int] = []
                for x in [base_deg, 0, 3, 4, 5, 2, 1]:
                    xi = int(x)
                    if xi not in cands:
                        cands.append(xi)

                active = _active_notes_16(bar_events)
                beat_rows = (0, 4, 8, 12)
                beat_pcs: list[str] = []
                for br in beat_rows:
                    nn = active[br] if br < len(active) else None
                    if nn is not None and nn in CHROMATIC_SET:
                        beat_pcs.append(_pc(nn))

                if (not beat_pcs) and strong_note is not None and strong_note in CHROMATIC_SET:
                    beat_pcs = [_pc(strong_note)]

                best_deg = int(base_deg)
                best_score = -1e18
                for cand in cands:
                    rt_c, th_c, fi_c, _ = _chord_up_for_degree(cand, bar_scale, bar_scale_up[0])
                    chord_pcs = {_pc(rt_c), _pc(th_c), _pc(fi_c)}
                    score = 0.0

                    for pc in beat_pcs:
                        score += (2.0 if pc in chord_pcs else -0.6)

                    if strong_note is not None and strong_note in CHROMATIC_SET:
                        score += (1.8 if _pc(strong_note) in chord_pcs else -1.2)

                    score -= abs(int(cand) - int(base_deg)) * 0.35
                    if score > best_score:
                        best_score = score
                        best_deg = int(cand)
                return best_deg

            use_deg = int(deg) if p_idx == 3 else _best_degree(int(deg))
            root, third, fifth, chord_up = _chord_up_for_degree(use_deg, bar_scale, bar_scale_up[0])

            # choose a non-crossing triad voicing (keeps accompaniment chord-consistent and "in tune")
            melody_ub = strong_note if (strong_note is not None and strong_note in CHROMATIC_SET) else None
            top, bass, inner = _choose_voicing(root, third, fifth, melody_ub, prev_voice)
            prev_voice[1] = top
            prev_voice[2] = bass
            prev_voice[3] = inner

            # basic harmony bed
            set_cell(p_idx, start_row, 1, top)
            set_cell(p_idx, start_row, 2, bass)
            set_cell(p_idx, start_row, 3, inner)

            # Harmony density can be nudged by variation.
            dens = max(0.15, min(0.90, 0.35 + 0.35 * max(0.0, min(1.5, float(variation)))) )
            if p_idx != 3 and rng.random() < dens:
                set_cell(p_idx, start_row + 8, 1, top)
                set_cell(p_idx, start_row + 8, 2, bass)
                # occasionally repeat the inner voice too, to glue the harmony together
                if rng.random() < (0.55 * dens):
                    set_cell(p_idx, start_row + 8, 3, inner)

            # pad pattern
            if p_idx == 3:
                if bar == 0:
                    hold = rng.choice([third, fifth, note_shift_safe(root, 12)])
                    hold = note_shift_safe(hold, 12) if hold.endswith('2') else hold
                    hold = hold if hold in CHROMATIC_SET else bar_scale_up[0]
                    set_cell(p_idx, start_row, 0, hold)
                elif bar == 1:
                    hold = rng.choice([fifth, third])
                    hold = note_shift_safe(hold, 12) if hold.endswith('2') else hold
                    hold = hold if hold in CHROMATIC_SET else bar_scale_up[0]
                    set_cell(p_idx, start_row, 0, hold)
                elif bar == 2:
                    hold = rng.choice([third, root])
                    hold = note_shift_safe(hold, 12) if hold.endswith('2') else hold
                    hold = hold if hold in CHROMATIC_SET else bar_scale_up[0]
                    set_cell(p_idx, start_row, 0, hold)
                else:
                    a = note_shift_safe(root, 12)
                    b = note_shift_safe(third, 12)
                    a = a if a in CHROMATIC_SET else bar_scale_up[0]
                    b = b if b in CHROMATIC_SET else bar_scale_up[0]
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
                bass2 = note_shift_safe(third, -12)
                bass5 = note_shift_safe(fifth, -12)
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
                hi = note_shift_safe(chord_up[2], 12)
                if hi in CHROMATIC_SET and rng.random() < 0.7:
                    set_cell(p_idx, start_row + 8, 1, hi)

            if p_idx == 9:
                # turnaround: a short answer in CH4
                tones = [chord_up[2], chord_up[1], chord_up[0], chord_up[1]]
                for i in range(8, 16, 2):
                    set_cell(p_idx, start_row + i, 3, tones[((i - 8) // 2) % len(tones)])

            

            if p_idx == 10:
                # Pop-gospel drive: walking-ish bass + gentle offbeat stabs
                bass2 = note_shift_safe(third, -12)
                bass5 = note_shift_safe(fifth, -12)
                seq = [bass, bass2, bass5, bass2]
                for i in range(0, 16, 4):
                    set_cell(p_idx, start_row + i, 2, seq[(i // 4) % len(seq)])
                stab = chord_up[2]
                for i in (2, 6, 10, 14):
                    set_cell(p_idx, start_row + i, 1, stab)

            if p_idx == 11:
                # Amen cadence: sustained top + small answer line
                hi = note_shift_safe(chord_up[2], 12)
                if hi in CHROMATIC_SET:
                    set_cell(p_idx, start_row, 1, hi)
                # gentle answer in CH4
                tones = [chord_up[1], chord_up[2], chord_up[1], chord_up[0]]
                for i in range(8, 16, 2):
                    set_cell(p_idx, start_row + i, 3, tones[((i - 8) // 2) % len(tones)])

            if p_idx == 12:
                # Pop lift: sync stabs + octave support
                stab = chord_up[1]
                for i in (4, 12):
                    set_cell(p_idx, start_row + i, 1, stab)
                o = note_shift_safe(chord_up[2], 12)
                if o in CHROMATIC_SET and rng.random() < 0.6:
                    set_cell(p_idx, start_row + 8, 1, o)

            if p_idx == 13:
                # Step-to-cadence: arpeggio shimmer in CH4
                arp = [chord_up[0], chord_up[1], chord_up[2], chord_up[1]]
                for i in range(0, 16, 2):
                    set_cell(p_idx, start_row + i, 3, arp[(i // 2) % len(arp)])

            if p_idx == 14:
                # Warm return: pedal-ish bass with chord pulses
                set_cell(p_idx, start_row, 2, bass)
                for i in (0, 4, 8, 12):
                    set_cell(p_idx, start_row + i, 1, chord_up[2])
                    set_cell(p_idx, start_row + i, 3, chord_up[1])

            if p_idx == 15:
                # Circle-ish: extra motion in bass and short top pickups
                bass5 = note_shift_safe(fifth, -12)
                seq = [bass, bass5, bass, bass5]
                for i in range(0, 16, 4):
                    set_cell(p_idx, start_row + i, 2, seq[(i // 4) % len(seq)])
                # Keep pickups on an 8th-grid (avoid row 1 "off-grid" feel)
                for i in (2, 10):
                    set_cell(p_idx, start_row + i, 1, chord_up[1])

            if p_idx == 16:
                # Gospel turn: short answering line in CH4 (call/response)
                tones = [chord_up[2], chord_up[1], chord_up[0], chord_up[1]]
                for i in range(0, 16, 2):
                    if rng.random() < 0.75:
                        set_cell(p_idx, start_row + i, 3, tones[(i // 2) % len(tones)])

            if p_idx == 17:
                # Breakdown: keep it sparser (remove some offbeats)
                if rng.random() < 0.8:
                    # clear some harmony repeats
                    for rr in (start_row + 8,):
                        set_cell(p_idx, rr, 1, None, sample=0)
                        set_cell(p_idx, rr, 3, None, sample=0)

            if p_idx == 18:
                # Climax: denser stabs (but still harmonic)
                stab1 = chord_up[2]
                stab2 = chord_up[1]
                for i in range(0, 16, 2):
                    set_cell(p_idx, start_row + i, 1, stab1 if (i % 4 == 0) else stab2)

            if p_idx == 19:
                # Descending close: descending bass line (approx) + top support
                b0 = bass
                b1 = note_shift(bass, -2)
                b2 = note_shift(bass, -4)
                b3 = note_shift(bass, -5)
                bseq = [x if x in CHROMATIC_SET else bass for x in (b0, b1, b2, b3)]
                for i in range(0, 16, 4):
                    set_cell(p_idx, start_row + i, 2, bseq[(i // 4) % len(bseq)])
                if rng.random() < 0.6:
                    set_cell(p_idx, start_row + 8, 1, chord_up[2])
            # keep legacy arpeggio feel in some patterns
            if p_idx in (2, 4) and rng.random() < 0.75:
                tones = [third, root, fifth, root]
                tones = [note_shift_safe(t, 12) if t.endswith('2') else t for t in tones]
                tones = [t if t in CHROMATIC_SET else bar_scale_up[0] for t in tones]
                for i in range(0, 16, 2):
                    set_cell(p_idx, start_row + i, 3, tones[(i // 2) % len(tones)])


    # Harmonize accompaniment channels per-bar so CH2..CH4 stay chord-consistent.
    # This reduces occasional "clashy" notes in highly varied patterns.
    def _nearest_note(note: str, allowed: list[str]) -> str:
        try:
            ni = CHROMATIC.index(note)
        except ValueError:
            return allowed[0]
        best = allowed[0]
        best_d = 999
        for a in allowed:
            try:
                ai = CHROMATIC.index(a)
            except ValueError:
                continue
            d = abs(ai - ni)
            if d < best_d:
                best_d = d
                best = a
        return best

    for p_idx, pat in enumerate(patterns):
        prog = progs[p_idx]
        for bar, deg in enumerate(prog):
            r0 = bar * 16
            use_alt = bool(use_mixed and bar_use_alt[p_idx][bar])
            hscale = scale_alt if use_alt else scale
            rt, th, fi = triad_from_degree(hscale, deg, octave_bias=0)
            allowed = [rt, th, fi, note_shift_safe(rt, 12), note_shift_safe(th, 12), note_shift_safe(fi, 12), note_shift_safe(rt, -12)]
            allowed = [n for n in allowed if n in CHROMATIC_SET]
            if not allowed:
                continue
            for row in range(r0, r0 + 16):
                # accompaniment channels: force chord tones
                for ch in (1, 2, 3):
                    if ch in drum_ch:
                        continue
                    note, samp, eff, par = pat[row][ch]
                    if note is not None and note not in allowed:
                        pat[row][ch] = (_nearest_note(note, allowed), samp, eff, par)
                # melody: keep freedom, but snap strong beats if wildly off chord (rare)
                if row in (r0, r0 + 8):
                    note, samp, eff, par = pat[row][0]
                    if note is not None and note not in allowed:
                        nn = _nearest_note(note, allowed)
                        try:
                            if abs(CHROMATIC.index(nn) - CHROMATIC.index(note)) >= 3:
                                pat[row][0] = (nn, samp, eff, par)
                        except ValueError:
                            pass

    # Ensure the selected speed/tempo is present in EVERY pattern.
    spd = max(1, min(31, int(speed)))
    bpm = max(32, min(255, int(tempo)))
    for pat in patterns:
        n, s, eff, par = pat[0][0]
        pat[0][0] = (n, s, 0x0F, spd)
        n, s, eff, par = pat[0][1]
        pat[0][1] = (n, s, 0x0F, bpm)

    return patterns, key_root, base_melody_name, base_meta, derive_used

def apply_drumsets_to_patterns(
    patterns: list,
    rng: random.Random,
    drum_channel_styles: dict[int, str],
    drum_sample_numbers: dict[str, dict[str, int]],
    variation: float = 1.0,
) -> None:
    """Overwrite selected channels with style-appropriate drum patterns.

    Patterns are assumed to be 64 rows, 4 bars of 16 rows.
    Notes are placed at C-3 (fixed) and instruments switch via sample numbers.
    """
    if not drum_channel_styles:
        return

    v = float(variation) if (variation == variation) else 1.0
    v = max(0.0, min(1.5, v))

    def _pick(style: str, name: str) -> int | None:
        m = drum_sample_numbers.get(style, {})
        return int(m.get(name)) if name in m else None

    def _hit(pat, row: int, ch: int, samp: int | None):
        if samp is None:
            return
        if 0 <= row < 64:
            pat[row][ch] = ("C-3", int(samp), 0x00, 0x00)

    def _is_intense(p_idx: int) -> float:
        # 0..19 -> 0..1 roughly
        return max(0.0, min(1.0, p_idx / max(1, (PATTERN_COUNT - 1))))

    for p_idx, pat in enumerate(patterns):
        # clear drum channels
        for ch in drum_channel_styles.keys():
            for r in range(64):
                pat[r][ch] = (None, 0, 0, 0)

        intensity = _is_intense(p_idx)

        for bar in range(4):
            base = bar * 16
            # small per-bar variation
            local = intensity * (0.65 + 0.35 * rng.random())

            for ch, style in drum_channel_styles.items():
                st = (style or "techno").lower()

                k = _pick(st, "Kick")
                s = _pick(st, "Snare")
                c = _pick(st, "Clap")
                hh = _pick(st, "CHat")
                oh = _pick(st, "OHat")
                tom = _pick(st, "Tom")
                crash = _pick(st, "Crash")
                perc = _pick(st, "Perc")

                # --- style templates on a 16-step grid (rows 0..15) ---
                if st == "techno":
                    # 4-on-the-floor + offbeat hats + clap
                    for r in (0, 4, 8, 12):
                        _hit(pat, base + r, ch, k)
                    for r in (4, 12):
                        _hit(pat, base + r, ch, c if c is not None else s)
                    # offbeats
                    for r in (2, 6, 10, 14):
                        _hit(pat, base + r, ch, hh)
                    # open hat on last offbeat sometimes
                    if rng.random() < (0.25 + 0.45 * local):
                        _hit(pat, base + 14, ch, oh)
                    # extra 16ths when intense
                    if local > 0.55 and rng.random() < (0.35 + 0.25 * v):
                        for r in (1, 3, 5, 7, 9, 11, 13, 15):
                            if rng.random() < 0.55:
                                _hit(pat, base + r, ch, hh)
                    # crash at bar start in some later patterns
                    if bar == 0 and intensity > 0.55 and rng.random() < 0.35:
                        _hit(pat, base + 0, ch, crash)

                elif st == "dubstep":
                    # half-time feel: snare on beat 3 (row 8)
                    _hit(pat, base + 8, ch, s)
                    if c is not None and rng.random() < 0.55:
                        _hit(pat, base + 8, ch, c)
                    # kick placements
                    _hit(pat, base + 0, ch, k)
                    if rng.random() < (0.35 + 0.35 * local):
                        _hit(pat, base + (12 if rng.random() < 0.6 else 10), ch, k)
                    if local > 0.55 and rng.random() < (0.30 + 0.35 * v):
                        _hit(pat, base + 14, ch, k)
                    # hats: syncopated, probabilistic
                    for r in range(16):
                        if r in (0, 8):
                            continue
                        prob = 0.10 + 0.35 * local + 0.10 * v
                        if r % 2 == 1:
                            prob += 0.10
                        if rng.random() < prob:
                            _hit(pat, base + r, ch, hh if (r % 4 != 0) else oh)
                    # occasional tom fill near end of bar
                    if local > 0.60 and rng.random() < 0.25:
                        for r in (12, 13, 14, 15):
                            if rng.random() < 0.55:
                                _hit(pat, base + r, ch, tom if tom is not None else perc)

                elif st == "hiphop":
                    # laid-back: kick + snare, hats with swing-ish randomness
                    for r in (0, 8):
                        _hit(pat, base + r, ch, k)
                    for r in (4, 12):
                        _hit(pat, base + r, ch, s)
                    # extra kicks
                    if rng.random() < (0.25 + 0.35 * local):
                        _hit(pat, base + 10, ch, k)
                    if rng.random() < (0.20 + 0.25 * local):
                        _hit(pat, base + 14, ch, k)
                    # hats
                    for r in (2, 6, 10, 14):
                        _hit(pat, base + r, ch, hh)
                    if local > 0.45 and rng.random() < 0.35:
                        for r in (3, 7, 11, 15):
                            if rng.random() < 0.55:
                                _hit(pat, base + r, ch, hh)
                    if rng.random() < 0.15:
                        _hit(pat, base + 12, ch, c if c is not None else perc)

                elif st == "folk":
                    # simple, "acoustic-ish": kick on 1&3, snare on 2&4, light hats
                    for r in (0, 8):
                        _hit(pat, base + r, ch, k)
                    for r in (4, 12):
                        _hit(pat, base + r, ch, s)
                    # hats on offbeats (sparse)
                    for r in (2, 6, 10, 14):
                        if rng.random() < (0.55 + 0.20 * v):
                            _hit(pat, base + r, ch, hh)
                    if local > 0.60 and rng.random() < 0.22:
                        _hit(pat, base + 15, ch, perc)

                elif st == "rock":
                    # straight rock: kick 1&3(+), snare 2&4, 8th hats, occasional crash
                    for r in (0, 8, 12):
                        _hit(pat, base + r, ch, k)
                    for r in (4, 12):
                        _hit(pat, base + r, ch, s)
                    for r in (0, 2, 4, 6, 8, 10, 12, 14):
                        _hit(pat, base + r, ch, hh)
                    if bar == 0 and rng.random() < (0.22 + 0.25 * local):
                        _hit(pat, base + 0, ch, crash)
                    if local > 0.55 and rng.random() < 0.25:
                        for r in (13, 14, 15):
                            _hit(pat, base + r, ch, tom if tom is not None else perc)

                else:  # pop / default
                    # pop groove: kick+snare, 8th hats, light extras
                    for r in (0, 8, 12):
                        if rng.random() < 0.85:
                            _hit(pat, base + r, ch, k)
                    for r in (4, 12):
                        _hit(pat, base + r, ch, s)
                    for r in (0, 2, 4, 6, 8, 10, 12, 14):
                        _hit(pat, base + r, ch, hh)
                    if rng.random() < (0.18 + 0.30 * local):
                        _hit(pat, base + 14, ch, oh)
                    if local > 0.55 and rng.random() < (0.22 + 0.22 * v):
                        _hit(pat, base + 15, ch, perc)

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


def generate_smart_order(rng: random.Random, n_patterns: int = PATTERN_COUNT) -> list[int]:
    """Generate a musically sensible pattern order with variety.

    This stays within 0..n_patterns-1 and tends to end with a cadence.
    """
    n_patterns = max(1, int(n_patterns))

    def _f(lst: list[int]) -> list[int]:
        return [x for x in lst if 0 <= x < n_patterns]

    base = _f([0])
    ornament = _f([1, 12])
    answer = _f([2, 4, 14])
    pad = _f([3, 17])
    cadence = _f([5, 11, 19])
    drive = _f([6, 10, 15, 18])
    arp = _f([7, 13])
    lift = _f([8, 18])
    turn = _f([9, 16])
    extra = _f([6, 7, 8, 9, 10, 13, 15, 16, 18])

    def pick(lst: list[int], fallback: list[int]) -> int:
        pool = lst if lst else fallback
        pool = pool if pool else [0]
        return rng.choice(pool)

    order: list[int] = []
    order.append(pick(base, [0]))
    if rng.random() < 0.35:
        order.append(pick(pad, [0]))

    # 2–4 blocks
    blocks = rng.randint(2, 4)
    for _ in range(blocks):
        order.append(pick(ornament, [0]))
        if rng.random() < 0.70:
            order.append(pick(answer, [0]))
        if rng.random() < 0.75:
            order.append(pick(drive, extra or [0]))
        if rng.random() < 0.55:
            order.append(pick(arp, extra or [0]))
        if rng.random() < 0.45:
            order.append(pick(lift, extra or [0]))
        if rng.random() < 0.40:
            order.append(pick(turn, extra or [0]))

    # pre-ending
    if rng.random() < 0.55:
        order.append(pick(pad, [0]))
    order.append(pick(cadence, [5] if n_patterns > 5 else [0]))

    # ensure reasonable length
    if len(order) < 6 and extra:
        while len(order) < 6:
            order.insert(-1, rng.choice(extra))
    if len(order) > 24:
        order = order[:23] + [order[-1]]
    return order


def validate_order(order: list[int], n_patterns: int = PATTERN_COUNT) -> None:
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
    scale_mode: str
    variation: float
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
    mute_channels: list[bool]
    stereo_width: float
    octave_spans: list[int] = field(default_factory=lambda: [3, 3, 3, 3])
    sample_names: list[str] = field(default_factory=list)
    drum_channel_styles: dict[int, str] = field(default_factory=dict)
    harmony_score: float = 0.0
    fadeout_pattern: bool = False
def _cell_to_text(cell: tuple[str | None, int, int, int]) -> str:
    note, samp, eff, par = cell
    n = note if note is not None else "---"
    s = f"{samp:02d}" if samp else "--"
    e = f"{eff:X}{par:02X}" if (eff or par) else "---"
    return f"{n} {s} {e}"




# -----------------------------
# Melody Plugin export (from generated songs)
# -----------------------------

_MIDI_TO_PT_NOTE: dict[int, str] = {}
try:
    # Build a small reverse map for the ProTracker note range we use.
    for _n in CHROMATIC:
        _m = _parse_note_token_to_midi(_n)
        if _m is not None:
            _MIDI_TO_PT_NOTE[int(_m)] = _n
except Exception:
    _MIDI_TO_PT_NOTE = {}


def _extract_melody_events_from_song(song: SongData, channel: int = 0) -> list[tuple[int | None, int]]:
    """Extract a monophonic melody event stream from a rendered song.

    We walk the final order and read note changes on the given channel, collapsing
    consecutive equal notes/rests. Output is (midi_note|None, dur_rows).
    """
    events: list[tuple[int | None, int]] = []
    last: int | None = None
    dur = 0
    try:
        order = list(song.order) if getattr(song, 'order', None) else list(song.order_original)
    except Exception:
        order = []

    for p_idx in order:
        if not isinstance(p_idx, int):
            continue
        if p_idx < 0 or p_idx >= len(song.patterns):
            continue
        pat = song.patterns[p_idx]
        for r in range(64):
            try:
                note = pat[r][channel][0]
            except Exception:
                note = None
            if note is None or note == "---":
                cur = None
            else:
                cur = _parse_note_token_to_midi(str(note))
            if cur == last:
                dur += 1
            else:
                if dur > 0:
                    events.append((last, dur))
                last = cur
                dur = 1

    if dur > 0:
        events.append((last, dur))

    # Split very long runs for readability.
    out: list[tuple[int | None, int]] = []
    for n, d in events:
        d = max(1, int(d))
        while d > 16:
            out.append((n, 16))
            d -= 16
        out.append((n, d))

    return out or [(60, 4), (62, 4), (64, 4), (65, 4)]


def plugin_export_text_from_song(mod_path: Path, song: SongData) -> str:
    """Return a melody-plugin compatible text block extracted from a generated song.

    This is designed so users can drop a saved parameter .txt into melody_plugins/<folder>/
    and the app can treat it as a base melody plugin.
    """
    name = (getattr(song, 'title_txt', '') or mod_path.stem).strip() or mod_path.stem    # derive basic metadata hints
    mode = ''
    try:
        mode = str(getattr(song, 'scale_mode', '') or '').strip().lower()
    except Exception:
        mode = ''

    # Prefer the actual song mode; fall back to base melody plugin meta if present.
    try:
        if not mode and getattr(song, 'base_melody_meta', None):
            mode = str(song.base_melody_meta.get('mode', '') or '').strip().lower()
    except Exception:
        pass

    if not mode:
        mode = 'major'

    # Normalize to the modes our plugin loader understands
    if mode.startswith('maj'):
        mode = 'major'
    elif mode.startswith('min') or 'moll' in mode or 'aeolian' in mode:
        mode = 'minor'
    elif mode.startswith('dor'):
        mode = 'dorian'
    elif mode.startswith('mixo'):
        mode = 'mixolydian'
    elif mode.startswith('mix'):
        mode = 'mixed'
    else:
        # default fallback
        mode = 'minor' if 'minor' in mode else 'major'

    tempo = int(getattr(song, 'tempo', DEFAULT_TEMPO) or DEFAULT_TEMPO)
    tmin = max(40, tempo - 20)
    tmax = min(255, tempo + 20)

    key_root = str(getattr(song, 'key_root', '') or '').strip() or 'C-2'
    key_hi = key_root
    try:
        km = _parse_note_token_to_midi(key_root)
        if km is not None and _MIDI_TO_PT_NOTE:
            # perfect fifth above (approx) for a friendly key range hint
            key_hi = _MIDI_TO_PT_NOTE.get(int(km) + 7, key_root)
    except Exception:
        key_hi = key_root

    events = _extract_melody_events_from_song(song, channel=0)
    bars = _events_to_4bars_degree_template(events)

    lines: list[str] = []
    lines.append('name: ' + name)
    lines.append(f'mode: {mode}')
    lines.append(f'tempo_hint: {tmin}-{tmax}')
    lines.append(f'preferred_key_range: {key_root}..{key_hi}')
    lines.append(f'source_mod: {mod_path.name}')
    lines.append(f'seed: {getattr(song, "seed", "")}')
    lines.append('# format: DEG OCT DUR  (DEG=0..6, OCT=-2..2, DUR=rows; use R for rest)')
    for bi, bar in enumerate(bars, start=1):
        lines.append(f'# bar {bi}')
        for deg, octv, dur in bar:
            if deg is None:
                lines.append(f'R 0 {int(dur)}')
            else:
                lines.append(f'{int(deg)} {int(octv)} {int(dur)}')
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'
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
    lines.append(f"scale_mode: {getattr(song, 'scale_mode', '')}")
    try:
        lines.append(f"variation: {float(getattr(song, 'variation', 0.0)):.3f}")
    except Exception:
        lines.append("variation: ")
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
    try:
        mc = getattr(song, 'mute_channels', [False, False, False, False])
        mc = [(bool(mc[i]) if i < len(mc) else False) for i in range(4)]
        lines.append(f"mute_channels: {''.join('1' if x else '0' for x in mc)}")
    except Exception:
        pass
    try:
        lines.append(f"stereo_width: {float(getattr(song, 'stereo_width', 1.0)):.3f}")
    except Exception:
        pass
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



    # --- melody plugin export block ---
    try:
        lines.append("")
        lines.append("# === MELODY PLUGIN EXPORT ===")
        lines.append("# Tip: Create a new subfolder in 'melody_plugins' and drop this .txt into it.")
        lines.append("# The app will treat it as a melody plugin (no renaming required).")
        lines.append("")
        lines.extend(plugin_export_text_from_song(mod_path, song).splitlines())
    except Exception:
        pass

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
    key_root_override: str | None = None,
    scale_mode: str | None = None,
    variation: float = 1.0,
    mute_channels: list[bool] | None = None,
    stereo_width: float = 1.0,
    octave_spans: list[int] | None = None,
    mod_signature: str | None = None,
    compat_mode: bool = True,
    fadeout_pattern: bool = False,
    quality_passes: int = 3,
) -> tuple[Path, SongData]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = random_seed_value()
    rng = random.Random(seed)
    rng_s = random.Random((int(seed) ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)

    sig = (mod_signature or DEFAULT_MOD_SIGNATURE).strip()
    if sig not in MOD_SIGNATURE_CHOICES:
        sig = DEFAULT_MOD_SIGNATURE

    inst_kinds = normalize_instrument_list(instruments)
    # Determine which channels are drum tracks (based on chosen "instrument").
    drum_channel_styles: dict[int, str] = {}
    for ch, k in enumerate(inst_kinds):
        st = drumset_style_from_kind(k)
        if st:
            drum_channel_styles[int(ch)] = str(st)

    ensemble_size = sum(1 for ch in range(len(inst_kinds)) if int(ch) not in drum_channel_styles)

    # --- sample allocation (up to 31 instruments) ---
    # We keep the first 4 sample slots reserved for the 4 channels (legacy-friendly),
    # then append drumkit samples as extra instruments (sample numbers 5..31).
    sample_cache: dict[tuple[str, bool, int], bytes] = {}
    samples_bytes: list[bytes] = []
    sample_names: list[str] = []
    sample_vols: list[int] = []

    # channel base samples (1..4)
    for ch, k in enumerate(inst_kinds):
        if ch in drum_channel_styles:
            # keep slot present, but silent (drums are handled via extra instruments)
            samples_bytes.append(b"\x00\x00")
            sample_names.append(k)
            sample_vols.append(0)
        else:
            ck = (k, bool(disable_vibrato), int(ensemble_size))
            if ck not in sample_cache:
                sample_cache[ck] = make_instrument_sample(k, rng_s, f0=REF_F0, disable_vibrato=bool(disable_vibrato), ensemble_size=int(ensemble_size))
            samples_bytes.append(sample_cache[ck])
            sample_names.append(k)
            sample_vols.append(int(INSTRUMENT_VOL.get(k, 48)))

    # drumkits (shared per style)
    drum_sample_numbers: dict[str, dict[str, int]] = {}
    unique_styles = []
    for st in drum_channel_styles.values():
        if st not in unique_styles:
            unique_styles.append(st)

    next_no = 5  # sample numbers are 1-based; 1..4 already used
    for st in unique_styles:
        kit = make_drumkit_samples(st, rng_s)
        drum_sample_numbers[st] = {}
        # If we're close to the 31-sample limit, keep only the core kit first.
        core_first = ["Kick", "Snare", "Clap", "CHat", "OHat", "Tom", "Crash", "Perc"]
        for d in core_first:
            if next_no > 31:
                break
            b = kit.get(d)
            if not b:
                continue
            samples_bytes.append(b)
            prefix = DRUM_STYLE_PREFIX.get(st, st[:3].upper())
            sample_names.append(f"{prefix} {d}")
            sample_vols.append(int(DRUM_VOL.get(d, 48)))
            drum_sample_numbers[st][d] = int(next_no)
            next_no += 1

    samples_float = [bytes_to_float_sample(b) for b in samples_bytes]

    patterns, key_root, base_melody, base_melody_meta, derive_used = make_patterns(
        rng,
        speed=speed,
        tempo=tempo,
        melody_name=melody_name,
        derive_mode=derive_mode,
        key_root_override=key_root_override,
        scale_mode=scale_mode,
        variation=variation,
        octave_spans=octave_spans,
        drum_channels=set(drum_channel_styles.keys()),
    )

    # Inject drum patterns (if any channels use a drumset preset).
    if drum_channel_styles:
        apply_drumsets_to_patterns(patterns, rng, drum_channel_styles, drum_sample_numbers, variation=float(variation))

    # ==========================================
    # QUALITY CHECKING WITH HARMONY ANALYSIS (configurable passes)
    # ==========================================
    harmony_score = 0.0
    if HARMONY_AVAILABLE and quality_passes > 0:
        quality_checker = MusicQualityChecker(quality_threshold=70.0)
        scale_mode_clean = str(scale_mode or 'major').strip().lower()
        if scale_mode_clean in ('auto', 'random'):
            scale_mode_clean = 'major'
        
        last_quality = None
        for pass_num in range(min(quality_passes, 5)):  # max 5 passes
            if pass_num == 0:
                # Pass 1: Basic harmonic analysis
                quality = quality_checker.check_quality_first_pass(patterns, scale_mode_clean, key_root)
                print(f"[Quality Check] Pass 1 - Overall: {quality.overall_score:.1f}, Harmony: {quality.harmony_score:.1f}")
            elif pass_num == 1:
                # Pass 2: Chord progression analysis
                quality = quality_checker.check_quality_second_pass(patterns, scale_mode_clean, key_root)
                print(f"[Quality Check] Pass 2 - Overall: {quality.overall_score:.1f}, Melody: {quality.melody_score:.1f}")
            else:
                # Pass 3+: Final verification
                quality = quality_checker.check_quality_third_pass(patterns, scale_mode_clean, key_root)
                print(f"[Quality Check] Pass {pass_num+1} (Final) - Overall: {quality.overall_score:.1f}")
            
            last_quality = quality
            if not quality.passed:
                print(f"[Quality Check] Pass {pass_num+1} FAILED - Issues: {quality.issues}")
            else:
                print(f"[Quality Check] Pass {pass_num+1} PASSED - Strengths: {quality.strengths}")
                harmony_score = quality.overall_score
                break  # Stop early if quality is good
        
        if last_quality and not last_quality.passed:
            print(f"[Quality Check] All {quality_passes} passes completed. Final issues: {last_quality.issues}")
            harmony_score = last_quality.overall_score
    
    # Inject drum patterns (if any channels use a drumset preset).
    if drum_channel_styles:
        apply_drumsets_to_patterns(patterns, rng, drum_channel_styles, drum_sample_numbers, variation=float(variation))

    if order is None:
        order = parse_order_string(DEFAULT_ORDER_STR)
    validate_order(order, n_patterns=len(patterns))

    order_original = list(order)

    order_for_write = list(order)
    
    # Add fade-out pattern if requested (empty pattern for natural instrument decay)
    if fadeout_pattern and len(order_for_write) > 0:
        # Create an empty pattern (all channels: no note, no sample, no effect)
        empty_pattern = [[(None, 0, 0, 0) for _ in range(4)] for _ in range(64)]
        patterns.append(empty_pattern)
        order_for_write.append(len(patterns) - 1)
        print(f"[MOD] Added empty fade-out pattern at position {len(patterns) - 1}")
    
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
    # Instrument headers (exactly 31)
    insts: list[bytes] = []
    for nm, sb, vol in zip(sample_names, samples_bytes, sample_vols):
        insts.append(inst_header(nm, sb, volume=int(vol)))
    empty_loop = 0 if compat_mode else 1
    empty = b"\x00" * 22 + struct.pack(">H", 0) + bytes([0]) + bytes([0]) + struct.pack(">H", 0) + struct.pack(">H", empty_loop)
    if len(insts) < 31:
        insts += [empty] * (31 - len(insts))
    else:
        insts = insts[:31]

    song_len = len(order_for_write)
    order_table = bytes(order_for_write + [0] * (128 - len(order_for_write)))

    mod = bytearray()
    mod += title
    for ih in insts:
        mod += ih
    mod += bytes([song_len])
    restart_byte = 0x7F if compat_mode else 0
    mod += bytes([restart_byte])  # restart byte
    mod += order_table
    mod += sig.encode("ascii", "ignore")[:4].ljust(4, b" ")
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
        scale_mode=(
            (str(scale_mode).strip() if scale_mode is not None else "Auto")
            if str(scale_mode or "").strip().lower() not in ("auto", "random", "")
            else str((base_melody_meta or {}).get('mode', 'major') or 'major')
        ),
        variation=float(variation),
        base_melody=base_melody,
        base_melody_meta=dict(base_melody_meta or {}),
        patterns=patterns,
        order_original=order_original,
        order=order_for_write,
        samples_bytes=samples_bytes,
        samples_float=samples_float,
        sample_names=sample_names,
        drum_channel_styles=dict(drum_channel_styles),
        instrument_kinds=inst_kinds,
        speed=int(speed),
        tempo=int(tempo),
        slowdown_enabled=bool(enable_slowdown),
        derive_mode=str(derive_used),
        vibrato_disabled=bool(disable_vibrato),
        mute_channels=(
            [(bool(mute_channels[i]) if i < len(mute_channels) else False) for i in range(4)]
            if mute_channels is not None else [False, False, False, False]
        ),
        stereo_width=float(stereo_width),
        octave_spans=(
            [int(octave_spans[i]) if octave_spans is not None and i < len(octave_spans) else 3 for i in range(4)]
        ),
        harmony_score=harmony_score,
        fadeout_pattern=bool(fadeout_pattern),
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
    # sample index per channel (0-based into song.samples_float)
    max_si = max(0, len(song.samples_float) - 1)
    chan_samp = [min(0, max_si), min(1, max_si), min(2, max_si), min(3, max_si)]
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
                        # MOD sample numbers are 1..31
                        max_si = max(0, len(song.samples_float) - 1)
                        chan_samp[ch] = max(0, min(max_si, int(samp) - 1))
                    chan_pos[ch] = 0.0

            row_secs = max(0.001, speed * _tick_seconds(tempo))
            n = int(row_secs * out_rate)
            # localize for speed
            sp = song.samples_float
            pos0, pos1, pos2, pos3 = chan_pos
            per0, per1, per2, per3 = chan_period
            sidx0, sidx1, sidx2, sidx3 = chan_samp
            vol0, vol1, vol2, vol3 = chan_vol

            for _ in range(n):
                # raw per-channel contributions (mono)
                c0 = c1 = c2 = c3 = 0.0

                # channel 0 (L)
                if per0 > 0:
                    step = _freq_from_period(per0) / out_rate
                    if 0 <= sidx0 < len(sp):
                        samp_arr = sp[sidx0]
                    else:
                        samp_arr = sp[0] if sp else []
                    i0 = int(pos0)
                    if i0 < len(samp_arr):
                        v = samp_arr[i0] * (vol0 / 64.0)
                        c0 = v
                    pos0 += step

                # channel 1 (R)
                if per1 > 0:
                    step = _freq_from_period(per1) / out_rate
                    if 0 <= sidx1 < len(sp):
                        samp_arr = sp[sidx1]
                    else:
                        samp_arr = sp[0] if sp else []
                    i1 = int(pos1)
                    if i1 < len(samp_arr):
                        v = samp_arr[i1] * (vol1 / 64.0)
                        c1 = v
                    pos1 += step

                # channel 2 (R)
                if per2 > 0:
                    step = _freq_from_period(per2) / out_rate
                    if 0 <= sidx2 < len(sp):
                        samp_arr = sp[sidx2]
                    else:
                        samp_arr = sp[0] if sp else []
                    i2 = int(pos2)
                    if i2 < len(samp_arr):
                        v = samp_arr[i2] * (vol2 / 64.0)
                        c2 = v
                    pos2 += step

                # channel 3 (L)
                if per3 > 0:
                    step = _freq_from_period(per3) / out_rate
                    if 0 <= sidx3 < len(sp):
                        samp_arr = sp[sidx3]
                    else:
                        samp_arr = sp[0] if sp else []
                    i3 = int(pos3)
                    if i3 < len(samp_arr):
                        v = samp_arr[i3] * (vol3 / 64.0)
                        c3 = v
                    pos3 += step

                # mutes + panning (tracker-ish). Width scales the pan amount.
                mutes = getattr(song, 'mute_channels', [False, False, False, False])
                try:
                    if len(mutes) < 4:
                        mutes = list(mutes) + [False] * (4 - len(mutes))
                except Exception:
                    mutes = [False, False, False, False]
                if mutes[0]:
                    c0 = 0.0
                if mutes[1]:
                    c1 = 0.0
                if mutes[2]:
                    c2 = 0.0
                if mutes[3]:
                    c3 = 0.0

                width = float(getattr(song, 'stereo_width', 1.0) or 1.0)
                if not (width == width):
                    width = 1.0
                width = max(0.0, min(2.0, width))
                pans = [-0.70, 0.70, 0.30, -0.30]
                pans = [p * width for p in pans]

                l = r = 0.0
                for c, pan in ((c0, pans[0]), (c1, pans[1]), (c2, pans[2]), (c3, pans[3])):
                    lg = 0.5 * (1.0 - pan)
                    rg = 0.5 * (1.0 + pan)
                    l += c * lg
                    r += c * rg

                # mild master gain to avoid clipping
                master = 0.40
                l *= master
                r *= master
                c0 *= master
                c1 *= master
                c2 *= master
                c3 *= master

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
    """Stereo L/R scopes (click visualizer to toggle from spectrum)."""

    def __init__(self, canvas, width: int = 560, height: int = 160):
        self.canvas = canvas
        self.width = int(width)
        self.height = int(height)
        self._pad = 6
        self._cleared = True

        self.canvas.configure(width=self.width, height=self.height, bg="#8f8f8f", highlightthickness=0)

        self._scope_ids: list[int] = []
        inner_h = self.height - 2 * self._pad
        self._slot_h = inner_h / 2.0  # L / R

        labels = ["L", "R"]
        for ch in range(2):
            x0 = self._pad
            x1 = self.width - self._pad
            y0 = self._pad + ch * self._slot_h
            y1 = y0 + self._slot_h

            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#6f6f6f", width=1)
            mid = (y0 + y1) * 0.5
            self.canvas.create_line(x0 + 1, mid, x1 - 1, mid, fill="#6f6f6f")

            self.canvas.create_text(x0 + 14, y0 + 10, text=labels[ch], fill="#1a1a1a", font=("Courier New", 12, "bold"))

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

    def update_from_pcm(self, pcm16: bytes, sr: int, sample_index: int, window: int = 1024):
        # pcm16: interleaved stereo int16 (L,R)
        if not pcm16:
            return
        total_frames = len(pcm16) // 4
        if total_frames <= 0:
            return

        i0 = max(0, min(total_frames - 1, int(sample_index)))
        i1 = min(total_frames, i0 + int(window))
        if i1 - i0 < 16:
            return

        x0 = self._pad
        x1 = self.width - self._pad
        w = max(16, int(x1 - x0 - 2))

        pts = min(360, i1 - i0, w)

        def get_lr(frame: int):
            off = frame * 4
            l = int.from_bytes(pcm16[off:off+2], "little", signed=True) / 32768.0
            r = int.from_bytes(pcm16[off+2:off+4], "little", signed=True) / 32768.0
            return l, r

        for ch in range(2):
            y0 = self._pad + ch * self._slot_h
            y1 = y0 + self._slot_h
            mid = (y0 + y1) * 0.5
            amp = (self._slot_h * 0.42)

            coords = []
            for p in range(pts):
                frame = i0 + int(p * (i1 - i0 - 1) / max(1, pts - 1))
                l, r = get_lr(frame)
                v = l if ch == 0 else r
                x = x0 + 1 + (p * (w - 1) / max(1, pts - 1))
                y = mid - (v * amp)
                coords.extend([x, y])

            self.canvas.coords(self._scope_ids[ch], *coords)

        self._cleared = False

class LightOrganView:
    """Classic light organ visualization with color bars responding to frequency bands."""

    def __init__(self, canvas, width: int = 560, height: int = 160):
        self.canvas = canvas
        self.width = int(width)
        self.height = int(height)
        self._pad = 6
        self._cleared = True

        self.canvas.configure(width=self.width, height=self.height, bg="#1a1a1a", highlightthickness=0)

        # Classic light organ colors: Low=red, Mid=orange/yellow, High=green
        self._colors = ["#ff0000", "#ff4400", "#ff8800", "#ffaa00", "#ffcc00", "#88ff00", "#44ff00", "#00ff00"]
        self._n_bands = len(self._colors)

        self._bar_ids: list[int] = []
        self._glow_ids: list[int] = []

        # Calculate bar dimensions
        slot_w = (self.width - 2 * self._pad) / self._n_bands

        for i, color in enumerate(self._colors):
            x0 = self._pad + i * slot_w
            x1 = x0 + slot_w - 4

            # Glow effect (larger, semi-transparent looking)
            glow = self.canvas.create_rectangle(x0 - 2, self.height - self._pad,
                                               x1 + 2, self.height - self._pad,
                                               outline="", fill=color, stipple="gray50")
            self._glow_ids.append(glow)

            # Main bar
            bar = self.canvas.create_rectangle(x0, self.height - self._pad,
                                              x1, self.height - self._pad,
                                              outline="", fill=color)
            self._bar_ids.append(bar)

        self._levels = [0.0] * self._n_bands
        self._slot_w = slot_w

    def reset(self):
        self._cleared = True
        self._levels = [0.0] * self._n_bands
        y_bottom = self.height - self._pad
        for bar_id, glow_id in zip(self._bar_ids, self._glow_ids):
            self.canvas.coords(bar_id,
                              self.canvas.coords(bar_id)[0], y_bottom,
                              self.canvas.coords(bar_id)[2], y_bottom)
            # Reset glow
            glow_coords = list(self.canvas.coords(glow_id))
            glow_coords[1] = y_bottom
            glow_coords[3] = y_bottom
            self.canvas.coords(glow_id, *glow_coords)

    def _compute_levels(self, mono: list[float], sr: int) -> list[float]:
        """Simple frequency band analysis."""
        n = len(mono)
        if n < 64:
            return [0.0] * self._n_bands

        if _HAS_NUMPY:
            x = _np.array(mono, dtype=_np.float32)
            # Simple energy in 3 frequency bands (low, mid, high)
            spec = _np.abs(_np.fft.rfft(x))
            freqs = _np.fft.rfftfreq(n, 1.0 / sr)

            # Map to 8 light organ bands
            bands = []
            f_edges = [60, 150, 300, 600, 1200, 2400, 4800, 8000, 12000]
            for i in range(self._n_bands):
                f0, f1 = f_edges[i], f_edges[i + 1]
                idx = _np.where((freqs >= f0) & (freqs < f1))[0]
                if idx.size == 0:
                    bands.append(0.0)
                else:
                    # RMS of band
                    energy = float(_np.sqrt(_np.mean(spec[idx] ** 2)))
                    bands.append(energy)
            return bands
        else:
            # Fallback: time-domain energy with simple bandpass simulation
            # Just use overall energy distributed across bands
            energy = sum(x * x for x in mono) / max(1, len(mono))
            energy = math.sqrt(energy)
            # Create a fake distribution
            return [energy * (0.5 + 0.5 * math.sin(i * 0.5)) for i in range(self._n_bands)]

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

        # Extract mono window
        mono: list[float] = []
        off = i0 * 4
        end = i1 * 4
        for j in range(off, end, 4):
            l = int.from_bytes(pcm16[j : j + 2], byteorder="little", signed=True)
            r = int.from_bytes(pcm16[j + 2 : j + 4], byteorder="little", signed=True)
            mono.append(((l + r) * 0.5) / 32768.0)

        raw = self._compute_levels(mono, sr)

        # Normalize and smooth
        mx = max(1e-9, max(raw))
        for i in range(self._n_bands):
            v = raw[i] / mx
            v = math.sqrt(v)  # Mild compression
            self._levels[i] = self._levels[i] * 0.6 + v * 0.4  # Smooth

        self._cleared = False

        # Draw bars
        y_bottom = self.height - self._pad
        full_h = self.height - 2 * self._pad

        for i, (bar_id, glow_id) in enumerate(zip(self._bar_ids, self._glow_ids)):
            level = max(0.0, min(1.0, self._levels[i]))
            h = full_h * level

            x0 = self._pad + i * self._slot_w
            x1 = x0 + self._slot_w - 4

            y0 = y_bottom - h

            # Update main bar
            self.canvas.coords(bar_id, x0, y0, x1, y_bottom)

            # Update glow (slightly larger)
            glow_h = h * 1.2
            glow_y0 = y_bottom - glow_h
            self.canvas.coords(glow_id, x0 - 2, glow_y0, x1 + 2, y_bottom)


# -----------------------------
# GUI (ProTracker-ish style)
# -----------------------------

def run_gui():
    closing = False
    ui_after_id = None
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
    root.title("ProTracker MOD Choral Generator (v2.0)")
    root.configure(bg="#8f8f8f")
    # Keep a stable window size (prevents width jitter from varying filename lengths)
    # but avoid cutting off the bottom on some Windows setups by starting taller.
    try:
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        w = min(1040, max(980, sw - 80))
        h = min(860, max(760, sh - 140))
        root.geometry(f"{w}x{h}")
        root.minsize(min(1040, w), min(720, h))
    except Exception:
        try:
            root.geometry("1040x840")
            root.minsize(1040, 760)
        except Exception:
            pass

    # allow the UI to expand when the window is resized
    try:
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
    except Exception:
        pass

    # Style (best-effort ProTracker vibe)
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    base_font = ("Courier New", 10, "bold")

    style.configure("PT.TButton", font=base_font, padding=(8, 3))
    # Map disabled state to ensure visual feedback
    style.map("PT.TButton",
              foreground=[('disabled', '#666666'), ('active', '#000000')],
              background=[('disabled', '#999999'), ('active', '#b0b0b0')])
    style.configure("PT.TLabel", font=base_font, background="#8f8f8f", foreground="#1a1a1a")
    style.configure("PT.TFrame", background="#8f8f8f")
    style.configure("PT.TCheckbutton", font=base_font, background="#8f8f8f")
    style.configure("PT.TCombobox", font=base_font)

    # layout frames
    main = ttk.Frame(root, style="PT.TFrame", padding=10)
    main.grid(row=0, column=0, sticky="nsew")

    try:
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)
    except Exception:
        pass

    left = tk.Frame(main, bg="#8f8f8f", bd=2, relief="ridge")
    left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

    right = tk.Frame(main, bg="#8f8f8f", bd=2, relief="ridge")
    right.grid(row=0, column=1, sticky="nsew")

    try:
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
    except Exception:
        pass

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)

    # --- i18n + tooltips (EN/DE/FR) ---
    LANG_CHOICES = ["English", "Deutsch", "Français"]
    LANG_CODE = {"English": "en", "Deutsch": "de", "Français": "fr"}

    UI_STR = {
        "en": {
            "LANGUAGE": "LANGUAGE",
            "PATTERN ORDER": "PATTERN ORDER",
            "SMART": "SMART",
            "BASE MELODY": "BASE MELODY",
            "MELODY DERIVATION": "MELODY DERIVATION",
            "BASE KEY (optional)": "BASE KEY (optional)",
            "SPEED": "SPEED",
            "TEMPO": "TEMPO",
            "SCALE MODE": "SCALE MODE",
            "VARIATION": "VARIATION",
            "SEED (optional)": "SEED (optional)",
            "NEW SEED EACH GENERATE": "NEW SEED EACH GENERATE",
            "RND": "RND",
            "BATCH": "BATCH",
            "MUTE CH": "MUTE CH",
            "STEREO %": "STEREO %",
            "Enable slowdown to the end of the song": "Enable slowdown to the end of the song",
            "Export rendered songs as WAV": "Export rendered songs as WAV",
            "Save song parameters": "Save song parameters",
            "Disable vibrato in samples": "Disable vibrato in samples",
            "Add empty fade-out pattern": "Add empty fade-out pattern",
            "PASSES": "PASSES",
            "INSTRUMENTS (CH1..CH4)": "INSTRUMENTS (CH1..CH4)",
            
            "OCTAVE SPAN": "OCTAVE SPAN",
            "CH1": "CH1",
            "CH2": "CH2",
            "CH3": "CH3",
            "CH4": "CH4",
            "HARMONY": "Harmony Score",
            "SAMPLES": "SAMPLES",
            "Sample Manager": "Sample Manager",
            "Import WAV": "Import WAV",
            "Play Sample": "Play",
            "Replace Sample": "Replace",
            "Reset Sample": "Reset",
            "Generated": "Generated",
            "Custom": "Custom",
            "Volume": "Volume",
            "Sample": "Sample","GENERATE": "GENERATE",
            "PLAY": "PLAY",
            "STOP": "STOP",
            "OPEN OUTPUT": "OPEN OUTPUT",
            "OPEN PLUGINS": "OPEN PLUGINS",
            "REFRESH": "REFRESH",
            "ADD AS PLUGIN": "ADD AS PLUGIN",
            "SPECTRUM ANALYZER": "SPECTRUM ANALYZER",
            "STEREO SCOPES": "STEREO SCOPES",
            "LIGHT ORGAN": "LIGHT ORGAN",
            "RE-GENERATE": "RE-GENERATE",
            "Click visualizer to toggle Spectrum / Scopes": "Click visualizer to toggle Spectrum / Scopes",
            "PATTERN PREVIEW": "PATTERN PREVIEW",
            "Generate a song, then hit PLAY.": "Generate a song, then hit PLAY.",
        },
        "de": {
            "LANGUAGE": "SPRACHE",
            "PATTERN ORDER": "PATTERN-REIHENFOLGE",
            "SMART": "SMART",
            "BASE MELODY": "BASISMELODIE",
            "MELODY DERIVATION": "MELODIE-ABLEITUNG",
            "BASE KEY (optional)": "BASISTONART (optional)",
            "SPEED": "GESCHWINDIGKEIT",
            "TEMPO": "TEMPO",
            "SCALE MODE": "TONART/MODUS",
            "VARIATION": "VARIATION",
            "SEED (optional)": "SEED (optional)",
            "NEW SEED EACH GENERATE": "NEUER SEED PRO GENERIERUNG",
            "RND": "ZUFALL",
            "BATCH": "STAPEL",
            "MUTE CH": "MUTE KANÄLE",
            "STEREO %": "STEREO %",
            "Enable slowdown to the end of the song": "Verlangsamung bis zum Songende",
            "Export rendered songs as WAV": "Gerenderte Songs als WAV exportieren",
            "Save song parameters": "Song-Parameter speichern",
            "Disable vibrato in samples": "Vibrato in Samples deaktivieren",
            "Add empty fade-out pattern": "Leeren Fade-Out-Pattern hinzufügen",
            "PASSES": "DURCHLÄUFE",
            "INSTRUMENTS (CH1..CH4)": "INSTRUMENTE (CH1..CH4)",
            
            "OCTAVE SPAN": "OKTAVEN",
            "CH1": "CH1",
            "CH2": "CH2",
            "CH3": "CH3",
            "CH4": "CH4",
            "HARMONY": "Harmonie-Score",
            "SAMPLES": "SAMPLES",
            "Sample Manager": "Sample Manager",
            "Import WAV": "WAV importieren",
            "Play Sample": "Abspielen",
            "Replace Sample": "Ersetzen",
            "Reset Sample": "Zurücksetzen",
            "Generated": "Generiert",
            "Custom": "Benutzerdefiniert",
            "Volume": "Lautstärke",
            "Sample": "Sample","GENERATE": "GENERIEREN",
            "PLAY": "ABSPIELEN",
            "STOP": "STOP",
            "OPEN OUTPUT": "AUSGABE ÖFFNEN",
            "OPEN PLUGINS": "PLUGINS ÖFFNEN",
            "REFRESH": "AKTUALISIEREN",
            "ADD AS PLUGIN": "ALS PLUGIN HINZUFÜGEN",
            "SPECTRUM ANALYZER": "SPEKTRUM-ANALYSATOR",
            "STEREO SCOPES": "STEREO-OSZILLOSCOPE",
            "LIGHT ORGAN": "LICHTORGEL",
            "RE-GENERATE": "NEU-GENERIEREN",
            "Click visualizer to toggle Spectrum / Scopes": "Visualizer klicken: Spektrum / Scopes",
            "PATTERN PREVIEW": "PATTERN-VORSCHAU",
            "Generate a song, then hit PLAY.": "Song generieren, dann PLAY drücken.",
        },
        "fr": {
            "LANGUAGE": "LANGUE",
            "PATTERN ORDER": "ORDRE DES PATTERNS",
            "SMART": "SMART",
            "BASE MELODY": "MÉLODIE DE BASE",
            "MELODY DERIVATION": "DÉRIVATION",
            "BASE KEY (optional)": "TONALITÉ (optionnel)",
            "SPEED": "VITESSE",
            "TEMPO": "TEMPO",
            "SCALE MODE": "MODE GAMME",
            "VARIATION": "VARIATION",
            "SEED (optional)": "GRAINE (optionnel)",
            "NEW SEED EACH GENERATE": "NOUVELLE GRAINE À CHAQUE GÉNÉRATION",
            "RND": "ALÉA",
            "BATCH": "LOT",
            "MUTE CH": "MUET CH",
            "STEREO %": "STÉRÉO %",
            "Enable slowdown to the end of the song": "Ralentir jusqu'à la fin",
            "Export rendered songs as WAV": "Exporter en WAV",
            "Save song parameters": "Sauver les paramètres",
            "Disable vibrato in samples": "Désactiver le vibrato",
            "Add empty fade-out pattern": "Ajouter un pattern de fade-out vide",
            "PASSES": "PASSES",
            "INSTRUMENTS (CH1..CH4)": "INSTRUMENTS (CH1..CH4)",
            
            "OCTAVE SPAN": "OCTAVES",
            "CH1": "CH1",
            "CH2": "CH2",
            "CH3": "CH3",
            "CH4": "CH4",
            "HARMONY": "Score d'harmonie",
            "SAMPLES": "SAMPLES",
            "Sample Manager": "Gestion des échantillons",
            "Import WAV": "Importer WAV",
            "Play Sample": "Lecture",
            "Replace Sample": "Remplacer",
            "Reset Sample": "Réinitialiser",
            "Generated": "Généré",
            "Custom": "Personnalisé",
            "Volume": "Volume",
            "Sample": "Échantillon","GENERATE": "GÉNÉRER",
            "PLAY": "JOUER",
            "STOP": "STOP",
            "OPEN OUTPUT": "OUVRIR SORTIE",
            "OPEN PLUGINS": "OUVRIR PLUGINS",
            "REFRESH": "RAFRAÎCHIR",
            "ADD AS PLUGIN": "AJOUTER COMME PLUGIN",
            "SPECTRUM ANALYZER": "ANALYSEUR DE SPECTRE",
            "STEREO SCOPES": "OSCILLOSCOPE STÉRÉO",
            "LIGHT ORGAN": "ORGUE LUMINEUX",
            "RE-GENERATE": "RE-GÉNÉRER",
            "Click visualizer to toggle Spectrum / Scopes": "Cliquer: Spectre / Oscillos",
            "PATTERN PREVIEW": "APERÇU PATTERN",
            "Generate a song, then hit PLAY.": "Générez un morceau, puis PLAY.",
        },
    }

    TT_STR = {
        "en": {
            "LANGUAGE": "Select the UI language (labels + tooltips).",
            "Click visualizer to toggle Spectrum / Scopes": "Click the visualizer to switch between Spectrum, Channel Scopes, and Light Organ.",
            "PATTERN ORDER": "Pattern playback order. You can type or pick a preset.",
            "SMART": "Generate a musically sensible order automatically.",
            "BASE MELODY": "Choose a base melody plugin. 'Pure Random' uses algorithmic melody.",
            "MELODY DERIVATION": "How strongly the base melody is transformed: Near / Far / Random.",
            "BASE KEY (optional)": "Optional song key root (e.g. C-2, F#-2, Bb-2). Leave empty for random.",
            "SPEED": "MOD speed (ticks per row). Typical: 6.",
            "TEMPO": "MOD tempo (BPM). Typical: 125.",
            "SCALE MODE": "Auto uses plugin 'mode' meta. Or force Major/Minor/Mixed/etc.",
            "VARIATION": "Overall variation amount: higher = more drive/ornaments.",
            "SEED (optional)": "Seed for reproducible generation. Same seed = same song.",
            "NEW SEED EACH GENERATE": "When enabled, each GENERATE uses a fresh random seed (uncheck to reuse the seed).",
            "BATCH": "Generate multiple songs in one run (seeds count up).",
            "MUTE CH": "Mute channels in preview rendering.",
            "STEREO %": "Stereo width for preview rendering.",
            "Enable slowdown to the end of the song": "Apply a slowdown effect only at the very end.",
            "Export rendered songs as WAV": "Save preview rendering as .wav next to the .mod.",
            "Save song parameters": "Save song parameters as .txt next to the .mod.",
            "Disable vibrato in samples": "Disable vibrato in synthesized instruments (more stable pitch).",
            "Add empty fade-out pattern": "Adds an empty pattern at the end so instruments can fade out naturally instead of stopping abruptly.",
            "PASSES": "Number of quality check passes (more passes = better harmony).",
            "OCTAVE SPAN": "Per channel: how many octaves (around the base key octave) notes may use. 1=only base octave, 3=base±1.",
            "CH1": "Channel 1 instrument and octave span.",
            "CH2": "Channel 2 instrument and octave span.",
            "CH3": "Channel 3 instrument and octave span.",
            "CH4": "Channel 4 instrument and octave span.",
            "HARMONY": "Harmonic quality score based on music theory analysis.",
            "SAMPLES": "Manage and replace instrument samples.",
            "Sample Manager": "View and manage instrument samples. Import custom WAV files to replace generated samples.",
            "Import WAV": "Import a custom WAV file to replace the selected sample.",
            "Play Sample": "Preview the selected sample.",
            "Replace Sample": "Replace generated sample with custom WAV.",
            "Reset Sample": "Reset to generated sample.",
            "Generated": "Procedurally generated sample.",
            "Custom": "User-imported custom sample.",
            "Volume": "Sample volume level.","GENERATE": "Generate a new .mod using the current settings.",
            "PLAY": "Render preview audio and play it.",
            "STOP": "Stop playback.",
            "OPEN OUTPUT": "Open the output folder (mods_out).",
            "OPEN PLUGINS": "Open the melody plugin folder (melody_plugins).",
            "REFRESH": "Reload melody plugins from disk.",
            "ADD AS PLUGIN": "Export the last generated song as a new melody plugin folder.",
            "PATTERN PREVIEW": "Preview patterns as tracker-like text. Select pattern number.",
            "RE-GENERATE": "Regenerate the current song with a new pattern order.",
        },
        "de": {
            "LANGUAGE": "GUI-Sprache auswählen (Labels + Tooltips).",
            "Click visualizer to toggle Spectrum / Scopes": "Visualizer klicken: zwischen Spektrum, Kanal-Scopes und Lichtorgel umschalten.",
            "PATTERN ORDER": "Pattern-Abspielreihenfolge. Du kannst tippen oder ein Preset wählen.",
            "SMART": "Erzeugt automatisch eine musikalisch sinnvolle Reihenfolge.",
            "BASE MELODY": "Basismelodie wählen. 'Pure Random' erzeugt die Melodie algorithmisch.",
            "MELODY DERIVATION": "Wie stark die Basismelodie verändert wird: Near / Far / Random.",
            "BASE KEY (optional)": "Optionale Tonart (z.B. C-2, F#-2, Bb-2). Leer = Zufall.",
            "SPEED": "MOD-Speed (Ticks pro Zeile). Typisch: 6.",
            "TEMPO": "MOD-Tempo (BPM). Typisch: 125.",
            "SCALE MODE": "Auto nutzt Plugin-Meta 'mode'. Oder Major/Minor/Mixed/etc erzwingen.",
            "VARIATION": "Variationsstärke: höher = mehr Drive/Ornamente.",
            "SEED (optional)": "Seed für reproduzierbare Generierung. Gleicher Seed = gleicher Song.",
            "NEW SEED EACH GENERATE": "Wenn aktiv, nutzt jede Generierung einen neuen Zufalls-Seed (deaktivieren = Seed wiederverwenden).",
            "BATCH": "Mehrere Songs in einem Lauf generieren (Seeds count up).",
            "MUTE CH": "Kanäle im Preview stummschalten.",
            "STEREO %": "Stereo-Breite für den Preview-Render.",
            "Enable slowdown to the end of the song": "Verlangsamung nur ganz am Ende anwenden.",
            "Export rendered songs as WAV": "Preview als .wav neben der .mod speichern.",
            "Save song parameters": "Song-Parameter als .txt neben der .mod speichern.",
            "Disable vibrato in samples": "Vibrato in Synth-Instrumenten deaktivieren (stabilere Tonhöhe).",
            "Add empty fade-out pattern": "Fügt einen leeren Pattern am Ende hinzu, damit Instrumente natürlich ausklingen können.",
            "PASSES": "Anzahl der Qualitätsprüf-Durchläufe (mehr = bessere Harmonie).",
            "OCTAVE SPAN": "Pro Kanal: über wie viele Oktaven (um die Basis-Oktave) Noten verteilt sein dürfen. 1=nur Basis-Oktave, 3=Basis±1.",
            "CH1": "Kanal 1 Instrument und Oktav-Spanne.",
            "CH2": "Kanal 2 Instrument und Oktav-Spanne.",
            "CH3": "Kanal 3 Instrument und Oktav-Spanne.",
            "CH4": "Kanal 4 Instrument und Oktav-Spanne.",
            "HARMONY": "Harmonische Qualitätsbewertung basierend auf Musiktheorie-Analyse.",
            "SAMPLES": "Instrument-Samples verwalten und ersetzen.",
            "Sample Manager": "Samples anzeigen und verwalten. Eigene WAV-Dateien importieren.",
            "Import WAV": "Eigene WAV-Datei importieren.",
            "Play Sample": "Vorschau des ausgewählten Samples.",
            "Replace Sample": "Generiertes Sample durch eigenes WAV ersetzen.",
            "Reset Sample": "Zurück zum generierten Sample.",
            "Generated": "Prozedural generiertes Sample.",
            "Custom": "Benutzerdefiniertes Sample.",
            "Volume": "Sample-Lautstärke.","GENERATE": "Erzeugt eine neue .mod Datei mit den aktuellen Einstellungen.",
            "PLAY": "Preview rendern und abspielen.",
            "STOP": "Playback stoppen.",
            "OPEN OUTPUT": "Ausgabeordner öffnen (mods_out).",
            "OPEN PLUGINS": "Plugin-Ordner öffnen (melody_plugins).",
            "REFRESH": "Plugins neu einlesen.",
            "ADD AS PLUGIN": "Export the last generated song as a new melody plugin folder.",
            "PATTERN PREVIEW": "Pattern-Vorschau (Tracker-Text). Pattern auswählen.",
            "RE-GENERATE": "Aktuellen Song mit neuer Pattern-Reihenfolge neu generieren.",
            "LIGHT ORGAN": "Song als Lichtorgel visualisieren.",
        },
        "fr": {
            "LANGUAGE": "Choisir la langue (libellés + info-bulles).",
            "Click visualizer to toggle Spectrum / Scopes": "Cliquer pour basculer Spectre / Oscilloscope / Orgue lumineux.",
            "PATTERN ORDER": "Ordre de lecture des patterns. Saisir ou choisir un preset.",
            "SMART": "Génère automatiquement un ordre musical cohérent.",
            "BASE MELODY": "Choisir une mélodie plugin. 'Pure Random' = mélodie algorithmique.",
            "MELODY DERIVATION": "Transformation: Near / Far / Random.",
            "BASE KEY (optional)": "Tonalité (ex: C-2, F#-2, Bb-2). Vide = aléatoire.",
            "SPEED": "Vitesse MOD (ticks/ligne). Typique: 6.",
            "TEMPO": "Tempo MOD (BPM). Typique: 125.",
            "SCALE MODE": "Auto utilise le meta 'mode'. Ou forcer Major/Minor/Mixed/etc.",
            "VARIATION": "Variation: plus élevé = plus d'ornements/drive.",
            "SEED (optional)": "Graine reproductible. Même graine = même morceau.",
            "NEW SEED EACH GENERATE": "Si activé, chaque génération utilise une nouvelle graine aléatoire (désactiver = réutiliser la graine).",
            "BATCH": "Générer plusieurs morceaux (seed incrémenté).",
            "MUTE CH": "Couper des canaux dans l'aperçu.",
            "STEREO %": "Largeur stéréo de l'aperçu.",
            "Enable slowdown to the end of the song": "Ralentir uniquement à la toute fin.",
            "Export rendered songs as WAV": "Enregistrer l'aperçu en .wav à côté du .mod.",
            "Save song parameters": "Enregistrer les paramètres en .txt à côté du .mod.",
            "Disable vibrato in samples": "Désactiver le vibrato (hauteur plus stable).",
            "Add empty fade-out pattern": "Ajoute un pattern vide à la fin pour laisser les instruments s'éteindre naturellement.",
            "PASSES": "Nombre de passes de vérification qualité (plus = meilleure harmonie).",
            "OCTAVE SPAN": "Par canal : nombre d’octaves autorisées (autour de l’octave de base). 1=octave de base, 3=base±1.",
            "CH1": "Canal 1 instrument et étendue d'octaves.",
            "CH2": "Canal 2 instrument et étendue d'octaves.",
            "CH3": "Canal 3 instrument et étendue d'octaves.",
            "CH4": "Canal 4 instrument et étendue d'octaves.",
            "HARMONY": "Score de qualité harmonique basé sur l'analyse musicale.",
            "SAMPLES": "Gérer et remplacer les échantillons.",
            "Sample Manager": "Gérer les échantillons. Importer des fichiers WAV personnalisés.",
            "Import WAV": "Importer un fichier WAV personnalisé.",
            "Play Sample": "Aperçu de l'échantillon sélectionné.",
            "Replace Sample": "Remplacer par un WAV personnalisé.",
            "Reset Sample": "Réinitialiser l'échantillon.",
            "Generated": "Échantillon généré.",
            "Custom": "Échantillon personnalisé.",
            "Volume": "Niveau de volume.","GENERATE": "Générer un nouveau .mod avec les réglages actuels.",
            "PLAY": "Rendre l'aperçu audio et le lire.",
            "STOP": "Arrêter la lecture.",
            "OPEN OUTPUT": "Ouvrir le dossier de sortie (mods_out).",
            "OPEN PLUGINS": "Ouvrir le dossier plugins (melody_plugins).",
            "REFRESH": "Recharger les plugins.",
            "ADD AS PLUGIN": "Exporter le dernier morceau comme plugin.",
            "PATTERN PREVIEW": "Aperçu des patterns (texte tracker).",
            "RE-GENERATE": "Régénérer le morceau actuel avec un nouvel ordre de patterns.",
        },
    }

    _cur_lang = {"code": "en"}
    lang_var = tk.StringVar(value="English")
    _i18n_bind: list[tuple[object, str]] = []  # (widget, key)

    def tr(key: str) -> str:
        code = _cur_lang["code"]
        return UI_STR.get(code, UI_STR["en"]).get(key, UI_STR["en"].get(key, key))

    def tt(key: str) -> str:
        code = _cur_lang["code"]
        return TT_STR.get(code, TT_STR["en"]).get(key, TT_STR["en"].get(key, ""))

    class TooltipManager:
        def __init__(self, root_: tk.Tk, delay_ms: int = 1000):
            self.root = root_
            self.delay_ms = delay_ms
            self._after = None
            self._tip = None
            self._key = None

        def bind(self, widget, key: str):
            widget.bind("<Enter>", lambda e, w=widget, k=key: self._schedule(w, k))
            widget.bind("<Leave>", lambda e: self._hide())
            widget.bind("<ButtonPress>", lambda e: self._hide())

        def _schedule(self, widget, key: str):
            self._hide()
            self._key = key
            self._after = widget.after(self.delay_ms, lambda: self._show(widget))

        def _show(self, widget):
            try:
                text_ = tt(self._key or "")
                if not text_:
                    return
                x = widget.winfo_rootx() + 10
                y = widget.winfo_rooty() + widget.winfo_height() + 8
                self._tip = tk.Toplevel(widget)
                self._tip.wm_overrideredirect(True)
                self._tip.wm_geometry(f"+{x}+{y}")
                lbl = tk.Label(self._tip, text=text_, justify="left", wraplength=340,
                               bg="#ffffe0", fg="#000000", bd=1, relief="solid",
                               font=("Segoe UI", 9))
                lbl.pack(ipadx=6, ipady=4)
            except Exception:
                pass

        def _hide(self):
            try:
                if self._after is not None:
                    try:
                        self.root.after_cancel(self._after)
                    except Exception:
                        pass
            finally:
                self._after = None
            try:
                if self._tip is not None:
                    self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    tips = TooltipManager(root, delay_ms=1000)

    def _bind_i18n(widget, key: str):
        _i18n_bind.append((widget, key))

    def apply_language():
        # Update all registered widgets' displayed text.
        for w, key in list(_i18n_bind):
            try:
                w.configure(text=tr(key))
            except Exception:
                pass

    # --- left controls ---
    def pt_label(parent, key: str):
        w = ttk.Label(parent, text=tr(key), style="PT.TLabel")
        _bind_i18n(w, key)
        return w

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



    def _add_last_as_plugin():
        """Create a new melody plugin folder from the most recently generated song.

        Strategy: we copy the saved song-parameter .txt (which includes a "MELODY PLUGIN EXPORT"
        block), so users can also manually edit it later.
        """
        nonlocal last_song, last_mod_path
        if last_song is None:
            try:
                log("No song yet - generate one first.")
            except Exception:
                pass
            return

        try:
            plugin_root = _PLUGIN_ROOT
        except Exception:
            plugin_root = _default_plugin_root()
        try:
            plugin_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Prefer the real parameter file if it exists.
        src_txt: Path | None = None
        if last_mod_path is not None:
            p = last_mod_path.with_suffix('.txt')
            if p.exists():
                src_txt = p

        # If it doesn't exist (checkbox disabled), create a minimal plugin txt on the fly.
        if src_txt is None:
            try:
                base_mod = last_mod_path or (Path('mods_out') / 'generated.mod')
                tmp = Path('mods_out') / (base_mod.stem + '_plugin.txt')
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(plugin_export_text_from_song(base_mod, last_song), encoding='utf-8')
                src_txt = tmp
            except Exception as e:
                try:
                    log(f"Add as plugin failed: {e}")
                except Exception:
                    pass
                return

        # Create new folder
        base_name = (getattr(last_song, 'title_txt', '') or (last_mod_path.stem if last_mod_path else 'generated')).strip() or 'generated'
        slug = _slugify(base_name)
        dest_dir = plugin_root / slug
        i = 2
        while dest_dir.exists():
            dest_dir = plugin_root / f"{slug}_{i}"
            i += 1
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            shutil.copyfile(str(src_txt), str(dest_dir / 'melody.txt'))
        except Exception:
            # fallback: plain copy
            try:
                (dest_dir / 'melody.txt').write_text(src_txt.read_text(encoding='utf-8', errors='ignore'), encoding='utf-8')
            except Exception as e:
                try:
                    log(f"Add as plugin failed: {e}")
                except Exception:
                    pass
                return

        # Create/overwrite info.txt only if missing
        try:
            info_p = dest_dir / 'info.txt'
            if not info_p.exists():
                info_p.write_text(_default_plugin_info_text(base_name), encoding='utf-8')
        except Exception:
            pass

        try:
            log(f"Added melody plugin: {dest_dir.name}")
        except Exception:
            pass

        try:
            _refresh_plugins()
        except Exception:
            pass

    # Pattern order + quick "smart order" generator
    pt_label(left, "PATTERN ORDER").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))

    def _smart_order_generate() -> str:
        # Use seed if provided for reproducibility.
        try:
            s = seed_var.get().strip()
            base_seed = int(s) if s else random_seed_value()
        except Exception:
            base_seed = random_seed_value()
        rr = random.Random(base_seed ^ 0xA5A5)
        return ", ".join(str(x) for x in generate_smart_order(rr, n_patterns=PATTERN_COUNT))

    smart_btn = ttk.Button(left, text=tr("SMART"), style="PT.TButton", command=lambda: order_var.set(_smart_order_generate()))
    _bind_i18n(smart_btn, "SMART")
    tips.bind(smart_btn, "SMART")
    smart_btn.grid(row=0, column=1, sticky="e", padx=8, pady=(8, 2))

    order_var = tk.StringVar(value=DEFAULT_ORDER_STR)
    order_combo = ttk.Combobox(left, textvariable=order_var, values=ORDER_PRESETS, width=32, style="PT.TCombobox", state="normal")
    order_combo.grid(row=1, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))
    tips.bind(order_combo, "PATTERN ORDER")

    pt_label(left, "BASE MELODY").grid(row=2, column=0, columnspan=2, sticky="w", padx=8)
    melody_var = tk.StringVar(value="Pure Random")
    melody_combo = ttk.Combobox(left, textvariable=melody_var, values=MELODY_CHOICES, width=32, style="PT.TCombobox", state="readonly")
    melody_combo.grid(row=3, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))
    tips.bind(melody_combo, "BASE MELODY")

    pt_label(left, "MELODY DERIVATION").grid(row=4, column=0, columnspan=2, sticky="w", padx=8)
    derive_var = tk.StringVar(value="Random")
    derive_combo = ttk.Combobox(left, textvariable=derive_var, values=["Random", "Near", "Far"], width=32, style="PT.TCombobox", state="readonly")
    derive_combo.grid(row=5, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))
    tips.bind(derive_combo, "MELODY DERIVATION")

    pt_label(left, "BASE KEY (optional)").grid(row=6, column=0, sticky="w", padx=8)
    key_var = tk.StringVar(value="C-2")
    key_row = tk.Frame(left, bg="#8f8f8f")
    key_row.grid(row=6, column=1, sticky="e", padx=8, pady=2)
    key_entry = tk.Entry(key_row, textvariable=key_var, width=8, font=base_font, bg="#9b9b9b", fg="#000000", relief="sunken")
    key_entry.pack(side="left")
    tips.bind(key_entry, "BASE KEY (optional)")
    rnd_key_btn = ttk.Button(key_row, text=tr("RND"), style="PT.TButton", command=lambda: key_var.set(random_key_root()))
    _bind_i18n(rnd_key_btn, "RND")
    rnd_key_btn.pack(side="left", padx=(6, 0))
    tips.bind(rnd_key_btn, "BASE KEY (optional)")
# Advanced options (kept in a compact panel to avoid clutter)
    adv = tk.Frame(left, bg="#8f8f8f", bd=2, relief="ridge")
    adv.grid(row=9, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 10))
    adv.columnconfigure(1, weight=1)

    pt_label(adv, "SCALE MODE").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
    scale_mode_var = tk.StringVar(value="Major")
    modsig_var = tk.StringVar(value=DEFAULT_MOD_SIGNATURE)
    compat_var = tk.BooleanVar(value=True)
    scale_combo = ttk.Combobox(adv, textvariable=scale_mode_var, values=SCALE_MODE_CHOICES, width=14, style="PT.TCombobox", state="readonly")
    scale_combo.grid(row=0, column=1, sticky="e", padx=6, pady=(6, 2))
    tips.bind(scale_combo, "SCALE MODE")

    pt_label(adv, "VARIATION").grid(row=1, column=0, sticky="w", padx=6, pady=(2, 2))
    variation_var = tk.IntVar(value=65)
    variation_scale = tk.Scale(adv, from_=0, to=100, orient="horizontal", variable=variation_var, length=180, bg="#8f8f8f", highlightthickness=0)
    variation_scale.grid(row=1, column=1, sticky="e", padx=6, pady=(2, 2))
    tips.bind(variation_scale, "VARIATION")

    pt_label(adv, "SEED (optional)").grid(row=2, column=0, sticky="w", padx=6, pady=(2, 2))
    seed_var = tk.StringVar(value="")
    seed_row = tk.Frame(adv, bg="#8f8f8f")
    seed_row.grid(row=2, column=1, sticky="e", padx=6, pady=(2, 2))
    seed_entry = tk.Entry(seed_row, textvariable=seed_var, width=16, font=base_font, bg="#9b9b9b", fg="#000000", relief="sunken")
    seed_entry.pack(side="left")
    tips.bind(seed_entry, "SEED (optional)")
    rnd_seed_btn = ttk.Button(seed_row, text=tr("RND"), style="PT.TButton", command=lambda: seed_var.set(str(random_seed_value())))
    _bind_i18n(rnd_seed_btn, "RND")
    rnd_seed_btn.pack(side="left", padx=(6, 0))
    tips.bind(rnd_seed_btn, "SEED (optional)")

    auto_seed_var = tk.BooleanVar(value=True)
    auto_seed_cb = ttk.Checkbutton(adv, text=tr("NEW SEED EACH GENERATE"), variable=auto_seed_var, style="PT.TCheckbutton")
    _bind_i18n(auto_seed_cb, "NEW SEED EACH GENERATE")
    auto_seed_cb.grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))
    tips.bind(auto_seed_cb, "NEW SEED EACH GENERATE")

    pt_label(adv, "BATCH").grid(row=4, column=0, sticky="w", padx=6, pady=(2, 6))
    batch_var = tk.IntVar(value=1)
    batch_spin = tk.Spinbox(adv, from_=1, to=50, textvariable=batch_var, width=6, font=base_font, bg="#9b9b9b", fg="#000000", relief="sunken")
    batch_spin.grid(row=4, column=1, sticky="e", padx=6, pady=(2, 6))
    tips.bind(batch_spin, "BATCH")

    pt_label(adv, "MUTE CH").grid(row=5, column=0, sticky="w", padx=6, pady=(2, 6))
    mute_vars = [tk.BooleanVar(value=False) for _ in range(4)]
    mute_row = tk.Frame(adv, bg="#8f8f8f")
    mute_row.grid(row=5, column=1, sticky="e", padx=6, pady=(2, 6))
    for i in range(4):
        cb = ttk.Checkbutton(mute_row, text=f"{i+1}", variable=mute_vars[i], style="PT.TCheckbutton")
        cb.pack(side="left")
        tips.bind(cb, "MUTE CH")

    pt_label(adv, "STEREO %").grid(row=6, column=0, sticky="w", padx=6, pady=(0, 6))
    width_var = tk.IntVar(value=100)
    width_scale = tk.Scale(adv, from_=0, to=200, orient="horizontal", variable=width_var, length=180, bg="#8f8f8f", highlightthickness=0)
    width_scale.grid(row=6, column=1, sticky="e", padx=6, pady=(0, 6))
    tips.bind(width_scale, "STEREO %")

    pt_label(adv, "PASSES").grid(row=7, column=0, sticky="w", padx=6, pady=(2, 6))
    passes_var = tk.IntVar(value=3)
    passes_combo = ttk.Combobox(adv, textvariable=passes_var, values=["1", "2", "3", "4", "5"], width=6, style="PT.TCombobox", state="readonly")
    passes_combo.grid(row=7, column=1, sticky="e", padx=6, pady=(2, 6))
    tips.bind(passes_combo, "Quality check passes (more = better harmony)")

    slowdown_var = tk.BooleanVar(value=False)

    export_wav_var = tk.BooleanVar(value=True)
    save_params_var = tk.BooleanVar(value=True)
    vibrato_var = tk.BooleanVar(value=False)
    fadeout_var = tk.BooleanVar(value=True)

    pt_label(left, "SPEED").grid(row=7, column=0, sticky="w", padx=8)
    speed_var = tk.StringVar(value=str(DEFAULT_SPEED))
    speed_entry = tk.Entry(left, textvariable=speed_var, width=6, font=base_font, bg="#9b9b9b", fg="#000000", relief="sunken")
    speed_entry.grid(row=7, column=1, sticky="e", padx=8, pady=2)
    tips.bind(speed_entry, "SPEED")

    pt_label(left, "TEMPO").grid(row=8, column=0, sticky="w", padx=8)
    tempo_var = tk.StringVar(value=str(DEFAULT_TEMPO))
    # small hint label
    try:
        oct_hint = pt_label(left, "OCTAVE SPAN")
        oct_hint.grid(row=11, column=0, sticky="w", padx=160)
        tips.bind(oct_hint, "OCTAVE SPAN")
        _bind_i18n(oct_hint, "OCTAVE SPAN")
    except Exception:
        pass

    def _randomize_instruments():
        # curated palettes that tend to blend well
        palettes = [
            ["Piano", "Piano", "Piano", "Piano"],
            ["Piano", "Strings", "Choir Aah", "Organ"],
            ["Electric Piano", "Synth Pad", "Strings", "Choir Ooh"],
            ["Organ", "Choir Aah", "French Horn", "Bassoon"],
            ["Harp", "Strings", "Choir Ooh", "Flute"],
            ["Synth Pad", "Synth Lead", "Square Lead", "Bassoon"],
            ["Clarinet", "Sax", "French Horn", "Tuba"],
        ]
        try:
            # deterministic with seed if provided
            st = seed_var.get().strip()
            rr = random.Random(int(st) if st else int(time.time() * 1000))
        except Exception:
            rr = random.Random(int(time.time() * 1000))
        pal = rr.choice(palettes)
        for i in range(4):
            try:
                inst_vars[i].set(pal[i])
            except Exception:
                pass

    rnd_inst_btn = ttk.Button(left, text=tr("RND"), style="PT.TButton", command=_randomize_instruments)
    _bind_i18n(rnd_inst_btn, "RND")
    rnd_inst_btn.grid(row=11, column=1, sticky="e", padx=8)
    tips.bind(rnd_inst_btn, "INSTRUMENTS (CH1..CH4)")

    inst_vars = [tk.StringVar(value=DEFAULT_INSTRUMENTS[i]) for i in range(4)]
    oct_vars = [tk.StringVar(value="3"), tk.StringVar(value="3"), tk.StringVar(value="2"), tk.StringVar(value="3")]  # CH3 default = 2

    def add_inst_row(r: int, label: str, var: tk.StringVar, octv: tk.StringVar):
        pt_label(left, label).grid(row=r, column=0, sticky="w", padx=8, pady=2)
        rowf = tk.Frame(left, bg=PT_BG)
        rowf.grid(row=r, column=1, sticky="e", padx=8, pady=2)
        cb = ttk.Combobox(rowf, textvariable=var, values=INSTRUMENT_CHOICES, width=16, style="PT.TCombobox", state="readonly")
        cb.pack(side="left")
        oc = ttk.Combobox(rowf, textvariable=octv, values=["1","2","3"], width=3, style="PT.TCombobox", state="readonly")
        oc.pack(side="left", padx=(6,0))
        tips.bind(cb, "INSTRUMENTS (CH1..CH4)")
        tips.bind(oc, "OCTAVE SPAN")

    add_inst_row(12, "CH1", inst_vars[0], oct_vars[0])
    add_inst_row(13, "CH2", inst_vars[1], oct_vars[1])
    add_inst_row(14, "CH3", inst_vars[2], oct_vars[2])
    add_inst_row(15, "CH4", inst_vars[3], oct_vars[3])

    # Language selector (affects labels + tooltips)
    pt_label(left, "LANGUAGE").grid(row=16, column=0, sticky="w", padx=8, pady=(10, 2))
    lang_combo = ttk.Combobox(left, textvariable=lang_var, values=LANG_CHOICES, width=14, style="PT.TCombobox", state="readonly")
    lang_combo.grid(row=16, column=1, sticky="e", padx=8, pady=(10, 2))
    tips.bind(lang_combo, "LANGUAGE")

    def _on_lang_change(_evt=None):
        _cur_lang["code"] = LANG_CODE.get(lang_var.get(), "en")
        apply_language()
        # Update a few non-ttk labels/vars (best effort)
        try:
            hint_lbl.configure(text=tr("Click visualizer to toggle Spectrum / Scopes"))
        except Exception:
            pass
        try:
            patt_title.configure(text=tr("PATTERN PREVIEW"))
        except Exception:
            pass
        try:
            if viz_mode == "spectrum":
                viz_title_var.set(tr("SPECTRUM ANALYZER"))
            else:
                viz_title_var.set(tr("STEREO SCOPES"))
        except Exception:
            pass

    lang_combo.bind("<<ComboboxSelected>>", _on_lang_change)

    # Keep the left panel compact; song details are written to the log on the right.

    # buttons
    btn_frame = tk.Frame(left, bg="#8f8f8f")
    btn_frame.grid(row=17, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 10))

    gen_btn = ttk.Button(btn_frame, text=tr("GENERATE"), style="PT.TButton")
    regen_btn = ttk.Button(btn_frame, text=tr("RE-GENERATE"), style="PT.TButton")
    # Use tk.Button instead of ttk.Button for play_btn to avoid style issues
    play_btn = tk.Button(btn_frame, text=tr("PLAY"), font=base_font, bg="#9b9b9b", fg="#000000",
                         activebackground="#b0b0b0", activeforeground="#000000",
                         relief="raised", bd=2, state="disabled", disabledforeground="#666666")
    stop_btn = ttk.Button(btn_frame, text=tr("STOP"), style="PT.TButton")
    _bind_i18n(gen_btn, "GENERATE")
    _bind_i18n(regen_btn, "RE-GENERATE")
    _bind_i18n(play_btn, "PLAY")
    _bind_i18n(stop_btn, "STOP")
    tips.bind(gen_btn, "GENERATE")
    tips.bind(regen_btn, "RE-GENERATE")
    tips.bind(play_btn, "PLAY")
    tips.bind(stop_btn, "STOP")

    gen_btn.grid(row=0, column=0, sticky="we", padx=(0, 6))
    regen_btn.grid(row=0, column=1, sticky="we", padx=(0, 6))
    play_btn.grid(row=0, column=2, sticky="we", padx=(0, 6))
    stop_btn.grid(row=0, column=3, sticky="we")

    open_out_btn = ttk.Button(btn_frame, text=tr("OPEN OUTPUT"), style="PT.TButton", command=_open_output_folder)
    open_plg_btn = ttk.Button(btn_frame, text=tr("OPEN PLUGINS"), style="PT.TButton", command=_open_plugin_folder)
    refresh_plg_btn = ttk.Button(btn_frame, text=tr("REFRESH"), style="PT.TButton", command=_refresh_plugins)
    _bind_i18n(open_out_btn, "OPEN OUTPUT")
    _bind_i18n(open_plg_btn, "OPEN PLUGINS")
    _bind_i18n(refresh_plg_btn, "REFRESH")
    tips.bind(open_out_btn, "OPEN OUTPUT")
    tips.bind(open_plg_btn, "OPEN PLUGINS")
    tips.bind(refresh_plg_btn, "REFRESH")

    open_out_btn.grid(row=1, column=0, sticky="we", padx=(0, 6), pady=(6, 0))
    open_plg_btn.grid(row=1, column=1, sticky="we", padx=(0, 6), pady=(6, 0))
    refresh_plg_btn.grid(row=1, column=2, columnspan=2, sticky="we", pady=(6, 0))

    add_plg_btn = ttk.Button(btn_frame, text=tr("ADD AS PLUGIN"), style="PT.TButton", command=_add_last_as_plugin)
    _bind_i18n(add_plg_btn, "ADD AS PLUGIN")
    tips.bind(add_plg_btn, "ADD AS PLUGIN")
    add_plg_btn.grid(row=2, column=0, columnspan=4, sticky="we", pady=(6, 0))

    # initial states - tk.Button uses config, ttk uses configure
    _dummy = None
    try:
        play_btn.config(state="disabled")
        regen_btn.configure(state="disabled")
        stop_btn.configure(state="disabled")
        add_plg_btn.configure(state="disabled")
    except Exception:
        pass

    btn_frame.columnconfigure(0, weight=1)
    btn_frame.columnconfigure(1, weight=1)
    btn_frame.columnconfigure(2, weight=1)
    btn_frame.columnconfigure(3, weight=1)

    # --- Notebook with Main (Visualizer+Logs), Samples, and Options ---
    right_notebook = ttk.Notebook(right, style="PT.TNotebook")
    right_notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # Tab 1: MAIN - Visualizer + Render Status + Pattern Preview + Logs
    main_tab = tk.Frame(right_notebook, bg="#8f8f8f")
    right_notebook.add(main_tab, text=tr("MAIN"))

    # Visualizer section
    title_bar = tk.Frame(main_tab, bg="#8f8f8f")
    title_bar.pack(fill="x", pady=(10, 2))

    viz_title_var = tk.StringVar(value=tr("SPECTRUM ANALYZER"))
    viz_title_lbl = tk.Label(title_bar, textvariable=viz_title_var, bg="#8f8f8f", fg="#1a1a1a", font=("Courier New", 11, "bold"))
    viz_title_lbl.pack(anchor="w")

    hint_lbl = tk.Label(title_bar, text=tr("Click visualizer to toggle Spectrum / Scopes"), bg="#8f8f8f", fg="#2a2a2a", font=("Courier New", 12, "bold"))
    hint_lbl.pack(anchor="w")

    canvas = tk.Canvas(main_tab)
    canvas.pack(fill="x", pady=(0, 10))
    tips.bind(canvas, "Click visualizer to toggle Spectrum / Scopes")

    viz_mode = "spectrum"  # spectrum | scope | lightorgan
    viz_view = None

    def set_viz_mode(mode: str):
        nonlocal viz_mode, viz_view
        mode = (mode or "").strip().lower()
        if mode not in ("spectrum", "scope", "lightorgan"):
            mode = "spectrum"
        viz_mode = mode
        try:
            canvas.delete("all")
        except Exception:
            pass
        if viz_mode == "spectrum":
            viz_title_var.set(tr("SPECTRUM ANALYZER"))
            viz_view = SpectrumAnalyzer(canvas, bars=32, width=560, height=160, segments=22)
        elif viz_mode == "scope":
            viz_title_var.set(tr("STEREO SCOPES"))
            viz_view = OscilloscopeView(canvas, width=560, height=160)
        else:  # lightorgan
            viz_title_var.set(tr("LIGHT ORGAN"))
            viz_view = LightOrganView(canvas, width=560, height=160)
        try:
            viz_view.reset()
        except Exception:
            pass

    def _toggle_viz(_evt=None):
        # Cycle through: spectrum -> scope -> lightorgan -> spectrum
        modes = ["spectrum", "scope", "lightorgan"]
        current_idx = modes.index(viz_mode) if viz_mode in modes else 0
        next_mode = modes[(current_idx + 1) % len(modes)]
        set_viz_mode(next_mode)

    canvas.bind("<Button-1>", _toggle_viz)
    set_viz_mode("spectrum")

    # Render status and Harmony display
    status_frame = tk.Frame(main_tab, bg="#8f8f8f")
    status_frame.pack(fill="x", pady=(0, 6))
    status_frame.columnconfigure(0, weight=1)
    status_frame.columnconfigure(1, weight=1)

    render_var = tk.StringVar(value="")
    progress_lbl = tk.Label(
        status_frame,
        textvariable=render_var,
        bg="#8f8f8f",
        fg="#1a1a1a",
        font=("Courier New", 14, "bold"),
        anchor="w",
        justify="left",
    )
    progress_lbl.grid(row=0, column=0, sticky="w", pady=(0, 6))

    harmony_var = tk.StringVar(value="Harmony: --%")
    harmony_lbl = tk.Label(
        status_frame,
        textvariable=harmony_var,
        bg="#8f8f8f",
        fg="#2a6a2a",
        font=("Courier New", 10, "bold"),
        anchor="e",
    )
    harmony_lbl.grid(row=0, column=1, sticky="e", pady=(0, 6))
    tips.bind(harmony_lbl, "HARMONY")

    # Log output
    info_txt = tk.Text(main_tab, height=8, font=("Courier New", 9), bg="#9b9b9b", fg="#000000", relief="sunken", bd=2)
    info_txt.pack(fill="x", padx=5, pady=5)
    info_txt.insert("end", tr("Generate a song, then hit PLAY.") + "\n")
    info_txt.config(state="disabled")

    # Pattern preview header
    patt_header = tk.Frame(main_tab, bg="#8f8f8f")
    patt_header.pack(fill="x", pady=(10, 2))
    patt_title = tk.Label(patt_header, text=tr("PATTERN PREVIEW"), bg="#8f8f8f", fg="#1a1a1a", font=("Courier New", 11, "bold"))
    patt_title.pack(side="left")

    patt_sel_var = tk.StringVar(value="0")
    patt_combo = ttk.Combobox(patt_header, textvariable=patt_sel_var, values=["0"], width=6, style="PT.TCombobox", state="readonly")
    patt_combo.pack(side="left", padx=(12, 0))
    tips.bind(patt_combo, "PATTERN PREVIEW")

    # Pattern preview frame
    patt_frame = tk.Frame(main_tab, bg="#8f8f8f")
    patt_frame.pack(fill="both", expand=True, padx=5, pady=5)
    patt_frame.columnconfigure(0, weight=1)
    patt_frame.rowconfigure(0, weight=1)

    patt_txt = tk.Text(patt_frame, height=10, font=("Courier New", 9), bg="#9b9b9b", fg="#000000", relief="sunken", bd=2, wrap="none")
    patt_txt.grid(row=0, column=0, sticky="nsew")
    patt_y = tk.Scrollbar(patt_frame, orient="vertical", command=patt_txt.yview)
    patt_y.grid(row=0, column=1, sticky="ns")
    patt_x = tk.Scrollbar(patt_frame, orient="horizontal", command=patt_txt.xview)
    patt_x.grid(row=1, column=0, sticky="we")
    patt_txt.configure(yscrollcommand=patt_y.set, xscrollcommand=patt_x.set)

    def log(msg: str):
        info_txt.config(state="normal")
        info_txt.insert("end", msg.rstrip() + "\n")
        info_txt.see("end")
        info_txt.config(state="disabled")

    def post_log(msg: str):
        try:
            (not closing) and root.after(0, lambda: log(msg))
        except Exception:
            pass

    # Tab 2: Samples Manager
    samples_tab = tk.Frame(right_notebook, bg="#8f8f8f")
    right_notebook.add(samples_tab, text=tr("SAMPLES"))

    # Sample Manager UI
    pt_label(samples_tab, "Sample Manager").pack(anchor="w", pady=(10, 5))

    # Sample list frame
    sample_list_frame = tk.Frame(samples_tab, bg="#8f8f8f", bd=2, relief="ridge")
    sample_list_frame.pack(fill="x", padx=5, pady=5)

    # Headers
    header_frame = tk.Frame(sample_list_frame, bg="#7f7f7f")
    header_frame.pack(fill="x")
    tk.Label(header_frame, text=tr("Sample"), bg="#7f7f7f", fg="#000000", font=("Courier New", 9, "bold"), width=12).pack(side="left", padx=5)
    tk.Label(header_frame, text=tr("Instrument"), bg="#7f7f7f", fg="#000000", font=("Courier New", 9, "bold"), width=15).pack(side="left", padx=5)
    tk.Label(header_frame, text="Status", bg="#7f7f7f", fg="#000000", font=("Courier New", 9, "bold"), width=12).pack(side="left", padx=5)
    tk.Label(header_frame, text=tr("Volume"), bg="#7f7f7f", fg="#000000", font=("Courier New", 9, "bold"), width=8).pack(side="left", padx=5)

    # Sample rows (4 channels)
    sample_vars = []
    sample_status_vars = []
    sample_volume_vars = []
    sample_custom_paths = [None, None, None, None]  # Store custom WAV paths

    for ch in range(4):
        row = tk.Frame(sample_list_frame, bg="#8f8f8f")
        row.pack(fill="x", pady=2)

        # Channel label
        tk.Label(row, text=f"CH{ch+1}", bg="#8f8f8f", fg="#000000", font=("Courier New", 9, "bold"), width=12).pack(side="left", padx=5)

        # Instrument label (updates when generated)
        inst_lbl = tk.Label(row, text="--", bg="#9b9b9b", fg="#000000", font=("Courier New", 9), width=15, relief="sunken")
        inst_lbl.pack(side="left", padx=5)

        # Status (Generated/Custom)
        status_var = tk.StringVar(value=tr("Generated"))
        sample_status_vars.append(status_var)
        status_lbl = tk.Label(row, textvariable=status_var, bg="#8f8f8f", fg="#2a6a2a", font=("Courier New", 9), width=12)
        status_lbl.pack(side="left", padx=5)

        # Volume slider
        vol_var = tk.DoubleVar(value=1.0)
        sample_volume_vars.append(vol_var)
        vol_scale = tk.Scale(row, from_=0.0, to=2.0, resolution=0.1, orient="horizontal",
                            variable=vol_var, length=80, bg="#8f8f8f", highlightthickness=0)
        vol_scale.pack(side="left", padx=5)

        # Action buttons
        def _make_play_btn(channel):
            return lambda: _play_sample(channel)
        def _make_import_btn(channel):
            return lambda: _import_sample(channel)
        def _make_reset_btn(channel):
            return lambda: _reset_sample(channel)

        sample_play_btn = ttk.Button(row, text=tr("Play Sample"), style="PT.TButton", command=_make_play_btn(ch), width=8)
        sample_play_btn.pack(side="left", padx=2)
        tips.bind(sample_play_btn, "Play Sample")

        import_btn = ttk.Button(row, text=tr("Replace Sample"), style="PT.TButton", command=_make_import_btn(ch), width=10)
        import_btn.pack(side="left", padx=2)
        tips.bind(import_btn, "Replace Sample")

        reset_btn = ttk.Button(row, text=tr("Reset Sample"), style="PT.TButton", command=_make_reset_btn(ch), width=8)
        reset_btn.pack(side="left", padx=2)
        tips.bind(reset_btn, "Reset Sample")

        sample_vars.append({
            'inst_lbl': inst_lbl,
            'status_var': status_var,
            'vol_var': vol_var,
            'row': row
        })

    # Global import button
    import_all_btn = ttk.Button(samples_tab, text=tr("Import WAV"), style="PT.TButton",
                                 command=lambda: _import_all_samples())
    import_all_btn.pack(anchor="w", padx=5, pady=(10, 5))
    tips.bind(import_all_btn, "Import WAV")

    # Sample info text
    sample_info = tk.Text(samples_tab, height=8, font=("Courier New", 9), bg="#9b9b9b", fg="#000000", relief="sunken", bd=2)
    sample_info.pack(fill="both", expand=True, padx=5, pady=5)
    sample_info.insert("end", tr("Sample Manager") + "\n")
    sample_info.insert("end", "- " + tr("Generated") + ": " + tt("Generated") + "\n")
    sample_info.insert("end", "- " + tr("Custom") + ": " + tt("Custom") + "\n")
    sample_info.insert("end", "\n" + tr("Import WAV") + ": " + tt("Import WAV") + "\n")
    sample_info.config(state="disabled")

    # Tab 3: Options - Checkbox controls moved here
    options_tab = tk.Frame(right_notebook, bg="#8f8f8f")
    right_notebook.add(options_tab, text=tr("OPTIONS"))

    pt_label(options_tab, "OPTIONS").pack(anchor="w", pady=(10, 5), padx=10)

    # Export options frame
    export_frame = tk.Frame(options_tab, bg="#8f8f8f", bd=2, relief="ridge")
    export_frame.pack(fill="x", padx=10, pady=5)

    export_wav_cb = ttk.Checkbutton(export_frame, text=tr("Export rendered songs as WAV"), variable=export_wav_var, style="PT.TCheckbutton")
    _bind_i18n(export_wav_cb, "Export rendered songs as WAV")
    export_wav_cb.pack(anchor="w", padx=5, pady=2)
    tips.bind(export_wav_cb, "Export rendered songs as WAV")

    save_params_cb = ttk.Checkbutton(export_frame, text=tr("Save song parameters"), variable=save_params_var, style="PT.TCheckbutton")
    _bind_i18n(save_params_cb, "Save song parameters")
    save_params_cb.pack(anchor="w", padx=5, pady=2)
    tips.bind(save_params_cb, "Save song parameters")

    vibrato_cb = ttk.Checkbutton(export_frame, text=tr("Disable vibrato in samples"), variable=vibrato_var, style="PT.TCheckbutton")
    _bind_i18n(vibrato_cb, "Disable vibrato in samples")
    vibrato_cb.pack(anchor="w", padx=5, pady=2)
    tips.bind(vibrato_cb, "Disable vibrato in samples")

    # Fadeout checkbox moved to OPTIONS tab
    fadeout_cb = ttk.Checkbutton(export_frame, text=tr("Add empty fade-out pattern"), variable=fadeout_var, style="PT.TCheckbutton")
    _bind_i18n(fadeout_cb, "Add empty fade-out pattern")
    fadeout_cb.pack(anchor="w", padx=5, pady=2)
    tips.bind(fadeout_cb, "Adds empty pattern at end for instruments to fade out naturally")

    # Slowdown checkbox moved to OPTIONS tab
    slowdown_cb = ttk.Checkbutton(export_frame, text=tr("Enable slowdown to the end of the song"), variable=slowdown_var, style="PT.TCheckbutton")
    _bind_i18n(slowdown_cb, "Enable slowdown to the end of the song")
    slowdown_cb.pack(anchor="w", padx=5, pady=2)
    tips.bind(slowdown_cb, "Enable slowdown to the end of the song")

    # Info text for options
    options_info = tk.Text(options_tab, height=10, font=("Courier New", 9), bg="#9b9b9b", fg="#000000", relief="sunken", bd=2)
    options_info.pack(fill="both", expand=True, padx=10, pady=10)
    options_info.insert("end", "Export Options\n")
    options_info.insert("end", "- Export WAV: Automatically save rendered audio\n")
    options_info.insert("end", "- Save Params: Save song parameters to text file\n")
    options_info.insert("end", "- Disable Vibrato: Turn off vibrato in generated samples\n")
    options_info.insert("end", "- Fade-out: Add empty pattern for natural instrument decay\n")
    options_info.insert("end", "- Slowdown: Enable ending slowdown effect\n")
    options_info.config(state="disabled")

    # Initial sample display - show default instruments
    try:
        for ch, inst in enumerate(DEFAULT_INSTRUMENTS):
            if ch < len(sample_vars):
                sample_vars[ch]['inst_lbl'].configure(text=inst)
    except Exception:
        pass

    def _play_sample(channel: int):
        """Play the sample for the specified channel."""
        try:
            # Check for custom sample first
            if sample_custom_paths[channel] is not None:
                path = sample_custom_paths[channel]
                if Path(path).exists():
                    log(f"Playing CH{channel+1} custom sample: {Path(path).name}")
                    with open(path, 'rb') as f:
                        wav_data = f.read()
                    player.play(wav_data)
                    log(f"Custom sample playback started for CH{channel+1}")
                    return
                else:
                    log(f"Custom sample file not found: {path}")
                    return
            
            # Check if we have a generated song
            if last_song is not None and channel < len(last_song.instrument_kinds):
                inst_kind = last_song.instrument_kinds[channel]
                log(f"Playing CH{channel+1}: {inst_kind}")
                
                # Play the preview if available
                if preview_wav is not None:
                    player.play(preview_wav)
                    log(f"Sample playback started for CH{channel+1}")
                else:
                    log(f"No preview available - generate a song first")
            else:
                # No song generated yet
                inst_name = inst_vars[channel].get() if channel < len(inst_vars) else f"CH{channel+1}"
                log(f"CH{channel+1} ({inst_name}): Generate a song first to hear the sample")
        except Exception as e:
            log(f"Play sample error: {e}")

    def _import_sample(channel: int):
        """Import a custom WAV file for the specified channel."""
        try:
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                title=f"Import WAV for CH{channel+1}",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            if path:
                sample_custom_paths[channel] = path
                sample_status_vars[channel].set(tr("Custom"))
                log(f"CH{channel+1}: Imported custom sample: {Path(path).name}")
        except Exception as e:
            log(f"Import error: {e}")

    def _reset_sample(channel: int):
        """Reset to generated sample."""
        try:
            sample_custom_paths[channel] = None
            sample_status_vars[channel].set(tr("Generated"))
            log(f"CH{channel+1}: Reset to generated sample")
        except Exception as e:
            log(f"Reset error: {e}")

    def _import_all_samples():
        """Import multiple samples at once."""
        try:
            from tkinter import filedialog
            paths = filedialog.askopenfilenames(
                title="Import WAV files",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            if paths:
                for i, path in enumerate(paths[:4]):  # Max 4 samples
                    if i < 4:
                        sample_custom_paths[i] = path
                        sample_status_vars[i].set(tr("Custom"))
                log(f"Imported {len(paths[:4])} sample(s)")
        except Exception as e:
            log(f"Import error: {e}")

    def _update_sample_display():
        """Update sample display when a new song is generated."""
        try:
            if last_song is not None and hasattr(last_song, 'instrument_kinds'):
                for ch, inst_kind in enumerate(last_song.instrument_kinds[:4]):
                    if ch < len(sample_vars):
                        sample_vars[ch]['inst_lbl'].configure(text=inst_kind)
                        # Reset custom status on new generation
                        if sample_custom_paths[ch] is None:
                            sample_status_vars[ch].set(tr("Generated"))
        except Exception:
            pass

    # analyzer update loop
    after_id = None

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
            if closing:
                return
            if play_state == "playing":
                idx = int(max(0.0, time.perf_counter() - play_started_t) * preview_sr)
                if viz_mode == "spectrum":
                    if preview_pcm and viz_view is not None:
                        try:
                            viz_view.update_from_pcm(preview_pcm, preview_sr, idx, window=1024)
                        except Exception:
                            pass
                elif viz_mode == "scope":
                    if preview_pcm and viz_view is not None:
                        try:
                            viz_view.update_from_pcm(preview_pcm, preview_sr, idx, window=1024)
                        except Exception:
                            pass
                else:  # lightorgan
                    if preview_pcm and viz_view is not None:
                        try:
                            viz_view.update_from_pcm(preview_pcm, preview_sr, idx, window=1024)
                        except Exception:
                            pass
                after_id = (None if closing else root.after(50, analyzer_tick))
            else:
                # nothing playing -> snap back to 0
                if viz_view is not None and not getattr(viz_view, "_cleared", False):
                    try:
                        viz_view.reset()
                    except Exception:
                        pass
                after_id = (None if closing else root.after(200, analyzer_tick))
        except BaseException:
            # Never let the visualizer crash the app.
            try:
                after_id = (None if closing else root.after(200, analyzer_tick))
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
            validate_order(order_list, n_patterns=PATTERN_COUNT)

            spd = parse_int_field("Speed", speed_var.get(), 1, 31)
            bpm = parse_int_field("Tempo", tempo_var.get(), 32, 255)

            instruments = [v.get() for v in inst_vars]

            # Octave span per channel (1..3). "1" means: stay in the base key octave only.
            spans_used: list[int] = []
            for _i, _v in enumerate(oct_vars, start=1):
                try:
                    s = int((_v.get() or "3").strip())
                except Exception:
                    s = 3
                spans_used.append(max(1, min(3, s)))

            
# seed/batch
            seed_base: int | None = None
            st = ""
            try:
                st = seed_var.get().strip()
            except Exception:
                st = ""

            if auto_seed_var.get():
                seed_base = random_seed_value()
                try:
                    seed_var.set(str(seed_base))
                except Exception:
                    pass
            else:
                try:
                    seed_base = int(st) if st else None
                except Exception:
                    seed_base = None
                if seed_base is None:
                    seed_base = random_seed_value()
                    try:
                        seed_var.set(str(seed_base))
                    except Exception:
                        pass

            try:
                batch_n = int(batch_var.get())
            except Exception:
                batch_n = 1
            batch_n = max(1, min(50, batch_n))

            last_path = None
            last_song_local = None

            scale_mode = scale_mode_var.get()
            variation = max(0.0, min(1.5, float(variation_var.get()) / 100.0))
            mutes = [mv.get() for mv in mute_vars]
            stereo_width = max(0.0, min(2.0, float(width_var.get()) / 100.0))

            for i in range(batch_n):
                seed_i = int(seed_base) + i
                path, song = generate_song(
                    order=order_list,
                    seed=seed_i,
                    enable_slowdown=slowdown_var.get(),
                    speed=spd,
                    tempo=bpm,
                    instruments=instruments,
                    melody_name=melody_var.get(),
                    derive_mode=derive_var.get(),
                    disable_vibrato=vibrato_var.get(),
                    key_root_override=key_var.get(),
                    scale_mode=scale_mode,
                    variation=variation,
                    mute_channels=mutes,
                    stereo_width=stereo_width,
                    octave_spans=spans_used,
                    mod_signature=modsig_var.get() if 'modsig_var' in locals() else DEFAULT_MOD_SIGNATURE,
                    compat_mode=compat_var.get() if 'compat_var' in locals() else True,
                    fadeout_pattern=fadeout_var.get(),
                    quality_passes=passes_var.get(),
                )
                last_path = path
                last_song_local = song
                log(f"Generated: {path}")
                if batch_n > 1:
                    try:
                        log(f"  batch {i+1}/{batch_n} | seed={seed_i}")
                    except Exception:
                        pass

            last_song = last_song_local
            last_mod_path = last_path

            # invalidate preview cache
            preview_pcm = None
            preview_wav = None
            preview_frames = 0
            preview_sr = 44100
            preview_ch = None

            song = last_song
            path = last_mod_path
            if song is None or path is None:
                raise RuntimeError("Generation failed")

            # Update harmony probability display
            try:
                if hasattr(song, 'harmony_score') and song.harmony_score is not None:
                    harmony_pct = int(song.harmony_score)
                    harmony_var.set(f"Harmony: {harmony_pct}%")
                    log(f"Harmonic quality: {harmony_pct}%")
                else:
                    harmony_var.set("Harmony: N/A")
            except Exception as e:
                harmony_var.set("Harmony: --%")
                log(f"Harmony display error: {e}")

            derive_txt = getattr(song, "derive_mode", "")
            vib_txt = "OFF" if getattr(song, "vibrato_disabled", False) else "ON"
            log(f"Melody: {song.base_melody}")
            meta_disp = get_plugin_metadata_display(song.base_melody)
            if meta_disp:
                log(f"Melody meta: {meta_disp}")
            log(f"Derive: {derive_txt} | Scale: {getattr(song, 'scale_mode', '')} | Var: {getattr(song, 'variation', 0):.2f} | Vibrato: {vib_txt}")
            log(f"Instruments: {', '.join(song.instrument_kinds)}")
            try:
                log(f"Mute: {''.join(['1' if x else '0' for x in getattr(song, 'mute_channels', [False, False, False, False])])} | Stereo: {getattr(song,'stereo_width',1.0):.2f}")
            except Exception:
                pass

            try:
                _update_sample_display()
            except Exception:
                pass

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
                _set_btn_states(can_generate=True, can_play=True, can_stop=False)
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
        try:
            gen_btn.configure(state="normal" if can_generate else "disabled")
            regen_btn.configure(state="normal" if (can_generate and last_song is not None) else "disabled")
            play_btn.config(state="normal" if can_play else "disabled")
            stop_btn.configure(state="normal" if can_stop else "disabled")
            try:
                add_plg_btn.configure(state="normal" if (last_song is not None) else "disabled")
            except Exception:
                pass
        except Exception:
            pass

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
            nonlocal ui_after_id
            try:
                if not closing:
                    ui_after_id = root.after(120, _ui_tick)
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
        nonlocal closing, ui_after_id
        closing = True
        try:
            render_cancel.set()
        except BaseException:
            pass
        try:
            player.stop()
        except Exception:
            pass
        try:
            stop_analyzer()
        except Exception:
            pass
        if ui_after_id is not None:
            try:
                root.after_cancel(ui_after_id)
            except Exception:
                pass
            ui_after_id = None
        try:
            root.quit()
        except Exception:
            pass
        try:
            root.after(0, root.destroy)
        except Exception:
            try:
                root.destroy()
            except Exception:
                pass

    def on_regenerate():
        """Regenerate the same song keeping the current pattern order."""
        nonlocal last_song, last_mod_path
        if last_song is None:
            try:
                messagebox.showerror("Error", "No song generated yet. Generate a song first.")
            except Exception:
                pass
            return

        # Stop playback if playing
        if play_state == "playing":
            try:
                player.stop()
            except Exception:
                pass

        # Keep the current pattern order - do NOT modify order_var
        # Just regenerate with current settings for variation in patterns
        log(f"Re-generating with current pattern order: {order_var.get()}")

        # Now call on_generate to create the new version
        on_generate()

    gen_btn.config(command=on_generate)
    regen_btn.config(command=on_regenerate)
    play_btn.config(command=on_play)
    stop_btn.config(command=on_stop)

    root.protocol("WM_DELETE_WINDOW", on_close)

    left.columnconfigure(0, weight=1)
    left.columnconfigure(1, weight=1)

    
    # Final geometry adjustment so nothing is clipped (based on requested size).
    try:
        root.update_idletasks()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        reqw = root.winfo_reqwidth()
        reqh = root.winfo_reqheight()
        w = min(sw - 60, max(reqw + 20, 1040))
        h = min(sh - 80, max(reqh + 20, 860))
        root.geometry(f"{w}x{h}")
        root.minsize(min(w, reqw), min(h, reqh))
    except Exception:
        pass

    # Start GUI event loop
    root.mainloop()


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate churchy ProTracker .MOD files (GUI by default).")
    ap.add_argument("-nogui", action="store_true", help="Run in CLI mode (do not show GUI).")
    ap.add_argument("-modsig", type=str, default=None, help="MOD signature: M.K. or M!K! (default: M!K!).")
    ap.add_argument("-nocompat", action="store_true", help="Disable compatibility safeguards (default is ON).")
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
    ap.add_argument("-key", type=str, default=None, help="CLI: base key root (e.g. C-2, F#-2, Bb-2). Empty=Random.")
    ap.add_argument("-scale", type=str, default="Auto", choices=SCALE_MODE_CHOICES, help="CLI: scale/mode (Auto/Major/Minor/Mixed/Dorian/Mixolydian).")
    ap.add_argument("-variation", type=float, default=0.65, help="CLI: variation strength 0..1.5 (default 0.65).")
    ap.add_argument("-seed", type=int, default=None, help="CLI: seed for deterministic generation.")
    ap.add_argument("-batch", type=int, default=1, help="CLI: generate N songs (seeds will increment).")
    ap.add_argument("-mute", type=str, default="0000", help="CLI: mute channels as 4-bit string, e.g. 0101 mutes CH2+CH4.")
    ap.add_argument("-stereo", type=float, default=1.0, help="CLI: stereo width 0..2 (default 1.0).")

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

    # mutes
    m = str(args.mute or "0000").strip()
    m = (m + "0000")[:4]
    mute_channels = [(c == "1") for c in m]

    batch_n = max(1, min(50, int(args.batch or 1)))
    seed_base = int(args.seed) if args.seed is not None else int(time.time() * 1000) ^ (os.getpid() << 8)

    for i in range(batch_n):
        path, _ = generate_song(
            enable_slowdown=not args.noslowdown,
            speed=int(speed),
            tempo=int(tempo),
            instruments=instruments,
            order=order_list,
            melody_name=(args.melody if args.melody else None),
            derive_mode=args.derive,
            disable_vibrato=bool(args.novibrato),
            key_root_override=(args.key if args.key else None),
            seed=seed_base + i,
            scale_mode=args.scale,
            variation=float(args.variation),
            mute_channels=mute_channels,
            stereo_width=float(args.stereo),
        )
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
