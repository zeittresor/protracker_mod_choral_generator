# protracker_mod_choral_generator.py
# source: github.com/zeittresor

#!/usr/bin/env python3
from __future__ import annotations
import argparse
import io
import math
import os
import random
import re
import struct
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

DEFAULT_ORDER_STR = "0, 1, 2, 3, 2, 4, 1, 4, 2, 5"
ORDER_PRESETS = [
    "5, 5, 1, 5, 0, 2, 3, 4, 2, 5, 0",
    "5, 0, 1, 5, 2, 3, 1, 4, 2, 5, 0",
    "0, 1, 2, 3, 2, 4, 1, 4, 2, 5",
    "0, 1, 2, 3, 2, 4, 5",
]

# Reference fundamental for all generated samples (Hz). Tuned so that C-3 plays consistently across instruments.
REF_F0 = 261.63

INSTRUMENT_CHOICES = [
    "Piano",
    "Clarinet",
    "Sax",
    "Synth Pad",
    "Violin",
    "Tuba",
    "Banjo",
    "Panflute",
    "Acoustic Guitar",
    "Flamenco Guitar",
    "Organ",
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


def make_instrument_sample(kind: str, rng: random.Random, length: int = 32768, sr: int = 8287, f0: float = REF_F0) -> bytes:
    kind = (kind or "").strip()
    if kind not in INSTRUMENT_CHOICES:
        kind = "Piano"

    if kind == "Piano":
        return make_pianoish_sample(rng, length=length, sr=sr, f0=f0)

    detune = rng.uniform(0.9990, 1.0015)
    vib_rate = rng.uniform(4.5, 6.2)
    vib_amt = rng.uniform(0.0, 0.0030) if kind in ("Violin", "Synth Pad", "Panflute", "Flute") else rng.uniform(0.0, 0.0015)

    if kind == "Organ":
        vib_amt = 0.0

    # Envelope choices (kept conservative so pitch feels stable)
    if kind in ("Synth Pad", "Violin", "Panflute", "Clarinet", "Sax", "Flute", "Oboe", "Organ"):
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


def make_patterns(rng: random.Random, speed: int = DEFAULT_SPEED, tempo: int = DEFAULT_TEMPO):
    NUM_CH = 4
    ROWS = 64
    patterns: list[list[list[tuple[str | None, int, int, int]]]] = []

    key_root = rng.choice(["C-2", "G-2", "F-2", "D-2"])
    scale = major_scale(key_root)

    for _ in range(6):
        pat = [[(None, 0, 0, 0) for _ in range(NUM_CH)] for _ in range(ROWS)]
        patterns.append(pat)

    def set_cell(p: int, row: int, ch: int, note: str | None = None, sample: int | None = None, effect: int = 0x00, param: int = 0x00):
        # default sample per channel: 1..4
        if note is None:
            samp = 0 if sample is None else sample
        else:
            samp = (ch + 1) if sample is None else sample
        patterns[p][row][ch] = (note, samp, effect, param)

    # Set initial speed + tempo in pattern 0 (standard ProTracker trick)
    set_cell(0, 0, 0, None, 0, 0x0F, max(1, min(31, int(speed))))
    set_cell(0, 0, 1, None, 0, 0x0F, max(32, min(255, int(tempo))))

    progs = [
        pick_progression(rng),
        [0, 3, 0, 4],
        [5, 3, 4, 0],
        [3, 0, 4, 0],
        [0, 2, 3, 4],
        [5, 3, 4, 0],
    ]

    last_note = rng.choice([note_shift(key_root, 12), note_shift(key_root, 14)])

    for p_idx in range(6):
        prog = progs[p_idx]
        for bar, deg in enumerate(prog):
            r0 = bar * 16

            root, third, fifth = triad_from_degree(scale, deg, octave_bias=0)
            bass = note_shift(root, -12)
            top = fifth

            set_cell(p_idx, r0, 1, top)
            set_cell(p_idx, r0, 2, bass)
            set_cell(p_idx, r0, 3, third)

            if p_idx not in (3,) and rng.random() < 0.55:
                set_cell(p_idx, r0 + 8, 1, top)
                set_cell(p_idx, r0 + 8, 2, bass)

            if p_idx == 3:
                if bar == 0:
                    hold = rng.choice([third, fifth, note_shift(root, 12)])
                    hold = note_shift(hold, 12) if hold.endswith("2") else hold
                    hold = hold if hold in CHROMATIC_SET else note_shift(key_root, 12)
                    set_cell(p_idx, r0, 0, hold)
                elif bar == 1:
                    hold = rng.choice([fifth, third])
                    hold = note_shift(hold, 12) if hold.endswith("2") else hold
                    hold = hold if hold in CHROMATIC_SET else note_shift(key_root, 12)
                    set_cell(p_idx, r0, 0, hold)
                elif bar == 2:
                    hold = rng.choice([third, root])
                    hold = note_shift(hold, 12) if hold.endswith("2") else hold
                    hold = hold if hold in CHROMATIC_SET else note_shift(key_root, 12)
                    set_cell(p_idx, r0, 0, hold)
                else:
                    a = note_shift(root, 12)
                    b = note_shift(third, 12)
                    a = a if a in CHROMATIC_SET else note_shift(key_root, 12)
                    b = b if b in CHROMATIC_SET else note_shift(key_root, 12)
                    set_cell(p_idx, r0, 0, a)
                    set_cell(p_idx, r0 + 8, 0, b)
            else:
                chord_up = (note_shift(root, 12), note_shift(third, 12), note_shift(fifth, 12))
                chord_up = tuple(n if n in CHROMATIC_SET else note_shift(key_root, 12) for n in chord_up)

                bar_events = build_bar_melody(
                    rng,
                    scale=[note_shift(n, 12) for n in scale],
                    chord=chord_up,
                    base_note=last_note,
                )
                r = r0
                for note, dur in bar_events:
                    if note is not None and note in CHROMATIC_SET:
                        set_cell(p_idx, r, 0, note)
                        last_note = note
                    r += dur

            if p_idx in (2, 4) and rng.random() < 0.75:
                tones = [third, root, fifth, root]
                tones = [note_shift(t, 12) if t.endswith("2") else t for t in tones]
                tones = [t if t in CHROMATIC_SET else note_shift(key_root, 12) for t in tones]
                for i in range(0, 16, 2):
                    set_cell(p_idx, r0 + i, 3, tones[(i // 2) % len(tones)])

    return patterns, key_root


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


def validate_order(order: list[int], n_patterns: int = 6) -> None:
    if len(order) > 128:
        raise ValueError("Order is too long (max 128 positions).")
    bad = [x for x in order if x < 0 or x >= n_patterns]
    if bad:
        raise ValueError(f"Order contains out-of-range pattern numbers {bad}. Allowed: 0..{n_patterns-1}")


@dataclass
class SongData:
    title_txt: str
    key_root: str
    patterns: list
    order: list[int]
    samples_bytes: list[bytes]
    samples_float: list[list[float]]
    instrument_kinds: list[str]
    speed: int
    tempo: int


def generate_song(
    out_dir: str = "mods_out",
    seed: int | None = None,
    order: list[int] | None = None,
    enable_slowdown: bool = True,
    speed: int = DEFAULT_SPEED,
    tempo: int = DEFAULT_TEMPO,
    instruments: list[str] | None = None,
) -> tuple[Path, SongData]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = int(time.time() * 1000) ^ (os.getpid() << 8)
    rng = random.Random(seed)

    inst_kinds = normalize_instrument_list(instruments)

    # Generate sample bytes (4 slots). If the same instrument is selected, we still keep distinct sample numbers.
    sample_cache: dict[str, bytes] = {}
    samples_bytes: list[bytes] = []
    for k in inst_kinds:
        if k not in sample_cache:
            sample_cache[k] = make_instrument_sample(k, rng, f0=REF_F0)
        samples_bytes.append(sample_cache[k])

    samples_float = [bytes_to_float_sample(b) for b in samples_bytes]

    patterns, key_root = make_patterns(rng, speed=speed, tempo=tempo)

    if order is None:
        order = parse_order_string(DEFAULT_ORDER_STR)
    validate_order(order, n_patterns=len(patterns))

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
    section3 = ["is_at", "move_to", "will_meet", "save_the", "want_see", "went_fast", "dance_fame", "just_get", "have_meet", "move_on", "make_on_to_a", "get_on_a", "linked_by_a"]
    section4 = ["at_dancefloor__", "the_DJ__", "at_poolparty__", "at_busstation__", "to_heaven__", "ready_to_rock__", "disco__", "crazy__", "party__", "roll_around__", "fight__", "as_a_sausage__", "at_phonecall__"]
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
        key_root=key_root,
        patterns=patterns,
        order=order_for_write,
        samples_bytes=samples_bytes,
        samples_float=samples_float,
        instrument_kinds=inst_kinds,
        speed=int(speed),
        tempo=int(tempo),
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


def render_song_to_pcm16(song: SongData, out_rate: int = 44100, progress_cb=None, cancel_event: threading.Event | None = None) -> tuple[bytes, int]:
    # Very small MOD subset renderer (enough for our generator):
    # - Note on, sample select, Fxx speed/tempo.
    # - No finetune, no loops, no other effects.
    # - Fixed panning (ch1+ch4 left, ch2+ch3 right).

    # channel state
    chan_period = [0, 0, 0, 0]
    chan_samp = [0, 1, 2, 3]
    chan_pos = [0.0, 0.0, 0.0, 0.0]
    chan_vol = [48, 48, 48, 48]

    speed = int(song.speed)
    tempo = int(song.tempo)

    # Pre-calc panning
    pan_l = [1.0, 0.0, 0.0, 1.0]
    pan_r = [0.0, 1.0, 1.0, 0.0]

    mix_l = array("h")
    mix_r = array("h")

    patterns = song.patterns

    total_rows = max(1, len(song.order) * 64)
    done_rows = 0

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

                # channel 0
                if per0 > 0:
                    step = _freq_from_period(per0) / out_rate
                    samp_arr = (sp0, sp1, sp2, sp3)[sidx0]
                    i0 = int(pos0)
                    if i0 < len(samp_arr):
                        v = samp_arr[i0] * (vol0 / 64.0)
                        l += v
                    pos0 += step

                # channel 1
                if per1 > 0:
                    step = _freq_from_period(per1) / out_rate
                    samp_arr = (sp0, sp1, sp2, sp3)[sidx1]
                    i1 = int(pos1)
                    if i1 < len(samp_arr):
                        v = samp_arr[i1] * (vol1 / 64.0)
                        r += v
                    pos1 += step

                # channel 2
                if per2 > 0:
                    step = _freq_from_period(per2) / out_rate
                    samp_arr = (sp0, sp1, sp2, sp3)[sidx2]
                    i2 = int(pos2)
                    if i2 < len(samp_arr):
                        v = samp_arr[i2] * (vol2 / 64.0)
                        r += v
                    pos2 += step

                # channel 3
                if per3 > 0:
                    step = _freq_from_period(per3) / out_rate
                    samp_arr = (sp0, sp1, sp2, sp3)[sidx3]
                    i3 = int(pos3)
                    if i3 < len(samp_arr):
                        v = samp_arr[i3] * (vol3 / 64.0)
                        l += v
                    pos3 += step

                # mild master gain to avoid clipping
                l *= 0.25
                r *= 0.25

                # clamp
                l = max(-1.0, min(1.0, l))
                r = max(-1.0, min(1.0, r))

                mix_l.append(int(l * 32767))
                mix_r.append(int(r * 32767))

            chan_pos = [pos0, pos1, pos2, pos3]

            done_rows += 1
            if progress_cb is not None:
                try:
                    progress_cb(done_rows, total_rows)
                except Exception:
                    pass

    interleaved = array("h")
    for i in range(len(mix_l)):
        interleaved.append(mix_l[i])
        interleaved.append(mix_r[i])

    return interleaved.tobytes(), out_rate


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int, nch: int = 2) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return bio.getvalue()


class Player:
    def __init__(self):
        self._is_windows = sys.platform.startswith("win")
        self._play_obj = None
        self._start_t = 0.0
        self._duration_s = 0.0
        self._wav_bytes: bytes | None = None
        self._sr = 44100

    def play(self, wav_bytes: bytes, sample_rate: int, total_frames: int):
        self.stop()
        self._wav_bytes = wav_bytes
        self._sr = sample_rate
        self._duration_s = total_frames / float(sample_rate)
        self._start_t = time.perf_counter()

        # Windows: built-in winsound
        if self._is_windows:
            try:
                import winsound  # type: ignore

                winsound.PlaySound(wav_bytes, winsound.SND_ASYNC | winsound.SND_MEMORY)
                self._play_obj = "winsound"
                return
            except Exception:
                self._play_obj = None

        # Cross-platform fallback: simpleaudio if available
        try:
            import simpleaudio  # type: ignore

            wave_obj = simpleaudio.WaveObject(wav_bytes)
            self._play_obj = wave_obj.play()
            return
        except Exception:
            self._play_obj = None
            raise RuntimeError(
                "Playback backend not available. On Windows this should work via winsound; otherwise install 'simpleaudio'."
            )

    def stop(self):
        if self._play_obj is None:
            return

        if self._play_obj == "winsound":
            try:
                import winsound  # type: ignore

                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        else:
            try:
                self._play_obj.stop()
            except Exception:
                pass

        self._play_obj = None
        self._start_t = 0.0
        self._duration_s = 0.0
        self._wav_bytes = None

    def is_playing(self) -> bool:
        if self._play_obj is None:
            return False
        if self._play_obj == "winsound":
            # winsound doesn't expose state; approximate via time
            return (time.perf_counter() - self._start_t) < self._duration_s
        try:
            return self._play_obj.is_playing()
        except Exception:
            return False

    def playback_sample_index(self) -> int:
        if self._start_t <= 0.0:
            return 0
        t = max(0.0, time.perf_counter() - self._start_t)
        return int(t * self._sr)


# -----------------------------
# Spectrum analyzer
# -----------------------------

try:
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


class SpectrumAnalyzer:
    def __init__(self, canvas, bars: int = 32, width: int = 520, height: int = 110):
        self.canvas = canvas
        self.bars = bars
        self.width = width
        self.height = height
        self._ids: list[int] = []
        self._levels = [0.0] * bars

        self.canvas.configure(width=width, height=height, bg="#8f8f8f", highlightthickness=0)

        pad = 6
        slot_w = (width - 2 * pad) / bars
        for i in range(bars):
            x0 = pad + i * slot_w
            x1 = x0 + slot_w - 2
            y0 = pad
            y1 = height - pad
            # slot outline
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#6f6f6f", width=1)
            # bar fill
            rid = self.canvas.create_rectangle(x0 + 1, y1, x1 - 1, y1, outline="", fill="#28ff28")
            self._ids.append(rid)

        # precompute band edges
        self._fmin = 60.0
        self._fmax = 5200.0
        self._edges = [self._fmin * ((self._fmax / self._fmin) ** (i / bars)) for i in range(bars + 1)]

    def _compute_levels(self, mono: list[float], sr: int) -> list[float]:
        n = len(mono)
        if n < 64:
            return [0.0] * self.bars

        # window
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
        centers = [math.sqrt(self._edges[i] * self._edges[i + 1]) for i in range(self.bars)]
        levels: list[float] = []
        for f in centers:
            w = 2.0 * math.pi * f / sr
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
        # pcm16 is interleaved stereo int16
        if not pcm16:
            return

        total_frames = len(pcm16) // 4
        if total_frames <= 0:
            return

        i0 = max(0, min(total_frames - 1, sample_index))
        i1 = min(total_frames, i0 + window)
        if i1 - i0 < 64:
            return

        # extract mono window
        mono: list[float] = []
        # byte offset
        off = i0 * 4
        end = i1 * 4
        # manual unpack for speed
        for j in range(off, end, 4):
            l = int.from_bytes(pcm16[j : j + 2], byteorder="little", signed=True)
            r = int.from_bytes(pcm16[j + 2 : j + 4], byteorder="little", signed=True)
            mono.append(((l + r) * 0.5) / 32768.0)

        raw = self._compute_levels(mono, sr)

        # normalize + smooth
        mx = max(1e-9, max(raw))
        for i in range(self.bars):
            v = raw[i] / mx
            # dB-ish compression
            v = math.sqrt(v)
            self._levels[i] = self._levels[i] * 0.75 + v * 0.25

        # draw
        pad = 6
        y_bottom = self.height - pad
        slot_w = (self.width - 2 * pad) / self.bars

        for i, rid in enumerate(self._ids):
            x0 = pad + i * slot_w + 2
            x1 = x0 + slot_w - 6
            h = (self.height - 2 * pad) * self._levels[i]
            y0 = y_bottom - h
            self.canvas.coords(rid, x0, y0, x1, y_bottom)


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

    render_lock = threading.Lock()
    render_thread: threading.Thread | None = None

    root = tk.Tk()
    root.title("ProTracker MOD Choral Generator (v1.4.1)")
    root.configure(bg="#8f8f8f")

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

    pt_label(left, "PATTERN ORDER").grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 2))

    order_var = tk.StringVar(value=DEFAULT_ORDER_STR)
    order_combo = ttk.Combobox(left, textvariable=order_var, values=ORDER_PRESETS, width=32, style="PT.TCombobox", state="normal")
    order_combo.grid(row=1, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))

    pt_label(left, "SPEED").grid(row=2, column=0, sticky="w", padx=8)
    speed_var = tk.StringVar(value=str(DEFAULT_SPEED))
    speed_entry = tk.Entry(left, textvariable=speed_var, width=6, font=base_font, bg="#9b9b9b", fg="#000000", relief="sunken")
    speed_entry.grid(row=2, column=1, sticky="e", padx=8, pady=2)

    pt_label(left, "TEMPO").grid(row=3, column=0, sticky="w", padx=8)
    tempo_var = tk.StringVar(value=str(DEFAULT_TEMPO))
    tempo_entry = tk.Entry(left, textvariable=tempo_var, width=6, font=base_font, bg="#9b9b9b", fg="#000000", relief="sunken")
    tempo_entry.grid(row=3, column=1, sticky="e", padx=8, pady=2)

    slowdown_var = tk.BooleanVar(value=True)
    slowdown_cb = ttk.Checkbutton(left, text="Enable slowdown to the end of the song", variable=slowdown_var, style="PT.TCheckbutton")
    slowdown_cb.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 10))

    pt_label(left, "INSTRUMENTS (CH1..CH4)").grid(row=5, column=0, columnspan=2, sticky="w", padx=8)

    inst_vars = [tk.StringVar(value=DEFAULT_INSTRUMENTS[i]) for i in range(4)]

    def add_inst_row(r: int, label: str, var: tk.StringVar):
        pt_label(left, label).grid(row=r, column=0, sticky="w", padx=8, pady=2)
        cb = ttk.Combobox(left, textvariable=var, values=INSTRUMENT_CHOICES, width=18, style="PT.TCombobox", state="readonly")
        cb.grid(row=r, column=1, sticky="e", padx=8, pady=2)

    add_inst_row(6, "CH1", inst_vars[0])
    add_inst_row(7, "CH2", inst_vars[1])
    add_inst_row(8, "CH3", inst_vars[2])
    add_inst_row(9, "CH4", inst_vars[3])

    status_var = tk.StringVar(value="")
    status = tk.Label(left, textvariable=status_var, bg="#8f8f8f", fg="#1a1a1a", font=("Courier New", 9, "bold"), justify="left", anchor="w")
    status.grid(row=10, column=0, columnspan=2, sticky="we", padx=8, pady=(10, 8))

    # buttons
    btn_frame = tk.Frame(left, bg="#8f8f8f")
    btn_frame.grid(row=11, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 10))

    gen_btn = ttk.Button(btn_frame, text="GENERATE", style="PT.TButton")
    play_btn = ttk.Button(btn_frame, text="PLAY", style="PT.TButton")
    stop_btn = ttk.Button(btn_frame, text="STOP", style="PT.TButton")

    gen_btn.grid(row=0, column=0, sticky="we", padx=(0, 6))
    play_btn.grid(row=0, column=1, sticky="we", padx=(0, 6))
    stop_btn.grid(row=0, column=2, sticky="we")

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

    # --- right: analyzer panel ---
    title_bar = tk.Frame(right, bg="#8f8f8f")
    title_bar.pack(fill="x", padx=10, pady=(10, 4))

    tk.Label(title_bar, text="SPECTRUM ANALYZER", bg="#8f8f8f", fg="#1a1a1a", font=("Courier New", 11, "bold")).pack(anchor="w")

    canvas = tk.Canvas(right)
    canvas.pack(fill="x", padx=10, pady=(0, 10))
    analyzer = SpectrumAnalyzer(canvas, bars=32, width=560, height=120)

    info_bar = tk.Frame(right, bg="#8f8f8f")
    info_bar.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    info_txt = tk.Text(info_bar, height=7, font=("Courier New", 9), bg="#9b9b9b", fg="#000000", relief="sunken", bd=2)
    info_txt.pack(fill="both", expand=True)
    info_txt.insert("end", "Generate a song, then hit PLAY.\n")
    info_txt.config(state="disabled")

    # analyzer update loop
    after_id = None

    def log(msg: str):
        info_txt.config(state="normal")
        info_txt.insert("end", msg.rstrip() + "\n")
        info_txt.see("end")
        info_txt.config(state="disabled")

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
        if preview_pcm and player.is_playing():
            idx = player.playback_sample_index()
            analyzer.update_from_pcm(preview_pcm, preview_sr, idx, window=1024)
            after_id = root.after(50, analyzer_tick)
        else:
            after_id = root.after(200, analyzer_tick)

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
        nonlocal last_song, last_mod_path, preview_pcm, preview_wav, preview_frames, preview_sr
        try:
            order_list = parse_order_string(order_var.get())
            validate_order(order_list, n_patterns=6)

            spd = parse_int_field("Speed", speed_var.get(), 1, 31)
            bpm = parse_int_field("Tempo", tempo_var.get(), 32, 255)

            instruments = [v.get() for v in inst_vars]

            path, song = generate_song(
                order=order_list,
                enable_slowdown=slowdown_var.get(),
                speed=spd,
                tempo=bpm,
                instruments=instruments,
            )

            last_song = song
            last_mod_path = path

            # invalidate preview cache
            preview_pcm = None
            preview_wav = None
            preview_frames = 0
            preview_sr = 44100

            status_var.set(f"MOD saved:\n{path.name}\nKey: {song.key_root}  |  Samples: 1..4")
            log(f"Generated: {path}")
            log(f"Instruments: {', '.join(song.instrument_kinds)}")
            try:
                play_btn.state(["!disabled"])
                stop_btn.state(["disabled"])
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", str(e))


    # --- render/playback state ---
    state_lock = threading.Lock()
    render_cancel = threading.Event()
    render_progress = 0.0
    render_error: str | None = None
    is_rendering = False
    auto_play_after_render = False

    render_var = tk.StringVar(value="")
    render_lbl = tk.Label(left, textvariable=render_var, bg="#8f8f8f", fg="#1a1a1a",
                          font=("Courier New", 9, "bold"), justify="left", anchor="w")
    render_lbl.grid(row=12, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 10))

    def _set_btn_states(*, can_generate: bool, can_play: bool, can_stop: bool):
        gen_btn.state(["!disabled"] if can_generate else ["disabled"])
        play_btn.state(["!disabled"] if can_play else ["disabled"])
        stop_btn.state(["!disabled"] if can_stop else ["disabled"])

    def _render_preview_worker(song: SongData):
        nonlocal preview_pcm, preview_wav, preview_frames, preview_sr, render_error

        def _prog(done: int, total: int):
            nonlocal render_progress
            with state_lock:
                render_progress = 0.0 if total <= 0 else (done / float(total))

        try:
            pcm16, sr = render_song_to_pcm16(song, out_rate=44100, progress_cb=_prog, cancel_event=render_cancel)
            wavb = pcm16_to_wav_bytes(pcm16, sr, nch=2)
            with render_lock:
                preview_pcm = pcm16
                preview_wav = wavb
                preview_sr = sr
                preview_frames = len(pcm16) // 4
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
        try:
            if last_song is None:
                raise ValueError("No song generated yet.")

            # Already playing? Ignore.
            if player.is_playing():
                return

            with render_lock:
                ready = (preview_wav is not None and preview_pcm is not None and preview_frames > 0)

            if ready:
                assert preview_wav is not None
                player.play(preview_wav, preview_sr, preview_frames)
                render_var.set("PLAYING")
                log("PLAY")
                _set_btn_states(can_generate=True, can_play=False, can_stop=True)
                return

            # Not ready -> render first, then auto-play.
            log("Rendering preview...")
            _start_render(last_song, auto_play=True)

        except Exception as e:
            messagebox.showerror("Playback", str(e))

    def _ui_tick():
        nonlocal is_rendering, auto_play_after_render

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
                                sr = preview_sr
                            if wavb is not None and frames > 0:
                                player.play(wavb, sr, frames)
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

        # Playback finished?
        if (not is_rendering) and (last_song is not None) and (not player.is_playing()):
            if render_var.get() == "PLAYING":
                render_var.set("")
            _set_btn_states(can_generate=True, can_play=True, can_stop=False)

        root.after(120, _ui_tick)

    _ui_tick()

    def on_stop():
        nonlocal auto_play_after_render
        auto_play_after_render = False
        # Cancel render if running
        try:
            render_cancel.set()
        except Exception:
            pass
        try:
            player.stop()
        except Exception:
            pass
        render_var.set("")
        log("STOP")
        _set_btn_states(can_generate=True, can_play=(last_song is not None), can_stop=False)

    def on_close():
        try:
            render_cancel.set()
        except Exception:
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
    )
    print(f"Generated: {path}")


if __name__ == "__main__":
    main()
