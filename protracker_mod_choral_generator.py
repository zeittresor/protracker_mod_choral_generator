# protracker_mod_choral_generator.py
# source: github.com/zeittresor

#!/usr/bin/env python3
import math
import os
import random
import struct
import time
from pathlib import Path

# -----------------------------
# ProTracker note period table (C-1 .. B-3) for standard Amiga / ProTracker tuning
# -----------------------------
PERIODS = {
    "C-1":1712, "C#1":1616, "D-1":1524, "D#1":1440, "E-1":1356, "F-1":1280, "F#1":1208, "G-1":1140, "G#1":1076, "A-1":1016, "A#1":960, "B-1":906,
    "C-2":856,  "C#2":808,  "D-2":762,  "D#2":720,  "E-2":678,  "F-2":640,  "F#2":604,  "G-2":570,  "G#2":538,  "A-2":508,  "A#2":480, "B-2":453,
    "C-3":428,  "C#3":404,  "D-3":381,  "D#3":360,  "E-3":339,  "F-3":320,  "F#3":302,  "G-3":285,  "G#3":269,  "A-3":254,  "A#3":240, "B-3":226,
}

CHROMA = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"]
OCTAVES = [1, 2, 3]
CHROMATIC = [f"{n}{o}" for o in OCTAVES for n in CHROMA]
CHROMATIC_SET = set(CHROMATIC)

def note_shift(note: str, semitones: int) -> str:
    i = CHROMATIC.index(note)
    j = i + semitones
    j = max(0, min(len(CHROMATIC) - 1, j))
    return CHROMATIC[j]

def pack_cell(note_name=None, sample=0, effect=0, param=0) -> bytes:
    period = 0 if note_name is None else PERIODS[note_name]
    samp = sample & 0x1F
    b0 = ((samp & 0x10) << 4) | ((period >> 8) & 0x0F)
    b1 = period & 0xFF
    b2 = ((samp & 0x0F) << 4) | (effect & 0x0F)
    b3 = param & 0xFF
    return bytes([b0, b1, b2, b3])

def inst_header(name: str, sample_bytes: bytes, finetune=0, volume=48, loop_start=0, loop_len_words=1) -> bytes:
    name_b = name.encode("ascii", "ignore")[:22].ljust(22, b"\x00")
    length_words = (len(sample_bytes) // 2) & 0xFFFF
    return (
        name_b +
        struct.pack(">H", length_words) +
        bytes([finetune & 0x0F]) +
        bytes([max(0, min(64, volume))]) +
        struct.pack(">H", loop_start & 0xFFFF) +
        struct.pack(">H", loop_len_words & 0xFFFF)
    )

def make_pianoish_sample(rng: random.Random, length=32768, sr=8287) -> bytes:
    f0 = rng.choice([220.0, 246.94, 261.63, 293.66])  # slight variation in tone base
    attack = int(sr * rng.uniform(0.004, 0.008))
    decay = rng.uniform(0.9, 1.6)  # longer sustain for "andacht"
    detune = rng.uniform(1.002, 1.008)

    h2 = rng.uniform(0.35, 0.50)
    h3 = rng.uniform(0.18, 0.28)
    h4 = rng.uniform(0.10, 0.20)
    d2 = rng.uniform(0.06, 0.14)

    data = bytearray()
    for n in range(length):
        t = n / sr

        x = (math.sin(2 * math.pi * f0 * t) * 1.00 +
             math.sin(2 * math.pi * f0 * 2 * t) * h2 +
             math.sin(2 * math.pi * f0 * 3 * t) * h3 +
             math.sin(2 * math.pi * f0 * 4 * t) * h4 +
             math.sin(2 * math.pi * (f0 * detune) * t) * d2)

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

def major_scale(root_note: str) -> list[str]:
    # root_note is e.g. "C-2" or "G-2"
    intervals = [0, 2, 4, 5, 7, 9, 11]
    return [note_shift(root_note, i) for i in intervals]

def triad_from_degree(scale: list[str], degree: int, octave_bias: int = 0) -> tuple[str, str, str]:
    # degree: 0..6 (I..VII)
    r = scale[degree % 7]
    t = scale[(degree + 2) % 7]
    f = scale[(degree + 4) % 7]
    if octave_bias != 0:
        r = note_shift(r, 12 * octave_bias)
        t = note_shift(t, 12 * octave_bias)
        f = note_shift(f, 12 * octave_bias)
    return r, t, f

def pick_progression(rng: random.Random) -> list[int]:
    # Degrees in major: I(0), ii(1), iii(2), IV(3), V(4), vi(5)
    # Churchy-safe cadences, end on I.
    start = rng.choice([0, 5])  # I or vi
    mid_pool = [1, 3, 4, 5, 2]
    prog = [start]
    for _ in range(2):
        prog.append(rng.choice(mid_pool))
    prog.append(rng.choice([4, 3]))  # set up cadence
    return prog

def build_bar_melody(rng: random.Random, scale: list[str], chord: tuple[str, str, str], base_note: str) -> list[tuple[str | None, int]]:
    # 16 rows per bar. Return list of (note_or_None, duration_rows).
    chord_tones = list(chord)
    if base_note in chord_tones:
        current = base_note
    else:
        current = rng.choice(chord_tones)

    events = []
    remaining = 16

    n_events = rng.choice([3, 4, 5])
    durs = []
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

def make_patterns(rng: random.Random):
    NUM_CH = 4
    ROWS = 64
    patterns = []

    key_root = rng.choice(["C-2", "G-2", "F-2", "D-2"])
    scale = major_scale(key_root)

    # Build 6 patterns: intro, theme, variation, Ruhephase, reprise, ending
    for _ in range(6):
        pat = [[(None, 0, 0, 0) for _ in range(NUM_CH)] for _ in range(ROWS)]
        patterns.append(pat)

    def set_cell(p, row, ch, note=None, sample=1, effect=0x00, param=0x00):
        patterns[p][row][ch] = (note, sample if note is not None else 0, effect, param)

    # Speed/tempo at start
    set_cell(0, 0, 0, None, 0, 0x0F, 0x06)  # speed 6
    set_cell(0, 0, 1, None, 0, 0x0F, 0x7D)  # tempo 125

    # Chord plans (per pattern: 4 bars)
    progs = [
        pick_progression(rng),
        [0, 3, 0, 4],                          # I IV I V
        [5, 3, 4, 0],                          # vi IV V I
        [3, 0, 4, 0],                          # "Ruhe" base: IV I V I
        [0, 2, 3, 4],                          # I iii IV V
        [5, 3, 4, 0],                          # ending cadence
    ]

    last_note = rng.choice([note_shift(key_root, 12), note_shift(key_root, 14)])  # around octave 3

    for p_idx in range(6):
        prog = progs[p_idx]
        for bar, deg in enumerate(prog):
            r0 = bar * 16

            root, third, fifth = triad_from_degree(scale, deg, octave_bias=0)
            # Voice the chord into usable registers:
            bass = note_shift(root, -12)   # down an octave
            mid  = root
            top  = fifth

            # Ensure in our supported range (C-1..B-3)
            for n in [bass, mid, top, third]:
                if n not in CHROMATIC_SET:
                    pass

            # Place chord (channels 1..3)
            set_cell(p_idx, r0, 1, top)
            set_cell(p_idx, r0, 2, bass)
            set_cell(p_idx, r0, 3, third)

            # Optional gentle re-strike mid-bar
            if p_idx not in (3,) and rng.random() < 0.55:
                set_cell(p_idx, r0 + 8, 1, top)
                set_cell(p_idx, r0 + 8, 2, bass)

            # Melody: channel 0
            if p_idx == 3:
                # Ruhephase: long holds + silence
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
                    # final bar: small cadence movement then rest
                    a = note_shift(root, 12)
                    b = note_shift(third, 12)
                    a = a if a in CHROMATIC_SET else note_shift(key_root, 12)
                    b = b if b in CHROMATIC_SET else note_shift(key_root, 12)
                    set_cell(p_idx, r0, 0, a)
                    set_cell(p_idx, r0 + 8, 0, b)
            else:
                base = note_shift(root, 12)
                base = base if base in CHROMATIC_SET else note_shift(key_root, 12)
                chord_up = (note_shift(root, 12), note_shift(third, 12), note_shift(fifth, 12))
                chord_up = tuple(n if n in CHROMATIC_SET else note_shift(key_root, 12) for n in chord_up)

                bar_events = build_bar_melody(rng, scale=[note_shift(n, 12) for n in scale], chord=chord_up, base_note=last_note)
                r = r0
                for note, dur in bar_events:
                    if note is not None and note in CHROMATIC_SET:
                        set_cell(p_idx, r, 0, note)
                        last_note = note
                    r += dur

            # Variation: arpeggio in channel 3 for pattern 2 + 4 sometimes
            if p_idx in (2, 4) and rng.random() < 0.75:
                tones = [third, root, fifth, root]
                tones = [note_shift(t, 12) if t.endswith("2") else t for t in tones]
                tones = [t if t in CHROMATIC_SET else note_shift(key_root, 12) for t in tones]
                for i in range(0, 16, 2):
                    set_cell(p_idx, r0 + i, 3, tones[(i // 2) % len(tones)])

    # Ending tempo slowdown (pattern 5)
    set_cell(5, 0, 0, None, 0, 0x0F, rng.choice([0x64, 0x5A, 0x50]))  # tempo 100/90/80

    return patterns, key_root

def patterns_to_bytes(patterns):
    blob = bytearray()
    for pat in patterns:
        for r in range(64):
            for ch in range(4):
                note, samp, eff, par = pat[r][ch]
                blob += pack_cell(note, samp, eff, par)
        if len(blob) % 1024 != 0:
            raise RuntimeError("Pattern size mismatch")
    return bytes(blob)

def generate_mod(out_dir="mods_out", seed=None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = int(time.time() * 1000) ^ (os.getpid() << 8)
    rng = random.Random(seed)

    sample = make_pianoish_sample(rng)

    patterns, key_root = make_patterns(rng)
    pat_data = patterns_to_bytes(patterns)

    order = [0, 1, 2, 3, 2, 4, 5]  # ~1 min, with Ruhephase + reprise + ending
    song_len = len(order)
    order_table = bytes(order + [0] * (128 - len(order)))

    adjectives = ["Andacht", "Choral", "Vesper", "Cantus", "Abendlied", "Sanctus", "Pax", "Segen"]
    title_txt = f"{rng.choice(adjectives)} {rng.randint(1, 9999):04d}"
    title = title_txt.encode("ascii", "ignore")[:20].ljust(20, b"\x00")

    # Instruments (31). Only instrument 1 is used.
    insts = [inst_header("Piano-ish", sample, volume=48)]
    empty = b"\x00" * 22 + struct.pack(">H", 0) + bytes([0]) + bytes([0]) + struct.pack(">H", 0) + struct.pack(">H", 1)
    insts += [empty] * 30

    mod = bytearray()
    mod += title
    for ih in insts:
        mod += ih
    mod += bytes([song_len])
    mod += bytes([0])  # restart pos
    mod += order_table
    mod += b"M.K."     # 4 channels
    mod += pat_data
    mod += sample

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{title_txt.replace(' ', '_')}_{ts}_key_{key_root.replace('-', '').replace('#','s')}.mod"
    path = out_dir / fname
    path.write_bytes(mod)
    return path

def main():
    path = generate_mod()
    print(f"Generated: {path}")

if __name__ == "__main__":
    main()
