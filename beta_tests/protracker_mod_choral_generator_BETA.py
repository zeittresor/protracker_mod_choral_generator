# protracker_mod_choral_generator_BETA (current testversion)
# source: github.com/zeittresor

#!/usr/bin/env python3
import math
import os
import random
import struct
import time
from pathlib import Path
import argparse

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

def to_oct(note: str, target_oct: int) -> str:
    pc = note[:2]
    octv = int(note[2])
    delta = (target_oct - octv) * 12
    return note_shift(note, delta)

def safe(note: str) -> str:
    if note in CHROMATIC_SET:
        return note
    for _ in range(48):
        note = note_shift(note, 0)
        if note in CHROMATIC_SET:
            return note
    return "C-2"

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
    f0 = rng.choice([220.0, 246.94, 261.63, 293.66])
    attack = int(sr * rng.uniform(0.004, 0.008))
    decay = rng.uniform(0.9, 1.6)
    detune = rng.uniform(1.002, 1.008)
    h2 = rng.uniform(0.35, 0.50)
    h3 = rng.uniform(0.18, 0.28)
    h4 = rng.uniform(0.10, 0.20)
    d2 = rng.uniform(0.06, 0.14)

    data = bytearray()
    for n in range(length):
        t = n / sr
        x = (
            math.sin(2 * math.pi * f0 * t) * 1.00 +
            math.sin(2 * math.pi * f0 * 2 * t) * h2 +
            math.sin(2 * math.pi * f0 * 3 * t) * h3 +
            math.sin(2 * math.pi * f0 * 4 * t) * h4 +
            math.sin(2 * math.pi * (f0 * detune) * t) * d2
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

def major_scale(root_note: str) -> list[str]:
    intervals = [0, 2, 4, 5, 7, 9, 11]
    return [note_shift(root_note, i) for i in intervals]

def triad_from_degree(scale: list[str], degree: int) -> tuple[str, str, str]:
    r = scale[degree % 7]
    t = scale[(degree + 2) % 7]
    f = scale[(degree + 4) % 7]
    return r, t, f

def pick_progression(rng: random.Random, kind: str) -> list[int]:
    I, ii, iii, IV, V, vi = 0, 1, 2, 3, 4, 5
    if kind == "INTRO":
        return rng.choice([[I, IV, I, V], [I, vi, IV, V], [I, IV, V, I]])
    if kind == "A":
        return rng.choice([[I, vi, IV, V], [vi, IV, I, V], [I, iii, IV, V], [I, ii, V, I]])
    if kind == "B":
        return rng.choice([[I, IV, I, V], [IV, I, V, I], [I, IV, V, I]])
    if kind == "C":
        return rng.choice([[vi, IV, V, I], [I, V, vi, IV], [ii, V, I, I], [I, IV, vi, V]])
    if kind == "INTER":
        return rng.choice([[ii, V, I, V], [IV, V, vi, V]])
    if kind.startswith("RUHE"):
        return rng.choice([[IV, I, V, I], [I, IV, I, V], [vi, IV, I, V]])
    if kind == "ENDING":
        return rng.choice([[vi, IV, V, I], [IV, I, V, I], [I, IV, V, I]])
    return [I, IV, V, I]

def weighted_choice(rng: random.Random, items: list[tuple[object, float]]):
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for item, w in items:
        acc += w
        if r <= acc:
            return item
    return items[-1][0]

def diatonic_neighbors(scale: list[str], note: str) -> list[str]:
    if note not in scale:
        return [scale[0]]
    i = scale.index(note)
    res = []
    if i > 0:
        res.append(scale[i - 1])
    if i < len(scale) - 1:
        res.append(scale[i + 1])
    return res or [note]

def build_bar_melody_normal(rng: random.Random, scale3: list[str], chord3: tuple[str, str, str], last_note: str) -> tuple[list[tuple[str | None, int]], str]:
    chord_tones = list(chord3)
    if last_note not in scale3:
        last_note = rng.choice(chord_tones)

    style = weighted_choice(rng, [("flow", 0.55), ("step", 0.30), ("lift", 0.15)])

    if style == "flow":
        n_events = rng.choice([4, 5, 6])
        dur_pool = [2, 2, 4, 4, 6]
        rest_chance = 0.12
        max_leap = 4
    elif style == "step":
        n_events = rng.choice([3, 4, 5])
        dur_pool = [2, 4, 4, 8]
        rest_chance = 0.18
        max_leap = 2
    else:
        n_events = rng.choice([3, 4])
        dur_pool = [4, 4, 6, 8]
        rest_chance = 0.10
        max_leap = 7

    durations = []
    remaining = 16
    for i in range(n_events):
        if i == n_events - 1:
            durations.append(remaining)
        else:
            d = rng.choice(dur_pool)
            d = min(d, remaining - (n_events - i - 1) * 2)
            d = max(2, d)
            durations.append(d)
            remaining -= d

    events = []
    current = last_note
    for i, dur in enumerate(durations):
        if rng.random() < rest_chance and i != n_events - 1:
            events.append((None, dur))
            continue

        want_cadence = (i == n_events - 1) or (i == n_events - 2 and rng.random() < 0.45)

        if want_cadence:
            note = rng.choice(chord_tones)
        else:
            if rng.random() < 0.55:
                nbs = diatonic_neighbors(scale3, current)
                note = rng.choice(nbs)
            else:
                note = rng.choice(chord_tones)

            if abs(CHROMATIC.index(note) - CHROMATIC.index(current)) > max_leap:
                note = rng.choice(chord_tones)

        if note is not None:
            current = note
        events.append((note, dur))

    if all(n is None for n, _ in events):
        events[0] = (rng.choice(chord_tones), events[0][1])

    return events, current

def build_bar_melody_ruhez(rng: random.Random, scale3: list[str], chord3: tuple[str, str, str]) -> list[tuple[str | None, int]]:
    chord_tones = list(chord3)

    ruhe_style = weighted_choice(rng, [
        ("sustain", 0.35),
        ("two_holds", 0.25),
        ("suspension", 0.20),
        ("cadence_whisper", 0.20),
    ])

    if ruhe_style == "sustain":
        n = rng.choice(chord_tones)
        return [(n, 16)]

    if ruhe_style == "two_holds":
        n1 = rng.choice(chord_tones)
        n2 = rng.choice(chord_tones)
        if rng.random() < 0.35:
            return [(n1, 12), (None, 2), (n2, 2)]
        return [(n1, 8), (n2, 8)]

    if ruhe_style == "suspension":
        target = rng.choice(chord_tones)
        neigh = diatonic_neighbors(scale3, target)
        sus = rng.choice(neigh) if neigh else target
        if rng.random() < 0.5:
            return [(sus, 8), (target, 8)]
        return [(sus, 6), (None, 2), (target, 8)]

    # cadence_whisper
    start = rng.choice(chord_tones)
    seq = [start]
    for _ in range(2):
        seq.append(rng.choice(diatonic_neighbors(scale3, seq[-1]) or [seq[-1]]))
    seq.append(rng.choice(chord_tones))
    return [(seq[0], 4), (None, 2), (seq[1], 4), (None, 2), (seq[2], 2), (seq[3], 2)]

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

def make_song(rng: random.Random):
    key_root = rng.choice(["C-2", "G-2", "F-2", "D-2"])
    scale2 = major_scale(key_root)
    scale3 = [safe(to_oct(n, 3)) for n in scale2]

    FORM_TEMPLATES = [
        ["INTRO", "A", "B", "RUHE1", "B", "C", "ENDING"],
        ["INTRO", "A", "B", "C", "RUHE1", "A_VAR", "ENDING"],
        ["INTRO", "A", "RUHE1", "B", "C", "B", "ENDING"],
        ["INTRO", "A", "B", "RUHE1", "A_VAR", "RUHE2", "ENDING"],
        ["INTRO", "A", "INTER", "B", "RUHE1", "B", "ENDING"],
        ["INTRO", "A", "B", "C", "RUHE1", "C", "ENDING"],
    ]
    form = rng.choice(FORM_TEMPLATES)

    role_to_index = {}
    patterns = []

    def new_pattern():
        return [[(None, 0, 0, 0) for _ in range(4)] for _ in range(64)]

    def set_cell(pat, row, ch, note=None, sample=1, effect=0x00, param=0x00):
        pat[row][ch] = (note, sample if note is not None else 0, effect, param)

    def place_chord(pat, bar, deg, color="block"):
        r0 = bar * 16
        root, third, fifth = triad_from_degree(scale2, deg)
        bass = safe(to_oct(root, 1))
        mid  = safe(to_oct(root, 2))
        top  = safe(to_oct(fifth, 2))
        third2 = safe(to_oct(third, 2))

        set_cell(pat, r0, 1, top)
        set_cell(pat, r0, 2, bass)
        set_cell(pat, r0, 3, third2)

        if color in ("block", "warm") and rng.random() < 0.55:
            set_cell(pat, r0 + 8, 1, top)
            set_cell(pat, r0 + 8, 2, bass)

        return (safe(to_oct(root, 3)), safe(to_oct(third, 3)), safe(to_oct(fifth, 3)))

    def maybe_arp(pat, bar, chord3):
        if rng.random() < 0.65:
            tones = [chord3[1], chord3[0], chord3[2], chord3[0]]
            r0 = bar * 16
            for i in range(0, 16, 2):
                set_cell(pat, r0 + i, 3, tones[(i // 2) % len(tones)])

    def build_role_pattern(role: str, base_theme_from=None):
        pat = new_pattern()
        prog = pick_progression(rng, role if role != "A_VAR" else "A")

        last_note = safe(rng.choice([to_oct(key_root, 3), note_shift(to_oct(key_root, 3), 2), note_shift(to_oct(key_root, 3), 4)]))

        for bar in range(4):
            deg = prog[bar]
            color = "block"

            if role in ("C", "INTER"):
                color = "airy"
            if role.startswith("RUHE"):
                color = "warm"

            chord3 = place_chord(pat, bar, deg, color=color)

            if role in ("C", "A_VAR") and rng.random() < 0.75:
                maybe_arp(pat, bar, chord3)

            if role.startswith("RUHE"):
                events = build_bar_melody_ruhez(rng, scale3, chord3)
                r = bar * 16
                for note, dur in events:
                    if note is not None:
                        set_cell(pat, r, 0, note)
                    r += dur
                continue

            if role == "INTRO":
                intro_style = weighted_choice(rng, [("gentle", 0.7), ("answer", 0.3)])
                r0 = bar * 16
                if intro_style == "gentle":
                    n = rng.choice(chord3)
                    set_cell(pat, r0, 0, n)
                    if rng.random() < 0.45:
                        set_cell(pat, r0 + 8, 0, rng.choice(chord3))
                else:
                    set_cell(pat, r0, 0, chord3[0])
                    set_cell(pat, r0 + 8, 0, chord3[1])
                continue

            events, last_note = build_bar_melody_normal(rng, scale3, chord3, last_note)
            r = bar * 16
            for note, dur in events:
                if note is not None:
                    set_cell(pat, r, 0, note)
                r += dur

            if role in ("B", "A") and rng.random() < 0.20:
                r0 = bar * 16
                set_cell(pat, r0 + 12, 0, rng.choice(chord3))

        if role == "ENDING":
            slow = rng.choice([0x64, 0x5A, 0x50])  # 100/90/80
            set_cell(pat, 0, 0, None, 0, 0x0F, slow)

        return pat

    # Create needed roles (unique patterns per role token like RUHE1/RUHE2)
    for role in form:
        if role in role_to_index:
            continue
        role_to_index[role] = len(patterns)
        patterns.append(build_role_pattern(role))

    # Global tempo/speed + channel volumes at very start
    p0 = patterns[role_to_index[form[0]]]
    set_cell(p0, 0, 0, None, 0, 0x0F, 0x06)  # speed 6
    set_cell(p0, 0, 1, None, 0, 0x0F, 0x7D)  # tempo 125
    set_cell(p0, 0, 0, None, 0, 0x0C, 0x30)  # melody volume
    set_cell(p0, 0, 1, None, 0, 0x0C, 0x22)  # chord top
    set_cell(p0, 0, 2, None, 0, 0x0C, 0x20)  # bass
    set_cell(p0, 0, 3, None, 0, 0x0C, 0x1E)  # inner/arp

    # Build order with some extra musically sensible spice:
    base_order = [role_to_index[r] for r in form]

    if rng.random() < 0.40:
        intro_idx = role_to_index.get("INTRO")
        if intro_idx is not None:
            base_order = [intro_idx] + base_order  # double-intro

    if "A" in role_to_index and rng.random() < 0.35:
        a_idx = role_to_index["A"]
        insert_pos = min(len(base_order), rng.choice([3, 4]))
        base_order = base_order[:insert_pos] + [a_idx] + base_order[insert_pos:]

    if "ENDING" in role_to_index and rng.random() < 0.30:
        end_idx = role_to_index["ENDING"]
        base_order = base_order + [end_idx]  # lingering cadence

    # Keep within 128 entries
    if len(base_order) > 128:
        base_order = base_order[:128]

    return patterns, base_order, key_root

def generate_mod(out_dir="mods_out", seed=None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = int(time.time() * 1000) ^ (os.getpid() << 8)
    rng = random.Random(seed)

    sample = make_pianoish_sample(rng)
    patterns, order, key_root = make_song(rng)

    pat_data = patterns_to_bytes(patterns)
    song_len = len(order)
    order_table = bytes(order + [0] * (128 - len(order)))

    adjectives = ["Andacht", "Choral", "Vesper", "Cantus", "Abendlied", "Sanctus", "Pax", "Segen", "Lumen", "Credo"]
    title_txt = f"{rng.choice(adjectives)} {rng.randint(1, 9999):04d}"
    title = title_txt.encode("ascii", "ignore")[:20].ljust(20, b"\x00")

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
    mod += b"M.K."
    mod += pat_data
    mod += sample

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{title_txt.replace(' ', '_')}_{ts}_key_{key_root.replace('-', '').replace('#','s')}.mod"
    path = out_dir / fname
    path.write_bytes(mod)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mods_out", help="Output directory")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible output")
    args = ap.parse_args()

    path = generate_mod(out_dir=args.out, seed=args.seed)
    print(f"Generated: {path}")

if __name__ == "__main__":
    main()
