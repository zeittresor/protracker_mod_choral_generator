# protracker_mod_choral_generator.py
# source: github.com/zeittresor

#!/usr/bin/env python3
import argparse
import math
import os
import random
import re
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

DEFAULT_ORDER_STR = "0, 1, 2, 3, 2, 4, 1, 4, 2, 5"


DEFAULT_SPEED = 6
DEFAULT_TEMPO = 125
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

def build_bar_melody(rng: random.Random, scale: list[str], chord: tuple[str, str, str], base_note: str) -> list[tuple[str | None, int]]:
    chord_tones = list(chord)
    current = base_note if base_note in chord_tones else rng.choice(chord_tones)

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

def make_patterns(rng: random.Random, enable_slowdown: bool = True, speed: int = DEFAULT_SPEED, tempo: int = DEFAULT_TEMPO):
    NUM_CH = 4
    ROWS = 64
    patterns = []

    key_root = rng.choice(["C-2", "G-2", "F-2", "D-2"])
    scale = major_scale(key_root)

    for _ in range(6):
        pat = [[(None, 0, 0, 0) for _ in range(NUM_CH)] for _ in range(ROWS)]
        patterns.append(pat)

    def set_cell(p, row, ch, note=None, sample=1, effect=0x00, param=0x00):
        patterns[p][row][ch] = (note, sample if note is not None else 0, effect, param)

    set_cell(0, 0, 0, None, 0, 0x0F, speed)
    set_cell(0, 0, 1, None, 0, 0x0F, tempo)

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
                base = note_shift(root, 12)
                base = base if base in CHROMATIC_SET else note_shift(key_root, 12)
                chord_up = (note_shift(root, 12), note_shift(third, 12), note_shift(fifth, 12))
                chord_up = tuple(n if n in CHROMATIC_SET else note_shift(key_root, 12) for n in chord_up)

                bar_events = build_bar_melody(
                    rng,
                    scale=[note_shift(n, 12) for n in scale],
                    chord=chord_up,
                    base_note=last_note
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

    if enable_slowdown:
        set_cell(5, 0, 0, None, 0, 0x0F, rng.choice([0x64, 0x5A, 0x50]))

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

def parse_order_string(order_str: str) -> list[int]:
    parts = [p.strip() for p in re.split(r"[,\s]+", order_str.strip()) if p.strip()]
    if not parts:
        raise ValueError("Order is empty.")
    order = []
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

def generate_mod(out_dir: str = "mods_out", seed: int | None = None, order: list[int] | None = None, enable_slowdown: bool = True, speed: int = DEFAULT_SPEED, tempo: int = DEFAULT_TEMPO) -> Path:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = int(time.time() * 1000) ^ (os.getpid() << 8)
    rng = random.Random(seed)

    sample = make_pianoish_sample(rng)
    patterns, key_root = make_patterns(rng, enable_slowdown=enable_slowdown, speed=speed, tempo=tempo)
    pat_data = patterns_to_bytes(patterns)

    if order is None:
        order = parse_order_string(DEFAULT_ORDER_STR)
    validate_order(order, n_patterns=len(patterns))

    song_len = len(order)
    order_table = bytes(order + [0] * (128 - len(order)))

    section1 = ["The", "A", "A_dirty", "a_holy", "Another", "The_wildest", "A_crazy", "A_funny"]
    section2 = ["banana", "DJ", "pianist", "stardestroyer", "dentist", "pope", "dictator", "dancingqueen", "jungleman", "toilet", "strawberry"]
    section3 = ["is_at", "move_to", "will_meet", "save_the", "want_see", "went_to", "dance_fame", "just_get", "have_meet", "move_on", "make_on_to", "get_on", "linked_by"]
    section4 = ["a_dancefloor__", "the_DJ__", "at_poolparty__", "a_busstation__", "in_heaven__", "ready_to_rock__", "disco__", "crazy__", "party__", "roll_around__", "fight__", "a_sausage__", "a_phonecall__"]
    title_txt = f"{rng.choice(section1)}_{rng.choice(section2)}_{rng.choice(section3)}_{rng.choice(section4)}_{rng.randint(1, 9999):04d}"
    title = title_txt.encode("ascii", "ignore")[:20].ljust(20, b"\x00")

    insts = [inst_header("Piano-ish", sample, volume=48)]
    empty = b"\x00" * 22 + struct.pack(">H", 0) + bytes([0]) + bytes([0]) + struct.pack(">H", 0) + struct.pack(">H", 1)
    insts += [empty] * 30

    mod = bytearray()
    mod += title
    for ih in insts:
        mod += ih
    mod += bytes([song_len])
    mod += bytes([0])
    mod += order_table
    mod += b"M.K."
    mod += pat_data
    mod += sample

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{title_txt.replace(' ', '_')}_{ts}_key_{key_root.replace('-', '').replace('#','s')}.mod"
    path = out_dir_p / fname
    path.write_bytes(mod)
    return path

def run_gui():
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("ProTracker MOD Choral Generator")

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill="both", expand=True)

    tk.Label(frm, text="Pattern order (comma-separated):").grid(row=0, column=0, sticky="w")

    order_var = tk.StringVar(value=DEFAULT_ORDER_STR)
    order_entry = tk.Entry(frm, textvariable=order_var, width=44)
    order_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(2, 8))

    tk.Label(frm, text="Speed (ticks/row, 1-31):").grid(row=2, column=0, sticky="w")
    speed_var = tk.StringVar(value=str(DEFAULT_SPEED))
    speed_entry = tk.Entry(frm, textvariable=speed_var, width=10)
    speed_entry.grid(row=2, column=1, sticky="e", pady=(0, 6))

    tk.Label(frm, text="Tempo (BPM, 32-255):").grid(row=3, column=0, sticky="w")
    tempo_var = tk.StringVar(value=str(DEFAULT_TEMPO))
    tempo_entry = tk.Entry(frm, textvariable=tempo_var, width=10)
    tempo_entry.grid(row=3, column=1, sticky="e", pady=(0, 10))

    slowdown_var = tk.BooleanVar(value=True)
    slowdown_cb = tk.Checkbutton(frm, text="Enable slowdown in pattern 5 (ending)", variable=slowdown_var)
    slowdown_cb.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 10))

    out_label_var = tk.StringVar(value="")
    out_label = tk.Label(frm, textvariable=out_label_var, anchor="w", justify="left")
    out_label.grid(row=6, column=0, columnspan=2, sticky="we", pady=(8, 0))

    def on_generate():
        try:
            order = parse_order_string(order_var.get())
            validate_order(order, n_patterns=6)
            speed = int(speed_var.get().strip())
            tempo = int(tempo_var.get().strip())
            if speed < 1 or speed > 31:
                raise ValueError("Speed must be in range 1..31 (ticks per row).")
            if tempo < 32 or tempo > 255:
                raise ValueError("Tempo must be in range 32..255 (BPM).")
            p = generate_mod(order=order, enable_slowdown=slowdown_var.get(), speed=speed, tempo=tempo)
            out_label_var.set(f"Generated:\n{p}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    gen_btn = tk.Button(frm, text="Generate", command=on_generate, width=18)
    gen_btn.grid(row=5, column=0, sticky="w")

    quit_btn = tk.Button(frm, text="Quit", command=root.destroy, width=18)
    quit_btn.grid(row=5, column=1, sticky="e")

    frm.columnconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    root.mainloop()

def main():
    ap = argparse.ArgumentParser(description="Generate churchy ProTracker .MOD files (GUI by default).")
    ap.add_argument("-nogui", action="store_true", help="Run in CLI mode (do not show GUI).")
    ap.add_argument("-speed", type=int, default=None, help="CLI: MOD speed (ticks/row, 1..31).")
    ap.add_argument("-tempo", type=int, default=None, help="CLI: MOD tempo (BPM, 32..255).")
    ap.add_argument("-noslowdown", action="store_true", help="Disable ending slowdown (pattern 5 tempo change) in CLI mode.")
    args = ap.parse_args()

    if not args.nogui:
        run_gui()
        return

    speed = DEFAULT_SPEED if args.speed is None else args.speed
    tempo = DEFAULT_TEMPO if args.tempo is None else args.tempo
    if speed < 1 or speed > 31:
        raise SystemExit("Error: -speed must be in range 1..31.")
    if tempo < 32 or tempo > 255:
        raise SystemExit("Error: -tempo must be in range 32..255.")
    path = generate_mod(enable_slowdown=not args.noslowdown, speed=speed, tempo=tempo)
    print(f"Generated: {path}")

if __name__ == "__main__":
    main()
