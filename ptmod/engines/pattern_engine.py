from __future__ import annotations
from typing import Optional, List

try:
    from protracker_mod_choral_generator import _cell_to_text
except Exception:
    _cell_to_text = None

def format_pattern(pattern_rows: list, pattern_index: int = 0, order_positions: Optional[List[int]] = None) -> str:
    """
    Formats a single ProTracker pattern into a monospaced text preview.
    pattern_rows: list[row], each row: list[4] of (note, instrument, effect, param)
    """
    if order_positions is None:
        order_positions = []

    header = f"PATTERN {pattern_index:02d}"
    if order_positions:
        header += f"   (used in order positions: {', '.join(str(x) for x in order_positions)})"
    lines = [header, "-" * max(24, len(header))]

    for r, row in enumerate(pattern_rows):
        rr = f"{r:02X} | "
        parts = []
        for cell in row[:4]:
            if _cell_to_text:
                parts.append(_cell_to_text(cell))
            else:
                note, samp, eff, par = cell
                n = note if note else "---"
                s = f"{samp:02d}" if samp else "--"
                e = f"{eff:X}{par:02X}" if (eff or par) else "---"
                parts.append(f"{n} {s} {e}")
        lines.append(rr + "  ".join(parts))
    return "\n".join(lines) + "\n"

def order_positions_for_pattern(order: list[int], pattern_index: int) -> list[int]:
    return [i for i, p in enumerate(order or []) if int(p) == int(pattern_index)]
