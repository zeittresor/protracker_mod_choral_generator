from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

@dataclass
class SongConfig:
    # Output / determinism
    out_dir: str = "mods_out"
    seed: Optional[int] = None
    auto_seed_each_generate: bool = True
    batch_count: int = 1

    # Song structure
    order_str: str = ""                # if empty -> backend DEFAULT_ORDER_STR
    use_smart_order: bool = False      # if True -> ignore order_str and generate smart order

    # Music params
    speed: int = 6
    tempo: int = 125
    instruments: List[str] = field(default_factory=lambda: ["Piano","Piano","Piano","Piano"])
    octave_spans: List[int] = field(default_factory=lambda: [3,3,2,3])  # CH3 default 2 like Tk UI
    melody_name: Optional[str] = None          # None means Random
    derive_mode: str = "Random"                # Random / Near / Far
    key_root_override: Optional[str] = None    # e.g. C-2
    scale_mode: str = "Auto"                   # Auto/Major/Minor/Mixed/Dorian/Mixolydian
    variation_pct: int = 65                    # 0..100 (maps to 0..1.5 internally)
    stereo_width_pct: int = 100                # 0..200 (maps to 0..2.0 internally)
    mute_channels: List[bool] = field(default_factory=lambda: [False, False, False, False])

    # Export / compat
    export_wav: bool = True
    save_params: bool = True
    enable_slowdown: bool = False
    disable_vibrato: bool = False
    fadeout_pattern: bool = True
    compat_mode: bool = True                   # tracker compatibility safeguards
    mod_signature: Optional[str] = None         # M.K. / M!K! / 4CHN etc

    # Quality
    quality_passes: int = 3

    # Ralph loop (external meta-loop)
    ralph_loop: bool = False
    ralph_target_score: float = 90.0           # target for (harmony+melody)/2
    ralph_max_attempts: int = 50
    ralph_ignore_drumsets: bool = True   # ignore drumset channels in Ralph quality scoring

    # FX injection (optional)
    fx_insert_initial_speed_tempo: bool = True
    fx_vibrato_melody: bool = False
    fx_portamento_melody: bool = False
    fx_arpeggio_ornaments: bool = False
    fx_volume_motion: bool = False
    fx_note_cut: bool = False
    fx_retrig: bool = False
    fx_intensity: int = 50  # 0..100

    # Custom samples (optional): per channel path to wav
    custom_sample_paths: Dict[int, str] = field(default_factory=dict)

    def ensure_out_dir(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
