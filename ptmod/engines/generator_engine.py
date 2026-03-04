from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Tuple, Any, List
import os
import random
import time

from ptmod.config import SongConfig
from ptmod.engines.quality_engine import evaluate_patterns_for_ralph, QualityResult
from ptmod.engines.effects_engine import apply_fx_to_song
from ptmod.engines.mod_patch_engine import patch_mod_patterns_in_file, patch_mod_signature
from ptmod.engines.plugins_engine import reload_plugins
from ptmod.engines.scale_lock_engine import apply_scale_lock

# Legacy backend (proven song generator)
import protracker_mod_choral_generator as backend


StatusCb = Optional[Callable[[str], None]]
ProgressCb = Optional[Callable[[int], None]]
LogCb = Optional[Callable[[str], None]]

def _cb(cb, *args, **kwargs):
    try:
        if cb:
            cb(*args, **kwargs)
    except Exception:
        pass

def _seed_for_attempt(base_seed: int, attempt: int) -> int:
    # Spread seeds aggressively so retries are not near-duplicates
    return int(base_seed) ^ (attempt * 0x9E3779B1) ^ ((attempt + 17) * 0x85EBCA6B)

def _smart_order(seed: int) -> List[int]:
    rr = random.Random(int(seed) ^ 0xA5A5)
    return list(backend.generate_smart_order(rr, n_patterns=int(getattr(backend, 'PATTERN_COUNT', 20))))

def _parse_order(cfg: SongConfig, seed: int) -> List[int]:
    if bool(getattr(cfg, 'use_smart_order', False)):
        return _smart_order(seed)
    s = (getattr(cfg, 'order_str', '') or '').strip()
    if not s:
        s = str(getattr(backend, 'DEFAULT_ORDER_STR', '0,1,2,3,4,5'))
    try:
        return list(backend.parse_order_string(s))
    except Exception:
        # fall back to default if user typed something invalid
        return list(backend.parse_order_string(str(getattr(backend, 'DEFAULT_ORDER_STR', '0,1,2,3,4,5'))))

def _variation_float(cfg: SongConfig) -> float:
    v = float(int(getattr(cfg, 'variation_pct', 65)) / 100.0)
    if v != v:
        v = 0.65
    return max(0.0, min(1.5, v))

def _stereo_float(cfg: SongConfig) -> float:
    v = float(int(getattr(cfg, 'stereo_width_pct', 100)) / 100.0)
    if v != v:
        v = 1.0
    return max(0.0, min(2.0, v))

def generate_song_once(cfg: SongConfig,
                       status_cb: StatusCb = None,
                       progress_cb: ProgressCb = None,
                       log_cb: LogCb = None) -> Tuple[Path, Any, Optional[QualityResult]]:
    _cb(status_cb, "thinking...")
    _cb(progress_cb, 2)

    # Normalize melody selection
    mel = None if (cfg.melody_name in (None, "", "Random")) else cfg.melody_name

    # Order
    order = _parse_order(cfg, int(cfg.seed or backend.random_seed_value()))

    # Variation/stereo mapping from UI pct
    variation = _variation_float(cfg)
    stereo_width = _stereo_float(cfg)

    # Generate
    path, song = backend.generate_song(
        out_dir=cfg.out_dir,
        seed=cfg.seed,
        order=order,
        enable_slowdown=bool(cfg.enable_slowdown),
        speed=int(cfg.speed),
        tempo=int(cfg.tempo),
        instruments=list(cfg.instruments),
        melody_name=mel,
        derive_mode=str(cfg.derive_mode),
        disable_vibrato=bool(cfg.disable_vibrato),
        key_root_override=cfg.key_root_override,
        scale_mode=str(cfg.scale_mode),
        variation=float(variation),
        mute_channels=list(cfg.mute_channels),
        stereo_width=float(stereo_width),
        octave_spans=list(cfg.octave_spans),
        mod_signature=cfg.mod_signature,
        compat_mode=bool(cfg.compat_mode),
        pt2_compat_mode=bool(getattr(cfg, "pt2_compat_mode", True)),
        melody_influence_rows=int(getattr(cfg, 'melody_influence_rows', 64)),
        fadeout_pattern=bool(cfg.fadeout_pattern),
        quality_passes=int(cfg.quality_passes),
    )
    _cb(progress_cb, 55)


    # Force signature tag (magic number at offset 1080) even if backend ignores unknown tags
    try:
        if cfg.mod_signature:
            patch_mod_signature(Path(path), str(cfg.mod_signature))
    except Exception:
        pass

    # Post-process: scale-lock pitched notes (prevents accidental major/minor clashes).
    # This is conservative and only affects notes outside the selected scale.
    try:
        changed = apply_scale_lock(song, cfg, log_cb=log_cb)
        if changed:
            patch_mod_patterns_in_file(Path(path), song.patterns)
    except Exception as e:
        _cb(log_cb, f"[scale_lock] failed: {e}")

    # Quality for Ralph (optional)
    q = None
    try:
        root = getattr(song, "key_root", cfg.key_root_override or "C-2")
        scale = getattr(song, "scale_mode", cfg.scale_mode or "Auto")
        pats = song.patterns
        # Optional: ignore drumset channels in Ralph scoring (prevents drum hits from hurting harmony/melody metrics)
        try:
            if bool(getattr(cfg, 'ralph_ignore_drumsets', True)):
                drum_ch = set()
                try:
                    drum_ch = set(int(k) for k in getattr(song, 'drum_channel_styles', {}).keys())
                except Exception:
                    drum_ch = set(int(i) for i, kind in enumerate(getattr(cfg, 'instruments', []) or []) if backend.is_drumset_kind(str(kind)))
                if drum_ch:
                    # deep-ish copy patterns, blanking notes in drum channels
                    pats2 = []
                    for pat in pats:
                        pat2 = []
                        for row in pat:
                            row2 = list(row)
                            for ch in drum_ch:
                                if 0 <= int(ch) < 4:
                                    row2[int(ch)] = (None, 0, 0, 0)
                            pat2.append(row2)
                        pats2.append(pat2)
                    pats = pats2
        except Exception:
            pats = song.patterns
        q = evaluate_patterns_for_ralph(pats, str(scale), str(root))
    except Exception as e:
        _cb(log_cb, f"[quality] failed: {e}")

    # Optional FX injection (post-gen patching)
    try:
        any_fx = any([
            bool(getattr(cfg, 'fx_insert_initial_speed_tempo', True)),
            bool(getattr(cfg, 'fx_vibrato_melody', False)),
            bool(getattr(cfg, 'fx_portamento_melody', False)),
            bool(getattr(cfg, 'fx_arpeggio_ornaments', False)),
            bool(getattr(cfg, 'fx_volume_motion', False)),
            bool(getattr(cfg, 'fx_note_cut', False)),
            bool(getattr(cfg, 'fx_retrig', False)),
        ])
        if any_fx:
            fxsum = apply_fx_to_song(song, cfg)
            patch_mod_patterns_in_file(Path(path), song.patterns)
            try:
                _cb(log_cb, f"[fx] injected total={getattr(fxsum,'total',0)}  Fxx={getattr(fxsum,'initial_ftempo',0)} vib={getattr(fxsum,'vibrato',0)} porta={getattr(fxsum,'portamento',0)} arp={getattr(fxsum,'arpeggio',0)} vol={getattr(fxsum,'volume',0)} cut={getattr(fxsum,'notecut',0)} retrig={getattr(fxsum,'retrig',0)}")
            except Exception:
                pass
    except Exception as e:
        _cb(log_cb, f"[fx] failed: {e}")

    # Debug: drum channel activity (helps diagnose multi-drum setups)
    try:
        dcs = dict(getattr(song, 'drum_channel_styles', {}) or {})
        if dcs:
            for ch, st in sorted(dcs.items()):
                hits = 0
                for pat in getattr(song, 'patterns', []) or []:
                    for row in pat:
                        try:
                            n, s, e, p = row[int(ch)]
                            if n is not None and int(s) > 0:
                                hits += 1
                        except Exception:
                            pass
                _cb(log_cb, f"[drums] ch{int(ch)+1} style={st} hits={hits}")
    except Exception:
        pass
    _cb(progress_cb, 100)
    _cb(status_cb, "ready")
    return Path(path), song, q

def generate_with_ralph_loop(cfg: SongConfig,
                             status_cb: StatusCb = None,
                             progress_cb: ProgressCb = None,
                             log_cb: LogCb = None) -> Tuple[Path, Any, Optional[QualityResult]]:
    """External retry loop until harmony+melody >= target, keeping best."""
    if not cfg.ralph_loop:
        return generate_song_once(cfg, status_cb, progress_cb, log_cb)

    base_seed = int(cfg.seed) if cfg.seed is not None else int(backend.random_seed_value())

    best_path: Optional[Path] = None
    best_song: Any = None
    best_q: Optional[QualityResult] = None
    best_score = -1.0

    max_attempts = max(1, int(cfg.ralph_max_attempts or 50))
    target = float(cfg.ralph_target_score or 90.0)

    for attempt in range(max_attempts):
        attempt_seed = _seed_for_attempt(base_seed, attempt)
        cfg_local = SongConfig(**asdict(cfg))
        cfg_local.seed = attempt_seed

        _cb(status_cb, f"ralph is retrying (give him a chance) {max(0.0, min(100.0, best_score)):.1f}%")
        _cb(progress_cb, int((attempt / max_attempts) * 45) + 1)
        _cb(log_cb, f"[ralph] attempt {attempt+1}/{max_attempts}, seed={attempt_seed}")

        path, song, q = generate_song_once(cfg_local, status_cb=None, progress_cb=None, log_cb=log_cb)

        score = float(q.ralph_score) if q is not None else 0.0
        if q:
            _cb(log_cb, f"[ralph] score={score:.1f} (target={target:.1f})  harmony={q.harmony_score:.1f} melody={q.melody_score:.1f}")
        else:
            _cb(log_cb, f"[ralph] score={score:.1f} (target={target:.1f})")

        if score > best_score:
            # Remove previous best file to avoid clutter
            try:
                if best_path and best_path.exists():
                    best_path.unlink(missing_ok=True)  # py3.11
            except Exception:
                pass
            best_score = score
            best_path, best_song, best_q = Path(path), song, q
            _cb(status_cb, f"ralph is retrying (give him a chance) {best_score:.1f}%")
        else:
            # Not best -> delete this attempt's MOD
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

        if best_score >= target:
            _cb(log_cb, f"[ralph] target reached at attempt {attempt+1}: {best_score:.1f}")
            break

    _cb(progress_cb, 100)
    _cb(status_cb, "ready")
    if best_path is None:
        return generate_song_once(cfg, status_cb, progress_cb, log_cb)
    return best_path, best_song, best_q

def generate_batch(cfg: SongConfig,
                   status_cb: StatusCb = None,
                   progress_cb: ProgressCb = None,
                   log_cb: LogCb = None):
    """Generate 1..N songs sequentially (batch), yielding (mod_path, song, quality)."""
    n = max(1, int(getattr(cfg, 'batch_count', 1) or 1))

    # Determine base seed for deterministic batches
    base_seed = int(cfg.seed) if cfg.seed is not None else int(backend.random_seed_value())

    for i in range(n):
        cfg_i = SongConfig(**asdict(cfg))
        if bool(getattr(cfg, 'auto_seed_each_generate', True)):
            cfg_i.seed = int(backend.random_seed_value())
        else:
            cfg_i.seed = int(base_seed + i)

        _cb(log_cb, f"[batch] {i+1}/{n} seed={cfg_i.seed}")
        _cb(status_cb, "thinking...")
        _cb(progress_cb, int((i / n) * 100))

        mod_path, song, q = generate_with_ralph_loop(cfg_i, status_cb=status_cb, progress_cb=progress_cb, log_cb=log_cb)
        yield mod_path, song, q

    _cb(progress_cb, 100)
    _cb(status_cb, "ready")
