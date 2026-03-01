from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Any, Callable
import shutil

import protracker_mod_choral_generator as backend

LogCb = Optional[Callable[[str], None]]

def _log(cb: LogCb, msg: str):
    try:
        if cb:
            cb(msg)
    except Exception:
        pass

def plugin_root() -> Path:
    try:
        return backend._PLUGIN_ROOT  # type: ignore
    except Exception:
        try:
            return backend._default_plugin_root()  # type: ignore
        except Exception:
            return Path("melody_plugins")

def reload_plugins() -> List[str]:
    """Return available melody names for UI selection.

    We expose:
      - user plugin melodies (from melody_plugins/)
      - built-in melody templates shipped in the backend
    """
    names: List[str] = []
    try:
        names.extend(list(backend.reload_melody_plugins()))
    except Exception:
        try:
            names.extend(list(getattr(backend, 'MELODY_CHOICES', [])))
        except Exception:
            pass

    try:
        builtins = list(getattr(backend, 'MELODY_LIBRARY', {}).keys())
        names.extend(builtins)
    except Exception:
        pass

    # Deduplicate, stable sort
    out = sorted(set([str(n) for n in names if str(n).strip()]), key=lambda s: s.lower())
    return out


def open_folder(path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        backend.open_folder(path)  # type: ignore
    except Exception:
        # best effort: platform open
        import os, subprocess, sys
        p = str(path)
        if sys.platform.startswith('win'):
            os.startfile(p)  # type: ignore
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', p])
        else:
            subprocess.Popen(['xdg-open', p])

def add_last_as_plugin(last_mod_path: Optional[Path], last_song: Any, log_cb: LogCb = None) -> Optional[Path]:
    """Create a melody plugin folder from the last generated song (melody.txt + info.txt)."""
    if last_song is None:
        _log(log_cb, "No song to export as plugin.")
        return None

    root = plugin_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Prefer .txt params if exists, else create one on the fly.
    src_txt: Path | None = None
    if last_mod_path is not None:
        p = Path(last_mod_path).with_suffix('.txt')
        if p.exists():
            src_txt = p

    if src_txt is None:
        try:
            base_mod = Path(last_mod_path) if last_mod_path else (Path('mods_out') / 'generated.mod')
            tmp = Path('mods_out') / (base_mod.stem + '_plugin.txt')
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(backend.plugin_export_text_from_song(base_mod, last_song), encoding='utf-8')  # type: ignore
            src_txt = tmp
        except Exception as e:
            _log(log_cb, f"Add as plugin failed: {e}")
            return None

    base_name = (getattr(last_song, 'title_txt', '') or (last_mod_path.stem if last_mod_path else 'generated')).strip() or 'generated'
    slug = backend._slugify(base_name) if hasattr(backend, '_slugify') else base_name.replace(' ','_')  # type: ignore
    dest_dir = root / slug
    i = 2
    while dest_dir.exists():
        dest_dir = root / f"{slug}_{i}"
        i += 1
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        shutil.copyfile(str(src_txt), str(dest_dir / 'melody.txt'))
    except Exception:
        try:
            (dest_dir / 'melody.txt').write_text(src_txt.read_text(encoding='utf-8', errors='ignore'), encoding='utf-8')
        except Exception as e:
            _log(log_cb, f"Add as plugin failed: {e}")
            return None

    try:
        info_p = dest_dir / 'info.txt'
        if not info_p.exists():
            info_p.write_text(backend._default_plugin_info_text(base_name), encoding='utf-8')  # type: ignore
    except Exception:
        pass

    _log(log_cb, f"Added melody plugin: {dest_dir.name}")
    return dest_dir


def get_plugin_meta(name: str) -> dict[str, str]:
    """Return metadata dict for a melody plugin (or empty dict)."""
    try:
        pl = getattr(backend, 'PLUGIN_MELODIES', {}).get(str(name))
        if pl is None:
            return {}
        meta = getattr(pl, 'meta', None)
        if isinstance(meta, dict):
            return {str(k): str(v) for k, v in meta.items()}
    except Exception:
        return {}
    return {}

def get_recommended_pattern_order(name: str) -> str | None:
    """Return recommended pattern order from plugin metadata (keys: pattern_order/order/order_hint)."""
    meta = get_plugin_meta(name)
    for k in ('pattern_order', 'order', 'order_hint'):
        if k in meta and str(meta[k]).strip():
            return str(meta[k]).strip()
    return None

