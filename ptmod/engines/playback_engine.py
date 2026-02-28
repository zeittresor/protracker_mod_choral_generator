from __future__ import annotations

"""Audio playback abstraction.

Design goal: *works on Windows + Linux + macOS without drama*.

Priority order:
1) QtMultimedia (QMediaPlayer)
2) OS/player subprocess fallback (paplay/aplay/ffplay/afplay)
3) Legacy Player from protracker_mod_choral_generator

The engine also exposes a playhead frame index function used by the visualizer.
"""

from typing import Optional, List, Tuple
import io
import os
import subprocess
import sys
import tempfile
import time
import wave

try:
    from PyQt6.QtCore import QUrl
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    _QT_MEDIA_OK = True
except Exception:
    QUrl = None
    QMediaPlayer = None
    QAudioOutput = None
    _QT_MEDIA_OK = False

# Legacy fallback (winsound/simpleaudio wrapper in your backend)
from protracker_mod_choral_generator import Player


class PlaybackEngine:
    def __init__(self):
        self._fallback = Player()

        self._qt_player = None
        self._qt_audio = None
        if _QT_MEDIA_OK:
            try:
                self._qt_player = QMediaPlayer()
                self._qt_audio = QAudioOutput()
                self._qt_player.setAudioOutput(self._qt_audio)
            except Exception:
                self._qt_player = None
                self._qt_audio = None

        self._tmp_path: Optional[str] = None          # Qt temp wav
        self._sys_tmp_path: Optional[str] = None      # subprocess temp wav
        self._sys_proc: Optional[subprocess.Popen] = None

        # Playback state for visualizer / UI
        self._mode: str = "idle"  # idle|qt|sys|legacy
        self._sr: int = 44100
        self._nframes: int = 0
        self._start_mono: Optional[float] = None
        self._last_frame: int = 0

    # -------------------------- internals --------------------------

    @staticmethod
    def _wav_info(wav_bytes: bytes) -> Tuple[int, int]:
        """Return (sample_rate, nframes) for a WAV byte blob."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = int(wf.getframerate() or 44100)
                nf = int(wf.getnframes() or 0)
                return sr, nf
        except Exception:
            return 44100, 0

    def _cleanup_tmp(self):
        # Qt temp
        try:
            if self._tmp_path and os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
        except Exception:
            pass
        self._tmp_path = None

        # subprocess temp
        try:
            if self._sys_tmp_path and os.path.exists(self._sys_tmp_path):
                os.unlink(self._sys_tmp_path)
        except Exception:
            pass
        self._sys_tmp_path = None

    def _stop_sys_proc(self):
        p = self._sys_proc
        self._sys_proc = None
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    p.wait(timeout=0.3)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        except Exception:
            pass

    def _candidate_system_players(self, wav_path: str) -> List[List[str]]:
        if sys.platform == "darwin":
            return [["afplay", wav_path]]
        if sys.platform.startswith("win"):
            # Windows handled by Qt + legacy fallback.
            return []
        # Linux / BSD
        return [
            ["paplay", wav_path],
            ["aplay", wav_path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
        ]

    def _try_system_player(self, wav_bytes: bytes) -> bool:
        """Try OS/player subprocess playback. Returns True if started."""
        try:
            self._stop_sys_proc()

            fd, path = tempfile.mkstemp(prefix="pt_sys_preview_", suffix=".wav")
            try:
                os.close(fd)
            except Exception:
                pass

            with open(path, "wb") as f:
                f.write(wav_bytes)

            self._sys_tmp_path = path

            for cmd in self._candidate_system_players(path):
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self._sys_proc = proc
                    return True
                except FileNotFoundError:
                    continue
                except Exception:
                    continue

        except Exception:
            pass

        self._stop_sys_proc()
        return False

    # -------------------------- public API --------------------------

    def stop(self):
        # Stop Qt first
        try:
            if self._qt_player is not None:
                self._qt_player.stop()
        except Exception:
            pass

        # Stop subprocess fallback
        self._stop_sys_proc()

        # Stop legacy fallback
        try:
            self._fallback.stop()
        except Exception:
            pass

        self._cleanup_tmp()

        # reset state
        self._mode = "idle"
        self._start_mono = None
        self._last_frame = 0
        self._nframes = 0

    def play_wav_bytes(self, wav_bytes: bytes):
        if not wav_bytes:
            return

        # Stop anything currently playing
        self.stop()

        # Track playback for visualizer (works even with subprocess fallbacks)
        self._sr, self._nframes = self._wav_info(wav_bytes)
        self._start_mono = time.monotonic()
        self._last_frame = 0

        # Prefer QtMultimedia
        if self._qt_player is not None and QUrl is not None:
            try:
                fd, path = tempfile.mkstemp(prefix="pt_qt_preview_", suffix=".wav")
                try:
                    os.close(fd)
                except Exception:
                    pass
                with open(path, "wb") as f:
                    f.write(wav_bytes)
                self._tmp_path = path

                self._qt_player.setSource(QUrl.fromLocalFile(path))
                self._qt_player.play()
                self._mode = "qt"
                return
            except Exception:
                self._cleanup_tmp()

        # System subprocess fallback (especially useful on Linux)
        if self._try_system_player(wav_bytes):
            self._mode = "sys"
            return

        # Legacy fallback
        try:
            self._fallback.play(wav_bytes)
            self._mode = "legacy"
            return
        except Exception:
            # nothing else we can do
            self._mode = "idle"
            self._start_mono = None
            self._last_frame = 0
            self._nframes = 0

    def is_playing(self) -> bool:
        # Qt
        try:
            if self._qt_player is not None and QMediaPlayer is not None:
                st = self._qt_player.playbackState()
                return bool(st == QMediaPlayer.PlaybackState.PlayingState)
        except Exception:
            pass

        # subprocess
        try:
            if self._sys_proc is not None:
                return self._sys_proc.poll() is None
        except Exception:
            pass

        # legacy
        try:
            return bool(self._fallback.is_playing())
        except Exception:
            return False

    def playhead_frame_index(self) -> int:
        """Current playback position in *stereo frames*.

        For Qt playback we query the player position (ms).
        For subprocess/legacy we approximate from monotonic start time.
        """
        sr = int(self._sr or 44100)

        # Qt: query precise position if possible
        try:
            if self._mode == "qt" and self._qt_player is not None:
                ms = int(self._qt_player.position() or 0)
                fr = int((ms / 1000.0) * sr)
                if self._nframes > 0:
                    fr = max(0, min(fr, int(self._nframes)))
                self._last_frame = fr
                return fr
        except Exception:
            pass

        # Subprocess/legacy: approximate from start time
        try:
            if self._start_mono is not None:
                fr = int((time.monotonic() - float(self._start_mono)) * sr)
                if self._nframes > 0:
                    fr = max(0, min(fr, int(self._nframes)))
                self._last_frame = fr
                return fr
        except Exception:
            pass

        return int(self._last_frame or 0)

    def playback_sample_index(self) -> int:
        """Backwards-compatible alias."""
        return self.playhead_frame_index()
