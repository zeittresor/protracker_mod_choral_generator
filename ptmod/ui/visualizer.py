from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QWidget

class VisualizerWidget(QWidget):
    """Spectrum / Scope / LightOrgan visualizer (no external deps)."""

    MODES = ("spectrum", "scope", "lightorgan")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "spectrum"
        self.pcm = None
        # lightorgan smoothing state
        self._lamp_state = np.zeros(4, dtype=np.float32)
        self._lamp_peak = np.zeros(4, dtype=np.float32)            # np.int16 interleaved stereo
        self.sr = 44100
        self.playhead_cb = None    # callable -> frame index
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)
        self.setMinimumHeight(160)
        self.setAutoFillBackground(True)

    def set_audio(self, pcm16_bytes: bytes | None, sr: int = 44100):
        if not pcm16_bytes:
            self.pcm = None
            self.sr = int(sr)
            self.update()
            return
        arr = np.frombuffer(pcm16_bytes, dtype=np.int16)
        self.pcm = arr
        self.sr = int(sr)
        self.update()

    def set_playhead_callback(self, cb):
        self.playhead_cb = cb

    def mousePressEvent(self, ev):
        # cycle modes
        try:
            idx = self.MODES.index(self.mode)
        except ValueError:
            idx = 0
        self.mode = self.MODES[(idx + 1) % len(self.MODES)]
        self.update()

    def _tick(self):
        # repaint while playing
        self.update()

    def _playhead(self) -> int:
        try:
            if self.playhead_cb:
                return int(self.playhead_cb() or 0)
        except Exception:
            pass
        return 0

    def _get_window(self, win: int = 1024):
        if self.pcm is None or len(self.pcm) < 4:
            return None
        idx = self._playhead()
        # stereo frames -> interleaved -> frame index -> sample index*2
        pos = max(0, min((len(self.pcm)//2)-1, idx))
        start = max(0, pos - win//2)
        end = min((len(self.pcm)//2), start + win)
        # convert to float -1..1
        seg = self.pcm[start*2:end*2].astype(np.float32) / 32768.0
        if len(seg) < 4:
            return None
        L = seg[0::2]
        R = seg[1::2]
        return L, R

    def paintEvent(self, ev):
        p = QPainter(self)
        rect = self.rect()

        # background (use palette)
        p.fillRect(rect, self.palette().window())

        if self.pcm is None:
            # placeholder
            pen = QPen(self.palette().text().color())
            p.setPen(pen)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "(no audio)")
            return

        if self.mode == "scope":
            self._paint_scope(p, rect)
        elif self.mode == "lightorgan":
            self._paint_lightorgan(p, rect)
        else:
            self._paint_spectrum(p, rect)

    def _paint_scope(self, p: QPainter, rect):
        w = rect.width()
        h = rect.height()
        midL = rect.top() + h*0.33
        midR = rect.top() + h*0.75

        win = self._get_window(1024)
        if win is None:
            return
        L,R = win

        pen = QPen(self.palette().text().color())
        pen.setWidth(1)
        p.setPen(pen)

        def draw_wave(y0, arr):
            n = len(arr)
            if n < 2:
                return
            step = max(1, n // w)
            pts = []
            x = rect.left()
            for i in range(0, n, step):
                y = y0 + (-arr[i]) * (h*0.25)
                pts.append(QPointF(x, y))
                x += 1
                if x > rect.right():
                    break
            if len(pts) > 1:
                for i in range(len(pts)-1):
                    p.drawLine(pts[i], pts[i+1])

        draw_wave(midL, L)
        draw_wave(midR, R)

    def _paint_spectrum(self, p: QPainter, rect):
        win = self._get_window(2048)
        if win is None:
            return
        L,R = win
        x = (L + R) * 0.5
        # Hann window
        xw = x * np.hanning(len(x))
        spec = np.fft.rfft(xw)
        mag = np.abs(spec)
        mag = mag / (mag.max() + 1e-9)

        bars = 32
        w = rect.width()
        h = rect.height()
        bw = max(1, w // bars)
        # log-ish spacing
        idxs = np.geomspace(2, len(mag)-1, bars).astype(int)
        col = self.palette().highlight().color()
        for bi, mi in enumerate(idxs):
            v = float(mag[mi])
            bh = int(v * (h-8))
            x0 = rect.left() + bi * bw
            y0 = rect.bottom() - bh
            p.fillRect(x0, y0, bw-2, bh, col)

    def _paint_lightorgan(self, p: QPainter, rect):
        """80s-style 4-lamp light organ with AGC + smoothing.

        Goals:
        - visible motion even on quiet material (gamma + dB mapping)
        - avoid 'one tiny red bar' (banding by Hz + percentile, not mean)
        - stable but lively (attack/decay + peak hold)
        """
        win = self._get_window(4096)
        if win is None:
            return
        L, R = win
        x = (L + R) * 0.5

        # window + FFT
        xw = x * np.hanning(len(x))
        spec = np.fft.rfft(xw)
        mag = np.abs(spec).astype(np.float32)
        if mag.size < 8:
            return

        # Normalize by global max for stable dB mapping
        mag /= (mag.max() + 1e-9)

        # Frequency bands (Hz) tuned for tracker-ish content
        freqs = np.fft.rfftfreq(len(xw), d=1.0 / float(self.sr))
        bands_hz = [
            (50.0, 220.0),    # low
            (220.0, 900.0),   # mid
            (900.0, 2800.0),  # high
            (2800.0, 8000.0), # presence/air
        ]

        raw = []
        for lo, hi in bands_hz:
            msk = (freqs >= lo) & (freqs < hi)
            if not np.any(msk):
                raw.append(0.0)
                continue
            b = mag[msk]
            # use a high percentile to avoid dilution by many near-zero bins
            v = float(np.percentile(b, 92))
            # dB-ish mapping: values below ~-60dB go dark
            db = 20.0 * np.log10(v + 1e-9)  # <= 0
            val = (db + 60.0) / 60.0
            raw.append(max(0.0, min(1.0, val)))

        raw = np.array(raw, dtype=np.float32)

        # Boost low values + avoid 'dead' look
        raw = np.clip(raw * 1.35, 0.0, 1.0)
        raw = np.sqrt(raw)  # gamma

        # Attack/decay smoothing + peak hold
        attack = 0.45  # faster rise
        decay = 0.88   # slower fall
        peak_decay = 0.94

        for i in range(4):
            if raw[i] > self._lamp_state[i]:
                self._lamp_state[i] = self._lamp_state[i] + (raw[i] - self._lamp_state[i]) * attack
            else:
                self._lamp_state[i] = self._lamp_state[i] * decay

            if self._lamp_state[i] > self._lamp_peak[i]:
                self._lamp_peak[i] = self._lamp_state[i]
            else:
                self._lamp_peak[i] = self._lamp_peak[i] * peak_decay

        vals = self._lamp_state.tolist()
        peaks = self._lamp_peak.tolist()

        cols = [
            QColor(230, 70, 70),    # red
            QColor(70, 220, 100),   # green
            QColor(90, 150, 255),   # blue
            QColor(235, 210, 80),   # yellow
        ]

        w = rect.width()
        h = rect.height()
        pad = 10
        lamp_w = max(20, (w - pad * 5) // 4)
        max_h = max(40, h - 2 * pad)

        # Backplate
        back = self.palette().window().color()
        p.fillRect(rect, back)

        # Draw lamps
        for i, v in enumerate(vals):
            x0 = rect.left() + pad + i * (lamp_w + pad)
            y_top = rect.top() + pad
            y_bot = y_top + max_h

            lh = int(max_h * float(v))
            y0 = y_bot - lh

            # dim frame
            frame_pen = QPen(self.palette().text().color())
            frame_pen.setWidth(1)
            p.setPen(frame_pen)
            p.drawRect(x0, y_top, lamp_w, max_h)

            # lamp fill (with simple glow)
            c = cols[i]
            p.fillRect(x0 + 1, y0, lamp_w - 1, lh, c)
            glow = QColor(c)
            glow.setAlpha(90)
            p.fillRect(x0 + 1, y0, lamp_w - 1, min(lh + 8, max_h), glow)

            # peak marker
            ph = int(max_h * float(peaks[i]))
            py = y_bot - ph
            peak_pen = QPen(QColor(255, 255, 255, 180))
            peak_pen.setWidth(2)
            p.setPen(peak_pen)
            p.drawLine(x0 + 2, py, x0 + lamp_w - 3, py)

