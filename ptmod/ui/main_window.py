from __future__ import annotations

import os
import time
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QDesktopServices
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox, QCheckBox, QTextEdit,
    QFileDialog, QGroupBox, QProgressBar, QStatusBar, QSlider, QScrollArea, QSizePolicy, QFrame
)

import protracker_mod_choral_generator as backend

from ptmod.config import SongConfig
from ptmod import __version__
from ptmod.engines.playback_engine import PlaybackEngine
from ptmod.engines.sample_engine import SampleEngine, SamplePreviewSpec
from ptmod.engines.pattern_engine import format_pattern, order_positions_for_pattern
from ptmod.engines.generator_engine import generate_batch
from ptmod.engines.plugins_engine import reload_plugins, plugin_root, open_folder, add_last_as_plugin
from ptmod.ui.i18n import I18N, LANG_CHOICES
from ptmod.ui.visualizer import VisualizerWidget


class GenerationWorker(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    song_ready = pyqtSignal(object, object, object)  # (Path, SongData, QualityResult|None)
    preview_ready = pyqtSignal(bytes, bytes, int)    # (wav_bytes, pcm16_bytes, sample_rate)

    def __init__(self, cfg: SongConfig):
        super().__init__()
        self.cfg = cfg
        self._preview_rate = 44100
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.status.emit("thinking...")
            self.progress.emit(0)

            def _st(s: str):
                self.status.emit(s if (s and s.strip()) else "thinking...")

            def _pr(p: int):
                self.progress.emit(int(max(0, min(100, p))))

            def _lg(s: str):
                self.log.emit(str(s))

            last = None
            for mod_path, song, q in generate_batch(self.cfg, status_cb=_st, progress_cb=_pr, log_cb=_lg):
                if self._cancel:
                    _lg("[cancel] requested")
                    break
                last = (mod_path, song, q)
                self.song_ready.emit(mod_path, song, q)

                # Optional: export WAV and params per cfg
                if bool(getattr(self.cfg, 'save_params', True)):
                    try:
                        backend.save_song_parameters_txt(Path(mod_path), song)
                    except Exception as e:
                        _lg(f"[save_params] failed: {e}")
                if bool(getattr(self.cfg, 'export_wav', True)):
                    try:
                        self.status.emit("thinking...")
                        self.progress.emit(0)
                        def _render_prog(done: int, total: int):
                            pct = 0.0 if total <= 0 else (done / float(total)) * 100.0
                            self.status.emit(f"render {pct:.0f}%")
                            self.progress.emit(int(pct))
                        pcm16, sr, _chbufs = backend.render_song_to_pcm16(song, out_rate=self._preview_rate, progress_cb=_render_prog, cancel_event=None)
                        wavb = backend.pcm16_to_wav_bytes(pcm16, sr, nch=2)
                        wav_path = Path(mod_path).with_suffix('.wav')
                        if not wav_path.exists():
                            wav_path.write_bytes(wavb)
                        self.preview_ready.emit(wavb, pcm16, int(sr))
                    except Exception as e:
                        _lg(f"[export_wav] failed: {e}")

            # If not exporting WAV, still render preview for last song to enable PLAY + visualizer
            if last is not None and not bool(getattr(self.cfg, 'export_wav', True)):
                mod_path, song, q = last
                self.status.emit("thinking...")
                self.progress.emit(0)
                def _render_prog(done: int, total: int):
                    pct = 0.0 if total <= 0 else (done / float(total)) * 100.0
                    self.status.emit(f"render {pct:.0f}%")
                    self.progress.emit(int(pct))
                pcm16, sr, _chbufs = backend.render_song_to_pcm16(song, out_rate=self._preview_rate, progress_cb=_render_prog, cancel_event=None)
                wavb = backend.pcm16_to_wav_bytes(pcm16, sr, nch=2)
                self.preview_ready.emit(wavb, pcm16, int(sr))

            self.status.emit("ready")
            self.progress.emit(100)
        except Exception as e:
            self.log.emit(f"[error] {e}")
            self.status.emit("ready")
            self.progress.emit(0)


class MainWindow(QMainWindow):
    def __init__(self, themes: dict, apply_theme_cb):
        super().__init__()
        self.setWindowTitle(f"Protracker Music Generator - v{__version__}")
        self.resize(1120, 820)

        self.themes = themes
        self.apply_theme_cb = apply_theme_cb

        self.i18n = I18N("English")
        self.cfg = SongConfig()

        self.last_song: Optional[Any] = None
        self.last_mod_path: Optional[Path] = None
        self.last_quality: Optional[Any] = None
        self.preview_wav: Optional[bytes] = None
        self.preview_pcm16: Optional[bytes] = None
        self.preview_sr: int = 44100

        self.playback = PlaybackEngine()

        # Poll playback to keep STOP/PLAY state and visualizer running across backends
        self._play_poll = QTimer(self)
        self._play_poll.timeout.connect(self._poll_playback_state)
        self._play_poll.start(200)
        self._was_playing = False
        self.sample_engine = SampleEngine()
        self.worker: Optional[GenerationWorker] = None

        self._build_ui()
        self._apply_language()

    # ---------- UI ----------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # Left sidebar (scrollable controls + fixed action buttons)
        # Reason: sidebar content can exceed window height; Qt will then shrink groupboxes to near-zero
        # height (labels become unreadable). A scroll area preserves preferred sizes.
        left = QWidget()
        left_outer = QVBoxLayout(left)
        left_outer.setContentsMargins(8, 8, 8, 8)
        left_outer.setSpacing(8)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_content = QWidget()
        left_l = QVBoxLayout(left_content)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(8)

        # Pattern order
        grp_order = QGroupBox(self.i18n.tr("PATTERN ORDER"))
        g = QGridLayout(grp_order)
        self.btn_smart = QPushButton(self.i18n.tr("SMART"))
        self.btn_smart.clicked.connect(self._smart_order)
        self.order_combo = QComboBox()
        self.order_combo.setEditable(True)
        try:
            for p in getattr(backend, 'ORDER_PRESETS', []):
                self.order_combo.addItem(str(p))
        except Exception:
            pass
        self.order_combo.setCurrentText(getattr(backend, 'DEFAULT_ORDER_STR', '0,1,2,3,4,5'))
        g.addWidget(self.order_combo, 0, 0, 1, 2)
        g.addWidget(self.btn_smart, 1, 1)
        self.chk_use_smart = QCheckBox(self.i18n.tr("SMART"))
        self.chk_use_smart.setChecked(False)
        g.addWidget(self.chk_use_smart, 1, 0)
        left_l.addWidget(grp_order)

        # Base melody + derivation
        grp_mel = QGroupBox(self.i18n.tr("BASE MELODY"))
        gm = QGridLayout(grp_mel)
        self.melody_combo = QComboBox()
        self._reload_melodies(initial=True)
        gm.addWidget(self.melody_combo, 0, 0, 1, 2)
        gm.addWidget(QLabel(self.i18n.tr("MELODY DERIVATION")), 1, 0)
        self.derive_combo = QComboBox()
        self.derive_combo.addItems(["Random", "Near", "Far"])
        gm.addWidget(self.derive_combo, 1, 1)
        left_l.addWidget(grp_mel)

        # Key root + random
        grp_key = QGroupBox(self.i18n.tr("BASE KEY (optional)"))
        gk = QHBoxLayout(grp_key)
        self.key_edit = QLineEdit("")
        self.key_edit.setPlaceholderText("C-2")
        self.btn_key_rnd = QPushButton("RND")
        self.btn_key_rnd.clicked.connect(self._random_key)
        gk.addWidget(self.key_edit, 1)
        gk.addWidget(self.btn_key_rnd)
        left_l.addWidget(grp_key)

        # Speed / tempo
        grp_st = QGroupBox(self.i18n.tr("SPEED") + " / " + self.i18n.tr("TEMPO"))
        gst = QGridLayout(grp_st)
        gst.addWidget(QLabel(self.i18n.tr("SPEED")), 0, 0)
        self.speed = QSpinBox(); self.speed.setRange(1, 31); self.speed.setValue(6)
        gst.addWidget(self.speed, 0, 1)
        gst.addWidget(QLabel(self.i18n.tr("TEMPO")), 1, 0)
        self.tempo = QSpinBox(); self.tempo.setRange(32, 255); self.tempo.setValue(125)
        gst.addWidget(self.tempo, 1, 1)
        left_l.addWidget(grp_st)

        # Advanced
        grp_adv = QGroupBox("Advanced")
        ga = QGridLayout(grp_adv)

        ga.addWidget(QLabel(self.i18n.tr("SCALE MODE")), 0, 0)
        self.scale_combo = QComboBox()
        for s in getattr(backend, 'SCALE_MODE_CHOICES', ["Auto","Major","Minor"]):
            self.scale_combo.addItem(s)
        self.scale_combo.setCurrentText("Major")
        ga.addWidget(self.scale_combo, 0, 1)

        ga.addWidget(QLabel(self.i18n.tr("VARIATION")), 1, 0)
        self.var_slider = QSlider(Qt.Orientation.Horizontal)
        self.var_slider.setRange(0, 100)
        self.var_slider.setValue(65)
        ga.addWidget(self.var_slider, 1, 1)

        ga.addWidget(QLabel(self.i18n.tr("SEED (optional)")), 2, 0)
        self.seed_edit = QLineEdit("")
        self.btn_seed_rnd = QPushButton("RND")
        self.btn_seed_rnd.clicked.connect(self._random_seed)
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed_edit, 1)
        seed_row.addWidget(self.btn_seed_rnd)
        seed_wrap = QWidget(); seed_wrap.setLayout(seed_row)
        ga.addWidget(seed_wrap, 2, 1)

        self.chk_auto_seed = QCheckBox(self.i18n.tr("NEW SEED EACH GENERATE"))
        self.chk_auto_seed.setChecked(True)
        ga.addWidget(self.chk_auto_seed, 3, 0, 1, 2)

        ga.addWidget(QLabel(self.i18n.tr("BATCH")), 4, 0)
        self.batch = QSpinBox(); self.batch.setRange(1, 50); self.batch.setValue(1)
        ga.addWidget(self.batch, 4, 1)

        ga.addWidget(QLabel(self.i18n.tr("MUTE CH")), 5, 0)
        mute_row = QHBoxLayout()
        self.mute = [QCheckBox(str(i+1)) for i in range(4)]
        for cb in self.mute:
            mute_row.addWidget(cb)
        mute_wrap = QWidget(); mute_wrap.setLayout(mute_row)
        ga.addWidget(mute_wrap, 5, 1)

        ga.addWidget(QLabel(self.i18n.tr("STEREO %")), 6, 0)
        self.stereo_slider = QSlider(Qt.Orientation.Horizontal)
        self.stereo_slider.setRange(0, 200)
        self.stereo_slider.setValue(100)
        ga.addWidget(self.stereo_slider, 6, 1)

        ga.addWidget(QLabel(self.i18n.tr("PASSES")), 7, 0)
        self.passes = QComboBox()
        self.passes.addItems(["1","2","3","4","5"])
        self.passes.setCurrentText("3")
        ga.addWidget(self.passes, 7, 1)

        left_l.addWidget(grp_adv)

        # Instruments (CH1..CH4) + octave span
        grp_inst = QGroupBox(self.i18n.tr("INSTRUMENTS (CH1..CH4)"))
        gi = QGridLayout(grp_inst)
        self.btn_inst_rnd = QPushButton("RND")
        self.btn_inst_rnd.clicked.connect(self._randomize_instruments)
        gi.addWidget(self.btn_inst_rnd, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        self.inst = []
        self.octv = []
        inst_choices = list(getattr(backend, 'INSTRUMENT_CHOICES', ["Piano"]))
        for ch in range(4):
            gi.addWidget(QLabel(f"CH{ch+1}"), ch+1, 0)
            row = QHBoxLayout()
            cb = QComboBox(); cb.addItems(inst_choices); cb.setCurrentText("Piano")
            # Keep Samples tab labels in sync with instrument selection.
            cb.currentTextChanged.connect(lambda _t=None: self._refresh_sample_display())
            oc = QComboBox(); oc.addItems(["1","2","3"])
            oc.setCurrentText("2" if ch==2 else "3")
            row.addWidget(cb, 1)
            row.addWidget(oc)
            wrap = QWidget(); wrap.setLayout(row)
            gi.addWidget(wrap, ch+1, 1)
            self.inst.append(cb)
            self.octv.append(oc)
        left_l.addWidget(grp_inst)

        # Language
        grp_lang = QGroupBox(self.i18n.tr("LANGUAGE"))
        gl = QHBoxLayout(grp_lang)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(LANG_CHOICES)
        self.lang_combo.setCurrentText("English")
        self.lang_combo.currentTextChanged.connect(self._on_lang_change)
        gl.addWidget(self.lang_combo)
        left_l.addWidget(grp_lang)

        # Keep groupboxes readable: prefer fixed vertical sizing (scroll handles overflow)
        for _gb in (grp_order, grp_mel, grp_key, grp_st, grp_adv, grp_inst, grp_lang):
            _gb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            _gb.setMinimumHeight(_gb.sizeHint().height())

        left_l.addStretch(1)

        left_scroll.setWidget(left_content)
        left_outer.addWidget(left_scroll, 1)

        # Fixed action buttons (always visible)
        btn_panel = QWidget()
        btn_panel_l = QVBoxLayout(btn_panel)
        btn_panel_l.setContentsMargins(0, 0, 0, 0)
        btn_panel_l.setSpacing(8)

        btn_row1 = QHBoxLayout()
        self.btn_gen = QPushButton(self.i18n.tr("GENERATE"))
        self.btn_regen = QPushButton(self.i18n.tr("RE-GENERATE"))
        self.btn_play = QPushButton(self.i18n.tr("PLAY"))
        self.btn_stop = QPushButton(self.i18n.tr("STOP"))
        self.btn_gen.clicked.connect(self._on_generate)
        self.btn_regen.clicked.connect(self._on_regenerate)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_stop.clicked.connect(self._on_stop)

        btn_row1.addWidget(self.btn_gen)
        btn_row1.addWidget(self.btn_regen)
        btn_row1.addWidget(self.btn_play)
        btn_row1.addWidget(self.btn_stop)
        btn_wrap1 = QWidget(); btn_wrap1.setLayout(btn_row1)
        btn_panel_l.addWidget(btn_wrap1)

        btn_row2 = QGridLayout()
        self.btn_open_out = QPushButton(self.i18n.tr("OPEN OUTPUT"))
        self.btn_open_plg = QPushButton(self.i18n.tr("OPEN PLUGINS"))
        self.btn_refresh = QPushButton(self.i18n.tr("REFRESH"))
        self.btn_add_plg = QPushButton(self.i18n.tr("ADD AS PLUGIN"))
        self.btn_open_out.clicked.connect(self._open_output)
        self.btn_open_plg.clicked.connect(self._open_plugins)
        self.btn_refresh.clicked.connect(self._reload_melodies)
        self.btn_add_plg.clicked.connect(self._add_as_plugin)

        btn_row2.addWidget(self.btn_open_out, 0, 0)
        btn_row2.addWidget(self.btn_open_plg, 0, 1)
        btn_row2.addWidget(self.btn_refresh, 1, 0)
        btn_row2.addWidget(self.btn_add_plg, 1, 1)
        btn_wrap2 = QWidget(); btn_wrap2.setLayout(btn_row2)
        btn_panel_l.addWidget(btn_wrap2)

        left_outer.addWidget(btn_panel, 0)

        splitter.addWidget(left)

        # Right tabs
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        self._tab_main()
        self._tab_samples()
        self._tab_options()

        self._refresh_sample_display()


        # Status bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel("ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(220)
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.progress_bar)

        # Initial UI state
        self.btn_play.setEnabled(False)
        self.btn_regen.setEnabled(False)
        # STOP availability is managed by playback poller
        self.btn_add_plg.setEnabled(False)

    def _tab_main(self):
        w = QWidget()
        self.tabs.addTab(w, self.i18n.tr("MAIN"))
        layout = QVBoxLayout(w)

        self.viz_title = QLabel(self.i18n.tr("SPECTRUM ANALYZER"))
        self.viz_title.setStyleSheet("font-size: 14px;")
        self.viz_hint = QLabel(self.i18n.tr("Click visualizer to toggle Spectrum / Scopes") if hasattr(self.i18n, 'tr') else "Click visualizer to toggle")
        layout.addWidget(self.viz_title)
        layout.addWidget(self.viz_hint)

        self.visualizer = VisualizerWidget()
        self.visualizer.set_playhead_callback(self.playback.playhead_frame_index)
        layout.addWidget(self.visualizer)

        # Render/harmony row
        row = QHBoxLayout()
        self.render_lbl = QLabel("")
        self.render_lbl.setStyleSheet("font-size: 16px;")
        self.harmony_lbl = QLabel("Harmony: --%")
        row.addWidget(self.render_lbl, 1)
        row.addWidget(self.harmony_lbl, 0)
        layout.addLayout(row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.append("Generate a song, then hit PLAY.")
        layout.addWidget(self.log_text)

        # Pattern preview
        head = QHBoxLayout()
        head.addWidget(QLabel(self.i18n.tr("PATTERN PREVIEW")))
        self.pattern_sel = QComboBox()
        self.pattern_sel.addItem("0")
        self.pattern_sel.currentTextChanged.connect(self._update_pattern_preview)
        head.addWidget(self.pattern_sel)
        head.addStretch(1)
        layout.addLayout(head)

        self.pattern_text = QTextEdit()
        self.pattern_text.setReadOnly(True)
        font = QFont("Courier New")
        self.pattern_text.setFont(font)
        layout.addWidget(self.pattern_text, 1)

    def _tab_samples(self):
        w = QWidget()
        self.tabs.addTab(w, self.i18n.tr("SAMPLES"))
        layout = QVBoxLayout(w)

        self.sample_rows = []
        for ch in range(4):
            grp = QGroupBox(f"CH{ch+1}")
            g = QGridLayout(grp)
            self.sample_inst_lbl = QLabel("-")
            self.sample_status_lbl = QLabel("Generated")
            self.sample_vol_lbl = QLabel("-")
            btn_play = QPushButton(self.i18n.tr("PLAY"))
            btn_rep = QPushButton("Replace")
            btn_reset = QPushButton("Reset")
            btn_play.clicked.connect(lambda _=None, c=ch: self._play_sample(c))
            btn_rep.clicked.connect(lambda _=None, c=ch: self._replace_sample(c))
            btn_reset.clicked.connect(lambda _=None, c=ch: self._reset_sample(c))

            g.addWidget(QLabel("Instrument"), 0, 0)
            g.addWidget(self.sample_inst_lbl, 0, 1)
            g.addWidget(QLabel("Status"), 1, 0)
            g.addWidget(self.sample_status_lbl, 1, 1)
            g.addWidget(QLabel("Volume"), 2, 0)
            g.addWidget(self.sample_vol_lbl, 2, 1)

            g.addWidget(btn_play, 0, 2)
            g.addWidget(btn_rep, 1, 2)
            g.addWidget(btn_reset, 2, 2)

            self.sample_rows.append({
                "inst": self.sample_inst_lbl,
                "status": self.sample_status_lbl,
                "vol": self.sample_vol_lbl,
                "play": btn_play,
                "replace": btn_rep,
                "reset": btn_reset,
            })
            layout.addWidget(grp)

        btn_import_all = QPushButton("Import WAV (multi)")
        btn_import_all.clicked.connect(self._import_all_samples)
        layout.addWidget(btn_import_all)

        self.samples_info = QTextEdit()
        self.samples_info.setReadOnly(True)
        self.samples_info.setMinimumHeight(120)
        layout.addWidget(self.samples_info, 1)
        self._refresh_sample_display()

    def _tab_options(self):
        w = QWidget()
        self.tabs.addTab(w, self.i18n.tr("OPTIONS"))
        layout = QVBoxLayout(w)

        # Export options
        grp_export = QGroupBox("Export")
        ge = QVBoxLayout(grp_export)
        self.opt_export_wav = QCheckBox(self.i18n.tr("Export rendered songs as WAV"))
        self.opt_save_params = QCheckBox(self.i18n.tr("Save song parameters"))
        self.opt_disable_vibrato = QCheckBox(self.i18n.tr("Disable vibrato in samples"))
        self.opt_fadeout = QCheckBox(self.i18n.tr("Add empty fade-out pattern"))
        self.opt_slowdown = QCheckBox(self.i18n.tr("Enable slowdown to the end of the song"))

        self.opt_export_wav.setChecked(True)
        self.opt_save_params.setChecked(True)
        self.opt_disable_vibrato.setChecked(False)
        self.opt_fadeout.setChecked(True)
        self.opt_slowdown.setChecked(False)

        for cb in [self.opt_export_wav, self.opt_save_params, self.opt_disable_vibrato, self.opt_fadeout, self.opt_slowdown]:
            ge.addWidget(cb)
        layout.addWidget(grp_export)

        # Compatibility / signature
        grp_comp = QGroupBox(self.i18n.tr("Compatibility"))
        gc = QGridLayout(grp_comp)
        self.opt_compat = QCheckBox("compat_mode")
        self.opt_compat.setChecked(True)
        gc.addWidget(self.opt_compat, 0, 0, 1, 2)

        gc.addWidget(QLabel(self.i18n.tr("MOD Signature")), 1, 0)
        self.opt_sig = QComboBox()
        sig_items = [
            ("M.K. (ProTracker / 4ch)", "M.K."),
            ("M!K! (ProTracker alt / 4ch)", "M!K!"),
            ("FLT4 (StarTrekker / 4ch)", "FLT4"),
            ("4CHN (FastTracker / 4ch)", "4CHN"),
            ("N.T. (NoiseTracker / 4ch)", "N.T."),
            ("NSMS (NoiseTracker / 4ch)", "NSMS"),
            ("OKTA (Oktalyzer / 8ch tag)", "OKTA"),
            ("FLT8 (StarTrekker / 8ch tag)", "FLT8"),
            ("8CHN (FastTracker / 8ch tag)", "8CHN"),
        ]
        for label, sig in sig_items:
            self.opt_sig.addItem(label, sig)

        # Also include any backend-provided choices not already present
        try:
            for s in getattr(backend, 'MOD_SIGNATURE_CHOICES', []):
                s = str(s)
                if not any(str(self.opt_sig.itemData(i)) == s for i in range(self.opt_sig.count())):
                    self.opt_sig.addItem(f"{s} (Custom)", s)
        except Exception:
            pass

        default_sig = str(getattr(backend, 'DEFAULT_MOD_SIGNATURE', 'M!K!'))
        for i in range(self.opt_sig.count()):
            if str(self.opt_sig.itemData(i)) == default_sig:
                self.opt_sig.setCurrentIndex(i)
                break

        self.opt_sig.setToolTip(
            "Tracker signature tag (magic at offset 1080).\n"
            "Note: 8ch tags (OKTA/FLT8/8CHN) may confuse players because this generator outputs 4 channels."
        )
        gc.addWidget(self.opt_sig, 1, 1)
        layout.addWidget(grp_comp)

        # Ralph loop
        grp_r = QGroupBox(self.i18n.tr("Ralph-Loop"))
        gr = QGridLayout(grp_r)
        self.opt_ralph = QCheckBox(self.i18n.tr("Ralph-Loop"))
        self.opt_ralph.setChecked(False)
        gr.addWidget(self.opt_ralph, 0, 0, 1, 2)
        gr.addWidget(QLabel("Target %"), 1, 0)
        self.opt_r_target = QSpinBox(); self.opt_r_target.setRange(0, 100); self.opt_r_target.setValue(90)
        gr.addWidget(self.opt_r_target, 1, 1)
        gr.addWidget(QLabel("Max attempts"), 2, 0)
        self.opt_r_attempts = QSpinBox(); self.opt_r_attempts.setRange(1, 200); self.opt_r_attempts.setValue(50)
        gr.addWidget(self.opt_r_attempts, 2, 1)
        layout.addWidget(grp_r)

        # Theme
        grp_theme = QGroupBox(self.i18n.tr("Theme"))
        gt = QHBoxLayout(grp_theme)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(self.themes.keys()))
        self.theme_combo.setCurrentText("ProTracker Gray" if "ProTracker Gray" in self.themes else list(self.themes.keys())[0])
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_theme)
        gt.addWidget(self.theme_combo, 1)
        gt.addWidget(btn_apply)
        layout.addWidget(grp_theme)

        # FX injection
        grp_fx = QGroupBox(self.i18n.tr("FX Injection"))
        gfx = QGridLayout(grp_fx)
        self.fx_init = QCheckBox(self.i18n.tr("Insert initial speed/tempo"))
        self.fx_init.setChecked(True)
        self.fx_vib = QCheckBox(self.i18n.tr("Vibrato on melody"))
        self.fx_porta = QCheckBox(self.i18n.tr("Portamento transitions"))
        self.fx_arp = QCheckBox(self.i18n.tr("Arpeggio ornaments"))
        self.fx_vol = QCheckBox(self.i18n.tr("Volume motion"))
        self.fx_cut = QCheckBox(self.i18n.tr("Note cut"))
        self.fx_retrig = QCheckBox(self.i18n.tr("Retrig"))
        gfx.addWidget(self.fx_init, 0, 0, 1, 2)
        gfx.addWidget(self.fx_vib, 1, 0, 1, 2)
        gfx.addWidget(self.fx_porta, 2, 0, 1, 2)
        gfx.addWidget(self.fx_arp, 3, 0, 1, 2)
        gfx.addWidget(self.fx_vol, 4, 0, 1, 2)
        gfx.addWidget(self.fx_cut, 5, 0, 1, 2)
        gfx.addWidget(self.fx_retrig, 6, 0, 1, 2)
        gfx.addWidget(QLabel(self.i18n.tr("Intensity")), 7, 0)
        self.fx_intensity = QSlider(Qt.Orientation.Horizontal)
        self.fx_intensity.setRange(0, 100)
        self.fx_intensity.setValue(50)
        gfx.addWidget(self.fx_intensity, 7, 1)
        layout.addWidget(grp_fx)

        # tooltips for FX
        self.fx_init.setToolTip(self.i18n.tt("Insert initial speed/tempo"))
        self.fx_vib.setToolTip(self.i18n.tt("Vibrato on melody"))
        self.fx_porta.setToolTip(self.i18n.tt("Portamento transitions"))
        self.fx_arp.setToolTip(self.i18n.tt("Arpeggio ornaments"))
        self.fx_vol.setToolTip(self.i18n.tt("Volume motion"))
        self.fx_cut.setToolTip(self.i18n.tt("Note cut"))
        self.fx_retrig.setToolTip(self.i18n.tt("Retrig"))
        grp_fx.setToolTip(self.i18n.tt("FX Injection"))

        layout.addStretch(1)

    # ---------- language ----------
    def _on_lang_change(self, lang: str):
        self.i18n = I18N(lang)
        self._apply_language()

    def _apply_language(self):
        # For simplicity, only refresh a subset of visible labels.
        # (Full Qt translations can be added later.)
        self.btn_gen.setText(self.i18n.tr("GENERATE"))
        self.btn_regen.setText(self.i18n.tr("RE-GENERATE"))
        self.btn_play.setText(self.i18n.tr("PLAY"))
        self.btn_stop.setText(self.i18n.tr("STOP"))
        self.btn_open_out.setText(self.i18n.tr("OPEN OUTPUT"))
        self.btn_open_plg.setText(self.i18n.tr("OPEN PLUGINS"))
        self.btn_refresh.setText(self.i18n.tr("REFRESH"))
        self.btn_add_plg.setText(self.i18n.tr("ADD AS PLUGIN"))

    # ---------- helpers ----------
    def _append_log(self, s: str):
        self.log_text.append(str(s).rstrip())

    def _set_status(self, s: str):
        self.status_label.setText(s if (s and s.strip()) else "thinking...")
        self.render_lbl.setText(self.status_label.text())
        # Update title according to visualizer mode
        mode = getattr(self.visualizer, 'mode', 'spectrum')
        if mode == 'scope':
            self.viz_title.setText(self.i18n.tr("STEREO SCOPES"))
        elif mode == 'lightorgan':
            self.viz_title.setText(self.i18n.tr("LIGHT ORGAN"))
        else:
            self.viz_title.setText(self.i18n.tr("SPECTRUM ANALYZER"))

    def _set_progress(self, p: int):
        self.progress_bar.setValue(int(max(0, min(100, p))))

    def _smart_order(self):
        try:
            seed = int(self.seed_edit.text().strip() or backend.random_seed_value())
        except Exception:
            seed = int(backend.random_seed_value())
        rr = random.Random(seed ^ 0xA5A5)
        order = backend.generate_smart_order(rr, n_patterns=int(getattr(backend,'PATTERN_COUNT',20)))
        self.order_combo.setCurrentText(", ".join(str(x) for x in order))
        self.chk_use_smart.setChecked(True)

    def _random_key(self):
        try:
            self.key_edit.setText(backend.random_key_root())
        except Exception:
            self.key_edit.setText("C-2")

    def _random_seed(self):
        try:
            self.seed_edit.setText(str(backend.random_seed_value()))
        except Exception:
            self.seed_edit.setText(str(int(time.time()*1000)))

    def _randomize_instruments(self):
        palettes = [
            ["Piano", "Piano", "Piano", "Piano"],
            ["Piano", "Strings", "Choir Aah", "Organ"],
            ["Electric Piano", "Synth Pad", "Strings", "Choir Ooh"],
            ["Organ", "Choir Aah", "French Horn", "Bassoon"],
            ["Harp", "Strings", "Choir Ooh", "Flute"],
            ["Synth Pad", "Synth Lead", "Square Lead", "Bassoon"],
            ["Clarinet", "Sax", "French Horn", "Tuba"],
        ]
        try:
            st = self.seed_edit.text().strip()
            rr = random.Random(int(st) if st else int(time.time()*1000))
        except Exception:
            rr = random.Random(int(time.time()*1000))
        pal = rr.choice(palettes)
        for i in range(4):
            self.inst[i].setCurrentText(pal[i])

    def _open_output(self):
        try:
            open_folder(Path(self.cfg.out_dir))
        except Exception:
            open_folder(Path("mods_out"))

    def _open_plugins(self):
        open_folder(plugin_root())

    def _reload_melodies(self, initial: bool = False):
        names = reload_plugins()
        # Keep old selection
        prev = self.melody_combo.currentText() if hasattr(self, 'melody_combo') else ""
        if hasattr(self, 'melody_combo'):
            self.melody_combo.blockSignals(True)
            self.melody_combo.clear()
            # Ensure two canonical options
            self.melody_combo.addItem("Pure Random")
            self.melody_combo.addItem("Random")
            for n in names:
                if n in ("Pure Random", "Random"):
                    continue
                self.melody_combo.addItem(n)
            # restore
            if prev:
                self.melody_combo.setCurrentText(prev)
            elif initial:
                # Default to true random selection on first start
                self.melody_combo.setCurrentText("Random")
            self.melody_combo.blockSignals(False)
        if not initial:
            self._append_log(f"Reloaded {len(names)} melody plugin(s).")

    def _add_as_plugin(self):
        if self.last_song is None:
            self._append_log("No song yet.")
            return
        dest = add_last_as_plugin(self.last_mod_path, self.last_song, log_cb=self._append_log)
        if dest:
            self.btn_add_plg.setEnabled(True)

    # ---------- generation / playback ----------
    def _collect_cfg(self) -> SongConfig:
        cfg = SongConfig()

        cfg.out_dir = self.cfg.out_dir  # keep current unless we add picker later

        # seed
        seed_txt = self.seed_edit.text().strip()
        cfg.seed = int(seed_txt) if seed_txt else None
        cfg.auto_seed_each_generate = bool(self.chk_auto_seed.isChecked())
        cfg.batch_count = int(self.batch.value())

        # order
        cfg.order_str = self.order_combo.currentText()
        cfg.use_smart_order = bool(self.chk_use_smart.isChecked())

        # melody / derive / key
        m = self.melody_combo.currentText()
        if m in ("Random", ""):
            cfg.melody_name = None
        elif m == "Pure Random":
            cfg.melody_name = "Pure Random"
        else:
            cfg.melody_name = m
        cfg.derive_mode = self.derive_combo.currentText()
        k = self.key_edit.text().strip()
        cfg.key_root_override = k if k else None

        # speed/tempo
        cfg.speed = int(self.speed.value())
        cfg.tempo = int(self.tempo.value())

        cfg.scale_mode = self.scale_combo.currentText()
        cfg.variation_pct = int(self.var_slider.value())
        cfg.stereo_width_pct = int(self.stereo_slider.value())

        cfg.mute_channels = [cb.isChecked() for cb in self.mute]
        cfg.quality_passes = int(self.passes.currentText())

        cfg.instruments = [c.currentText() for c in self.inst]
        cfg.octave_spans = [int(c.currentText()) for c in self.octv]

        # options tab
        cfg.export_wav = bool(self.opt_export_wav.isChecked())
        cfg.save_params = bool(self.opt_save_params.isChecked())
        cfg.disable_vibrato = bool(self.opt_disable_vibrato.isChecked())
        cfg.fadeout_pattern = bool(self.opt_fadeout.isChecked())
        cfg.enable_slowdown = bool(self.opt_slowdown.isChecked())

        cfg.compat_mode = bool(self.opt_compat.isChecked())
        sig = self.opt_sig.currentData()
        cfg.mod_signature = str(sig) if sig else (self.opt_sig.currentText().strip()[:4] if self.opt_sig.currentText() else None)

        cfg.ralph_loop = bool(self.opt_ralph.isChecked())
        cfg.ralph_target_score = float(self.opt_r_target.value())
        cfg.ralph_max_attempts = int(self.opt_r_attempts.value())

        cfg.fx_insert_initial_speed_tempo = bool(self.fx_init.isChecked())
        cfg.fx_vibrato_melody = bool(self.fx_vib.isChecked())
        cfg.fx_portamento_melody = bool(self.fx_porta.isChecked())
        cfg.fx_arpeggio_ornaments = bool(self.fx_arp.isChecked())
        cfg.fx_volume_motion = bool(self.fx_vol.isChecked())
        cfg.fx_note_cut = bool(self.fx_cut.isChecked())
        cfg.fx_retrig = bool(self.fx_retrig.isChecked())
        cfg.fx_intensity = int(self.fx_intensity.value())

        return cfg

    def _on_generate(self):
        if self.worker and self.worker.isRunning():
            return
        self.cfg = self._collect_cfg()

        self.btn_gen.setEnabled(False)
        self.btn_regen.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = GenerationWorker(self.cfg)
        self.worker.status.connect(self._set_status)
        self.worker.progress.connect(self._set_progress)
        self.worker.log.connect(self._append_log)
        self.worker.song_ready.connect(self._on_song_ready)
        self.worker.preview_ready.connect(self._on_preview_ready)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

    def _on_regenerate(self):
        # Force a new seed unless auto_seed is off (then increment)
        try:
            if self.chk_auto_seed.isChecked():
                self.seed_edit.setText(str(backend.random_seed_value()))
            else:
                cur = int(self.seed_edit.text().strip() or backend.random_seed_value())
                self.seed_edit.setText(str(cur + 1))
        except Exception:
            self.seed_edit.setText(str(int(time.time()*1000)))
        self._on_generate()

    def _on_song_ready(self, mod_path: Path, song: Any, q: Any):
        self.last_mod_path = Path(mod_path)
        self.last_song = song
        self.last_quality = q
        self.btn_regen.setEnabled(True)
        self.btn_add_plg.setEnabled(True)

        # Harmony label
        try:
            hs = float(getattr(q, 'harmony_score', getattr(song, 'harmony_score', 0.0)))
            self.harmony_lbl.setText(f"Harmony: {hs:.1f}%")
        except Exception:
            self.harmony_lbl.setText("Harmony: --%")

        # Populate pattern selector
        try:
            n = len(song.patterns)
            self.pattern_sel.blockSignals(True)
            self.pattern_sel.clear()
            for i in range(n):
                self.pattern_sel.addItem(str(i))
            self.pattern_sel.setCurrentText("0")
            self.pattern_sel.blockSignals(False)
            self._update_pattern_preview()
        except Exception:
            pass

        self._refresh_sample_display()

    def _on_preview_ready(self, wavb: bytes, pcm16: bytes, sr: int):
        self.preview_wav = wavb
        self.preview_pcm16 = pcm16
        self.preview_sr = int(sr)
        self.visualizer.set_audio(self.preview_pcm16, sr=self.preview_sr)
        self.btn_play.setEnabled(True)

    def _on_worker_done(self):
        self.btn_gen.setEnabled(True)
        self.btn_regen.setEnabled(True if self.last_song is not None else False)
        # STOP availability is managed by playback poller
        if self.preview_wav:
            self.btn_play.setEnabled(True)
        self._set_status("ready")
        self._set_progress(100)

    def _on_play(self):
        if not self.preview_wav:
            return
        try:
            self.playback.play_wav_bytes(self.preview_wav)
            self._set_status("playing")
            # UI state (STOP usable during playback)
            self.btn_stop.setEnabled(True)
            self.btn_play.setEnabled(False)
        except Exception as e:
            self._append_log(f"[play] {e}")

    def _on_stop(self):
        # Stop playback
        try:
            self.playback.stop()
        except Exception:
            pass

        # Cancel generation if running
        if self.worker and self.worker.isRunning():
            try:
                self.worker.request_cancel()
            except Exception:
                pass

        # Update UI state
        self._set_status("ready")
        if self.preview_wav:
            self.btn_play.setEnabled(True)
        # STOP should only remain enabled if generation is still running
        self.btn_stop.setEnabled(bool(self.worker and self.worker.isRunning()))

    # ---------- pattern preview ----------
    def _update_pattern_preview(self):
        if self.last_song is None:
            self.pattern_text.setPlainText("")
            return
        try:
            idx = int(self.pattern_sel.currentText())
        except Exception:
            idx = 0
        try:
            pat = self.last_song.patterns[idx]
        except Exception:
            self.pattern_text.setPlainText("")
            return
        try:
            order = list(getattr(self.last_song, 'order', []) or getattr(self.last_song, 'order_original', []) or [])
            poss = order_positions_for_pattern(order, idx)
        except Exception:
            poss = []
        self.pattern_text.setPlainText(format_pattern(pat, idx, poss))

    # ---------- samples ----------
    def _refresh_sample_display(self):
        if not hasattr(self, 'sample_rows'):
            return
        # show instruments + status (custom/generated) + base volume
        kinds = [c.currentText() for c in getattr(self, 'inst', [])] if hasattr(self, 'inst') else ["Piano"]*4
        for ch in range(4):
            row = self.sample_rows[ch]
            inst_kind = kinds[ch] if ch < len(kinds) else f"CH{ch+1}"
            row["inst"].setText(inst_kind)
            custom = self.cfg.custom_sample_paths.get(ch)
            row["status"].setText("Custom" if custom else "Generated")
            try:
                vol = int(getattr(backend, 'INSTRUMENT_VOL', {}).get(inst_kind, 48))
            except Exception:
                vol = 48
            row["vol"].setText(str(vol))
        self.samples_info.setPlainText(
            "Sample Manager\n\n"
            "- Generated: procedural instrument\n"
            "- Custom: user WAV\n\n"
            "Replace: set a custom WAV for this channel.\n"
            "Reset: remove custom WAV.\n"
            "Play: preview immediately (even without generating a song).\n"
        )

    def _play_sample(self, ch: int):
        # custom path first
        p = self.cfg.custom_sample_paths.get(ch)
        if p:
            wavb = self.sample_engine.preview_wav_from_custom_path(p)
            if wavb:
                self.playback.play_wav_bytes(wavb)
                self._append_log(f"Playing CH{ch+1} custom sample: {Path(p).name}")
                return

        # use current instrument selection (works without song)
        inst_kind = self.inst[ch].currentText() if ch < len(self.inst) else "Piano"
        is_drum = False
        drum_style = "Kick"
        try:
            st = backend.drumset_style_from_kind(inst_kind)
            if st:
                is_drum = True
                drum_style = "Kick"
        except Exception:
            pass

        spec = SamplePreviewSpec(
            instrument_kind=inst_kind,
            disable_vibrato=bool(self.opt_disable_vibrato.isChecked()),
            seed=(int(self.seed_edit.text().strip() or backend.random_seed_value()) ^ (ch*1337)),
            is_drum=is_drum,
            drum_style=drum_style,
        )
        try:
            wavb = self.sample_engine.preview_wav_for(spec)
            self.playback.play_wav_bytes(wavb)
            self._append_log(f"Playing CH{ch+1}: {inst_kind}")
        except Exception as e:
            self._append_log(f"Play sample error: {e}")

    def _replace_sample(self, ch: int):
        path, _ = QFileDialog.getOpenFileName(self, f"Import WAV for CH{ch+1}", "", "WAV files (*.wav);;All files (*.*)")
        if not path:
            return
        self.cfg.custom_sample_paths[int(ch)] = str(path)
        self._refresh_sample_display()
        self._append_log(f"CH{ch+1}: Imported custom sample: {Path(path).name}")

    def _reset_sample(self, ch: int):
        if int(ch) in self.cfg.custom_sample_paths:
            self.cfg.custom_sample_paths.pop(int(ch), None)
        self._refresh_sample_display()
        self._append_log(f"CH{ch+1}: Reset to generated sample")

    def _import_all_samples(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import WAV files", "", "WAV files (*.wav);;All files (*.*)")
        if not paths:
            return
        for i, p in enumerate(paths[:4]):
            self.cfg.custom_sample_paths[int(i)] = str(p)
        self._refresh_sample_display()
        self._append_log(f"Imported {min(4,len(paths))} sample(s).")

    # ---------- theme ----------
    def _apply_theme(self):
        name = self.theme_combo.currentText()
        th = self.themes.get(name)
        if th:
            self.apply_theme_cb(th)
            self._append_log(f"Theme applied: {name}")

    def _on_tab_changed(self, idx: int):
        # Keep Samples tab labels consistent even if user never touched instrument combobox after load
        try:
            self._refresh_sample_display()
        except Exception:
            pass

    def _poll_playback_state(self):
        """Keep UI buttons consistent while audio is playing (Qt or subprocess)."""
        try:
            playing = bool(self.playback.is_playing())
        except Exception:
            playing = False

        gen_running = bool(self.worker and self.worker.isRunning())

        # STOP is available if playing OR generating (worker running)
        self.btn_stop.setEnabled(bool(playing or gen_running))

        # PLAY enabled only when we have a preview and not currently playing
        if self.preview_wav:
            self.btn_play.setEnabled(not playing)
        else:
            self.btn_play.setEnabled(False)

        # Auto-switch status back to ready when playback ends
        try:
            was = bool(getattr(self, "_was_playing", False))
        except Exception:
            was = False
        if was and (not playing) and (not gen_running):
            self._set_status("ready")
        self._was_playing = playing


