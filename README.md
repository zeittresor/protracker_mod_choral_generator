# ProTracker MOD Generator — PyQt6 UI (modular)

This is a **PyQt6-based GUI** for the existing generator backend (`protracker_mod_choral_generator.py`), with the codebase split into smaller “engines” (`ptmod/engines/`) so the project can grow without turning into one giant file.

## What this UI includes (parity + upgrades)

- Left-side control panel (close to the original Tk UI):
  - Pattern order presets + **SMART** order generation
  - Melody plugins + derivation mode
  - Key root + speed + tempo
  - Scale/mode, variation, seed, batch, mute channels, stereo width, quality passes
  - Instrument selection (CH1..CH4) + octave span
  - Language dropdown (EN/DE/FR) for core UI text

- Right-side tabs:
  - **MAIN**: Spectrum/Scopes/Light Organ visualizer (click to toggle), render status, harmony score, log, pattern preview
  - **SAMPLES**: per-channel sample manager (Play/Replace/Reset) — **samples are always playable**, even before generating a song
  - **OPTIONS**:
    - Export WAV, Save parameters
    - Disable vibrato, Fade-out pattern, End slowdown
    - Compatibility mode + MOD signature
    - **Ralph-Loop** (retry until harmony+melody >= target; keeps best)
    - Theme selection
    - **FX Injection** (optional ProTracker effects, only in empty slots)
      - Insert initial speed/tempo (Fxx) at song start (improves tracker compatibility)
      - Vibrato / Portamento / Arpeggio / Volume motion + intensity slider

## Quick start (Windows)

Double-click:

- `run_windows.bat`

It creates a `.venv`, installs dependencies, then launches `app.py`.


## Quick start (Linux)

1) Make the script executable (once):

```bash
chmod +x run_linux.sh
```

2) Run:

```bash
./run_linux.sh
```

### Linux notes (audio + Qt)

- Playback prefers **QtMultimedia**. On many distros it needs **GStreamer plugins**.
- If QtMultimedia cannot play (missing plugins), the app falls back to common system players (**paplay / aplay / ffplay**) when available.

Typical packages:

**Debian/Ubuntu:**
```bash
sudo apt update
sudo apt install -y python3-venv gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav pulseaudio-utils alsa-utils
```

**Fedora:**
```bash
sudo dnf install -y python3-virtualenv gstreamer1-plugins-base gstreamer1-plugins-good gstreamer1-plugins-bad-free gstreamer1-libav pulseaudio-utils alsa-utils
```

If you get an error about the Qt platform plugin `xcb`, install your distro's Qt6 XCB dependencies (package names vary; often includes `libxcb-cursor0`, `libxkbcommon-x11-0`, etc.).

## Manual start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

## Themes

Built-in themes:
- ProTracker Gray (default)
- Modern Dark / Modern Light
- Amiga ECS / Amiga MUI

You can also add custom themes:
- Drop `.qss` files into `themes/`
- Restart the app; the theme appears in the dropdown

## Structure

- `ptmod/engines/` contains the growing “core” parts:
  - `generator_engine.py`, `quality_engine.py`, `effects_engine.py`, `mod_patch_engine.py`, `sample_engine.py`, etc.
- The long legacy backend stays in `protracker_mod_choral_generator.py` for now.

