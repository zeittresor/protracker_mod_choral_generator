# protracker_music_generator
Generate Audio Song files for Windows (and Amiga Protracker compatible) .mod files.

Just start it and click "Generate". After a moment you can find a new song (each run different melody) in the output folder (where the script is).

## Quick start (Windows)

Double-click:

- `run_windows.bat`

It creates a `.venv`, installs dependencies, then launches `app.py`.

Or use the latest windows executable release (takes a moment to start at first run).

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


<br>

<b>Changelog:</b>

<i><b>New in Commit e101607 (02.03.2026) v6.12):</b></i>

- Hotfix added for a issue with the compatibility with the PT2Clone Project (https://github.com/8bitbubsy/pt2-clone)
- Compiled Windows executable for Windows added (v6.12)

https://github.com/user-attachments/assets/3675bf17-e072-40a0-b47a-be7cb615518b

<i><b>New in Commit 9ad9c31 (01.03.2026) v6.10):</b></i>

- Added vertical scrollbars for the app window (for monitors with a lower resolution)
- App is starting maximized now
- Added Presets
- Fixed a bug for tracks with multiple drumsets

<img width="2560" height="1041" alt="grafik" src="https://github.com/user-attachments/assets/af0306e4-617a-4148-8a13-c7e9b8142c38" />

<i><b>New in Commit be9f20f (01.03.2026 v6.9):</b></i>

- Added Presets
- Added Option for Ralph Loop option to ignore Drumsets
- Help System added (in options tab) based upon the Amiga Guide fileformat (incl. viewer for Windows / Linux)
- Bug fixes
  
<img width="2560" height="1040" alt="69" src="https://github.com/user-attachments/assets/33512d9d-077e-4cb3-99f6-b6028010cfd4" />

https://github.com/user-attachments/assets/646c4fe5-f415-4d6b-82bf-0d11f93ca2a9

<i><b>New in Commit 825e40b (28.02.2026 v6.7):</b></i>

- Some Option tweaks
- Added additional visual themes
- Minor bug fixes

  Note: The classic Version 2.0 is still included (if you dont like "ralph"). To run it just open protracker_mod_choral_generator.py instead of run_linux.sh for Linux or run_windows.bat for Windows.

Example Theme 1:
<img width="2167" height="881" alt="68" src="https://github.com/user-attachments/assets/7dd0760b-6f07-44a7-8a03-94e4c2c41b80" />

Example Theme 2 (Amiga ECS look):
<img width="1122" height="973" alt="ecs" src="https://github.com/user-attachments/assets/4bfab02d-912b-45bb-92c4-abc2f063bcc5" />

https://github.com/user-attachments/assets/1482b2a6-1650-4628-b719-c22ac27fd454

<i><b>New in Commit 825e40b (28.02.2026 v6.6):</b></i>

- Major changes have been made like:
- Interface Switched from using TkInter to PyQt6
- Theme Support
- Main Program splitted into different engine modules
- Fx Settings added
- Multiple bases / combinations of classical songs as a seed base for the randomize function
- bug fixes
- Setting Ralph Loop as a default :-)

<img width="1126" height="950" alt="ralph66" src="https://github.com/user-attachments/assets/dc9ee13a-155d-4489-ab5e-6387700af61c" />

<i><b>New in Commit 865d1b3 (27.02.2026 v2.0.3):</b></i>

- Tabs added Song / Samples / Options
- Samples tab let you preview each generated sample, or you could replace it with your own samples
- Ralph-Loop added in options (to let the script try as hard as it can to find a most harmonic / melodic song for a single seed)
- Empty Pattern Option added to the end of each song to make sure its not just stopping if you dont want to loop it
- Added another visulizer like a classical disco light
- Added many more minor options

<img width="1408" height="893" alt="grafik" src="https://github.com/user-attachments/assets/9b6178ae-fca8-444f-9aec-81c2bef09b9d" />

<i><b>New in Commit ee77f2e (23.02.2026 v1.8.1):</b></i>

- Some work on the harmonize engine
- Basic drum sets for music styles added
- Bug fixes

  Note: It might be currently better to change the octave span in channel 3 from 3 to 2 .. there is a little bug in the routine.

<img width="1276" height="961" alt="181" src="https://github.com/user-attachments/assets/2ca455cb-86a3-4234-9e9c-462b703d4692" />

https://github.com/user-attachments/assets/81f24185-98aa-4c3e-8edb-e3b5e04f2bfd

<i><b>New in Commit da7fe2b (21.02.2026 v1.7.8.4):</b></i>

- Bug fixes
- Channel Scope switched to show both stereo channels for rendered playback
- Harmonizer extended
- Channel limiter added
- Minor improovments

<img width="1277" height="960" alt="1784" src="https://github.com/user-attachments/assets/7972a877-b40e-4670-a7e5-b39212e1dbe8" />

https://github.com/user-attachments/assets/4682c9cb-2307-4253-9b9e-a9f44ea215d5

<i><b>New in Commit c65a690 (21.02.2026 v1.7.5):</b></i>

- Major changes with additional features, tooltips, languages, plugins, instruments, options..
- Bug fixes

<img width="1043" height="935" alt="175" src="https://github.com/user-attachments/assets/57d22303-1bc4-4ea1-840d-595cdaa52d8e" />

<i><b>New in Commit 88944e6 (19.02.2026 v1.6.6):</b></i>

- Feature added to use a generated song as a synthetic base for new songs
- Improoved Melody generation

<img width="1045" height="714" alt="166" src="https://github.com/user-attachments/assets/1d43b087-8afe-488d-bf33-fff2a43afdd6" />

<i><b>New in Commit afbff2a (18.02.2026 v1.6.5):</b></i>

- Added Patterns
- Added Instuments
- Added Base Key (optional)
- Bug fixes / improovments

<img width="1042" height="709" alt="165" src="https://github.com/user-attachments/assets/7a35db1b-4da7-4da6-b048-55d95b14f53c" />


<i><b>New in Commit e8055ea (05.02.2026 v1.6.3):</b></i>

- Pattern Preview added
- Metadata and Info Text option added for Melody Plugins
- Buttons to open the Plugin and Output folders (useful for the next Windows executable release)
- Button added to refresh the Melody Plugin listing (for example if you add melody plugins while the app is still running)
- Bugfixes (Pure Random Melody and some other stuff)

<img width="1043" height="712" alt="v163" src="https://github.com/user-attachments/assets/3e67d973-f817-42d0-81fe-19d5e29864ad" />

https://github.com/user-attachments/assets/e1878c28-d29d-47fc-b04f-089daad467f8


<i><b>New in Commit bab5bc3 (03.02.2026 v1.6.2):</b></i>

- Plugin Library added (you can add now TXT and/or MID/MIDI Files for own base melodies
- Bugfixes and Improvements added

<img width="1044" height="714" alt="v162" src="https://github.com/user-attachments/assets/2544a37b-1825-4df6-9093-ca51e0999c03" />

https://github.com/user-attachments/assets/83a63deb-0982-46e5-91ad-fbc8d5eadbab


<i><b>New in Commit d9dd7f1 (02.02.2026 v1.5.1):</b></i>

- Selectable Base Song / Random
- Unchecked Options
- Bugfixes

<img width="1040" height="716" alt="151" src="https://github.com/user-attachments/assets/84ba2126-dd2d-485b-ba7a-9d30f1d80a8b" />


<i><b>New in Commit 856b706 (01.01.2026 v1.5.0):</b></i>


- Added option to switch between Spectrum Analyzer and Channel Scope (by clicking on it)
- Improved Spectrum Analyzer graphics
- Bug fixes

<img width="1044" height="715" alt="grafik" src="https://github.com/user-attachments/assets/b28ec03c-f78a-4f15-b878-383dcc877b29" />
<img width="1045" height="717" alt="optimisations" src="https://github.com/user-attachments/assets/24444b76-bdef-4e81-9308-2816c012f8c9" />


<i><b>New in Commit c27098e (31.12.2025 v1.4.9):</b></i>


- Improvements for the Spectrum Analyzer
- WAV export function added (optional)
- TXT file export with all parameters / patterns added (optional)
  
  Example: https://github.com/zeittresor/protracker_mod_choral_generator/blob/main/mods_out/A_funny_pope_move_to_at_poolparty___9366_20251231_120230_key_D2.txt
- Playback backend is running seperated from the main process now
- Bug fixes
- Compiled Windows Executable Release Version 1.4.9 added

<img width="1042" height="720" alt="latest31" src="https://github.com/user-attachments/assets/81283d8e-8462-450b-a44c-09d37365cadd" />


<i><b>New in Commit 04fffbd (31.12.2025 v1.4.1):</b></i>


- New interface with build-in spectrum analyser and play function (after rendering the Amiga Protracker output for your PC).

<img width="1128" height="468" alt="v1_4_1" src="https://github.com/user-attachments/assets/789faa16-8d99-4c87-a560-377b3f629d64" />


<i><b>New in Commit dd96d8b (30.12.2025 v1.3):</b></i>


- Added the option to select presets from the pattern order field as a pulldown menu (all of the so far testet pattern orders)
- Changed the slowdown effect to the last pattern (if enabled) instead of explicit pattern 5

<img width="477" height="392" alt="v1_3" src="https://github.com/user-attachments/assets/8278d9e9-cf6d-471d-ba08-e7d2d785d3b2" />


https://github.com/user-attachments/assets/7e936bdb-bee0-4873-adc5-1dc91365918b


<i><b>New in Commit 78653e3 (30.12.2025 v1.2):</b></i>


- Added some more instruments to choose (Acoustic Guitar, Flamenco Guitar, Organ, Flute, Oboe)
- Bugfix for the Instruments (all instruments use now the same reference-note to make them more harmonic to the other instruments)

<img width="513" height="384" alt="grafik" src="https://github.com/user-attachments/assets/c3764adc-b564-4c7f-8ce2-0362e4e4e3ca" />


<i><b>New in Commit 00cc274 (29.12.2025 v1.1):</b></i>


- Changed the Samples of each of the 4 Protracker channels to different Samples (even if you stay by Piano it will be a different Sample Number to make it easier to change it later).
- Added some different (generated) default instuments (selectable) for each Channel (Piano, Clarinet, Sax, Synth Pad, Violin, Tuba, Bajo, Panflute)

<img width="533" height="392" alt="sax" src="https://github.com/user-attachments/assets/a4a271db-2c13-4668-a5a0-34df8468c821" />


<i><b>New in Commit c6b427a (28.12.2025):</b></i>


- Changed default mode from CLI mode to GUI mode
- Changed the Commandlineoption -gui to -nogui to use the console only instead of the GUI
- Added Options in the GUI to change the BPM / Speed of the generated songs
- Added more CLI parameters to alter the bpm / speed using the console
- Some Tests done with different song/pattern orders like 5, 5, 1, 5, 0, 2, 3, 4, 2, 5, 0

<img width="461" height="265" alt="grafik" src="https://github.com/user-attachments/assets/07a3e5ca-1996-4507-aa57-1c1acf22af59" />


https://github.com/user-attachments/assets/d87156f5-4f35-45bd-b5b7-c26b0f24e083


<i><b>New in Commit d424a19 (27.12.2025):</b></i>


- Generates more random songnames
  

<i><b>New in Commit 5f9ec5e (26.12.2025):</b></i>

Added (optional) GUI for order editing and generation options (the GUI is disabled by default).

 Use "-gui" commandline parameter to use a GUI
 
 Use "-noslowdown" commandline parameter to disable the slowdown to the song ending.

 Note: I have extended the default Pattern order a bit but you can change this back in GUI Mode to the previous order if you like.
 
 It was before "0, 1, 2, 3, 2, 4, 5" and now it is "0, 1, 2, 3, 2, 4, 1, 4, 2, 5" (makes each song a bit longer).
 
<br><br>

<b>Some Notes: </b>

The reason for some strange melodic songs is that i dont want the script to get uncreative at all, some uncommon notes are great in a specific context (just delete the worse songs).

How ever i would suggest to play the songs using the Protracker / Noisetracker / StarTrekker (by FLT) on a Amiga Computer or the Protracker 2 Clone / VLC using the Amiga Mod Player extension for Windows 10/11.

btw. i think the best way to just play the generated songs is to use the VLC (VideoLan) Mediaplayer because the generated Piano Sample is generated as a Chiptune like FM Synthesized Sample, most Trackers are interpreting it a bit strange, i use the Trackers for myself only to change the instrument but for pure playback VLC ist doing something a bit different and it sounds better with the original sample (also for a converting the generated mods to mp3).

Known bugs: It happens sometimes that most of the created songs have a strange melody (in gui mode) in that case just close the app and restart it (the reason is that the script have preselected "unbeauty" base-notes for the song generation - in that case the result is opposit to the default with strange sounds).
