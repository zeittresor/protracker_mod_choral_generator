# protracker_mod_choral_generator
Generate Amiga Protracker compatible .mod files in church/choral/classic style.

Just start it and click "Generate". After a moment you can find a new song (each run different melody) in the output folder (where the script is).

<br>

<b>Changelog:</b>


<i><b>New in Commit 856b706 (latest / 01.01.2026 v1.5.0):</b></i>


- Added option to switch between Spectrum Analyzer and Channel Scope (by clicking on it)
- Improved Spectrum Analyzer graphics
- Bug fixes

<img width="1044" height="715" alt="grafik" src="https://github.com/user-attachments/assets/b28ec03c-f78a-4f15-b878-383dcc877b29" />


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
