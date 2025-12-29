# protracker_mod_choral_generator
Generate Amiga Protracker compatible .mod files in church/choral style.

https://github.com/user-attachments/assets/d56f4303-fdbf-4ab1-bbb6-24efda2a6a44

https://github.com/user-attachments/assets/2c7fbb9c-8603-4fca-8788-cfaf6b71d2b0

Just start it. After a moment you can find a new song (each run different melody) in the output folder (where the script is).

Btw. not every song is great but most of them. ;-)
<br><br>

<b>Changelog:</b>

<i><b>New in Commit d424a19 (latest):</b></i>


- Generates more random songnames
  

<i><b>New in Commit 5f9ec5e:</b></i>

Added (optional) GUI for order editing and generation options (the GUI is disabled by default).

 Use "-gui" commandline parameter to use a GUI
 
<img width="461" height="265" alt="grafik" src="https://github.com/user-attachments/assets/07a3e5ca-1996-4507-aa57-1c1acf22af59" />

 Use "-noslowdown" commandline parameter to disable the slowdown to the song ending.

 Note: I have extended the default Pattern order a bit but you can change this back in GUI Mode to the previous order if you like.
 
 It was before "0, 1, 2, 3, 2, 4, 5" and now it is "0, 1, 2, 3, 2, 4, 1, 4, 2, 5" (makes each song a bit longer).
 
<br><br>

<b>Some Notes: </b>

The reason for some strange melodic songs is that i dont want the script to get uncreative at all, some uncommon notes are great in a specific context (just delete the worse songs).

How ever i would suggest to play the songs using the Protracker / Noisetracker / StarTrekker (by FLT) on a Amiga Computer or the Protracker 2 Clone / VLC using the Amiga Mod Player extension for Windows 10/11.

btw. i think the best way to just play the generated songs is to use the VLC (VideoLan) Mediaplayer because the generated Piano Sample is generated as a Chiptune like FM Synthesized Sample, most Trackers are interpreting it a bit strange, i use the Trackers for myself only to change the instrument but for pure playback VLC ist doing something a bit different and it sounds better with the original sample (also for a converting the generated mods to mp3).

Known bugs: It happens sometimes that most of the created songs have a strange melody (in gui mode) in that case just close the app and restart it (the reason is that the script have preselected "unbeauty" base-notes for the song generation - in that case the result is opposit to the default with strange sounds).
