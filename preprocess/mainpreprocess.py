import mido
import numpy as np
import os
import time
from selecttrack import SelectTrack
from processmidi import ProcessMIDI
import progressbar

directory = '../Test-MIDI'
tracksToPreprocess = []
trackSelector = SelectTrack()
TICKS_PER_BEAT_STANDARD = 480
out_directory = '../neuralnetwork/onesong'

'''Select tracks to preprocess'''
success = 0
fail = 0
t0 = time.time()

i = 0
with progressbar.ProgressBar(max_value=len(os.listdir(directory))) as bar:
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            #print(directory+'/'+filename)
            #track, ticks_per_beat = trackSelector.selectTrackFromMIDIFile2(directory+'/'+filename)
            track, ticks_per_beat = trackSelector.prepareCleanedMIDIFiles(directory+'/'+filename)
            if track is not None:
                filename = os.path.splitext(filename)[0] #remove .mid ending
                tracksToPreprocess.append((track, ticks_per_beat, filename))
                success = success + 1
            else:
                fail = fail + 1
                #print(filename + ' Did not have any appropriate instruments')
            i = i + 1
            bar.update(i)
t1 = time.time()
print('****** FINISHED PREPARING ******')
print('Runtime: ' + str(t1-t0))
print("Sucessfully prepared " + str(success) + " MIDI files for preprocessing")
print("Failed to prepare " + str(fail) + " MIDI files for preprocessing")
#exit()
'''Prepare for preprocessing and preprocess'''
pp = ProcessMIDI()
i = 0
with progressbar.ProgressBar(max_value=len(tracksToPreprocess)) as bar:
    for track, ticks_per_beat, filename in tracksToPreprocess:
        prepared = pp.prepareMIDITrackForPreprocessing(track, ticks_per_beat, TICKS_PER_BEAT_STANDARD) # If noteoff = noteon with 0 velocity, fix this and quantize track
        struct = pp.preprocessMIDITrack(track, ticks_per_beat, TICKS_PER_BEAT_STANDARD)
        pp.saveToTxt(struct, out_directory+filename)
        i = i + 1
        bar.update(i)
    print('****** FINISHED PREPROCESSING ******')
