import mido
import numpy as np
import os
from selecttrack import SelectTrack
from processmidi import ProcessMIDI

directory = '../MIDI'
tracksToPreprocess = []
trackSelector = SelectTrack()
TICKS_PER_BEAT_STANDARD = 480
out_directory = 'data/'

'''Select tracks to preprocess'''
for filename in os.listdir(directory):
    if filename.endswith(".mid"):
        track, ticks_per_beat = trackSelector.selectTrackFromMIDIFile(directory+'/'+filename)
        if track is not None:
            filename = os.path.splitext(filename)[0] #remove .mid ending
            tracksToPreprocess.append((track, ticks_per_beat, filename))
        else:
            print(filename + ' Did not have any appropriate instruments')
    else:
        print("Error while loading data. Could not find MIDI files")
        exit()

'''Prepare for preprocessing and preprocess'''
pp = ProcessMIDI()
for track, ticks_per_beat, filename in tracksToPreprocess:
    prepared = pp.prepareMIDITrackForPreprocessing(track, ticks_per_beat, TICKS_PER_BEAT_STANDARD) # If noteoff = noteon with 0 velocity, fix this and quantize track
    struct = pp.preprocessMIDITrack(track, ticks_per_beat, TICKS_PER_BEAT_STANDARD)
    pp.saveToTxt(struct, out_directory+filename)
print('done')
