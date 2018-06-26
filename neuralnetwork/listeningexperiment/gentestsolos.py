import sys
sys.path.insert(0, '../../neuralnetwork/')
sys.path.insert(0, '../../preprocess/')
sys.path.insert(0, '../../postprocess/')
import mido
import progressbar
import os
import numpy as np
from selecttrack import SelectTrack
from processmidi import ProcessMIDI
import recreate

'''Preprocess real MIDI solos to remove pitch bending and quantize data'''

def loadData(directory):
    filenames = []
    songdata = []
    i = 0
    with progressbar.ProgressBar(max_value=len(os.listdir(directory))) as bar:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                currentSong = np.loadtxt(directory+'/'+filename)
                newSong = []
                if (len(currentSong > 0)):
                    for timestep in currentSong:
                        if (len(np.nonzero(timestep)) > 0):
                            newSong.append(timestep)
                        #currentSong = currentSong[0:song_length] # Making song shorter
                    newSong = np.array(newSong)
                    songdata.append(newSong)
                    filenames.append(filename)
                    i = i + 1
                    bar.update(i)
        return songdata, filenames

directory = '../newlisteningexperiment'
rec = recreate.RecreateMIDI()
songdata, filenames = loadData(directory)
for index, song in enumerate(songdata):
    track = rec.recreateMIDI(song, 30)
    rec.createMIDITest(track, filenames[index])
