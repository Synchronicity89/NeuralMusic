from ngram import NGram
import editdistance
import sys
sys.path.insert(0, '../neuralnetwork/')
sys.path.insert(0, '../preprocess/')
import mido
import progressbar
import os
import numpy as np
from selecttrack import SelectTrack
from processmidi import ProcessMIDI

'''Calculate N-gram score and edit distance'''

class NGramCompare:
    def __init__(self):
        super(NGramCompare, self).__init__()
        self.TICKS_PER_BEAT_STANDARD = 480

    def importFiles(self, filename):
        trackSelector = SelectTrack()
        pp = ProcessMIDI()
        directory = '../neuralnetwork/bluessolodata'
        songdata = self.loadData(directory)
        track, ticks_per_beat = trackSelector.prepareCleanedMIDIFiles(filename)
        prepared = pp.prepareMIDITrackForPreprocessing(track, ticks_per_beat, self.TICKS_PER_BEAT_STANDARD) # If noteoff = noteon with 0 velocity, fix this and quantize track
        struct = pp.preprocessMIDITrack(prepared, ticks_per_beat, self.TICKS_PER_BEAT_STANDARD)
        return songdata, struct

    def loadData(self, directory):
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
                                if (np.nonzero(timestep)[0][0] != 128):
                                    newSong.append(timestep)
                        #currentSong = currentSong[0:song_length] # Making song shorter
                        newSong = np.array(newSong)
                        songdata.append(newSong)
                        i = i + 1
                        bar.update(i)
            return songdata

    def translateToString(self, solo):
        string = ""
        for cell in solo:
            indices = np.nonzero(cell)
            for index in indices:
                string += str(index)
        return string

if __name__ == '__main__':
    n = NGramCompare()
    solos, generatedSolo = n.importFiles('Recreated 2, 100 epochs, learning rate = 0.0055.mid')
    trans = []
    for solo in solos:
        trans.append(n.translateToString(solo))
    s1 = n.translateToString(generatedSolo)
    ngramScores = []
    editdistances = []
    for solo in trans:
        ngramScores.append(NGram.compare(s1, solo))
        editdistances.append(editdistance.eval(s1, solo))
    avgngram = 0
    avgedit = 0
    for g in ngramScores:
        avgngram += g
    avgngram = avgngram / len(ngramScores)
    for dist in editdistances:
        avgedit += dist
    avgedit = avgedit / len(editdistances)
    ngramScores.sort(reverse=True)
    editdistances.sort()
    print('Average n-gram score: ' + str(avgngram))
    print('Average edit-distance score: ' + str(avgedit))
    print('Best n-gram score: ' + str(ngramScores[0]))
    print('Best edit distance score: ' + str(editdistances[0]))
