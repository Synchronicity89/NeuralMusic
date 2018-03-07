import mido
import numpy as np
np.set_printoptions(threshold=np.nan)

class ProcessMIDI:
    def __init__(self):
        super(ProcessMIDI, self).__init__()

    '''----- Prepare track for preprocessing -----'''
    ''' Gets bass tracks '''
    def getBassTracks(mid):
        bassTracksFound = []
        for i, track in enumerate(mid.tracks):
            print('Track {}: {}'.format(i, track.name))
            for msg in track:
                if(msg.type is 'program_change'):
                    if(msg.program in BASS_TRACKS):
                        bassTracksFound.append(msg.program)
        return bassTracksFound

    ''' Gets guitar tracks '''
    def getGuitarTracks(self, mid):
        BASS_TRACKS = range(33, 41)
        GUITAR_TRACKS = range(25, 33)
        guitarTracksFound = []
        for i, track in enumerate(mid.tracks):
            #track.append(mido.Message('program_change', program=27, time=0)) #remove this. Only for testing data
            for msg in track:
                if(msg.type is 'program_change'):
                    if(msg.program in GUITAR_TRACKS):
                        guitarTracksFound.append(track)
        return guitarTracksFound

    ''' Print all info from all tracks in midi'''
    def printAllInfo(self, mid):
        for i, track in enumerate(mid.tracks):
            print('Track {}: {}'.format(i, track.name))
            for msg in track:
                print(msg)

    def printTrackInfo(self, track):
            for i, msg in enumerate(track):
                if (msg.type is 'note_on' or msg.type is 'note_off'):
                    print(msg)

    def createMIDITest(self, track):
            mid = mido.MidiFile()
            mid.tracks.append(track)
            mid.save('check.mid')

    def prepareMIDITrackForPreprocessing(self, track, ticks_per_beat, TICKS_PER_BEAT_STANDARD):
        for i, msg in enumerate(track):
            if (msg.type is 'note_on'):
                quant = self.quantize(msg, ticks_per_beat, TICKS_PER_BEAT_STANDARD)
                if(msg.velocity == 0):
                    track[i] = mido.Message('note_off', note=msg.note, time=msg.time, channel=msg.channel)
                else:
                    track[i] = mido.Message('note_on', note=msg.note, time=quant.time, channel=msg.channel)
            if (msg.type is 'note_off'):
                quant = self.quantize(msg, ticks_per_beat, TICKS_PER_BEAT_STANDARD)
                track[i] = mido.Message('note_off', note=msg.note, time=quant.time, channel=msg.channel)
        return track

    def getTempo(self, mid):
        for i, msg in enumerate(mid.tracks[0]):
            print(msg.type)
            if (msg.type == 'set_tempo'):
                return msg.tempo




    ''' ---- Preprocessing -----'''

    def quantize(self, msg, ticks_per_beat, TICKS_PER_BEAT_STANDARD):
        factor = 120 / (TICKS_PER_BEAT_STANDARD/ticks_per_beat) # 120 is standard 16th note
        mod = msg.time % factor
        if (mod > (factor / 2)):
            msg.time = int(msg.time + (factor - mod))
        else:
            msg.time = int(msg.time - mod)
        return msg

    def processTimestep(self, msg, finalMatrix, notesNotCompleted, timestepSize, sublist):
        '''For note_on: Add to notesNotCompleted list and add matrix to finalMatrix'''
        if (msg.type is 'note_on'):
            if(self.checkForChord(sublist)):
                notesInChord = self.getNotesInChords(sublist)
                return self.insertChord(finalMatrix, msg.note, notesInChord, notesNotCompleted)

            else:
                return self.insertNote(finalMatrix, msg.note, notesNotCompleted)
        '''For note_off: Remove from notesNotCompleted list'''
        if (msg.type is 'note_off'):
            removeNotes, updateIndexWithFactor = self.addToNotesNotCompleted(notesNotCompleted, sublist)
            removeNotes.append(msg.note) # add inital note
            for n in removeNotes:
                for index, note in enumerate(notesNotCompleted):
                    if (n == note):
                        del notesNotCompleted[index]
                        break
        return finalMatrix, notesNotCompleted, updateIndexWithFactor

    def insertEmpty(self, finalMatrix):
        b = np.zeros(shape=(1, 130))
        b.itemset((0, 128), 1) #can not concatenate with zero matrix
        newMatrix = np.concatenate([finalMatrix, b])
        newMatrix.itemset((len(newMatrix)-1, 128), 0) #change back to zero matrix
        return newMatrix

    def insertPause(self, finalMatrix):
        b = np.zeros(shape=(1, 130))
        b.itemset((0, 128), 1) #can not concatenate with zero matrix
        newMatrix = np.concatenate([finalMatrix, b])
        return newMatrix

    def insertHold(self, finalMatrix, notesNotCompleted):
        b = np.zeros(shape=(1, 130))
        for note in notesNotCompleted:
            b.itemset((0, note), 1)
        b.itemset((0, 129), 1) #can not concatenate with zero matrix
        newMatrix = np.concatenate([finalMatrix, b])
        return newMatrix

    def addToNotesNotCompleted(self, notesNotCompleted, msglist):
        notesToBeRemoved = []
        updateIndexWithFactor = 0
        for i, msg in enumerate(msglist):
            if (msg.type == 'note_off' and msg.time == 0):
                notesToBeRemoved.append(msg.note)
                updateIndexWithFactor = i
            else: break
        return notesToBeRemoved, updateIndexWithFactor

    def insertNote(self, finalMatrix, note, notesNotCompleted):
        b = np.zeros(shape=(1, 130))
        b.itemset((0, note), 1)
        notesNotCompleted.append(note)
        if (finalMatrix is None):
            newMatrix = b
        else:
            newMatrix = np.concatenate([finalMatrix, b])
        return newMatrix, notesNotCompleted, 0 # zero because no update in timestep

    def insertChord(self, finalMatrix, note, notesInChord, notesNotCompleted):
        updateIndexWithFactor = 0
        b = np.zeros(shape=(1, 130))
        b.itemset((0, note), 1)
        notesNotCompleted.append(note)
        for note in notesInChord:
            b.itemset((0, note), 1)
            notesNotCompleted.append(note)
            updateIndexWithFactor = updateIndexWithFactor + 1
        if (finalMatrix is None):
            newMatrix = b
        else:
            newMatrix = np.concatenate([finalMatrix, b])
        return newMatrix, notesNotCompleted, updateIndexWithFactor

    def getNotesInChords(self, msglist):
        chordList = []
        updateIndexWithFactor = 0
        for i, msg in enumerate(msglist):
            if (msg.type == 'note_on' and msg.time == 0):
                chordList.append(msg.note)
            else: break
        return chordList

    def checkForChord(self, msglist):
        for i, msg in enumerate(msglist):
            #print(msg)
            if (msg.type == 'note_on' and msg.time == 0):
                return True
            else:
                return False


    def preprocessMIDITrack(self, track, ticks_per_beat, TICKS_PER_BEAT_STANDARD):
        notesNotCompleted = []
        timestepSize = 120 / (TICKS_PER_BEAT_STANDARD/ticks_per_beat)
        finalMatrix = np.ones((1, 130))
        currentTimestep = 0
        forwardChordCheck = 3
        updateIndexWithFactor = 0

        for i, msg in enumerate(track):
            if updateIndexWithFactor != 0:
                updateIndexWithFactor = updateIndexWithFactor - 1
                continue
            if (msg.type == 'note_on' or msg.type == 'note_off'):
                if (msg.time == 0):
                    #check for chord here
                    finalMatrix, notesNotCompleted, updateIndexWithFactor = self.processTimestep(msg, finalMatrix, notesNotCompleted, timestepSize, track[i+1:i+forwardChordCheck])
                else:
                    check = msg.time - timestepSize
                    while (check != 0):
                        currentTimestep = currentTimestep + 1
                        if not finalMatrix.all():
                            if not notesNotCompleted: # if pause
                                finalMatrix = self.insertPause(finalMatrix)
                            else: # if hold
                                finalMatrix = self.insertHold(finalMatrix, notesNotCompleted)
                        check = check - timestepSize
                    if (check == 0):
                        currentTimestep = currentTimestep + 1
                        finalMatrix, notesNotCompleted, updateIndexWithFactor = self.processTimestep(msg, finalMatrix, notesNotCompleted, timestepSize, track[i+1:i+forwardChordCheck])
                        if (msg.type is 'note_off'):
                            if not notesNotCompleted: # if pause
                                finalMatrix = self.insertPause(finalMatrix)
                            else:
                                finalMatrix = self.insertHold(finalMatrix, notesNotCompleted)
                    else:
                        print('Error: Timesteps do not add up')
        return finalMatrix[1:len(finalMatrix)]


    def saveToTxt(self, array, filename):
        np.savetxt(filename + '.txt', array)


    '''----- recreate song from neural network structure -------'''
    def noteOnMsg(self, noteIn, timeIn):
        return mido.Message('note_on', note=noteIn, time=timeIn, channel=1)

'''
if __name__ == '__main__':
    mid = mido.MidiFile('../MIDI/unchain.mid')
    TICKS_PER_BEAT_STANDARD = 480
    pp = ProcessMIDI()
    t = pp.getGuitarTracks(mid)
    track = pp.prepareMIDITrackForPreprocessing(t[0], mid.ticks_per_beat, TICKS_PER_BEAT_STANDARD) # If noteoff = noteon with 0 velocity, fix this and quantize track
    #struct = pp.preprocessMIDITrack(track, mid.ticks_per_beat, TICKS_PER_BEAT_STANDARD)
'''
