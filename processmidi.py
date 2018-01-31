import mido
import numpy as np
np.set_printoptions(threshold=np.nan)

mid = mido.MidiFile('MIDI/test.mid')
print(mid.type)

# Ranges where midi guitars and basses are
BASS_TRACKS = range(33, 41)
GUITAR_TRACKS = range(25, 33)

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
def getGuitarTracks(mid):
    guitarTracksFound = []
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        track.append(mido.Message('program_change', program=27, time=0)) #remove this. Only for testing data
        for msg in track:
            if(msg.type is 'program_change'):
                if(msg.program in GUITAR_TRACKS):
                    guitarTracksFound.append(track)
    return guitarTracksFound

''' Print all info from all tracks in midi'''
def printAllInfo(mid):
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)

def processTimestep(msg, finalMatrix, notesNotCompleted, timestepSize):
    '''For note_on: Add to notesNotCompleted list and add matrix to finalMatrix'''
    if (msg.type is 'note_on'):
        b = np.zeros(shape=(1, 130))
        b.itemset((0, msg.note), 1)
        notesNotCompleted.append(msg.note)
        if (finalMatrix is None):
            newMatrix = b
        else:
            newMatrix = np.concatenate([finalMatrix, b])
        return newMatrix, notesNotCompleted
    '''For note_off: Remove from notesNotCompleted list'''
    if (msg.type is 'note_off'):
        for index, note in enumerate(notesNotCompleted):
            if (msg.note == note):
                del notesNotCompleted[index]
    return finalMatrix, notesNotCompleted

def insertEmpty(finalMatrix):
    b = np.zeros(shape=(1, 130))
    b.itemset((0, 128), 1) #can not concatenate with zero matrix
    newMatrix = np.concatenate([finalMatrix, b])
    newMatrix.itemset((len(newMatrix)-1, 128), 0) #change back to zero matrix
    return newMatrix

def insertPause(finalMatrix):
    b = np.zeros(shape=(1, 130))
    b.itemset((0, 128), 1) #can not concatenate with zero matrix
    newMatrix = np.concatenate([finalMatrix, b])
    return newMatrix

def insertHold(finalMatrix, note):
    b = np.zeros(shape=(1, 130))
    b.itemset((0, note), 1)
    b.itemset((0, 129), 1) #can not concatenate with zero matrix
    newMatrix = np.concatenate([finalMatrix, b])
    return newMatrix

def preprocessMIDITrack(track):
    notesNotCompleted = []
    timestepSize = 120
    finalMatrix = None
    currentTimestep = 0

    for i, msg in enumerate(track):
        if (msg.time == 0):
            finalMatrix, notesNotCompleted = processTimestep(msg, finalMatrix, notesNotCompleted, timestepSize)
        else:
            check = msg.time - timestepSize
            while (check != 0):
                currentTimestep = currentTimestep + 1
                for note in notesNotCompleted: # if notes are hold notes
                    finalMatrix = insertHold(finalMatrix, note)
                if not notesNotCompleted: # if pause
                    finalMatrix = insertPause(finalMatrix)
                check = check - timestepSize
            if (check == 0):
                currentTimestep = currentTimestep + 1
                finalMatrix, notesNotCompleted = processTimestep(msg, finalMatrix, notesNotCompleted, timestepSize)
                if (msg.type is 'note_off'):
                    for note in notesNotCompleted: # if notes are hold notes
                        finalMatrix = insertHold(finalMatrix, note)
                    if not notesNotCompleted: # if pause
                        finalMatrix = insertPause(finalMatrix)
            else:
                print('Error: Timesteps do not add up')
    print(finalMatrix)
    #print(finalMatrix.shape)

t = getGuitarTracks(mid)
preprocessMIDITrack(t[0])
