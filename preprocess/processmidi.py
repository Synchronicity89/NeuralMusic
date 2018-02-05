import mido
import numpy as np
np.set_printoptions(threshold=np.nan)

mid = mido.MidiFile('MIDI/littlered.mid')
print(mid.type)

# Ranges where midi guitars and basses are
BASS_TRACKS = range(33, 41)
GUITAR_TRACKS = range(25, 33)

TICKS_PER_BEAT_STANDARD = 480

#Limits for quantisizing
QUANTISIZE_LIMIT = 0.2

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
        #track.append(mido.Message('program_change', program=27, time=0)) #remove this. Only for testing data
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

def printTrackInfo(track):
        for i, msg in enumerate(track):
            if (msg.type is 'note_on' or msg.type is 'note_off'):
                print(msg)

def createMIDITest(track):
        mid = mido.MidiFile()
        mid.tracks.append(track)
        mid.save('testsong.mid')

def prepareMIDITrackForPreprocessing(track, dictionary):
    for i, msg in enumerate(track):
        if (msg.type is 'note_on'):
            quant = quantize(msg, dictionary)
            print(quant)
            if(msg.velocity == 0):
                track[i] = mido.Message('note_off', note=msg.note, time=msg.time, channel=msg.channel)
            else:
                track[i] = mido.Message('note_on', note=msg.note, time=quant.time, channel=msg.channel)
    return track

def getTempo(mid):
    for i, msg in enumerate(mid.tracks[0]):
        print(msg.type)
        if (msg.type == 'set_tempo'):
            return msg.tempo




''' ---- Preprocessing -----'''

def setNoteDict(ticks_per_beat):
    NOTE_DICT = {'1/1': 1920, '1/2': 960, '1/2t': 640, '1/4': 480, '1/4t': 320, '1/8': 240, '1/8t': 160, '1/16': 120}
    for key, value in NOTE_DICT.items():
        factor = ticks_per_beat/TICKS_PER_BEAT_STANDARD
        newConstant = value * factor
        newValue = range(int(newConstant - (newConstant * QUANTISIZE_LIMIT)), int(newConstant + (newConstant * QUANTISIZE_LIMIT))), newConstant
        NOTE_DICT[key] = newValue
    return NOTE_DICT


def quantize(msg, NOTE_DICT):
    lower_lim = NOTE_DICT['1/16'][0][0] # [Last element][Range in dict][Lowest value in range]
    for key, value in NOTE_DICT.items():
        if msg.time in value[0]:
            return msg
    if msg.time < lower_lim:
        print("Fixing zero value: " + str(msg.time))
        msg.time = 0
        return msg
    else:
        return msg


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

def preprocessMIDITrack(track, ticks_per_beat):
    notesNotCompleted = []
    timestepSize = TICKS_PER_BEAT_STANDARD * (ticks_per_beat/TICKS_PER_BEAT_STANDARD)
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
    print(finalMatrix.shape)


t = getGuitarTracks(mid)
dictionary = setNoteDict(mid.ticks_per_beat)
#preprocessMIDITrack(t[0])
track = prepareMIDITrackForPreprocessing(t[0], dictionary)
createMIDITest(track)
