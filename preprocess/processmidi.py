import mido
import numpy as np
np.set_printoptions(threshold=np.nan)

mid = mido.MidiFile('../MIDI/littlered.mid')
print(mid.type)

# Ranges where midi guitars and basses are
BASS_TRACKS = range(33, 41)
GUITAR_TRACKS = range(25, 33)

TICKS_PER_BEAT_STANDARD = 480


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

def prepareMIDITrackForPreprocessing(track, ticks_per_beat):
    for i, msg in enumerate(track):
        if (msg.type is 'note_on'):
            quant = quantize(msg, ticks_per_beat)
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

def quantize(msg, ticks_per_beat):
    factor = 120 / (TICKS_PER_BEAT_STANDARD/ticks_per_beat) # 120 is standard 16th note
    mod = msg.time % factor
    if (mod > (factor / 2)):
        msg.time = int(msg.time + (factor - mod))
    else:
        msg.time = int(msg.time - mod)
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
    timestepSize = 120 / (TICKS_PER_BEAT_STANDARD/ticks_per_beat)
    finalMatrix = np.ones((1, 130))
    currentTimestep = 0

    for i, msg in enumerate(track):
        print(msg)
        if (msg.type == 'note_on' or msg.type == 'note_off'):
            if (msg.time == 0):
                finalMatrix, notesNotCompleted = processTimestep(msg, finalMatrix, notesNotCompleted, timestepSize)
            else:
                check = msg.time - timestepSize
                while (check != 0):
                    currentTimestep = currentTimestep + 1
                    if not finalMatrix.all():
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
track = prepareMIDITrackForPreprocessing(t[0], mid.ticks_per_beat) # If noteoff = noteon with 0 velocity, fix this and quantize track
preprocessMIDITrack(track, mid.ticks_per_beat)
#createMIDITest(track)
