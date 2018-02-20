import mido
import numpy as np
np.set_printoptions(threshold=np.nan)

class Note:
    def __init__(self, note, time):
        super(Note, self).__init__()
        self.note = note
        self.time = time

class RecreateMIDI:
    def __init__(self):
        super(RecreateMIDI, self).__init__()

    def noteOnMsg(self, noteIn, timeIn):
        return mido.Message('note_on', note=noteIn, time=timeIn, channel=1)

    def noteOffMsg(self, noteIn, timeIn):
        return mido.Message('note_off', note=noteIn, time=timeIn, channel=1)

    def endNotes(self, notesNotCompleted, timestep, song):
        for i, note in enumerate(notesNotCompleted):
            song.append(self.noteOffMsg(note.note, timestep))
            del notesNotCompleted[i]
            timestep = 0
        return notesNotCompleted, timestep, song

    def checkIfAHoldShouldEnd(self, notesNotCompleted, messageArray, timestep, song):
        comparableList = []
        for note in notesNotCompleted:
            comparableList.append(note.note)
        notesToBeEnded = set(comparableList) - set(messageArray)
        if len(notesToBeEnded) > 0:
            for note in notesToBeEnded:
                song.append(self.noteOffMsg(note, timestep))
                timestep = 0
                for i, n in enumerate(notesNotCompleted):
                    if n.note == note:
                        del notesNotCompleted[i]
        return notesNotCompleted, song, timestep


    def recreateMIDI3(self, array):
        timefactor = 120
        pause = 128
        hold = 129
        song = []
        notesNotCompleted = []
        timestep = 0
        pauses = 0

        for x in array:
            messageArray = np.nonzero(x)[0]
            if (hold in messageArray):
                notesNotCompleted, song, timestep = self.checkIfAHoldShouldEnd(notesNotCompleted, messageArray, timestep, song)
                timestep = timestep + timefactor
            elif (pause in messageArray):
                if len(notesNotCompleted) > 0:
                    notesNotCompleted, timestep, song = self.endNotes(notesNotCompleted, timestep, song)
                timestep = timestep + timefactor
            else:
                if len(notesNotCompleted) > 0:
                    notesNotCompleted, timestep, song = self.endNotes(notesNotCompleted, timestep, song)
                for msg in messageArray:
                    note = Note(msg, timestep)
                    song.append(self.noteOnMsg(note.note, note.time))
                    notesNotCompleted.append(note)
                    timestep = 0
                timestep = timestep + timefactor
        for msg in song:
            print(msg)
        return song

    def addTrackHeader(self, track):
        header = mido.Message('program_change', program=27, time=0)
        return [header] + track

    def createMIDITest(self, track):
            mid = mido.MidiFile()
            track = self.addTrackHeader(track)
            mid.tracks.append(track)
            mid.save('recreated.mid')



if __name__ == '__main__':
    array = np.loadtxt("../neuralnetwork/testchord.txt")
    rec = RecreateMIDI()
    track = rec.recreateMIDI3(array)
    rec.createMIDITest(track)
