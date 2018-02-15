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

    def convertList(self, notesNotCompleted):
        newList = []
        for n in notesNotCompleted:
            newList.append(n.note)
        return newList

    def removeFromNotCompleted(self, notesNotCompleted, notesToDrop):
        for note in notesToDrop:
            for i, n in enumerate(notesNotCompleted):
                if note == n.note:
                    del notesNotCompleted[i]
        return notesNotCompleted


    def recreateMIDI(self, array):
        timefactor = 120
        pause = 128
        hold = 129
        song = []
        notesNotCompleted = []
        timePast = 0;

        for x in array:
            messageArray = np.nonzero(x)[0]
            for note in messageArray:
                if(hold in messageArray): # Check if hold note
                    for index, n in enumerate(notesNotCompleted):
                        if (note == n.note and note != hold):
                            n.time = n.time + timefactor
                elif pause in messageArray:
                    timePast = timePast + timefactor
                    break
                else:
                    for index, n in enumerate(notesNotCompleted): # Check if note already exists in notes not complete. If so append to song and remove
                        if (note == n.note):
                            song.append(self.noteOffMsg(n.note, n.time))
                            del notesNotCompleted[index]
                            timePast = 0
                    newNote = Note(note, timePast)
                    notesNotCompleted.append(newNote)
                    song.append(self.noteOnMsg(newNote.note, newNote.time))
                    timePast = timePast + timefactor
            temp = self.convertList(notesNotCompleted)
            notesNotToDrop = np.intersect1d(messageArray, temp)
            notesToDrop = set(notesNotCompleted) - set(temp)
            for note in notesToDrop:
                song.append(self.noteOffMsg(note.note, note.time))
            notesNotCompleted = self.removeFromNotCompleted(notesNotCompleted, notesNotToDrop)
        for msg in song:
            print(msg)

if __name__ == '__main__':
    array = np.loadtxt("../neuralnetwork/testchord.txt")
    rec = RecreateMIDI()
    rec.recreateMIDI(array)
