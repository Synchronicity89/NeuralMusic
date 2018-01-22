from music21 import *

def noteHandler(timestep):
    m21class = str(timestep.__class__)
    # If timestep is a chord
    if (m21class == '<class \'music21.chord.Chord\'>'):
        l = []
        for p in timestep.pitches: #iterates through all notes and converts to right pitch
            l.append(p.ps)
        t = {'class': 'chord', 'pitches': l, 'duration': timestep.duration} #Adds list of pitches in chord and duration to a dict.
        return t
    if (m21class == '<class \'music21.note.Rest\'>'):
        t = {'class1': 'rest', 'pitches': [], 'duration': timestep.duration}

    if (m21class == '<class \'music21.note.Note\'>'):
        l = []
        for p in timestep.pitches: # Find pitch of note
            l.append(p.ps)
        t = {'class': 'note', 'pitches': l, 'duration': timestep.duration}
        print(t)


s = converter.parse('MIDI/20th.mid')
s2 = instrument.partitionByInstrument(s)
#s2 = instrument.unbundleInstruments(s)

for p in s2.parts:
    unused = p.makeRests(fillGaps=True, inPlace=True)

#s2.show('text')
l = []
for i in s2:
    instrument = []
    for timestep in i:
        instrument.append(noteHandler(timestep))
    l.append(instrument)
    print(l)
