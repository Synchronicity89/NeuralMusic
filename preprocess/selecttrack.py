import mido
import numpy as np
np.set_printoptions(threshold=np.nan)

class Filters:
    def __init__(self):
        super(Filters, self).__init__()
        self.GUITAR_BASS_TRACKS = range(25, 41)
        self.BRASS_REED_PIPE_TRACKS = range(57, 80)

    def instrumentFilter(self, track):
        for msg in track:
            if(msg.type is 'program_change'):
                if(msg.program in self.GUITAR_BASS_TRACKS or msg.program in self.BRASS_REED_PIPE_TRACKS):
                    return track
        return None

    def countNotes(self, track):
        uniqueNotes = []
        for msg in track:
            if(msg.type is 'note_on'):
                if (msg.note not in uniqueNotes):
                    uniqueNotes.append(msg.note)
        return uniqueNotes

class SelectTrack:
    def __init__(self):
        super(SelectTrack, self).__init__()

    def selectTrackFromMIDIFile(self, midifile):
        mid = mido.MidiFile(midifile)
        selectedTrack = {'track': None, 'numnotes': 0}
        for i, track in enumerate(mid.tracks):
            uniqueNotes = []
            proceed = False
            for msg in track:
                if(msg.type is 'program_change'):
                    if(msg.program in GUITAR_BASS_TRACKS or msg.program in BRASS_REED_PIPE_TRACKS):
                        proceed = True
                        break
            if proceed is True:
                for msg in track:
                    if(msg.type is 'note_on'):
                        if (msg.note not in uniqueNotes):
                            uniqueNotes.append(msg.note)
            if (selectedTrack['track'] is None or selectedTrack['numnotes'] < len(uniqueNotes)):
                selectedTrack['track'] = track
                selectedTrack['numnotes'] = len(uniqueNotes)
        return selectedTrack['track'], mid.ticks_per_beat

    def prepareCleanedMIDIFiles(self, midifile): #Use me if dataset is properly cleaned
        mid = mido.MidiFile(midifile)
        for i, track in enumerate(mid.tracks):
            return track, mid.ticks_per_beat

#st = SelectTrack()
#st.selectTrackFromMIDIFile('MIDI/bags_groove_jh.mid')
