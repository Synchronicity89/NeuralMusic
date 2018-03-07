import mido
import numpy as np
np.set_printoptions(threshold=np.nan)

class SelectTrack:
    def __init__(self):
        super(SelectTrack, self).__init__()

    def selectTrackFromMIDIFile(self, midifile):
        GUITAR_BASS_TRACKS = range(25, 41)
        BRASS_REED_PIPE_TRACKS = range(57, 80)

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

#st = SelectTrack()
#st.selectTrackFromMIDIFile('MIDI/bags_groove_jh.mid')
