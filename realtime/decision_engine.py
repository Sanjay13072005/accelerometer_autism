class DecisionEngine:
    def __init__(self, move_thresh, sound_thresh):
        self.move_thresh = move_thresh
        self.sound_thresh = sound_thresh

    def is_distressed(self, motion_energy, sound_rms):
        return motion_energy > self.move_thresh and sound_rms > self.sound_thresh
