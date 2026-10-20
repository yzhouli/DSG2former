"""Strictly earlier, stage-restricted user states from observed macro graphs."""
class VisibleUserStates:
    def __init__(self):
        self.by_stage = [{}, {}, {}]
        self.pending = []
        self.cutoff = None

    def advance(self, cutoff):
        if self.cutoff is not None:
            if cutoff < self.cutoff:
                raise ValueError('State queries must be chronological')
            if cutoff > self.cutoff:
                for stage, uid, value in self.pending:
                    self.by_stage[stage][uid] = (self.cutoff, value)
                self.pending.clear()
        self.cutoff = cutoff

    def read(self, uid, stage):
        choices = [d[uid] for d in self.by_stage[:stage+1] if uid in d]
        if not choices:
            return None
        when, value = max(choices, key=lambda x: x[0])
        assert when < self.cutoff
        return when, value

    def observe(self, uid, stage, value):
        # Commit only after all queries at this timestamp. No candidate or
        # future-action membership is used to decide which states are stored.
        if self.cutoff is None:
            raise ValueError('Advance to an observation time first')
        self.pending.append((stage, uid, value))
