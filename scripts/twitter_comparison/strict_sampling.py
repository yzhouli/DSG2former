"""Strict cross-information negatives, distinct from window-only inactivity."""
import numpy as np


class StrictCrossInformationSampler:
    """Keep every positive row; sample only never-participating outside users.

    The source pool consists of raw records, so repeated observations retain
    their sampling frequency. Sampling uses replacement. A later NP50 mask can
    retain exactly half the generated negative rows across each dataset.
    """
    def __init__(self):
        self.chain_users = {}

    def __call__(self, thread, labels, groups, window, pool, rng):
        records = thread['reactions']; tid = thread['thread_id']
        if tid not in self.chain_users:
            self.chain_users[tid] = {thread['source']['uid'], *[r['uid'] for r in records]}
        participants = self.chain_users[tid]
        indices = np.flatnonzero((np.asarray(groups) == window) & (np.asarray(labels) >= 0))
        rows = [dict(uid=records[int(i)]['uid'], label=int(labels[int(i)]), reaction_index=int(i),
                     origin='actual', source_thread_id=tid, source_record_id=records[int(i)]['id'], window=window)
                for i in indices]
        negative = []; attempts = 0; limit = 10000 + 100 * len(rows)
        while len(negative) < len(rows):
            uid, other_tid, rid = pool[int(rng.integers(len(pool)))]
            if other_tid != tid and uid not in participants:
                negative.append(dict(uid=uid, label=2, reaction_index=-1, origin='other_information_never_participant',
                                     source_thread_id=other_tid, source_record_id=rid, window=window))
            attempts += 1
            if attempts > limit:
                raise RuntimeError(f'Insufficient cross-information never-participating users for {tid}')
        assert all(r['uid'] not in participants and r['source_thread_id'] != tid for r in negative)
        return rows + negative


def verify_future_participant_exclusion():
    thread = dict(thread_id='current', source=dict(uid='author'), reactions=[
        dict(uid='early', id='r1'), dict(uid='early', id='r2'), dict(uid='future', id='r3')])
    pool = [('future','other','o1'), ('author','other','o2'), ('absent','other','o3'),
            ('early','other','o4'), ('absent','other','o5')]
    sampler=StrictCrossInformationSampler()
    rows=sampler(thread,np.array([0,1,0]),np.array([0,0,2]),0,pool,np.random.default_rng(42))
    assert [(r['uid'],r['label']) for r in rows[:2]] == [('early',0),('early',1)]
    assert len(rows)==4 and [r['uid'] for r in rows[2:]]==['absent','absent']
    assert all(r['uid']!='future' for r in rows[2:])
    return dict(passed=True, future_participants_excluded=True, source_author_excluded=True,
                repeated_positive_actions_preserved=True, replacement_negative_rows_preserved=True)


if __name__=='__main__':
    print(verify_future_participant_exclusion())
