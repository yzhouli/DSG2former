"""Independent raw reconstruction of pooled test labels and hourly new users."""
from collections import defaultdict
import json
import numpy as np
from scripts.unified_action_comparison.common import ROOT,SPEC


def truth(name):
    spec=SPEC[name];root=ROOT/spec['root'];action=defaultdict(list);macro=defaultdict(list)
    for line in (root/'threads.jsonl').open():
        t=json.loads(line)
        if t['split']!='test':continue
        tid=t['thread_id'];start=t['source']['timestamp'];rows=json.loads((root/'hourly/rows'/(tid+'.json')).read_text())['rows']
        action['target'].extend(r['label'] for r in rows);action['thread_id'].extend([tid]*len(rows))
        groups=np.ceil((np.asarray([r['timestamp'] for r in t['reactions']])-start)/3600).astype(int)-1
        windows=spec['horizon'] if spec['horizon'] is not None else max(1,int(groups.max())+1) if len(groups) else 1
        by_window=defaultdict(set)
        for r,w in zip(t['reactions'],groups):
            if r['id']!=t['source']['id'] and r['timestamp']>start:by_window[int(w)].add(r['uid'])
        known={t['source']['uid']}
        for w in range(windows):
            active=by_window[w]
            for k,v in [('target',len(active-known)),('thread_id',tid),('window',w),('cutoff',start+w*3600),('end',start+(w+1)*3600)]:macro[k].append(v)
            known|=active
    return dict(action={k:np.asarray(v) for k,v in action.items()},macro={k:np.asarray(v) for k,v in macro.items()})
