"""All recoverable participant texts, time-varying labels, train-only calibration."""
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.unified_action_comparison.common import ROOT, OUT, SPEC, sha, write, link


def initialize(name):
    s=SPEC[name];root=ROOT/s['root'];parent=ROOT/s['parent'];root.mkdir(parents=True,exist_ok=True)
    for f in ['threads.jsonl','raw_audit.json']:link(parent/f,root/f)
    history=ROOT/s.get('history_parent',s['parent']+'/history')
    link(history,root/'history')
    return root


def build(name):
    root=initialize(name);spec=SPEC[name];path=root/'label_manifest.json'
    if path.exists():
        m=json.loads(path.read_text());assert m['source_hash']==sha(root/'threads.jsonl');return m
    threads=[json.loads(l) for l in (root/'threads.jsonl').open()]
    old=np.load(ROOT/spec['parent']/'action_labels.npz')
    offsets=old['reaction_offsets'];prob=None
    if name=='PHEME':prob=np.load(ROOT/spec['parent']/'stance_probabilities.npz')['probability']
    elif 'probability' in old:prob=old['probability']
    if prob is None:
        align=np.load(root/'history/alignment.npz')
        prob=np.load(root/'history/stance_probabilities.npy',mmap_mode='r')[align['reaction_rows']]
    eligible=[];train=[];retweets=[];splits=[]
    for t in threads:
        start=t['source']['timestamp']
        for r in t['reactions']:
            dt=r['timestamp']-start
            eligible.append(r.get('text_available',bool(r.get('text','').strip())) and r['id']!=t['source']['id']
                            and dt>0 and (spec['horizon'] is None or dt<=spec['horizon']*3600))
            train.append(t['split']=='train');splits.append(t['split']);retweets.append(r.get('is_source_retweet',False))
    eligible=np.asarray(eligible,bool);train=np.asarray(train,bool);retweets=np.asarray(retweets,bool);splits=np.asarray(splits)
    assert len(eligible)==len(prob)==offsets[-1]
    assert np.isfinite(prob).all() and np.allclose(prob[eligible].sum(1),1,atol=1e-5)
    from scripts.pheme_pipeline.build_information_dataset import fit_stance_threshold
    calibration=eligible&train&~retweets
    threshold=fit_stance_threshold(prob,calibration,spec['refute'])
    score=prob[:,1]/prob[:,:2].sum(1).clip(1e-12)
    label=np.full(len(prob),-1,np.int8);label[eligible]=(score[eligible]>threshold).astype(np.int8);label[eligible&retweets]=0
    np.savez_compressed(root/'action_labels.npz',label=label,related=eligible,forecastable=eligible,
        probability=prob,reaction_offsets=offsets,conditional_deny_score=score,
        similarity=old['similarity'],original_argmax=prob.argmax(1),original_confidence=prob.max(1))
    stats={split:{str(k):int(((splits==split)&(label==k)).sum()) for k in [0,1,-1]} for split in ['train','validation','test']}
    manifest=dict(complete=True,protocol='unified_action_v1',dataset=name,source_hash=sha(root/'threads.jsonl'),
        raw_records_deduplicated=False,all_recoverable_text_participation_used=True,relevance_filter_enabled=False,
        stance='Behavior-level weak labels can change within a user/information pair; preserve all raw rows',
        source_retweets_fixed_to_propagation=True,fit_partition='train only',conditional_deny_threshold=threshold,
        training_reply_refute_target=spec['refute'],training_calibration_rows=int(calibration.sum()),
        horizon_hours=spec['horizon'],statistics=stats,teacher_class_count=prob.shape[1],
        future_text_usage='Only target generation, never feature input',
        teacher_caveat='Frozen teacher weak labels; not human annotations. English stance teacher may overlap PHEME training data.',
        action_labels_sha256=sha(root/'action_labels.npz'))
    write(path,manifest);print(json.dumps(manifest),flush=True);return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(SPEC),required=True);build(p.parse_args().dataset)
