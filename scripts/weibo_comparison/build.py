"""72 fixed hourly forecasts, all target rows, bounded causal graph context.

Context is bounded for the native 16x16 Rumor-DPM raster. Target rows are split
into chunks, never discarded or deduplicated. Macro graphs never see candidates.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from dsg2former.paper_features import HistoricalCorpus,popularity_difference
from dsg2former.semantic_graph import shortest_positions
from dsg2former.multitask_data import action_rows,new_participants
from scripts.weibo_comparison.extract import sha

STAGE={'train':0,'validation':1,'test':2}


def context_users(thread,groups,w,limit=16):
    source=thread['source']['uid'];recent={source:thread['source']['timestamp']}
    for r,g in zip(thread['reactions'],groups):
        if 0<=g<w:recent[r['uid']]=max(recent.get(r['uid'],-np.inf),r['timestamp'])
    return {source,*[u for u in sorted(recent,key=lambda u:(-recent[u],u)) if u!=source][:limit-1]}


def graph(corpus,candidates,known,cutoff,src,pop,metadata):
    g=corpus.graph(candidates,known,cutoff,src,pop)
    social=g['positions'].copy();nodes=g['node_ids'];rows=[];owner=[]
    for i,u in enumerate(nodes):
        ix=corpus.eligible(u,cutoff,src);rows.extend(ix.tolist());owner.extend([i]*len(ix))
    edges=np.zeros((len(nodes),len(nodes)),bool);info=set();matches=0
    if rows:
        # Exact raw history is retained. Repeated texts/IDs supply repeated
        # evidence; adjacency necessarily has one entry per user pair.
        ix=np.asarray(rows,dtype=np.int64);who=np.asarray(owner,dtype=np.int64)
        emb=np.asarray(corpus.embeddings[ix],dtype=np.float64)
        emb/=np.linalg.norm(emb,axis=1,keepdims=True).clip(1e-12)
        q=np.asarray(corpus.embeddings[src],dtype=np.float64);q/=max(np.linalg.norm(q),1e-12)
        hit=emb@q>=.98;info=set(who[hit].tolist())
        # Compare in tiles to avoid an unbounded all-record-pairs matrix.
        for start in range(0,len(ix),128):
            a,b=np.nonzero(emb[start:start+128]@emb.T>=.98)
            left,right=who[a+start],who[b];cross=left!=right
            edges[left[cross],right[cross]]=True;matches+=int(cross.sum())
    if info:edges[np.ix_(list(info),list(info))]=True
    np.fill_diagonal(edges,False)
    g.update(metadata);g['social_positions']=social;g['positions']=shortest_positions((social==2)|edges)
    g['semantic_edges']=int(edges.sum());g['local_semantic_raw_pair_evidence']=matches
    return g


def build(root):
    torch.set_num_threads(1);out=root/'hourly';out.mkdir(exist_ok=True)
    if (out/'dataset_manifest.json').exists():return
    hist=root/'history';threads=[json.loads(l) for l in (root/'threads.jsonl').open()]
    records=[json.loads(l) for l in (hist/'records.jsonl').open()]
    emb=np.load(hist/'embeddings.npy',mmap_mode='r');sent=np.load(hist/'sentiment.npy',mmap_mode='r')
    lab=np.load(root/'action_labels.npz');stages={t['thread_id']:STAGE[t['split']] for t in threads}
    record_stage=np.asarray([stages[r['thread_id']] for r in records]);pools=defaultdict(list)
    for t in threads:pools[t['split']].extend((r['uid'],t['thread_id'],r['id']) for r in [t['source']]+t['reactions'])
    statistics={s:Counter() for s in STAGE};data=[];started=time.monotonic()
    # Keep only one stage-specific corpus resident at once (millions of users).
    for split in STAGE:
        corpus=HistoricalCorpus(records,emb,sent,5,.23,record_stage<=STAGE[split])
        for ti,t in enumerate(threads):
            if t['split']!=split:continue
            src=corpus.path_index[t['source']['archive_path']];start=t['source']['timestamp']
            groups=np.ceil((np.asarray([r['timestamp'] for r in t['reactions']])-start)/3600).astype(np.int64)-1
            offsets=lab['reaction_offsets'][ti:ti+2];labels=lab['label'][offsets[0]:offsets[1]]
            rng=np.random.default_rng(np.random.SeedSequence([42,int(t['thread_id'])%(2**32)]))
            known={t['source']['uid']};previous_active=known.copy();before_previous=set();previous_new=0
            nt=dict(thread_id=t['thread_id'],event=t['event'],split=split,snapshots=[],macro_snapshots=[],raw_reactions=len(t['reactions']))
            all_rows=[]
            for w in range(72):
                cutoff=start+w*3600;end=cutoff+3600;fc=cutoff if not w else float(np.nextafter(cutoff,np.inf))
                ix=np.flatnonzero(groups==w);new,active,rawcount=new_participants(t,ix,known)
                chosen=context_users(t,groups,w);pop=popularity_difference(len(previous_active),len(before_previous),0 if not w else 3600,900)
                metadata=dict(target=np.empty(0,np.int64),window=w,cutoff=cutoff,end=end,feature_cutoff_exclusive=fc,
                    row_start=len(all_rows),state_update_users=chosen.copy())
                macro=graph(corpus,[],chosen,fc,src,pop,metadata)
                macro.update(popularity_target=float(len(new)),window_active_users=len(active),raw_future_records=rawcount,
                    known_participants_before=sorted(known),history_summary=np.array([len(known)-1,previous_new,len(previous_active),w],np.float32))
                nt['macro_snapshots'].append(macro)
                selected=action_rows(t,labels,groups,w,pools[split],rng)
                if selected:
                    # Stable original row order; replacement negatives and
                    # repeated user actions remain separate supervised rows.
                    for a in range(0,len(selected),32):
                        chunk=selected[a:a+32];m=dict(metadata,target=np.array([r['label'] for r in chunk],np.int64),row_start=len(all_rows)+a)
                        micro=graph(corpus,[r['uid'] for r in chunk],chosen,fc,src,pop,m)
                        nt['snapshots'].append(micro)
                    all_rows.extend(selected)
                known|=active;before_previous,previous_active=previous_active,active;previous_new=len(new)
            expected=np.flatnonzero((labels>=0)&(groups>=0)&(groups<72)).tolist()
            assert sorted(r['reaction_index'] for r in all_rows if r['origin']=='actual')==expected
            nt['rows']=len(all_rows);data.append(nt)
            s=statistics[split];s.update(sources=1,windows=72,action_rows=len(all_rows),raw_reactions=len(t['reactions']),
                eligible_raw_reactions=sum(0<=g<72 for g in groups),macro_new_users=sum(g['popularity_target'] for g in nt['macro_snapshots']))
            for r in all_rows:s[['rumor','refute','nonparticipation'][r['label']]]+=1
            # Per-source shards make interruption resumable and independent
            # record/target audits possible, without all-corpus row JSON RAM.
            shard=out/'rows';shard.mkdir(exist_ok=True)
            (shard/(t['thread_id']+'.json')).write_text(json.dumps(dict(thread_id=t['thread_id'],split=split,rows=all_rows)))
            if len(data)%50==0:print(json.dumps(dict(sources=len(data),seconds=round(time.monotonic()-started,1))),flush=True)
            corpus.semantic_cache.clear()
        del corpus
    torch.save(data,out/'graphs.pt')
    with (out/'statistics.csv').open('w',newline='') as f:
        fields=['split']+list(statistics['train']);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for s,v in statistics.items():w.writerow(dict(split=s,**v))
    manifest=dict(complete=True,information_samples=len(threads),horizon_hours=72,window_seconds=3600,
        context_limit=16,supervised_rows_per_chunk=32,maximum_user_nodes=48,
        context_policy='Source plus15most recent observed chain participants; all supervised users retained via row chunks; no fixed test cohort',
        semantics='User-information projection and user-user any-raw-history cosine>=.98 inside each bounded causal graph; no global one-hop expansion',
        macro_graph='Only historical context; action candidates never used; one forecast per source/hour',
        target='First-time distinct participants in chain during next hour; source author/returning users/sampled negatives excluded',
        empty_windows='All72hours retained, archive-observed empty hours counted as0; collection censoring times unavailable',
        labels='Training-calibrated Chinese-compatible NLI weak stance, not human truth',
        negatives='One per weak-labeled action; replacement; prefer t+2, otherwise another rumor in same split',
        statistics={s:{k:int(value) for k,value in v.items()} for s,v in statistics.items()},deduplicate=False,
        source_hashes={str(p):sha(p) for p in [root/'threads.jsonl',root/'action_labels.npz',hist/'records.jsonl',hist/'embeddings.npy',hist/'sentiment.npy']},
        graph_sha256=sha(out/'graphs.pt'))
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);build(p.parse_args().root)
