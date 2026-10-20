"""Shared strict cross-information targets and explicitly typed causal semantics."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import numpy as np
import torch
from dsg2former.paper_features import HistoricalCorpus,popularity_difference
from dsg2former.semantic_graph import shortest_positions
from dsg2former.multitask_data import new_participants
from scripts.twitter_comparison.strict_sampling import StrictCrossInformationSampler
from scripts.weibo_comparison.build import context_users
from scripts.unified_action_comparison.common import ROOT,OUT,SPEC,sha,write

STAGE={'train':0,'validation':1,'test':2}


def windows(t,spec):
    groups=np.ceil((np.asarray([r['timestamp'] for r in t['reactions']])-t['source']['timestamp'])/3600).astype(np.int64)-1
    count=spec['horizon'] if spec['horizon'] is not None else max(1,int(groups.max())+1) if len(groups) else 1
    return groups,count


def targets(name):
    spec=SPEC[name];root=ROOT/spec['root'];out=root/'hourly';out.mkdir(exist_ok=True)
    if (out/'targets_completion.json').exists():return
    threads=[json.loads(l) for l in (root/'threads.jsonl').open()];lab=np.load(root/'action_labels.npz')
    pools=defaultdict(list);positive=Counter()
    for i,t in enumerate(threads):
        pools[t['split']].extend((r['uid'],t['thread_id'],r['id']) for r in [t['source']]+t['reactions'])
        a,b=lab['reaction_offsets'][i:i+2];positive[t['split']]+=int((lab['label'][a:b]>=0).sum())
    counts=np.array([positive[s] for s in STAGE]);quotas=counts//2
    for i in sorted(range(3),key=lambda i:(-(counts[i]%2),i))[:int(counts.sum()//2-quotas.sum())]:quotas[i]+=1
    masks={};cursor=Counter();stats={s:Counter() for s in STAGE}
    for i,s in enumerate(STAGE):
        mask=np.zeros(counts[i],bool);rng=np.random.default_rng(np.random.SeedSequence([42,i,271828]))
        mask[rng.choice(counts[i],int(quotas[i]),replace=False)]=True;masks[s]=mask
    sampler=StrictCrossInformationSampler();(out/'rows').mkdir(exist_ok=True)
    for i,t in enumerate(threads):
        tid=t['thread_id'];split=t['split'];groups,count=windows(t,spec);a,b=lab['reaction_offsets'][i:i+2]
        labels=lab['label'][a:b];rng=np.random.default_rng(np.random.SeedSequence([42,int(tid)%(2**32)]));rows=[]
        known={t['source']['uid']};macro=[]
        for w in range(count):
            selected=sampler(t,labels,groups,w,pools[split],rng)
            kept=[]
            for r in selected:
                if r['label']!=2:kept.append(r)
                else:
                    if masks[split][cursor[split]]:kept.append(r)
                    cursor[split]+=1
            # The permutation uses neither label values nor class-dependent partitions.
            shuffle=np.random.default_rng(np.random.SeedSequence([42,int(tid)%(2**32),w,314159]))
            rows.extend(kept[int(j)] for j in shuffle.permutation(len(kept)))
            new,active,rawcount=new_participants(t,np.flatnonzero(groups==w),known)
            macro.append(dict(window=w,target=len(new),active_users=len(active),raw_records=rawcount));known|=active
        expected=np.flatnonzero((labels>=0)&(groups>=0)&(groups<count)).tolist()
        assert sorted(r['reaction_index'] for r in rows if r['origin']=='actual')==expected
        chain={t['source']['uid'],*[r['uid'] for r in t['reactions']]}
        assert all(r['uid'] not in chain and r['source_thread_id']!=tid for r in rows if r['label']==2)
        s=stats[split];s.update(sources=1,windows=count,action_rows=len(rows),raw_reactions=len(t['reactions']),
            eligible_raw_reactions=int(((groups>=0)&(groups<count)).sum()),macro_new_users=sum(v['target'] for v in macro))
        for r in rows:s[['rumor','refute','nonparticipation'][r['label']]]+=1
        write(out/'rows'/(tid+'.json'),dict(thread_id=tid,split=split,rows=rows))
        write(out/'rows'/(tid+'.macro.json'),dict(thread_id=tid,split=split,windows=macro))
    for i,s in enumerate(STAGE):assert cursor[s]==counts[i] and stats[s]['nonparticipation']==quotas[i]
    proof=dict(complete=True,protocol='unified_action_v1',dataset=name,statistics={s:dict(v) for s,v in stats.items()},
        source_hash=sha(root/'threads.jsonl'),label_hash=sha(root/'action_labels.npz'),
        all_positive_rows_preserved=True,all_negative_users_excluded_from_entire_chain=True,
        negative_origin='Other information in same partition; raw-record pool; replacement; no deduplication',
        nonparticipation_fraction_of_positive_rows=.5,nonparticipation_quota=quotas.tolist(),
        label_unit='Raw behavior row, may vary with time for same user/information',
        same_window_candidate_permutation='Seeded, independent of label values; before32row chunking',
        horizon_hours=spec['horizon'],window_seconds=3600,
        macro_target='Next hour first-time distinct participants; all observed raw actions, including missing-text actions',
        source_code_sha256={str(Path(p).relative_to(ROOT)):sha(p) for p in [__file__,ROOT/'scripts/twitter_comparison/strict_sampling.py']})
    write(out/'targets_completion.json',proof);print(json.dumps(proof),flush=True)


def semantic_graph(corpus,candidates,known,cutoff,source,pop,metadata):
    g=corpus.graph(candidates,known,cutoff,source,pop);social=g['positions'].copy();nodes=g['node_ids']
    rows=[];owners=[]
    for j,u in enumerate(nodes):
        ix=corpus.eligible(u,cutoff,source);rows.extend(ix.tolist());owners.extend([j]*len(ix))
    edge=np.zeros((len(nodes),len(nodes)),bool);information=set();evidence=0
    if rows:
        # Share identical vector computations, then restore raw multiplicities
        # in the evidence weights. No raw observations or supervision removed.
        raw=np.asarray(corpus.embeddings[np.asarray(rows)],np.float64)
        raw/=np.linalg.norm(raw,axis=1,keepdims=True).clip(1e-12)
        unique,inverse=np.unique(raw,axis=0,return_inverse=True)
        counts=np.zeros((len(unique),len(nodes)),np.int64);np.add.at(counts,(inverse,np.asarray(owners)),1)
        query=np.asarray(corpus.embeddings[source],np.float64);query/=max(np.linalg.norm(query),1e-12)
        information=set(np.flatnonzero(counts[unique@query>=.98].sum(0)).tolist())
        for a in range(0,len(unique),128):
            matches=(unique[a:a+128]@unique.T)>=.98
            links=counts[a:a+128].T@matches.astype(np.int64)@counts
            edge|=links>0;evidence+=int(links.sum()-np.trace(links))
    direct=edge.copy();np.fill_diagonal(direct,False)
    if information:edge[np.ix_(list(information),list(information))]=True
    np.fill_diagonal(edge,False)
    g.update(metadata);g.update(social_positions=social,positions=shortest_positions((social==2)|edge),
        semantic_information_users=sorted(information),semantic_user_edges=direct,
        semantic_edges=int(edge.sum()),local_semantic_raw_pair_evidence=evidence)
    return g


def graphs(name):
    torch.set_num_threads(1);spec=SPEC[name];root=ROOT/spec['root'];out=root/'hourly'
    if (out/'dataset_manifest.json').exists():return
    targets(name);threads=[json.loads(l) for l in (root/'threads.jsonl').open()]
    records=[json.loads(l) for l in (root/'history/records.jsonl').open()]
    stages={t['thread_id']:STAGE[t['split']] for t in threads}
    record_stage=np.array([stages.get(r['thread_id'],-1) for r in records])
    emb=np.load(root/'history/embeddings.npy',mmap_mode='r');sent=np.load(root/'history/sentiment.npy',mmap_mode='r')
    shards=out/'graph_shards';shards.mkdir(exist_ok=True);started=time.monotonic();done=0
    for split in STAGE:
        pending=[t for t in threads if t['split']==split and not(shards/(t['thread_id']+'.pt')).exists()]
        if not pending:continue
        corpus=HistoricalCorpus(records,emb,sent,5,.23,record_stage<=STAGE[split])
        for t in pending:
            tid=t['thread_id'];rows=json.loads((out/'rows'/(tid+'.json')).read_text())['rows'];by_window=defaultdict(list)
            for r in rows:by_window[r['window']].append(r)
            src=corpus.path_index[t['source']['archive_path']];start=t['source']['timestamp'];groups,count=windows(t,spec)
            known={t['source']['uid']};previous_active=known.copy();before_previous=set();previous_new=0;row_start=0
            nt=dict(thread_id=tid,event=t['event'],split=split,snapshots=[],macro_snapshots=[],raw_reactions=len(t['reactions']),rows=len(rows))
            for w in range(count):
                cutoff=start+w*3600;fc=cutoff if not w else float(np.nextafter(cutoff,np.inf))
                new,active,rawcount=new_participants(t,np.flatnonzero(groups==w),known)
                context=context_users(t,groups,w);pop=popularity_difference(len(previous_active),len(before_previous),0 if not w else 3600,900)
                meta=dict(target=np.empty(0,np.int64),window=w,cutoff=cutoff,end=cutoff+3600,
                    feature_cutoff_exclusive=fc,row_start=row_start,state_update_users=context.copy())
                macro=semantic_graph(corpus,[],context,fc,src,pop,meta)
                macro.update(popularity_target=float(len(new)),window_active_users=len(active),raw_future_records=rawcount,
                    known_participants_before=sorted(known),history_summary=np.array([len(known)-1,previous_new,len(previous_active),w],np.float32))
                nt['macro_snapshots'].append(macro)
                selected=by_window[w]
                for a in range(0,len(selected),32):
                    chunk=selected[a:a+32];m=dict(meta,target=np.array([r['label'] for r in chunk],np.int64),row_start=row_start+a)
                    nt['snapshots'].append(semantic_graph(corpus,[r['uid'] for r in chunk],context,fc,src,pop,m))
                row_start+=len(selected);known|=active;before_previous,previous_active=previous_active,active;previous_new=len(new)
            assert row_start==len(rows)
            path=shards/(tid+'.pt');tmp=path.with_suffix('.part');torch.save(nt,tmp);tmp.replace(path)
            corpus.semantic_cache.clear();done+=1
            if done%25==0:print(json.dumps(dict(dataset=name,graphs_completed=done,seconds=round(time.monotonic()-started,1))),flush=True)
        del corpus
    data=[torch.load(shards/(t['thread_id']+'.pt'),weights_only=False,map_location='cpu') for t in threads]
    torch.save(data,out/'graphs.pt')
    proof=json.loads((out/'targets_completion.json').read_text())
    proof.update(graph_sha256=sha(out/'graphs.pt'),semantic_threshold=.98,
        semantics='Explicit user-information and user-user any-visible-history cosine edges; original social graph retained',
        context_limit=16,supervised_rows_per_chunk=32,maximum_user_nodes=48,
        macro_graph='Historical context only; never action candidates',
        history_policy='Only timestamps preceding feature cutoff; known source allowed at trigger; no later partition history',
        graph_builder_sha256=sha(__file__))
    write(out/'dataset_manifest.json',proof)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['targets','graphs']);p.add_argument('--dataset',choices=list(SPEC),required=True)
    a=p.parse_args();globals()[a.mode](a.dataset)
