"""Reuse native preparation with English resources and record-preserving NP50."""
import argparse
from collections import Counter, defaultdict
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.twitter_comparison.common import ROOT, setup, sha, write


def graphs(root):
    from scripts.weibo_comparison.build import build
    build(root)
    out=root/'hourly'
    if (out/'np50_selection.json').exists(): return
    threads=[json.loads(l) for l in (root/'threads.jsonl').open()]
    original={s:Counter() for s in ['train','validation','test']}
    for t in threads:
        rows=json.loads((out/'rows'/(t['thread_id']+'.json')).read_text())['rows']
        original[t['split']].update(r['label'] for r in rows)
    splits=list(original);counts=np.array([original[s][2] for s in splits]);quotas=counts//2
    extra=int(counts.sum()//2-quotas.sum())
    for i in sorted(range(3),key=lambda i:(-(counts[i]%2),i))[:extra]:quotas[i]+=1
    masks={};cursor=defaultdict(int)
    for i,s in enumerate(splits):
        mask=np.zeros(counts[i],bool);rng=np.random.default_rng(np.random.SeedSequence([42,int(root.name[7:9]),i]))
        mask[rng.choice(counts[i],size=quotas[i],replace=False)]=True;masks[s]=mask
    data=torch.load(out/'graphs.pt',weights_only=False,map_location='cpu');statistics={s:Counter() for s in splits}
    parent=out/'parent_full_negative_graphs.pt';(out/'graphs.pt').replace(parent)
    selections=[]
    for t in data:
        path=out/'rows'/(t['thread_id']+'.json');doc=json.loads(path.read_text());rows=doc['rows'];split=t['split'];keep=[]
        for j,r in enumerate(rows):
            if r['label']!=2: keep.append(j)
            else:
                if masks[split][cursor[split]]:keep.append(j)
                cursor[split]+=1
        selected=np.zeros(len(rows),bool);selected[keep]=True;n=0
        for snap in t['snapshots']:
            mask=selected[n:n+len(snap['target'])];n+=len(mask)
            for key in ['target','sample_nodes','seen_before','seen_before_chain']:
                if key in snap:snap[key]=np.asarray(snap[key])[mask]
            # row_start is the start within the reduced supervision sequence.
            snap['row_start']=int(selected[:n-len(mask)].sum())
        assert n==len(rows);newrows=[rows[j] for j in keep];t['rows']=len(newrows)
        assert [r for r in newrows if r['label']!=2]==[r for r in rows if r['label']!=2]
        doc['rows']=newrows;write(path,doc);statistics[split].update(r['label'] for r in newrows)
        selections.append(dict(thread_id=t['thread_id'],original_rows=len(rows),kept_parent_rows=keep))
    torch.save(data,out/'graphs.pt')
    manifest=json.loads((out/'dataset_manifest.json').read_text())
    for i,s in enumerate(splits):
        assert cursor[s]==counts[i] and statistics[s][2]==quotas[i]
        manifest['statistics'][s].update(action_rows=sum(statistics[s].values()),rumor=statistics[s][0],refute=statistics[s][1],nonparticipation=statistics[s][2])
    write(out/'row_selection.json',selections)
    proof=dict(complete=True,keep_fraction=.5,seed=42,original_negatives=dict(zip(splits,counts.tolist())),
               retained_negatives=dict(zip(splits,quotas.tolist())),all_positive_rows_preserved=True,
               graph_nodes_edges_and_macro_targets_preserved=True,parent_graph_sha256=sha(parent),
               row_selection_sha256=sha(out/'row_selection.json'))
    manifest.update(graph_sha256=sha(out/'graphs.pt'),negative_subsampling=proof,
                    labels='PHEME frozen English BERTweet weak stance; train-only relevance and refutation threshold',
                    negatives='Half of original t+2 / other-information negative supervision; no deduplication',
                    raw_audit_sha256=sha(root/'raw_audit.json'))
    write(out/'dataset_manifest.json',manifest);write(out/'np50_selection.json',proof)


def features():
    from scripts.baseline_comparison import prepare_features as m
    from scripts.weibo_comparison.prepare_features import paired, branch_views
    if Path('data/pheme_paper_baselines/completion.json').exists(): return
    m.paired=paired;m.branch_views=branch_views
    original=m.text_resources
    def masked_resources(records,embeddings,teacher,train_mask,out,epochs):
        available=np.array([r['text_available'] for r in records])
        text,emotion,resource=original(records,embeddings,teacher,train_mask&available,out,epochs)
        text[~available]=0;emotion[~available]=0
        np.savez_compressed(out/'text_record_features.npz',text=text,emotion=emotion)
        return text,emotion,resource
    m.text_resources=masked_resources
    source=inspect.getsource(m.build)
    old="for micro,macro in zip(gthread['snapshots'],gthread['macro_snapshots']):"
    assert source.count(old)==1
    source=source.replace(old,"for micro,macro,use_micro,use_macro in paired(gthread):")
    old="for branch,snapshot in [('snapshots',micro),('macro_snapshots',macro)]:"
    assert source.count(old)==1
    source=source.replace(old,"for branch,snapshot in branch_views(micro,macro,use_micro,use_macro):")
    exec(compile(source,'<Twitter-independent-hourly-chunks>','exec'),vars(m))
    m.build(SimpleNamespace(output='data/pheme_paper_baselines',text_epochs=10))


def trgcn():
    from scripts.trgcn_baseline import prepare as m
    m.main()
    previous=json.loads((ROOT/'results/pheme_trgcn_baseline/plan.json').read_text())
    options=dict(previous['options']);options['input_dim']=json.loads(Path('data/pheme_trgcn/completion.json').read_text())['feature_dim']
    Path('results/trgcn').mkdir(parents=True,exist_ok=True)
    path=Path('results/trgcn/plan.json')
    if not path.exists():write(path,dict(options=options,prepared='data/pheme_trgcn/graphs.pt',prepared_sha256=sha('data/pheme_trgcn/graphs.pt'),
                                      source_sha256={str(Path(__file__).resolve()):sha(__file__)},test_used=False,adapter='Same English train-only TF-IDF as PHEME'))


def resources():
    from scripts.weibo_comparison import prepare_features as adapter
    from scripts.baseline_comparison import knowledge_messages as messages
    def parent_record(record,visible):
        candidates=[r for r in visible if r.get('tweet_id')==record.get('reply_to_id')
                    and r['uid']==record.get('reply_to_uid') and r['timestamp']<=record['timestamp']
                    and r['archive_path']!=record['archive_path']]
        if not candidates:return record['reply_to_id']
        return max(candidates,key=lambda r:(r['timestamp'],r['archive_path']))['id']
    def bounded(graphs):
        source=inspect.getsource(messages.add_message_entities)
        source=source.replace("visible=[source]+[r for r in thread['reactions'] if r['timestamp']<feature_cutoff]",
            "visible=[source]+[r for r in thread['reactions'] if r['timestamp']<feature_cutoff and r['uid'] in s['node_ids']][-32:]")
        source=source.replace("parent=lookup.get('message:'+r['reply_to_id'])","parent=lookup.get('message:'+twitter_parent_record(r,visible))")
        messages.twitter_parent_record=parent_record
        exec(compile(source,'<Twitter-causal-message-record-identity>','exec'),vars(messages))
        return messages.add_message_entities(graphs)
    adapter.bounded_messages=bounded
    adapter.resources()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['graphs','features','improved','resources','trgcn']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);root=a.root.resolve()
    if a.mode=='graphs':graphs(root)
    else:
        setup(root)
        if a.mode=='features':features()
        elif a.mode=='trgcn':trgcn()
        else:
            from scripts.weibo_comparison.prepare_features import improved_features
            (improved_features if a.mode=='improved' else resources)()
