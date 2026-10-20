"""Reuse audited feature/native-resource modules in isolated dataset workspaces."""
import argparse
from collections import Counter,defaultdict
import inspect
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scripts.weibo_comparison.extract import sha

STOP=set('的了是在和与及也就都而或把被之着这那你我他她它们地得啊吗呀呢吧')


def tokens(text):
    text=re.sub(r'https?://\S+','',text.lower())
    return re.findall(r'[\u4e00-\u9fff]|[#@]?[a-z][a-z\x27_-]*',text)


def keywords(ws):return [w for w in ws if w not in STOP and not w.startswith('@')]


def paired(t):
    micros=defaultdict(list)
    for s in t['snapshots']:micros[s['window']].append(s)
    for macro in t['macro_snapshots']:
        batch=micros[macro['window']]
        if not batch:yield macro,macro,False,True
        else:
            for i,micro in enumerate(batch):yield micro,macro,True,i==0


def branch_views(micro,macro,use_micro,use_macro):
    if use_micro:yield 'snapshots',micro
    if use_macro:yield 'macro_snapshots',macro


def setup(root):
    root=root.resolve();project=Path(__file__).resolve().parents[2];work=root/'runtime';work.mkdir(exist_ok=True)
    (work/'data').mkdir(exist_ok=True);(work/'models').mkdir(exist_ok=True)
    for link,target in [(work/'data/pheme_multitask_hourly',root/'hourly'),(work/'data/pheme_history',root/'history'),
        (work/'models/pheme_bert',project/'models/weibo_bert_fp16'),
        (work/'dsg2former',project/'dsg2former'),(work/'scripts',project/'scripts')]:
        if not link.exists():link.symlink_to(target,target_is_directory=True)
    for name in ['threads.jsonl','action_labels.npz','raw_audit.json']:
        p=root/'hourly'/name
        if not p.exists():p.symlink_to(root/name)
    os.chdir(work)
    return project,work


def baseline_features():
    from scripts.baseline_comparison import prepare_features as m
    if Path('data/pheme_paper_baselines/completion.json').exists():return
    # Match the native lexical module's file schema with a train-only induced
    # Chinese lexicon, rather than treating all Chinese tokens as English OOV.
    records=[json.loads(l) for l in Path('data/pheme_history/records.jsonl').open()]
    teacher=np.load('data/pheme_history/sentiment.npy',mmap_mode='r')
    raw=torch.load('data/pheme_multitask_hourly/graphs.pt',weights_only=False,map_location='cpu')
    last_train=max(s['feature_cutoff_exclusive'] for t in raw if t['split']=='train' for s in t['macro_snapshots'])
    del raw
    total=Counter();polarity=Counter()
    for i,r in enumerate(records):
        if r['split']!='train' or r['timestamp']>=last_train:continue
        for w in tokens(r['text']):total[w]+=1;polarity[w]+=float(teacher[i,0]-teacher[i,1])
    lex=Path('external/vaderSentiment/vader_lexicon.txt');lex.parent.mkdir(parents=True,exist_ok=True)
    lex.write_text('\n'.join(w+'\t'+str(4*polarity[w]/n) for w,n in total.items()))
    m.tokens=tokens;m.keywords=keywords;m.paired=paired;m.branch_views=branch_views
    source=inspect.getsource(m.build)
    source=source.replace("for micro,macro in zip(gthread['snapshots'],gthread['macro_snapshots']):",
        "for micro,macro,use_micro,use_macro in paired(gthread):")
    source=source.replace("for branch,snapshot in [('snapshots',micro),('macro_snapshots',macro)]:",
        "for branch,snapshot in branch_views(micro,macro,use_micro,use_macro):")
    exec(compile(source,str(Path(__file__).resolve()),'exec'),m.__dict__)
    m.build(SimpleNamespace(output='data/pheme_paper_baselines',text_epochs=10))
    p=Path('data/pheme_paper_baselines/completion.json');proof=json.loads(p.read_text())
    proof.update(dataset_specific_adapter_sha256=sha(__file__),chinese_lexicon='Training-only mean frozen Chinese sentiment polarity per token',
        chinese_tokenizer='Single Chinese characters plus Latin words; fixed train vocabulary and original32token feature limit',
        independent_micro_chunks_and_unique_macro_windows=True)
    p.write_text(json.dumps(proof,indent=2))


def improved_features():
    from scripts.model_improvement.prepare_cache import node_features,count_history
    out=Path('data/pheme_model_improvement');out.mkdir(exist_ok=True)
    if (out/'completion.json').exists():return
    data=torch.load('data/pheme_paper_baselines/graphs.pt',weights_only=False,map_location='cpu')
    raw=torch.load('data/pheme_multitask_hourly/graphs.pt',weights_only=False,map_location='cpu')
    records=[json.loads(l) for l in Path('data/pheme_history/records.jsonl').open()];paths={r['archive_path']:i for i,r in enumerate(records)}
    threads={t['thread_id']:t for t in map(json.loads,Path('data/pheme_multitask_hourly/threads.jsonl').open())}
    text=np.load('data/pheme_paper_baselines/text_record_features.npz')['text'];train=[];result=[]
    for t,r in zip(data,raw):
        assert t['thread_id']==r['thread_id'];src=threads[t['thread_id']]['source'];source_text=text[paths[src['archive_path']]]
        nt={k:v for k,v in t.items() if k not in ['snapshots','macro_snapshots']};nt.update(snapshots=[],macro_snapshots=[])
        macro_by_window={s['window']:s for s in t['macro_snapshots']};past=[];history={}
        for w in range(72):
            s=macro_by_window[w];history[w]=np.r_[s['history_summary'],count_history(past,s['cutoff'])].astype(np.float32)
            past.append(float(s['popularity_target']))
        for branch in ['snapshots','macro_snapshots']:
            assert len(t[branch])==len(r[branch])
            for s,o in zip(t[branch],r[branch]):
                for k in ['node_ids','sample_nodes','target','cutoff','window','end']:np.testing.assert_array_equal(s[k],o[k])
                x=node_features(s,o,source_text);n=s['user_count'];N=len(x)
                adj=np.abs(s['adjacency']).astype(np.float32);adj[:n,n]=1;adj[n,:n]=1;np.fill_diagonal(adj,1)
                pos=np.zeros((N,N),np.int16);pos[:n,:n]=o['positions'];pos[adj>0]=2;np.fill_diagonal(pos,1)
                drivers=np.zeros((N,2,3),np.float32);drivers[:n]=o['drivers'];role=np.ones(N,np.int64);role[:n]=0
                ns=dict(x=x,drivers=drivers,adjacency=adj,positions=pos,role=role,user_count=n,
                    node_ids=o['node_ids'],sample_nodes=o['sample_nodes'],target=o['target'],window=o['window'],
                    cutoff=o['cutoff'],end=o['end'],state_update_users=o['state_update_users'],
                    history_summary=history[s['window']],popularity_target=macro_by_window[s['window']]['popularity_target'])
                nt[branch].append(ns)
                if t['split']=='train' and branch=='snapshots':train.append(x)
        result.append(nt)
    scaler=StandardScaler().fit(np.concatenate(train));joblib.dump(scaler,out/'normalizer.joblib')
    for t in result:
        for branch in ['snapshots','macro_snapshots']:
            for s in t[branch]:s['raw_x']=s['x'];s['x']=scaler.transform(s['x']).clip(-12,12).astype(np.float32)
    torch.save(result,out/'graphs.pt');(out/'completion.json').write_text(json.dumps(dict(complete=True,
        dim=184,prepared_sha256=sha(out/'graphs.pt'),source_sha256=sha(__file__),
        fit_scaler_train_only=True,all_targets_preserved=True,macro_windows_unique=True),indent=2))


def bounded_messages(graphs):
    # Knowledge pretraining keeps native user/message relation types. Retain
    # all original evidence for represented users, but bound message context
    # to their most recent32visible records so large graphs stay tractable.
    from scripts.baseline_comparison import knowledge_messages as m
    source=inspect.getsource(m.add_message_entities)
    source=source.replace("visible=[source]+[r for r in thread['reactions'] if r['timestamp']<feature_cutoff]",
        "visible=[source]+[r for r in thread['reactions'] if r['timestamp']<feature_cutoff and r['uid'] in s['node_ids']][-32:]")
    exec(compile(source,str(Path(__file__).resolve()),'exec'),m.__dict__)
    return m.add_message_entities(graphs)


def resources():
    from scripts.baseline_comparison import prepare_method_resources as m
    m.add_message_entities=bounded_messages
    m.main(SimpleNamespace(input='data/pheme_paper_baselines',output='data/pheme_paper_baseline_resources'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['features','resources','improved']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();setup(a.root)
    torch.set_num_threads(1)
    {'features':baseline_features,'resources':resources,'improved':improved_features}[a.mode]()
