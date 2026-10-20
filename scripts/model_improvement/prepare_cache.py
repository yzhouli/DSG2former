"""Richer causal features; immutable labels, rows, source split, and macro graph."""
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from scripts.baseline_comparison.prepare_features import sha


def node_features(s,raw,source_text):
 n=s['user_count'];base=s['base'];a=np.abs(s['adjacency'][:n,:n]);a=a/a.sum(1,keepdims=True).clip(1);context=a@base
 src=s['x'][n].copy();src[16:]=source_text
 source=np.repeat(src[None],n,0);global_features=np.repeat(np.r_[s['history_summary'],np.mean(base,axis=0)][None],n,0)
 f=np.c_[base,s['ics'],s['game'],s['fractions'],s['history_signal'],s['dpm_factors'].reshape(n,-1),s['effect_input'][:,:2].reshape(n,-1),context,source,global_features,np.sign(raw['drivers'].reshape(n,6))*np.log1p(np.abs(raw['drivers'].reshape(n,6)))]
 x=np.zeros((len(s['x']),f.shape[1]),np.float32);x[:n]=f
 x[n:,:32]=s['x'][n:];x[n,16:32]=source_text;x[n:,110:142]=src[None]
 # Source vectors are public at the prediction boundary, unlike future replies.
 return x


def count_history(history,now):
 h=np.zeros(8,np.float32);h[-min(8,len(history)):]=history[-8:] if history else h
 return np.r_[h,np.log1p(np.sum(history)),np.sin(now/86400*2*np.pi),np.cos(now/86400*2*np.pi)].astype(np.float32)


def main():
 torch.set_num_threads(1);out=Path('data/pheme_model_improvement');out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return
 started=time.monotonic();base=torch.load('data/pheme_paper_baselines/graphs.pt',weights_only=False,map_location='cpu');raw=torch.load('data/pheme_multitask_hourly/graphs.pt',weights_only=False,map_location='cpu');records=list(map(json.loads,Path('data/pheme_history/records.jsonl').read_text().splitlines()));paths={r['archive_path']:i for i,r in enumerate(records)};threads={t['thread_id']:t for t in map(json.loads,Path('data/pheme_multitask_hourly/threads.jsonl').read_text().splitlines())};text=np.load('data/pheme_paper_baselines/text_record_features.npz')['text'];data=[];train=[];audit={'labels_unchanged':True,'macro_graph_candidates_excluded':True,'snapshots':0,'source_texts_known_at_cutoff':True,'rows':{}}
 for t,r in zip(base,raw):
  assert t['thread_id']==r['thread_id'];source=threads[t['thread_id']]['source'];source_text=text[paths[source['archive_path']]];nt={'thread_id':t['thread_id'],'event':t['event'],'split':t['split'],'snapshots':[],'macro_snapshots':[]};past=[]
  for i,(small,large) in enumerate(zip(t['snapshots'],t['macro_snapshots'])):
   extra=count_history(past,large['cutoff'])
   for branch,s,o in [('snapshots',small,r['snapshots'][i]),('macro_snapshots',large,r['macro_snapshots'][i])]:
    for k in ['node_ids','sample_nodes','target','cutoff','window','end']:np.testing.assert_array_equal(s[k],o[k])
    x=node_features(s,o,source_text);n=s['user_count'];N=len(x);assert len(o['node_ids'])==n
    adj=np.abs(s['adjacency']).astype(np.float32);adj[:n,n]=1.;adj[n,:n]=1.;np.fill_diagonal(adj,1.)
    pos=np.zeros((N,N),np.int16);pos[:n,:n]=o['positions'];pos[adj>0]=2;np.fill_diagonal(pos,1)
    # Non-social source-context edges do not alter historical activity drivers.
    drivers=np.zeros((N,2,3),np.float32);drivers[:n]=o['drivers'];role=np.ones(N,np.int64);role[:n]=0
    ns={'x':x,'drivers':drivers,'adjacency':adj,'positions':pos,'role':role,'user_count':n,'node_ids':o['node_ids'],'sample_nodes':np.asarray(o['sample_nodes']),'target':np.asarray(o['target']),'window':o['window'],'cutoff':o['cutoff'],'end':o['end'],'state_update_users':o['state_update_users'],'history_summary':np.r_[s['history_summary'],extra].astype(np.float32),'popularity_target':s['popularity_target']}
    assert np.isfinite(x).all();nt[branch].append(ns);audit['snapshots']+=1
    if t['split']=='train' and branch=='snapshots' and len(o['target']):train.append(x)
   # Only append after constructing both current prediction inputs.
   past.append(float(large['popularity_target']))
  data.append(nt)
  if len(data)%300==0:print(json.dumps({'sources':len(data),'seconds':time.monotonic()-started}),flush=True)
 scaler=StandardScaler().fit(np.concatenate(train));joblib.dump(scaler,out/'normalizer.joblib')
 for t in data:
  for branch in ['snapshots','macro_snapshots']:
   for s in t[branch]:s['raw_x']=s['x'];s['x']=scaler.transform(s['x']).clip(-12,12).astype(np.float32)
 for split in ['train','validation','test']:audit['rows'][split]=sum(len(s['target']) for t in data if t['split']==split for s in t['snapshots'])
 torch.save(data,out/'graphs.pt');meta={'complete':True,'raw_graph_sha256':sha('data/pheme_multitask_hourly/graphs.pt'),'base_graph_sha256':sha('data/pheme_paper_baselines/graphs.pt'),'prepared_sha256':sha(out/'graphs.pt'),'source_sha256':sha(__file__),'dim':data[0]['snapshots'][0]['x'].shape[1],'history_dim':len(data[0]['snapshots'][0]['history_summary']),'fit_scaler_train_only':True,'source_bottleneck':'Reuse train-fitted BERT PCA16; no future action text is input','extensions':['historical feature residual','source-content token','local+global graph processing','eight-hour causal count history'],'audit':audit,'seconds':time.monotonic()-started};(out/'completion.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
