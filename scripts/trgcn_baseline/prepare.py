"""Train-only TF-IDF, strictly cutoff-limited user text; unchanged raw graphs."""
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch,joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.baseline_comparison.prepare_features import sha
OUT=Path('data/pheme_trgcn');STAGE={'train':0,'validation':1,'test':2}

def eligible(indices,times,stages,cutoff,stage,source):
    ix=indices[(times[indices]<cutoff)&(stages[indices]<=stage)]
    if source in indices and times[source]==cutoff and source not in ix:ix=np.r_[ix,source]
    return ix

def depths(adj,root):
    n=len(adj);d=np.full(n,n,dtype=np.int64);d[root]=0;todo=[root];connected=(adj>0)|(adj.T>0)
    for i in todo:
        for j in np.flatnonzero(connected[i]):
            if d[j]>d[i]+1:d[j]=d[i]+1;todo.append(int(j))
    # Unreachable context/negative users have depth0, explicitly not a tree.
    d[d==n]=0;return d

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():return
    raw=torch.load('data/pheme_multitask_hourly/graphs.pt',weights_only=False,map_location='cpu');records=[json.loads(l) for l in Path('data/pheme_history/records.jsonl').read_text().splitlines()];threads=[json.loads(l) for l in Path('data/pheme_multitask_hourly/threads.jsonl').read_text().splitlines()];stage={t['thread_id']:STAGE[t['split']] for t in raw};times=np.array([r['timestamp'] for r in records]);stages=np.array([stage.get(r['thread_id'],-1) for r in records]);last_train=max(s['feature_cutoff_exclusive'] for t in raw if t['split']=='train' for s in t['macro_snapshots']);fit=np.flatnonzero((stages<=0)&(times<last_train))
    vectorizer=TfidfVectorizer(max_features=5000,sublinear_tf=False,dtype=np.float32);vectorizer.fit([records[i]['text'] for i in fit]);text=vectorizer.transform([r['text'] for r in records]);joblib.dump(vectorizer,OUT/'tfidf.joblib');paths={r['archive_path']:i for i,r in enumerate(records)};users={}
    for i,r in enumerate(records):users.setdefault(r['uid'],[]).append(i)
    users={u:np.asarray(ix,dtype=np.int64) for u,ix in users.items()};cache={};result=[];lookups=0
    for ti,(t,source_thread) in enumerate(zip(raw,threads)):
        src=paths[source_thread['source']['archive_path']];source_uid=records[src]['uid'];new={k:v for k,v in t.items() if k not in ['snapshots','macro_snapshots']}
        for branch in ['snapshots','macro_snapshots']:
            out=[]
            for s in t[branch]:
                xx=[]
                for uid in s['node_ids']:
                    ix=eligible(users.get(uid,np.empty(0,dtype=np.int64)),times,stages,s['feature_cutoff_exclusive'],STAGE[t['split']],src);key=(uid,tuple(ix))
                    if key not in cache:
                        vec=sparse.csr_matrix(text[ix].sum(0),dtype=np.float32) if len(ix) else sparse.csr_matrix((1,text.shape[1]),dtype=np.float32);norm=np.sqrt(vec.multiply(vec).sum());cache[key]=vec/max(norm,1e-12)
                    assert np.all((times[ix]<s['feature_cutoff_exclusive'])|((ix==src)&(times[ix]==s['feature_cutoff_exclusive'])));assert np.all(stages[ix]<=STAGE[t['split']]);xx.append(cache[key]);lookups+=1
                # Same causal relation graph as original model; no new edges.
                adj=(np.asarray(s['positions'])==2).astype(np.float32);root=s['node_ids'].index(source_uid) if source_uid in s['node_ids'] else 0
                row={k:s[k] for k in ['node_ids','sample_nodes','target','window','cutoff','end','feature_cutoff_exclusive']};row.update(x=sparse.vstack(xx,format='csr'),adjacency=adj,depth=depths(np.asarray(s['social_positions'])==2,root))
                if branch=='macro_snapshots':row.update({k:s[k] for k in ['history_summary','popularity_target']})
                out.append(row)
            new[branch]=out
        result.append(new)
        if ti%300==0:print('prepared',ti,flush=True)
    torch.save(result,OUT/'graphs.pt');meta={'complete':True,'prepared_sha256':sha(OUT/'graphs.pt'),'original_sha256':sha('data/pheme_multitask_hourly/graphs.pt'),'records_sha256':sha('data/pheme_history/records.jsonl'),'vectorizer_sha256':sha(OUT/'tfidf.joblib'),'feature_dim':text.shape[1],'fit_records':len(fit),'cutoff_checked_user_lookups':lookups,'strict_past_except_initial_source':True,'raw_targets_sampling_and_duplicates_unchanged':True,'source_sha256':sha(__file__),'adaptation':'Posts are replaced by user nodes with summed/L2-normalized visible past TF-IDF records. Candidate-independent macro graphs retained. Train-only standard TF-IDF vocabulary/IDF, top5000; paper IDF equation differs from standard and author processed file absent.'};(OUT/'completion.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
