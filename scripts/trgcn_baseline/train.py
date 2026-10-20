"""Fixed paper-parameter TRGCN three-action adapter; val-only early stopping."""
import argparse,json,sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,numpy as np
from torch.nn import functional as F
from scipy import sparse
from dsg2former.baselines.trgcn import TRGCN
from dsg2former.baselines.graph_models import CountAdapter
from dsg2former.experiment import set_seed
from scripts.baseline_comparison.train_method import snapshots,action_summary,rank,train_count,count_predict,save_predictions
from scripts.baseline_comparison.prepare_features import sha
R=Path('results/pheme_trgcn_baseline')

def batch_graph(rows):
    counts=[len(s['node_ids']) for _,s in rows];n=max(counts);b=len(rows);d=rows[0][1]['x'].shape[1];xx=[];valid=torch.zeros(b,n,dtype=torch.bool);a=torch.zeros(b,n,n);depth=torch.zeros(b,n,dtype=torch.long)
    for i,((t,s),size) in enumerate(zip(rows,counts)):
        xx.extend([s['x'],sparse.csr_matrix((n-size,d),dtype=np.float32)]);valid[i,:size]=True;a[i,:size,:size]=torch.as_tensor(s['adjacency']);depth[i,:size]=torch.as_tensor(s['depth'])
    coo=sparse.vstack(xx,format='coo');x=torch.sparse_coo_tensor(torch.as_tensor(np.stack([coo.row,coo.col]),dtype=torch.long),torch.as_tensor(coo.data,dtype=torch.float32),size=coo.shape).coalesce();return x,a,valid,depth

def gather(logits,rows):
    return torch.cat([logits[i,torch.as_tensor(s['sample_nodes'])] for i,(_,s) in enumerate(rows)]),np.concatenate([s['target'] for _,s in rows]),np.concatenate([np.repeat(t['thread_id'],len(s['target'])) for t,s in rows])

def groups(rows,rng=None):
    # Length buckets only affect padding/compute; every graph/duplicate row kept.
    order=sorted(range(len(rows)),key=lambda i:len(rows[i][1]['node_ids']));chunks=[order[i:i+64] for i in range(0,len(order),64)]
    if rng is not None:rng.shuffle(chunks)
    for chunk in chunks:yield chunk

@torch.no_grad()
def classify(model,rows):
    model.eval();ys=[];ps=[];ids=[]
    for i in range(0,len(rows),64):
        batch=rows[i:i+64];logits,_=model(*batch_graph(batch));selected,y,tid=gather(logits,batch);ys.append(y);ps.append(selected.softmax(-1).numpy());ids.append(tid)
    return np.concatenate(ys),np.concatenate(ps),np.concatenate(ids)

@torch.no_grad()
def macro_features(model,rows):
    model.eval();result={k:[] for k in ['representation','history','count','target','thread_id','window','cutoff','end']}
    for start in range(0,len(rows),64):
        batch=rows[start:start+64];_,h=model(*batch_graph(batch))
        for i,(t,s) in enumerate(batch):
            n=len(s['node_ids']);values=[h[i,:n].mean(0).numpy(),s['history_summary'],n,s['popularity_target'],t['thread_id'],s['window'],s['cutoff'],s['end']]
            for k,v in zip(result,values):result[k].append(v)
    return {k:np.asarray(v) for k,v in result.items()}

def main(a):
    torch.set_num_threads(1);set_seed(a.seed);plan=json.loads((R/'plan.json').read_text())
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    assert sha(plan['prepared'])==plan['prepared_sha256'];data=torch.load(plan['prepared'],weights_only=False,map_location='cpu');out=R/f'seed_{a.seed}';out.mkdir(parents=True,exist_ok=True);options=plan['options'];model=TRGCN(**options)
    if a.mode!='fit':
        selection=json.loads((R/'selection.json').read_text());assert selection['frozen_before_test'];cp=torch.load(out/'model.pt',weights_only=False,map_location='cpu');model.load_state_dict(cp['state_dict']);y,p,ids=classify(model,snapshots(data,'test'));macro=macro_features(model,snapshots(data,'test','macro_snapshots'));c=torch.load(out/'count_model.pt',weights_only=False,map_location='cpu');count=CountAdapter(c['input_dim']);count.load_state_dict(c['state_dict']);folder=out/('test_predictions' if a.mode=='test' else 'exact_replay');print(json.dumps(save_predictions(folder,y,p,ids,macro,count_predict(count,macro))),flush=True);return
    if (out/'completion.json').exists():return
    tr=snapshots(data,'train');va=snapshots(data,'validation');gcns=[*model.conv1.parameters(),*model.conv2.parameters()];gid={id(p) for p in gcns};optimizer=torch.optim.Adam([{'params':[p for p in model.parameters() if id(p) not in gid]},{'params':gcns,'lr':.0001}],lr=.0005,weight_decay=1e-4);rng=np.random.default_rng(a.seed);best=None;stale=0;history=[];total=sum(len(s['target']) for _,s in tr);started=time.monotonic()
    for epoch in range(1,101):
        model.train();loss_sum=0.
        for indices in groups(tr,rng):
            batch=[tr[i] for i in indices];optimizer.zero_grad(set_to_none=True);logits,_=model(*batch_graph(batch));selected,y,_=gather(logits,batch);loss=F.cross_entropy(selected,torch.as_tensor(y,dtype=torch.long));(loss*len(y)*len(tr)/(total*len(batch))).backward();optimizer.step();loss_sum+=float(loss.detach())*len(y)
        y,p,ids=classify(model,va);m=action_summary(y,p);quality=rank(m);improved=best is None or quality>best
        if improved:best=quality;state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        row={'epoch':epoch,'training_ce':loss_sum/total,'validation':m,'improved':improved,'elapsed':time.monotonic()-started};history.append(row);print(json.dumps({'seed':a.seed,**row}),flush=True)
        if stale>=10:break
    model.load_state_dict(state);torch.save({'state_dict':state,'options':options,'best_epoch':best_epoch,'seed':a.seed,'native_classes':4,'adapted_classes':3},out/'model.pt');(out/'training_history.json').write_text(json.dumps(history,indent=2));y,p,ids=classify(model,va)
    print('encoding causal count train/validation',flush=True);mt=macro_features(model,snapshots(data,'train','macro_snapshots'));mv=macro_features(model,snapshots(data,'validation','macro_snapshots'));count,epoch=train_count(mt,mv,a.seed,out);validation=save_predictions(out/'validation_predictions',y,p,ids,mv,count_predict(count,mv));(out/'completion.json').write_text(json.dumps({'complete':True,'validation':validation,'count_epoch':epoch,'test_used':False,'elapsed':time.monotonic()-started},indent=2));print('FIT COMPLETE',a.seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','test','replay']);p.add_argument('--seed',required=True,type=int);main(p.parse_args())
