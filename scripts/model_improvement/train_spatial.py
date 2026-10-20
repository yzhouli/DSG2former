"""Validation-only spatial search for the explicit DSG2former+ extension."""
import argparse,json,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from torch.nn import functional as F
from dsg2former.enhanced_model import EnhancedDSG2former
from dsg2former.experiment import set_seed
from scripts.baseline_comparison.prepare_features import sha
from scripts.baseline_comparison.train_method import action_summary,rank

CONFIGS={
 'd48_regular':{'dim':48,'lr':.001,'dropout':.2,'weight_decay':.001},
 'd48_slow':{'dim':48,'lr':.0003,'dropout':.2,'weight_decay':.001},
 'd48_hier':{'dim':48,'lr':.001,'dropout':.2,'weight_decay':.001,'hierarchical':True},
 'd96_regular':{'dim':96,'lr':.0005,'dropout':.2,'weight_decay':.001},
 'd24_regular':{'dim':24,'lr':.001,'dropout':.1,'weight_decay':.001},
 'd48_global':{'dim':48,'lr':.001,'dropout':.2,'weight_decay':.001,'local':False},
 'd48_strong_reg':{'dim':48,'lr':.001,'dropout':.35,'weight_decay':.01},
 'feature_mlp_control':{'dim':48,'lr':.001,'dropout':.2,'weight_decay':.001,'blocks':0,'eligible':False}}


def pack(batch):
 N=max(len(s['x']) for _,s in batch);B=len(batch);D=batch[0][1]['x'].shape[1];x=torch.zeros(B,N,D);drivers=torch.zeros(B,N,2,3);positions=torch.zeros(B,N,N,dtype=torch.long);adj=torch.zeros(B,N,N);role=torch.zeros(B,N,dtype=torch.long);valid=torch.zeros(B,N,dtype=torch.bool);counts=[]
 for i,(_,s) in enumerate(batch):
  n=len(s['x']);x[i,:n]=torch.from_numpy(s['x']);drivers[i,:n]=torch.from_numpy(s['drivers']);positions[i,:n,:n]=torch.as_tensor(s['positions'],dtype=torch.long);adj[i,:n,:n]=torch.from_numpy(s['adjacency']);role[i,:n]=torch.from_numpy(s['role']);valid[i,:n]=True;counts.append(s['user_count'])
 return x,drivers,positions,adj,role,valid,torch.tensor(counts,dtype=torch.long)


def gather(logits,representation,batch):
 y=[];p=[];z=[];ids=[]
 for i,(t,s) in enumerate(batch):
  idx=torch.as_tensor(s['sample_nodes']);p.append(logits[i,idx]);z.append(representation[i,idx]);y.extend(s['target']);ids.extend([t['thread_id']]*len(idx))
 return torch.cat(p),torch.cat(z),torch.tensor(y,dtype=torch.long),np.array(ids)

@torch.no_grad()
def evaluate(model,batch,save=None):
 model.eval();truth=[];prob=[];ids=[]
 for start in range(0,len(batch),32):
  b=batch[start:start+32];logits,z=model.spatial(*pack(b));l,h,y,tid=gather(logits,z,b);truth.append(y.numpy());prob.append(l.softmax(-1).numpy());ids.append(tid)
 y,p,tid=np.concatenate(truth),np.concatenate(prob),np.concatenate(ids);m=action_summary(y,p)
 if save:
  save.mkdir(parents=True,exist_ok=True);np.savez_compressed(save/'action_predictions.npz',target=y,probability=p,thread_id=tid);(save/'metrics.json').write_text(json.dumps(m,indent=2))
 return m


def run(a):
 torch.set_num_threads(1);set_seed(a.seed);root=Path(a.root);out=root/a.config/f'seed_{a.seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'spatial_completion.json').exists():return
 plan=json.loads((root/'spatial_plan.json').read_text());cfg=plan['configurations'][a.config]
 for p,h in plan['source_sha256'].items():assert sha(p)==h,p
 assert sha('data/pheme_model_improvement/graphs.pt')==plan['prepared_sha256'];data=torch.load('data/pheme_model_improvement/graphs.pt',weights_only=False,map_location='cpu');train=[(t,s) for t in data if t['split']=='train' for s in t['snapshots'] if len(s['target'])];val=[(t,s) for t in data if t['split']=='validation' for s in t['snapshots'] if len(s['target'])];D=train[0][1]['x'].shape[1];options={k:cfg[k] for k in ['dim','dropout','local','hierarchical','blocks'] if k in cfg};options['input_dim']=D;model=EnhancedDSG2former(**options);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=cfg['lr']*.1);rng=np.random.default_rng(a.seed);total=sum(len(s['target']) for _,s in train);best=None;state=None;history=[];stale=0;started=time.monotonic()
 for epoch in range(1,101):
  model.train();order=rng.permutation(len(train));loss_sum=0.
  for start in range(0,len(order),32):
   b=[train[int(i)] for i in order[start:start+32]];optimizer.zero_grad(set_to_none=True);l,z=model.spatial(*pack(b));logits,_,target,_=gather(l,z,b);loss=F.cross_entropy(logits,target,reduction='sum',label_smoothing=.02);objective=loss*len(train)/(total*len(b));objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())
  scheduler.step();m=evaluate(model,val);r=rank(m);improved=best is None or r>best
  if improved:best=r;state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
  else:stale+=1
  history.append({'epoch':epoch,'train_ce':loss_sum/total,'validation':m,'improved':improved});print(json.dumps({'config':a.config,'seed':a.seed,'epoch':epoch,'validation':m,'seconds':time.monotonic()-started}),flush=True)
  if epoch>=20 and stale>=12:break
 model.load_state_dict(state);torch.save({'state_dict':state,'options':options,'config':a.config,'seed':a.seed,'best_epoch':best_epoch,'prepared_sha256':plan['prepared_sha256']},out/'spatial_model.pt');evaluate(model,val,save=out/'spatial_validation');(out/'spatial_history.json').write_text(json.dumps(history,indent=2));(out/'spatial_completion.json').write_text(json.dumps({'complete':True,'best_epoch':best_epoch,'validation':history[best_epoch-1]['validation'],'seconds':time.monotonic()-started,'test_evaluated':False},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--root',default='results/pheme_model_improvement');run(p.parse_args())
