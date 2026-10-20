"""Train/validate one fully traced original-paper PHEME adaptation."""
import argparse,json,sys,time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1]/'pheme_pipeline')]
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from dsg2former.experiment import set_seed
from dsg2former.baselines.native import ImprovedCuckooSearch,rbf_svm,RumorDPMCNN
from dsg2former.baselines.graph_models import KGraphSAGE,RAMGAT,TRSGCN,CountAdapter
from dsg2former.paper_optimizer import TensorFlowAdam
from dsg2former.pooled_metrics import pooled_action_metrics,pooled_popularity_metrics
from scripts.baseline_comparison.prepare_features import sha


def snapshots(data,split,branch='snapshots'):
 return [(t,s) for t in data if t['split']==split for s in t[branch] if branch=='macro_snapshots' or len(s['target'])]

def action_summary(y,p):
 m=pooled_action_metrics(y,p);m['cross_entropy']=float(-np.log(np.maximum(p[np.arange(len(y)),y].astype(float),1e-12)).mean());return m

def rank(m):return (round(m['accuracy'],10),-m['cross_entropy'],m['auc'])

def graph_batch(batch,method):
 ss=[s for _,s in batch];counts=[s['user_count'] if method=='RAM-GAT' else len(s.get('k_x',s['x'])) if method=='K-GraphSAGE' else len(s['x']) for s in ss];size=max(counts);features=16 if method=='RAM-GAT' else 32;b=len(ss)
 x=torch.zeros(b,size,features);valid=torch.zeros(b,size,dtype=torch.bool);adj=torch.zeros(b,size,size);views=torch.zeros(b,3,size,size);effect=torch.zeros(b,size,3,13);proportion=torch.zeros(b,size,3);edge_features=torch.zeros(b,4,size,size) if method=='TRS-GCN' and all('edge_features' in s for s in ss) else None
 for i,(s,n) in enumerate(zip(ss,counts)):
  x[i,:n]=torch.as_tensor(s['ram_x'] if method=='RAM-GAT' else s.get('k_normalized_x',s['normalized_x']) if method=='K-GraphSAGE' else s['normalized_x']);valid[i,:n]=True
  adj[i,:n,:n]=torch.as_tensor(s['ram_adjacency'] if method=='RAM-GAT' else s.get('k_adjacency',s['adjacency']) if method=='K-GraphSAGE' else s['adjacency'])
  if method=='TRS-GCN':
   views[i,:,:n,:n]=torch.as_tensor(s['views'])
   if edge_features is not None:edge_features[i,:,:n,:n]=torch.as_tensor(s['edge_features'])
  if method=='RAM-GAT':effect[i,:n]=torch.as_tensor(s['effect_input']);proportion[i,:n]=torch.as_tensor(s['fractions'])
 return x,valid,adj,views,effect,proportion,edge_features

def forward(model,batch,method):
 if method=='Rumor-DPM':
  image=torch.as_tensor(np.stack([np.concatenate((s['completed_image'][-8:],s['game_image'])) for _,s in batch]));return model(image)
 x,valid,adj,views,effect,proportion,edge_features=graph_batch(batch,method)
 if method=='RAM-GAT':return model(x,adj,valid,effect,proportion)
 if method=='TRS-GCN':return model(x,views,valid,edge_features)
 return model(x,adj,valid)

def gather(logits,h,batch,method):
 selected=[];representations=[];truth=[];ids=[]
 for i,(t,s) in enumerate(batch):
  indices=s['sample_nodes'];truth.append(s['target']);ids.extend([t['thread_id']]*len(indices))
  if method=='Rumor-DPM':
   pixel=s['pixel_coordinates'][indices];selected.append(logits[i,torch.as_tensor(pixel[:,0]),torch.as_tensor(pixel[:,1])]);representations.append(h[i].expand(len(indices),-1))
  else:selected.append(logits[i,torch.as_tensor(indices)]);representations.append(h[i,torch.as_tensor(indices)])
 return torch.cat(selected),torch.cat(representations),np.concatenate(truth),np.array(ids)

@torch.no_grad()
def classify(model,batch,method,svm=None):
 model.eval();prob=[];truth=[];ids=[];representation=[]
 for start in range(0,len(batch),32):
  b=batch[start:start+32];logits,h=forward(model,b,method);selected,z,y,tid=gather(logits,h,b,method);representation.append(z.numpy());truth.append(y);ids.append(tid)
  if svm is None:prob.append(selected.softmax(-1).numpy())
 z=np.concatenate(representation);y=np.concatenate(truth);p=np.concatenate(prob) if svm is None else svm.predict_proba(z)
 return y,p,np.concatenate(ids),z

def fit_linear_svm(z,y,seed):
 estimator=make_pipeline(StandardScaler(),LinearSVC(C=1.,dual='auto',max_iter=5000,random_state=seed))
 return CalibratedClassifierCV(estimator=estimator,method='sigmoid',cv=3,n_jobs=1).fit(z,y)

@torch.no_grad()
def macro_features(model,batch,method,scaler=None):
 vectors=[];history=[];count=[];target=[];ids=[];window=[];cutoff=[];end=[]
 if model is not None:model.eval()
 for start in range(0,len(batch),32):
  b=batch[start:start+32]
  if method!='ICS-SVM':_,h=forward(model,b,method)
  for i,(t,s) in enumerate(b):
   n=s['user_count'];v=scaler.transform(s['ics']).mean(0) if method=='ICS-SVM' else (h[i].numpy() if method=='Rumor-DPM' else h[i,:n].mean(0).numpy())
   vectors.append(v);history.append(s['history_summary']);count.append(n);target.append(s['popularity_target']);ids.append(t['thread_id']);window.append(s['window']);cutoff.append(s['cutoff']);end.append(s['end'])
 return {'representation':np.stack(vectors),'history':np.stack(history),'count':np.array(count,dtype=np.float32),'target':np.array(target,dtype=np.float32),'thread_id':np.array(ids),'window':np.array(window),'cutoff':np.array(cutoff),'end':np.array(end)}

def train_count(train,val,seed,out):
 set_seed(seed+71);model=CountAdapter(train['representation'].shape[1]);optimizer=TensorFlowAdam(model.parameters(),lr=.001);rng=np.random.default_rng(seed+71);before_test={};history=[];best=None;state=None;stale=0
 tensors={name:{k:torch.as_tensor(d[k],dtype=torch.float32) for k in ['representation','history','count','target']} for name,d in [('train',train),('validation',val)]}
 for epoch in range(1,101):
  model.train();losses=[];tr=tensors['train']
  for _ in range(int(np.ceil(len(train['target'])/256))):
   idx=torch.as_tensor(rng.integers(len(train['target']),size=256));optimizer.zero_grad(set_to_none=True);p=model(tr['representation'][idx],tr['history'][idx],tr['count'][idx]);loss=F.mse_loss(torch.log1p(p),torch.log1p(tr['target'][idx]));loss.backward();optimizer.step();losses.append(float(loss.detach()))
  model.eval();v=tensors['validation']
  with torch.no_grad():p=model(v['representation'],v['history'],v['count']).numpy()
  m=pooled_popularity_metrics(val['target'],p);r=(-m['msle'],-m['mse']);improved=best is None or r>best
  if improved:best=r;state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
  else:stale+=1
  history.append({'epoch':epoch,'training_msle':float(np.mean(losses)),'validation':m,'improved':improved})
  if epoch>=20 and stale>=10:break
 model.load_state_dict(state);torch.save({'state_dict':state,'input_dim':train['representation'].shape[1],'best_epoch':best_epoch,'seed':seed},out/'count_model.pt');(out/'count_history.json').write_text(json.dumps(history,indent=2))
 return model,best_epoch

@torch.no_grad()
def count_predict(model,features):
 model.eval();return model(*[torch.as_tensor(features[k],dtype=torch.float32) for k in ['representation','history','count']]).numpy()

def save_predictions(out,y,p,ids,macro,estimated):
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'action_predictions.npz',target=y,probability=p,thread_id=ids)
 np.savez_compressed(out/'popularity_predictions.npz',prediction=estimated,**{k:v for k,v in macro.items() if k not in ['representation','history','count']},previous_new_users=macro['history'][:,1])
 m={'action':action_summary(y,p),'popularity':pooled_popularity_metrics(macro['target'],estimated)};(out/'metrics.json').write_text(json.dumps(m,indent=2));return m


def pretrain_knowledge(model,train,seed,out):
 rng=np.random.default_rng(seed);optimizer=torch.optim.Adam(model.parameters(),lr=.003);history=[]
 for epoch in range(1,21):
  model.train();order=rng.permutation(len(train));losses=[]
  for start in range(0,len(order),32):
   b=[train[int(i)] for i in order[start:start+32]];x,valid,adj,*_=graph_batch(b,'K-GraphSAGE');entity=model.entity(x);_,h=model(x,adj,valid);terms=[]
   for i,(_,s) in enumerate(b):
    triples=s.get('k_triples',s['triples']);n=len(s.get('k_x',s['x']))
    if len(triples):
     if len(triples)>64:triples=triples[rng.choice(len(triples),64,replace=False)]
     triples=torch.as_tensor(triples);negative=torch.as_tensor(rng.integers(n,size=len(triples)));terms.append(model.knowledge.margin_loss(entity[i,triples[:,0]],triples[:,1],entity[i,triples[:,2]],entity[i,negative]))
    edges=np.transpose(np.nonzero(s.get('k_adjacency',s['adjacency'])>0))
    if len(edges):
     edges=edges[rng.choice(len(edges),min(64,len(edges)),replace=False)];negative=torch.as_tensor(rng.integers(n,size=(len(edges),5)));terms.append(model.neighborhood_loss(h[i,:n],torch.as_tensor(edges),negative))
   if not terms:continue
   optimizer.zero_grad(set_to_none=True);loss=torch.stack(terms).mean();loss.backward();optimizer.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':float(np.mean(losses)) if losses else 0.});print(json.dumps({'phase':'TranSparse_GraphSAGE_pretrain','seed':seed,**history[-1]}),flush=True)
 # Entity and relation embeddings are the preceding representation stage.
 for component in [model.entity,model.knowledge]:
  for p in component.parameters():p.requires_grad_(False)
 torch.save({'state_dict':model.state_dict(),'history':history},out/'knowledge_pretrained.pt')


def run(a):
 torch.set_num_threads(1);set_seed(a.seed);root=Path(a.root);plan=json.loads((root/'plan.json').read_text());cfg=plan['configurations'][a.method][a.variant];out=root/a.method/a.variant/f'seed_{a.seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return
 for p,h in plan['source_sha256'].items():assert sha(p)==h,p
 data_path=Path(plan['prepared'])/'graphs.pt';assert sha(data_path)==plan['prepared_sha256'];data=torch.load(data_path,weights_only=False,map_location='cpu');train=snapshots(data,'train');val=snapshots(data,'validation');started=time.monotonic();rng=np.random.default_rng(a.seed);classifier=None;ics_scaler=None;model=None;model_options={}
 if a.method=='ICS-SVM':
  def arrays(batch):return np.concatenate([s['ics'][s['sample_nodes']] for _,s in batch]),np.concatenate([s['target'] for _,s in batch]),np.concatenate([np.repeat(t['thread_id'],len(s['target'])) for t,s in batch])
  tx,ty,_=arrays(train);vx,vy,vid=arrays(val);ics_scaler=StandardScaler().fit(tx);tx=ics_scaler.transform(tx);vx=ics_scaler.transform(vx);evaluations=[];cache={}
  def objective(log_parameters):
   key=tuple(np.round(log_parameters,8))
   if key in cache:return cache[key]
   c,sigma=2.**np.asarray(log_parameters);svm=rbf_svm(c,sigma,a.seed);svm.fit(tx,ty);accuracy=float(np.mean(svm.predict(vx)==vy));cache[key]=accuracy;evaluations.append({'log2_parameters':list(key),'accuracy':accuracy});return accuracy
  best,trace=ImprovedCuckooSearch(seed=a.seed,nests=cfg['nests'],iterations=cfg['iterations']).optimize(objective,[-5,-8],[10,5],initial=np.log2([2,.033]));c,sigma=2.**best;classifier=rbf_svm(c,sigma,a.seed,True).fit(tx,ty);joblib.dump({'classifier':classifier,'scaler':ics_scaler,'C':c,'sigma':sigma},out/'svm.joblib');(out/'search_history.json').write_text(json.dumps({'trace':trace,'evaluations':evaluations},indent=2));yv,pv,idv=vy,classifier.predict_proba(vx),vid
  torch.save({'model':'ICS-SVM','native_classes':2,'adapted_classes':3,'C':c,'sigma':sigma,'seed':a.seed,'prepared_sha256':plan['prepared_sha256']},out/'model.pt')
 else:
  if a.method=='K-GraphSAGE':
   heads=[set() for _ in range(5)];tails=[set() for _ in range(5)]
   for t,s in train:
    entities=s['k_entity_ids']
    for h,r,t0 in s['k_triples']:heads[r].add(entities[h]);tails[r].add(entities[t0])
   model_options={'head_counts':[max(1,len(x)) for x in heads],'tail_counts':[max(1,len(x)) for x in tails],'seed':a.seed};model=KGraphSAGE(**model_options);pretrain_knowledge(model,train,a.seed,out)
  elif a.method=='RAM-GAT':
   model_options={'classes':3};model=RAMGAT(**model_options);reg=joblib.load(Path(plan['prepared'])/'ram_influence_regression.joblib')
   with torch.no_grad():model.effect.weight.copy_(torch.as_tensor(reg.coef_)[None]);model.effect.bias.copy_(torch.as_tensor([reg.intercept_]))
   for p in model.effect.parameters():p.requires_grad_(False)
  elif a.method=='TRS-GCN':model_options={'classes':3};model=TRSGCN(**model_options)
  elif a.method=='Rumor-DPM':model_options={'channels':10,'classes':3,'variant':a.variant};model=RumorDPMCNN(**model_options)
  else:raise ValueError(a.method)
  total_training_rows=sum(len(s['target']) for _,s in train)
  optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg.get('weight_decay',0.));history=[];best=None;best_state=None;best_classifier=None;stale=0
  for epoch in range(1,cfg['epochs']+1):
   model.train();order=rng.permutation(len(train));loss_sum=0.;rows=0
   for start in range(0,len(order),32):
    batch=[train[int(i)] for i in order[start:start+32]];optimizer.zero_grad(set_to_none=True);logits,h=forward(model,batch,a.method);selected,z,y,_=gather(logits,h,batch,a.method);target=torch.as_tensor(y,dtype=torch.long);loss=F.cross_entropy(selected,target)
    # Global row objective under a uniform graph permutation; no information-
    # equal loss or duplicate-row removal.
    weight=len(train)/(total_training_rows*len(batch));scaled=loss*len(y)*weight;scaled.backward();optimizer.step();loss_sum+=float(loss.detach())*len(y);rows+=len(y)
   classifier_now=None
   if a.method=='RAM-GAT' and (epoch%10==0 or epoch==cfg['epochs']):
    yt,_,_,zt=classify(model,train,a.method);classifier_now=fit_linear_svm(zt,yt,a.seed)
   if a.method=='RAM-GAT' and classifier_now is None:
    history.append({'epoch':epoch,'training_ce':loss_sum/max(rows,1),'classifier_evaluated':False});continue
   yv,pv,idv,_=classify(model,val,a.method,classifier_now);m=action_summary(yv,pv);r=rank(m);improved=best is None or r>best
   if improved:best=r;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch;best_classifier=classifier_now;stale=0
   else:stale+=1
   history.append({'epoch':epoch,'training_ce':loss_sum/max(rows,1),'validation':m,'improved':improved,'final_classifier':'calibrated linear SVM' if classifier_now else 'softmax'})
   print(json.dumps({'method':a.method,'variant':a.variant,'seed':a.seed,'epoch':epoch,'validation':m,'seconds':time.monotonic()-started}),flush=True)
   if epoch>=cfg.get('min_epochs',0) and stale>=cfg.get('patience',10):break
  model.load_state_dict(best_state);model.eval();classifier=best_classifier
  if classifier is not None:joblib.dump(classifier,out/'svm.joblib')
  (out/'training_history.json').write_text(json.dumps(history,indent=2));torch.save({'model':a.method,'options':model_options,'state_dict':best_state,'seed':a.seed,'best_epoch':best_epoch,'adapted_classes':3,'native_classes':4 if a.method=='RAM-GAT' else (2 if a.method in ['TRS-GCN','Rumor-DPM'] else 3),'prepared_sha256':plan['prepared_sha256']},out/'model.pt');yv,pv,idv,_=classify(model,val,a.method,classifier)
 macro_train=macro_features(model,snapshots(data,'train','macro_snapshots'),a.method,ics_scaler);macro_val=macro_features(model,snapshots(data,'validation','macro_snapshots'),a.method,ics_scaler);count_model,count_epoch=train_count(macro_train,macro_val,a.seed,out)
 validation=save_predictions(out/'validation_predictions',yv,pv,idv,macro_val,count_predict(count_model,macro_val));(out/'selection.json').write_text(json.dumps({'classifier_frozen_before_test':True,'count_epoch':count_epoch,'validation':validation,'no_test_scores_used':True},indent=2))
 # Variant screening is validation-only. Held-out inference is a separate mode.
 (out/'completion.json').write_text(json.dumps({'complete':True,'validation':validation,'seconds':time.monotonic()-started,'stage':'fit'},indent=2))


def test(a):
 torch.set_num_threads(1);root=Path(a.root);plan=json.loads((root/'plan.json').read_text());selected=json.loads((root/'selection.json').read_text())['selected_variants'];assert selected[a.method]==a.variant
 out=root/a.method/a.variant/f'seed_{a.seed}';data=torch.load(Path(plan['prepared'])/'graphs.pt',weights_only=False,map_location='cpu');cp=torch.load(out/'model.pt',weights_only=False,map_location='cpu');classifier=None;scaler=None;model=None
 if a.method=='ICS-SVM':
  d=joblib.load(out/'svm.joblib');classifier=d['classifier'];scaler=d['scaler'];batch=snapshots(data,'test');x=np.concatenate([s['ics'][s['sample_nodes']] for _,s in batch]);y=np.concatenate([s['target'] for _,s in batch]);ids=np.concatenate([np.repeat(t['thread_id'],len(s['target'])) for t,s in batch]);prob=classifier.predict_proba(scaler.transform(x))
 else:
  cls={'K-GraphSAGE':KGraphSAGE,'RAM-GAT':RAMGAT,'TRS-GCN':TRSGCN,'Rumor-DPM':RumorDPMCNN}[a.method];model=cls(**cp['options']);model.load_state_dict(cp['state_dict']);model.eval();classifier=joblib.load(out/'svm.joblib') if a.method=='RAM-GAT' else None;y,prob,ids,_=classify(model,snapshots(data,'test'),a.method,classifier)
 counts=macro_features(model,snapshots(data,'test','macro_snapshots'),a.method,scaler);saved=torch.load(out/'count_model.pt',weights_only=False,map_location='cpu');count_model=CountAdapter(saved['input_dim']);count_model.load_state_dict(saved['state_dict']);directory=out/('test_predictions' if a.mode=='test' else 'exact_replay');m=save_predictions(directory,y,prob,ids,counts,count_predict(count_model,counts));print(json.dumps({'method':a.method,'seed':a.seed,'test':m}),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','test','replay']);p.add_argument('--method',required=True);p.add_argument('--variant',default='paper');p.add_argument('--seed',type=int,required=True);p.add_argument('--root',default='results/pheme_paper_baselines');a=p.parse_args();run(a) if a.mode=='fit' else test(a)
