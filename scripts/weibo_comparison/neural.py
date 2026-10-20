"""Fresh matched two-block/three-backbone SDS and DSG fits; no test selection."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from dsg2former.experiment import set_seed
from dsg2former.sds_aligned import PaperAlignedDSG2former,align_graph_positions
from dsg2former.neural_popularity import observed_history,pool_micro_state,final_micro_state
from dsg2former.memory_count_gate import MemoryCountGate
from dsg2former.pooled_metrics import pooled_popularity_metrics
from scripts.model_improvement.train_spatial import pack,gather
from scripts.baseline_comparison.train_method import rank,action_summary
from scripts.macro_residual_study.train import fit_base,fit_residual,load_predict,normalized_risk
from scripts.weibo_comparison.prepare_features import setup
from scripts.weibo_comparison.extract import sha

CONFIGS={'d24_regular':dict(dim=24,dropout=.1,lr=.001,weight_decay=.001),
    'd96_regular':dict(dim=96,dropout=.2,lr=.0005,weight_decay=.001),
    'd48_strong_reg':dict(dim=48,dropout=.35,lr=.001,weight_decay=.01)}
PUBLIC_TRAINING=dict(spatial_epochs=100,spatial_min_epochs=20,spatial_patience=12,batch_size=64,temporal_epochs=150,temporal_min_epochs=20,temporal_patience=15)
PUBLIC_ATTENTION='dense'
SEEDS=[42,2025,2026];R=Path('results/neural')
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2))


def spatial_outputs(model,batch,device):
    from dsg2former.sparse_aligned import SparseDSG2former
    if isinstance(model,SparseDSG2former):
        from dsg2former.sparse_data import pack_sparse_snapshots
        return model.spatial_packed(*pack_sparse_snapshots(batch,device))
    return model.spatial(*[v.to(device) for v in pack(batch)])


@torch.no_grad()
def evaluate(model,batch,device):
    model.eval();yy=[];pp=[]
    for a in range(0,len(batch),64):
        b=batch[a:a+64];logits,h=spatial_outputs(model,b,device);l,_,y,_=gather(logits,h,b)
        yy.append(y.numpy());pp.append(l.softmax(-1).cpu().numpy())
    return action_summary(np.concatenate(yy),np.concatenate(pp))


def fit_spatial(data,backbone,seed,device):
    out=R/backbone/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'spatial_model.pt').exists():return
    set_seed(seed);cfg=CONFIGS[backbone];options=dict(input_dim=184,dim=cfg['dim'],heads=3,blocks=2,
        dropout=cfg['dropout'],bias_size=50,mask_rates=(1.,.5),norm_eps=1e-6)
    model=PaperAlignedDSG2former(**options).to(device)
    train=[(t,s) for t in data if t['split']=='train' for s in t['snapshots'] if len(s['target'])]
    val=[(t,s) for t in data if t['split']=='validation' for s in t['snapshots'] if len(s['target'])]
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=PUBLIC_TRAINING['spatial_epochs'],eta_min=cfg['lr']*.1)
    rng=np.random.default_rng(seed);total=sum(len(s['target']) for _,s in train);best=None;history=[];stale=0
    for epoch in range(1,PUBLIC_TRAINING['spatial_epochs']+1):
        model.train();order=rng.permutation(len(train));loss_sum=0
        for a in range(0,len(order),PUBLIC_TRAINING['batch_size']):
            b=[train[int(i)] for i in order[a:a+PUBLIC_TRAINING['batch_size']]];opt.zero_grad(set_to_none=True)
            l,h=spatial_outputs(model,b,device);logits,_,y,_=gather(l,h,b)
            loss=F.cross_entropy(logits,y.to(device),reduction='sum',label_smoothing=.02)
            (loss*len(train)/(total*len(b))).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=float(loss.detach())
        schedule.step();m=evaluate(model,val,device);q=rank(m);improved=best is None or q>best
        if improved:best=q;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
        else:stale+=1
        row=dict(epoch=epoch,training_ce=loss_sum/total,validation=m,improved=improved);history.append(row)
        print(json.dumps(dict(backbone=backbone,seed=seed,stage='spatial',**row)),flush=True)
        if epoch>=PUBLIC_TRAINING['spatial_min_epochs'] and stale>=PUBLIC_TRAINING['spatial_patience']:break
    torch.save(dict(architecture='SparseDSG2former' if PUBLIC_ATTENTION=='sparse' else 'PaperAlignedDSG2former',attention_backend=PUBLIC_ATTENTION,options=options,state_dict=state,
        seed=seed,best_epoch=best_epoch,data_sha256=sha('data/pheme_model_improvement/graphs.pt')),out/'spatial_model.pt')
    write(out/'spatial_history.json',history)


@torch.no_grad()
def cached(model,data,split,device='cpu'):
    """All chunks at the same cutoff read the same strictly earlier cache."""
    model=model.to(device).eval();selected=[t for t in data if t['split']==split];outputs={}
    flat=[(t,s,branch,i) for t in selected for branch in ['snapshots','macro_snapshots'] for i,s in enumerate(t[branch])]
    for a in range(0,len(flat),64):
        b=flat[a:a+64];l,h=spatial_outputs(model,[(t,s) for t,s,_,_ in b],device)
        for j,(t,s,branch,i) in enumerate(b):
            n=len(s['x']);outputs[t['thread_id'],branch,i]=(l[j,:n].cpu().clone(),h[j,:n].cpu().clone())
    model=model.cpu();action=defaultdict(list);common=defaultdict(list);states=defaultdict(list)
    for t in selected:
        for branch in ['snapshots','macro_snapshots']:
            cache={};times={};groups=defaultdict(list)
            for i,s in enumerate(t[branch]):groups[s['window']].append((i,s))
            for w,items in groups.items():
                updates=defaultdict(list)
                for i,s in items:
                    logits,h=outputs.pop((t['thread_id'],branch,i));n=s['user_count'];current=h[:n]
                    previous=torch.zeros_like(current);mask=torch.zeros(n,dtype=torch.bool);age=torch.zeros(n)
                    for j,u in enumerate(s['node_ids']):
                        if u in cache:
                            assert times[u]<s['cutoff'];previous[j]=cache[u];mask[j]=True;age[j]=(s['cutoff']-times[u])/3600
                    temporal=model.action_cached(current,logits[:n],previous,mask,age)
                    if branch=='snapshots':
                        idx=torch.as_tensor(s['sample_nodes'])
                        for k,v in [('current',current[idx]),('logits',logits[idx]),('previous',previous[idx]),('mask',mask[idx]),('age',age[idx]),('target',torch.as_tensor(s['target'],dtype=torch.long))]:action[k].append(v)
                        for k,v in [('thread_id',t['thread_id']),('window',w),('cutoff',s['cutoff']),('end',s['end'])]:action[k].extend([v]*len(idx))
                    else:
                        sequence,_=observed_history(s);common['sequence'].append(sequence)
                        # Final Transformer d coordinates, class probs, count;
                        # same small state input as the PHEME residual study.
                        d=model.encoder.dim
                        for variant,z,score in [('spatial',current,logits[:n]),('temporal',final_micro_state(model,current,previous,mask,age,True),temporal)]:
                            pooled=pool_micro_state(z,score).numpy();states[variant].append(np.r_[pooled[:d],pooled[-4:]])
                        for k,v in [('target',s['popularity_target']),('thread_id',t['thread_id']),('window',w),('cutoff',s['cutoff']),('end',s['end']),('previous_new_users',s['history_summary'][1])]:common[k].append(v)
                    for j,u in enumerate(s['node_ids']):
                        if u in s['state_update_users']:updates[u].append(current[j])
                for u,values in updates.items():cache[u]=torch.stack(values).mean(0);times[u]=items[0][1]['cutoff']
    keys=['current','logits','previous','mask','age','target']
    action={k:torch.cat(v) if k in keys else np.asarray(v) for k,v in action.items()}
    return dict(action=action,macro={k:np.asarray(v) for k,v in common.items()},state={k:np.asarray(v,dtype=np.float32) for k,v in states.items()})


@torch.no_grad()
def action_probability(model,a,variant):
    if variant=='spatial':return a['logits'].softmax(-1).numpy()
    model=model.cpu().eval();parts=[]
    for i in range(0,len(a['target']),2048):
        ix=slice(i,i+2048);parts.append(model.action_cached(*[a[k][ix] for k in ['current','logits','previous','mask','age']]).softmax(-1).numpy())
    return np.concatenate(parts)


def fit_temporal(model,dev,seed,out):
    if (out/'temporal_model.pt').exists():return
    model=model.cpu();tr,va=dev['train']['action'],dev['validation']['action']
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('gate.','temporal_head.')))
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed+211);m=action_summary(va['target'].numpy(),action_probability(model,va,'temporal'))
    best=rank(m);state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0;history=[dict(epoch=0,validation=m,improved=True)]
    for epoch in range(1,PUBLIC_TRAINING['temporal_epochs']+1):
        model.train();order=rng.permutation(len(tr['target']))
        for a in range(0,len(order),512):
            ix=torch.as_tensor(order[a:a+512]);opt.zero_grad(set_to_none=True)
            l=model.action_cached(*[tr[k][ix] for k in ['current','logits','previous','mask','age']]);F.cross_entropy(l,tr['target'][ix],label_smoothing=.02).backward()
            torch.nn.utils.clip_grad_norm_(params,1);opt.step()
        m=action_summary(va['target'].numpy(),action_probability(model,va,'temporal'));q=rank(m);improved=q>best
        if improved:best=q;state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,validation=m,improved=improved))
        if epoch>=PUBLIC_TRAINING['temporal_min_epochs'] and stale>=PUBLIC_TRAINING['temporal_patience']:break
    model.load_state_dict(state);options=torch.load(out/'spatial_model.pt',weights_only=False)['options']
    torch.save(dict(architecture='SparseDSG2former' if PUBLIC_ATTENTION=='sparse' else 'PaperAlignedDSG2former',attention_backend=PUBLIC_ATTENTION,options=options,state_dict=state,best_epoch=best_epoch,spatial_sha256=sha(out/'spatial_model.pt')),out/'temporal_model.pt');write(out/'temporal_history.json',history)


def fit(args):
    data=torch.load('data/pheme_model_improvement/graphs.pt',weights_only=False,map_location='cpu');align_graph_positions(data)
    out=R/args.backbone/f'seed_{args.seed}';fit_spatial(data,args.backbone,args.seed,args.device)
    cp=torch.load(out/'spatial_model.pt',weights_only=False,map_location='cpu');model=PaperAlignedDSG2former(**cp['options']);model.load_state_dict(cp['state_dict'])
    cache=out/'development.pt'
    if not cache.exists():torch.save({s:cached(model,data,s,args.device) for s in ['train','validation']},cache)
    dev=torch.load(cache,weights_only=False,map_location='cpu');fit_temporal(model,dev,args.seed,out)
    # Spatial encoding can be shared, but the final temporal macro state must
    # be recomputed under the selected temporal gate, not its initialization.
    model.load_state_dict(torch.load(out/'temporal_model.pt',weights_only=False,map_location='cpu')['state_dict'])
    final=out/'development_final.pt'
    if not final.exists():torch.save({s:cached(model,data,s,args.device) for s in ['train','validation']},final)
    write(out/'completion.json',dict(complete=True,test_used=False))


def causal_memory(data):
    threads=[json.loads(l) for l in Path('data/pheme_multitask_hourly/threads.jsonl').open()]
    lookup={t['thread_id']:t for t in threads};by_author=defaultdict(list);stages={'train':0,'validation':1,'test':2}
    for t in data:
        first=t['macro_snapshots'][0];author=lookup[t['thread_id']]['source']['uid']
        by_author[author].append((first['end'],stages[t['split']],t['thread_id'],np.log1p(first['popularity_target'])))
    result={}
    for t in data:
        src=lookup[t['thread_id']]['source'];candidates=by_author[src['uid']]
        for s in t['macro_snapshots']:
            eligible=[v for end,stage,tid,v in candidates if end<=s['cutoff'] and stage<=stages[t['split']] and tid!=t['thread_id']]
            result[t['thread_id'],s['window']]=(float(np.mean(eligible)) if eligible else 0.,len(eligible))
    return result


def numeric_common(data):
    action=np.concatenate([s['raw_x'][s['sample_nodes'],32:35] for t in data if t['split']=='train' for s in t['snapshots']])
    scaler=StandardScaler().fit(action);memory=causal_memory(data);out={}
    for split in ['train','validation','test']:
        d=defaultdict(list)
        for t in data:
            if t['split']!=split:continue
            for s in t['macro_snapshots']:
                n=s['user_count'];d['ics'].append(scaler.transform(s['raw_x'][:n,32:35]).mean(0));d['profile'].append(s['raw_x'][:n,:16].mean(0))
                d['observed'].append(np.r_[s['history_summary'][:4],n]);d['sequence'].append(observed_history(s)[0])
                mean,count=memory[t['thread_id'],s['window']];d['memory_mean'].append(mean);d['memory_count'].append(count)
                for k,v in [('target',s['popularity_target']),('thread_id',t['thread_id']),('window',s['window']),('cutoff',s['cutoff']),('end',s['end']),('previous_new_users',s['history_summary'][1])]:d[k].append(v)
        out[split]={k:np.asarray(v) for k,v in d.items()}
    return out


def gate_inputs(a,p):
    return tuple(torch.as_tensor(x,dtype=torch.bool if i==3 else torch.float64) for i,x in enumerate([p,a['memory_mean'],a['memory_count'],a['window']==0]))


def fit_gate(out,tr,va,tp,vp):
    if (out/'memory_gate.pt').exists():return
    head=MemoryCountGate(confidence=5);opt=torch.optim.Adam(head.parameters(),lr=.05);tx=gate_inputs(tr,tp);vx=gate_inputs(va,vp)
    y=torch.as_tensor(tr['target'],dtype=torch.float64);root=tx[-1];scale=max(1.,float(np.mean(tr['target']**2)))
    reference=pooled_popularity_metrics(va['target'],vp);best=None;stale=0;history=[]
    for epoch in range(1,501):
        opt.zero_grad(set_to_none=True);p=head(*tx);loss=(torch.log1p(p[root])-torch.log1p(y[root])).square().mean()+.25*(p[root]-y[root]).square().mean()/scale
        loss.backward();opt.step()
        with torch.no_grad():pred=head(*vx).numpy()
        m=pooled_popularity_metrics(va['target'],pred);q=(normalized_risk(m,reference),m['msle']/reference['msle']+m['mse']/reference['mse']);improved=best is None or q<best
        if improved:best=q;state=copy.deepcopy(head.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,loss=float(loss.detach()),validation=m,improved=improved))
        if epoch>=30 and stale>=50:break
    torch.save(dict(options=dict(confidence=5),state_dict=state,best_epoch=best_epoch),out/'memory_gate.pt');write(out/'memory_history.json',history)


def count_fit(args):
    if not (R/'common.pt').exists():raise FileNotFoundError('Prepare common cache first')
    common=torch.load(R/'common.pt',weights_only=False,map_location='cpu');out=R/args.backbone/f'seed_{args.seed}'
    dev=torch.load(out/'development_final.pt',weights_only=False,map_location='cpu');tr,va=common['train'],common['validation'];seed=args.seed
    base=R/'numeric_base'/f'seed_{seed}'
    if not (base/'completion.json').exists():fit_base(base,tr,va,'ics',seed)
    cfg=dict(first_cap=.05,continuation_cap=.5,fine_tune=False,raw_weight=0.,penalty=.25)
    reference=json.loads((base/'completion.json').read_text())['validation']
    for v in ['spatial','temporal']:
        folder=out/v;ts,vs=dev['train']['state'][v],dev['validation']['state'][v]
        fit_residual(folder,tr,va,ts,vs,cfg,seed,base,reference)
        fit_gate(folder,tr,va,load_predict(folder,tr,ts),load_predict(folder,va,vs))
    write(out/'count_completion.json',dict(complete=True,test_used=False))


def predict(args):
    plan=json.loads((R/'inference_plan.json').read_text())
    assert plan['source_sha256']==sha(__file__)
    assert plan['input_sha256']==sha('data/pheme_model_improvement/graphs.pt')
    for path,h in plan['checkpoint_sha256'].items():assert sha(path)==h,path
    data=torch.load('data/pheme_model_improvement/graphs.pt',weights_only=False,map_location='cpu');align_graph_positions(data)
    common=torch.load(R/'common.pt',weights_only=False,map_location='cpu')['test'];out=R/args.backbone/f'seed_{args.seed}'
    cp=torch.load(out/'temporal_model.pt',weights_only=False,map_location='cpu');model=PaperAlignedDSG2former(**cp['options']);model.load_state_dict(cp['state_dict'])
    d=cached(model,data,'test',args.device)
    for v in ['spatial','temporal']:
        folder=out/v;dest=folder/('test_predictions' if args.mode=='predict' else 'exact_replay');dest.mkdir(exist_ok=True)
        a=d['action'];p=action_probability(model,a,v);parent=load_predict(folder,common,d['state'][v])
        c=torch.load(folder/'memory_gate.pt',weights_only=False,map_location='cpu');gate=MemoryCountGate(**c['options']);gate.load_state_dict(c['state_dict'])
        with torch.no_grad():forecast=gate(*gate_inputs(common,parent)).numpy()
        for key in ['target','thread_id','window','cutoff','end']:np.testing.assert_array_equal(d['macro'][key],common[key])
        np.savez_compressed(dest/'action_predictions.npz',target=a['target'].numpy(),probability=p,**{k:a[k] for k in ['thread_id','window','cutoff','end']})
        np.savez_compressed(dest/'popularity_predictions.npz',prediction=forecast,parent=parent,**{k:common[k] for k in ['target','thread_id','window','cutoff','end','previous_new_users']})
        write(dest/'metrics.json',dict(action=action_summary(a['target'].numpy(),p),popularity=pooled_popularity_metrics(common['target'],forecast)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','common','base','count','freeze','predict','replay']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--backbone',choices=list(CONFIGS));p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cpu');a=p.parse_args();setup(a.root);torch.set_num_threads(1);R.mkdir(parents=True,exist_ok=True)
    if a.mode=='fit':fit(a)
    elif a.mode=='common':
        if not (R/'common.pt').exists():torch.save(numeric_common(torch.load('data/pheme_model_improvement/graphs.pt',weights_only=False,map_location='cpu')),R/'common.pt')
    elif a.mode=='base':
        c=torch.load(R/'common.pt',weights_only=False,map_location='cpu');fit_base(R/'numeric_base'/f'seed_{a.seed}',c['train'],c['validation'],'ics',a.seed)
    elif a.mode=='count':count_fit(a)
    elif a.mode=='freeze':
        files=[p for p in R.rglob('*.pt') if p.name not in ['common.pt','development.pt','development_final.pt']]
        write(R/'inference_plan.json',dict(frozen_before_test=True,checkpoint_sha256={str(p):sha(p) for p in files},source_sha256=sha(__file__),input_sha256=sha('data/pheme_model_improvement/graphs.pt')))
    else:predict(a)
