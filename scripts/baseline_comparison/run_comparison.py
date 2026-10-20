"""Frozen validation selection and three-seed published-baseline comparison."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.baseline_comparison.prepare_features import sha

CONFIGS={
 'ICS-SVM':{'paper':{'nests':8,'iterations':10}},
 'K-GraphSAGE':{'paper':{'lr':.003,'epochs':120,'min_epochs':20,'patience':10}},
 'RAM-GAT':{'paper':{'lr':.005,'epochs':60,'weight_decay':.5,'min_epochs':60,'patience':10}},
 'TRS-GCN':{'paper':{'lr':.01,'epochs':50,'weight_decay':.0005,'min_epochs':20,'patience':10}},
 'Rumor-DPM':{v:{'lr':.01,'epochs':20,'min_epochs':20,'patience':10} for v in ['A','B','C']}}


def run(a):
 root=Path(a.root);root.mkdir(parents=True,exist_ok=True);prepared=Path(a.prepared);assert json.loads((prepared/'completion.json').read_text())['complete']
 path=root/'plan.json'
 if not path.exists():
  sources=[*Path('dsg2former/baselines').glob('*.py'),*Path('scripts/baseline_comparison').glob('*.py')]
  plan={'version':1,'created_before_fits':True,'accepted_protocol':'Native module verification plus explicitly named three-action PHEME task adapters, confirmed by user','prepared':str(prepared),'prepared_sha256':sha(prepared/'graphs.pt'),'raw_graph_sha256':sha('data/pheme_multitask_hourly/graphs.pt'),'reference_model':'results/pheme_pooled_optimization/hourly_spatial_fresh','seeds':[42,2025,2026],'configurations':CONFIGS,'source_sha256':{str(p):sha(p) for p in sources},
   'selection':'All fixed native configurations run for all three seeds. DPM A/B/C screen seed42, best2 by pooled validation Accuracy then CE then AUC confirm all seeds; choose by three-seed mean same rank. All choices frozen before new baseline test inference. SDS is a fixed spatial-only ablation of the already validated hourly encoder.',
   'metrics':'Pool every action row/window within each seed. Accuracy/micro Recall/micro F1; micro OvR probability AUC; MSE/RMSE/MSLE. Mean/sample SD after scoring each seed separately.',
   'data':'Unchanged source chronological8:1:1, fixed1h windows, labels, repeated action rows, sampled nonparticipants, candidate-independent macro graph; no new test labeling or sampling.',
   'classification_training':'Graph models use raw-row weighted CE under uniform graph permutations. ICS tunes RBF C/sigma on validation using ICS. RAM hidden representation trained with CE then final train-only calibrated linear SVM checked each10epochs. K uses native knowledge/neighborhood pretraining20epochs, frozen entity/relation stage, then GraphSAGE classifier training. DPM uses full SDNE/tSNE/grid/GAN pipeline, 8 observed slices plus two game channels and TableIII A/B/C.',
   'macro_extension':'All baselines add frozen-encoder CountAdapter: H mean plus causal history/count, hidden12 Tanh/Softplus, pooled MSLE, Adam.001,max100,min20,patience10. DPM uses its global CNN image representation. Candidate-independent inputs only; no summing sampled candidate probabilities.',
   'preprocessing':'Fixed seed42 shared feature resources. Scaler, sparse dictionary, TCA, SDNE, GAN and DeepLex fitted on train only. TCA uses128+128 training landmarks and inductive transforms; tSNE at inference uses only current causal graph without labels or encoder updates.',
   'native_outputs':'ICS binary; KGraphSAGE3; RumorDPM binary; RAM4 including promotion; TRS binary with0.6native decision; SDS3. Native constructors/kernels independently tested. Only adapted3class outputs enter shared comparison.',
   'limits':'Missing demographics are zero/missing, archive replies proxy for retweets/followers, native ELmo/Chinese lexicon/LLM keyword source are replaced by declared cached-BERT/English-lexicon/keyword adapters. Original paper datasets/scores and ambiguous unspecified original code are not recovered.',
   'reproduction_spec':'artifacts/pheme_baseline_reproduction_spec.json','reproduction_spec_sha256':sha('artifacts/pheme_baseline_reproduction_spec.json'),
   'test_history':'Reference test scores were already observed; this is not a fresh blind test.'}
  path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
 plan=json.loads(path.read_text())
 for p,h in plan['source_sha256'].items():assert sha(p)==h,p
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def launch(tasks,mode='fit'):
  queue=list(tasks);running=[]
  while queue or running:
   while queue and len(running)<a.workers:
    method,variant,seed=queue.pop(0);folder=root/method/variant/f'seed_{seed}';done=folder/('completion.json' if mode=='fit' else 'test_predictions/metrics.json')
    if done.exists():continue
    log=(root/f'{mode}_{method}_{variant}_seed_{seed}.log').open('w');proc=subprocess.Popen([sys.executable,'scripts/baseline_comparison/train_method.py',mode,'--root',str(root),'--method',method,'--variant',variant,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT);running.append((proc,log,method,variant,seed));print(json.dumps({'launched':method,'variant':variant,'seed':seed,'mode':mode}),flush=True)
   pending=[]
   for proc,log,method,variant,seed in running:
    code=proc.poll()
    if code is None:pending.append((proc,log,method,variant,seed))
    else:
     log.close()
     if code:raise RuntimeError(f'{method} {variant} {seed} {mode} failed; inspect log')
     print(json.dumps({'completed':method,'variant':variant,'seed':seed,'mode':mode}),flush=True)
   running=pending
   if running:time.sleep(1)
 def metrics(method,variant,seed):return json.loads((root/method/variant/f'seed_{seed}/validation_predictions/metrics.json').read_text())['action']
 def rank(rows):return (round(float(np.mean([r['accuracy'] for r in rows])),10),-float(np.mean([r['cross_entropy'] for r in rows])),float(np.mean([r['auc'] for r in rows])))
 launch([(m,v,42) for m,variants in plan['configurations'].items() for v in variants])
 dpm=sorted(CONFIGS['Rumor-DPM'],key=lambda v:rank([metrics('Rumor-DPM',v,42)]),reverse=True)[:2]
 (root/'screening.json').write_text(json.dumps({'Rumor-DPM_confirmation':dpm,'seed':42,'test_scores_used':False},indent=2))
 confirmed={m:(dpm if m=='Rumor-DPM' else list(vs)) for m,vs in CONFIGS.items()}
 launch([(m,v,s) for m,vs in confirmed.items() for v in vs for s in plan['seeds']])
 selected={m:max(vs,key=lambda v:rank([metrics(m,v,s) for s in plan['seeds']])) for m,vs in confirmed.items()}
 selection={'selected_variants':selected,'frozen_before_test':True,'selection_uses_only_validation':True,'plan_sha256':sha(path),'confirmed':{m:{v:[metrics(m,v,s) for s in plan['seeds']] for v in vs} for m,vs in confirmed.items()}}
 selection_path=root/'selection.json'
 if selection_path.exists():assert json.loads(selection_path.read_text())==selection
 else:selection_path.write_text(json.dumps(selection,indent=2))
 print(json.dumps(selection,indent=2),flush=True);launch([(m,v,s) for m,v in selected.items() for s in plan['seeds']],mode='test')
 (root/'completion.json').write_text(json.dumps({'complete':True,'selected_variants':selected,'test_runs':15,'SDSformer_runs':3,'reference_runs':3},indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='results/pheme_paper_baselines');p.add_argument('--prepared',default='data/pheme_paper_baseline_resources');p.add_argument('--workers',type=int,default=3);run(p.parse_args())
