"""Train-only scalers, sparse/TCA transfer, and SDNE/t-SNE/GAN image pipeline."""
import argparse,hashlib,json,sys,time,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import torch
from dsg2former.baselines.native import SparseDictionary,GaussianTCA,SDNE,ImageCompensator,spread_pixels
from dsg2former.baselines.native import binary_game
from scripts.baseline_comparison.prepare_features import sha
from scripts.baseline_comparison.knowledge_messages import add_message_entities


def key(s):
 n=s['user_count'];adj=(s['views'][1,:n,:n]>0).astype(np.float32);return str(n)+':'+hashlib.sha256(adj.tobytes()).hexdigest(),adj


def padded_adjacency(items,width=256):
 a=torch.zeros(len(items),max(len(v) for v in items),width);valid=torch.zeros(len(items),width,dtype=torch.bool)
 for i,v in enumerate(items):n=len(v);a[i,:n,:n]=torch.as_tensor(v);valid[i,:n]=True
 return a,valid


def image_for(s,coordinates,channels=8):
 image=np.zeros((channels,16,16),np.float32);mask=np.zeros((16,16),bool);game=np.zeros((2,16,16),np.float32)
 for i,(x,y) in enumerate(coordinates):image[:,x,y]=s['history_signal'][i,-channels:];mask[x,y]=s['observed'][i];game[:,x,y]=s['dpm_game'][i]
 return image,mask,game


def main(a):
 torch.set_num_threads(1);torch.manual_seed(42);rng=np.random.default_rng(42);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return
 started=time.monotonic();graphs=torch.load(Path(a.input)/'graphs.pt',weights_only=False,map_location='cpu');train=[s for t in graphs if t['split']=='train' for s in t['snapshots'] if len(s['target'])]
 x=np.concatenate([s['x'] for s in train]);scaler=StandardScaler().fit(x);joblib.dump(scaler,out/'normalizer.joblib')
 knowledge_audit=add_message_entities(graphs);knowledge_scaler=StandardScaler().fit(np.concatenate([s['k_x'] for s in train]));joblib.dump(knowledge_scaler,out/'knowledge_normalizer.joblib')
 for t in graphs:
  for branch in ['snapshots','macro_snapshots']:
   for s in t[branch]:s['normalized_x']=scaler.transform(s['x']).astype(np.float32);s['k_normalized_x']=knowledge_scaler.transform(s['k_x']).astype(np.float32)
 print(json.dumps({'phase':'normalization','training_graphs':len(train),'training_nodes':len(x)}),flush=True)
 # Original papers explicitly fit multiple linear regression coefficients.
 # Strategy-specific targets only enter these TRAINING fits. Promotion has
 # no observed label in PHEME and is never equated with nonparticipation.
 dpm_x=[];ram_x=[];strategy_y=[]
 for s in train:
  i=s['sample_nodes'];dpm_x.append(s['dpm_factors'][i]);ram_x.append(s['effect_input'][i,:2]);strategy_y.append(np.eye(3)[s['target']][:,:2])
 y=np.concatenate(strategy_y).reshape(-1);dpm_reg=LinearRegression().fit(np.concatenate(dpm_x).reshape(-1,2),y);ram_reg=LinearRegression().fit(np.concatenate(ram_x).reshape(-1,13),y)
 joblib.dump(dpm_reg,out/'dpm_influence_regression.joblib');joblib.dump(ram_reg,out/'ram_influence_regression.joblib')
 for t in graphs:
  for branch in ['snapshots','macro_snapshots']:
   for s in t[branch]:
    effect=dpm_reg.predict(s['dpm_factors'].reshape(-1,2)).reshape(-1,2)
    s['dpm_game']=binary_game(torch.as_tensor(effect),torch.as_tensor(s['fractions'][:,:2])).numpy().astype(np.float32)
 # Sparse dictionary + transfer map are fitted without any held-out rows.
 training_user=np.concatenate([s['normalized_x'][:s['user_count']] for s in train]);dictionary=SparseDictionary(components=16,nonzero=4,iterations=40).fit(training_user);code=dictionary.transform(training_user)
 half=len(code)//2;left=code[rng.choice(half,min(128,half),replace=False)];right=code[half+rng.choice(len(code)-half,min(128,len(code)-half),replace=False)];tca=GaussianTCA(dim=16,sigma=1.,regularization=1.).fit(left,right);joblib.dump({'dictionary':dictionary,'tca':tca,'training_landmarks_only':True,'domain_definition':'earlier versus later training snapshot records','seed':42},out/'ram_preprocessing.joblib')
 print(json.dumps({'phase':'dictionary_tca','training_users':len(training_user),'landmarks':len(tca.landmarks)}),flush=True)
 transform_cache={}
 for t in graphs:
  for branch in ['snapshots','macro_snapshots']:
   for s in t[branch]:
    n=s['user_count'];v=s['normalized_x'][:n];digest=hashlib.sha256(v.tobytes()).hexdigest()
    if digest not in transform_cache:transform_cache[digest]=tca.transform(dictionary.transform(v))
    z=transform_cache[digest];s['ram_x']=z;adj=(s['views'][1,:n,:n]>0).astype(np.float32);similarity=z@z.T;np.fill_diagonal(similarity,0)
    for i in range(n):
     if not s['observed'][i]:continue
     order=np.argsort(-similarity[i],kind='stable')
     for j in [int(j) for j in order if s['observed'][j] and j!=i and similarity[i,j]>=.8][:3]:adj[i,j]=1.
    s['ram_adjacency']=adj
 # SDNE learns actual training adjacency topologies. Sharing a computed layout
 # across identical matrices is only a computation cache; action records and
 # their repeated loss weights are never removed.
 topology={}
 for s in train:
  k,adj=key(s);topology.setdefault(k,adj)
 sdne=SDNE();optimizer=torch.optim.Adam(sdne.parameters(),lr=.001);structures=list(topology.values());sdne_history=[]
 for epoch in range(1,21):
  sdne.train();losses=[];order=rng.permutation(len(structures))
  for start in range(0,len(order),16):
   items=[structures[int(i)] for i in order[start:start+16]];adj,valid=padded_adjacency(items);optimizer.zero_grad(set_to_none=True);loss=sdne.loss(adj,valid);loss.backward();optimizer.step();losses.append(float(loss.detach()))
  sdne_history.append({'epoch':epoch,'loss':float(np.mean(losses))});print(json.dumps({'phase':'SDNE',**sdne_history[-1]}),flush=True)
 torch.save({'state_dict':sdne.state_dict(),'history':sdne_history,'training_topologies':len(topology)},out/'sdne.pt');sdne.eval();layouts={};layout_started=time.monotonic()
 for ti,t in enumerate(graphs):
  for branch in ['snapshots','macro_snapshots']:
   for s in t[branch]:
    k,adj=key(s);n=len(adj)
    if k not in layouts:
     with torch.no_grad():padded,valid=padded_adjacency([adj]);z=sdne(padded)[0][0,:n].numpy()
     if n<=2:coordinates=np.stack((np.arange(n),np.zeros(n)),-1)
     else:coordinates=TSNE(n_components=2,perplexity=min(30.,max(1.,(n-1)/3)),max_iter=250,init='random',learning_rate='auto',method='exact',random_state=42).fit_transform(z)
     # Algorithm1 cuberoot grid step, followed by the declared fixed
     # 16x16 raster adapter needed by a common image classifier.
     extent=max(float(np.max(coordinates))**2,1e-12);step=np.cbrt(extent/max(n,1));native_grid=np.rint(coordinates/step)
     layouts[k]=spread_pixels(native_grid,16)
    image,mask,game=image_for(s,layouts[k]);s['pixel_coordinates']=layouts[k];s['historical_image']=image;s['observed_pixels']=mask;s['game_image']=game
  if ti%200==0:print(json.dumps({'phase':'t-SNE_pixelation','sources':ti,'computed_layouts':len(layouts),'seconds':time.monotonic()-layout_started}),flush=True)
 torch.save(layouts,out/'pixel_layouts.pt');gan=ImageCompensator(channels=8);gopt=torch.optim.Adam(gan.generator.parameters(),lr=.0002,betas=(.5,.999));dopt=torch.optim.Adam(gan.discriminator.parameters(),lr=.0002,betas=(.5,.999));ghistory=[]
 for epoch in range(1,11):
  order=rng.permutation(len(train));dl=[];gl=[];gan.train()
  for start in range(0,len(order),32):
   image=torch.as_tensor(np.stack([train[int(i)]['historical_image'] for i in order[start:start+32]]));mask=torch.as_tensor(np.stack([train[int(i)]['observed_pixels'] for i in order[start:start+32]]));noise=torch.randn_like(image)
   dopt.zero_grad(set_to_none=True);fake=gan.complete(image,mask,noise);d,_=gan.losses(gan.discriminator(image),gan.discriminator(fake.detach()));d.backward();dopt.step()
   gopt.zero_grad(set_to_none=True);fake=gan.complete(image,mask,noise);_,g=gan.losses(gan.discriminator(image),gan.discriminator(fake));g.backward();gopt.step();dl.append(float(d.detach()));gl.append(float(g.detach()))
  ghistory.append({'epoch':epoch,'discriminator':float(np.mean(dl)),'generator':float(np.mean(gl))});print(json.dumps({'phase':'GAN',**ghistory[-1]}),flush=True)
 torch.save({'state_dict':gan.state_dict(),'channels':8,'history':ghistory,'printed_eq12_ambiguity':'Canonical minimax log(1-Dfake) used; printed1-logDfake is unbounded in D'},out/'gan.pt');gan.eval()
 # Fixed inference noise is independent of UID, target, source size and split.
 fixed_noise=torch.as_tensor(np.random.default_rng(42).normal(size=(1,8,16,16)).astype(np.float32))
 for t in graphs:
  for branch in ['snapshots','macro_snapshots']:
   for start in range(0,len(t[branch]),32):
    batch=t[branch][start:start+32];image=torch.as_tensor(np.stack([s['historical_image'] for s in batch]));mask=torch.as_tensor(np.stack([s['observed_pixels'] for s in batch]))
    with torch.no_grad():completed=gan.complete(image,mask,fixed_noise.expand(len(batch),-1,-1,-1)).numpy()
    for s,v in zip(batch,completed):
     # Both model inputs are solely historical; the fixed observed pixels
     # remain exact, including original zero observations.
     assert np.array_equal(v[:,s['observed_pixels']],s['historical_image'][:,s['observed_pixels']]);s['completed_image']=v
 torch.save(graphs,out/'graphs.pt');metadata={'complete':True,'base_features_sha256':sha(Path(a.input)/'graphs.pt'),'source_sha256':sha(__file__),'output_graphs_sha256':sha(out/'graphs.pt'),'knowledge_message_audit':knowledge_audit,'train_only_scaler_dictionary_tca_sdne_gan':True,'preprocessing_seed':42,'tca_landmarks':len(tca.landmarks),'SDNE_training_structures':len(structures),'total_TSNE_layouts':len(layouts),'raw_supervised_rows_preserved':True,'causal_macro_images_never_use_action_candidates':True,'seconds':time.monotonic()-started}
 (out/'completion.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',default='data/pheme_paper_baselines');p.add_argument('--output',default='data/pheme_paper_baseline_resources');main(p.parse_args())
