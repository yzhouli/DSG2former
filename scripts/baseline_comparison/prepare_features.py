"""Causal original-paper feature proxies over unchanged hourly PHEME targets."""
import argparse,hashlib,json,re,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch
from torch.nn import functional as F
from dsg2former.paper_features import HistoricalCorpus
from dsg2former.baselines.text_models import DeepLex,best_keyword_cluster,lexical_sentiment,sentiment_transition,sif_phrase
from dsg2former.baselines.native import influence_kgraph,binary_game


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def tokens(text):return re.findall(r"[#@]?[a-z][a-z'_-]*",re.sub(r'https?://\S+','',text.lower()))
def keywords(words):return [w for w in words if w not in ENGLISH_STOP_WORDS and len(w)>2 and not w.startswith('@')]
def prefix(rows,times,cutoff,source,source_uid,uid):
 chosen=rows[:np.searchsorted(times[rows],cutoff,side='left')]
 if uid==source_uid and times[source]==cutoff:chosen=np.r_[chosen,source]
 return chosen

def text_resources(records,embeddings,teacher,train_mask,out,epochs):
 if (out/'text_record_features.npz').exists():
  resource=joblib.load(out/'text_resources.joblib')
  if resource.get('source_sha256')==sha('dsg2former/baselines/text_models.py') and resource.get('schema_version')==3:
   z=np.load(out/'text_record_features.npz');return z['text'],z['emotion'],resource
 torch.manual_seed(42);rng=np.random.default_rng(42);words=[tokens(r['text']) for r in records];train=np.flatnonzero(train_mask)
 counts=Counter(w for i in train for w in words[int(i)]);vocab={'<pad>':0,'<unk>':1}
 for w,_ in counts.most_common(12000):vocab[w]=len(vocab)
 lexicon={}
 for line in Path('external/vaderSentiment/vader_lexicon.txt').read_text().splitlines():
  bits=line.split('\t')
  if len(bits)>=2:lexicon[bits[0]]=float(bits[1])/4
 from transformers import AutoTokenizer,AutoModel
 tok=AutoTokenizer.from_pretrained('models/pheme_bert',local_files_only=True);bert=AutoModel.from_pretrained('models/pheme_bert',local_files_only=True)
 table=bert.get_input_embeddings().weight.detach().cpu().numpy();lookup=tok(list(vocab),add_special_tokens=False)['input_ids'];word_matrix=np.stack([table[ids].mean(0) if ids else table[tok.unk_token_id] for ids in lookup]);del bert
 pca=PCA(n_components=16,svd_solver='randomized',random_state=42).fit(np.asarray(embeddings[train]));text=pca.transform(embeddings).astype(np.float32)
 word_matrix=pca.transform(word_matrix).astype(np.float32);word_matrix[0]=0;word_matrix/=np.maximum(np.linalg.norm(word_matrix,axis=1,keepdims=True),1e-8)
 df=Counter(w for i in train for w in set(words[int(i)]));idf={w:float(np.log((1+len(train))/(1+df[w]))+1) for w in vocab};length=32
 ids=np.zeros((len(records),length),np.int64);tfidf=np.zeros_like(ids,dtype=np.float32);lex=np.zeros_like(tfidf)
 for i,ws in enumerate(words):
  counts_i=Counter(ws)
  for j,w in enumerate(ws[:length]):ids[i,j]=vocab.get(w,1);tfidf[i,j]=idf.get(w,1)*counts_i[w]/max(1,len(ws));lex[i,j]=lexicon.get(w,0.)
 model=DeepLex(word_matrix);optimizer=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');best_state=None;history=[];patience=0
 boundary=max(1,int(.95*len(train)));fit,dev=train[:boundary],train[boundary:];dev=dev if len(dev) else fit[-1:]
 for epoch in range(1,epochs+1):
  model.train();losses=[]
  for start in range(0,len(fit),256):
   if start==0:permutation=rng.permutation(fit)
   idx=permutation[start:start+256];optimizer.zero_grad(set_to_none=True);logits,_=model(torch.as_tensor(ids[idx]),torch.as_tensor(tfidf[idx]),torch.as_tensor(lex[idx]));loss=F.kl_div(logits.log_softmax(-1),torch.as_tensor(np.asarray(teacher[idx])),reduction='batchmean');loss.backward();optimizer.step();losses.append(float(loss.detach()))
  model.eval();total=0.
  with torch.no_grad():
   for start in range(0,len(dev),256):
    idx=dev[start:start+256];logits,_=model(torch.as_tensor(ids[idx]),torch.as_tensor(tfidf[idx]),torch.as_tensor(lex[idx]));total+=float(F.kl_div(logits.log_softmax(-1),torch.as_tensor(np.asarray(teacher[idx])),reduction='sum'))
  loss=total/len(dev);history.append({'epoch':epoch,'training_kl':float(np.mean(losses)),'training_inner_holdout_kl':loss});print(json.dumps({'phase':'DeepLex','epoch':epoch,'validation_kl':loss}),flush=True)
  if loss<best:best=loss;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
  else:patience+=1
  if patience>=3:break
 model.load_state_dict(best_state);model.eval();probability=[]
 with torch.no_grad():
  for start in range(0,len(records),256):
   sl=slice(start,start+256);logits,_=model(torch.as_tensor(ids[sl]),torch.as_tensor(tfidf[sl]),torch.as_tensor(lex[sl]));probability.append(logits.softmax(-1).numpy())
 probability=np.concatenate(probability);rational_weight=1-float(np.mean([any(w.startswith(('#','@')) for w in words[int(i)]) or max(probability[i,:2])>.9 for i in train]));emotion=np.stack([lexical_sentiment(ws,[lexicon.get(w,0.) for w in ws],p,rational_weight) for ws,p in zip(words,probability)]).astype(np.float32)
 resource={'vocab':vocab,'word_matrix':word_matrix,'frequency':{w:counts[w]/max(sum(counts.values()),1) for w in vocab},'idf':idf,'lexicon':lexicon,'pca':pca,'training_rows':train,'rational_weight':rational_weight,'train_mask_sha256':hashlib.sha256(train_mask.tobytes()).hexdigest(),'source_sha256':sha('dsg2former/baselines/text_models.py'),'schema_version':3}
 joblib.dump(resource,out/'text_resources.joblib');np.savez_compressed(out/'text_record_features.npz',text=text,emotion=emotion);torch.save({'state_dict':best_state,'embedding':word_matrix,'history':history,'target':'Frozen prior-text positive/negative/neutral probabilities; not future actions'},out/'deeplex.pt')
 return text,emotion,resource


def build(a):
 torch.set_num_threads(1);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
 if (out/'completion.json').exists():return
 graph_hash=sha('data/pheme_multitask_hourly/graphs.pt');graphs=torch.load('data/pheme_multitask_hourly/graphs.pt',weights_only=False,map_location='cpu')
 threads=list(map(json.loads,Path('data/pheme_multitask_hourly/threads.jsonl').read_text().splitlines()));records=list(map(json.loads,Path('data/pheme_history/records.jsonl').read_text().splitlines()));embeddings=np.load('data/pheme_history/embeddings.npy',mmap_mode='r');teacher=np.load('data/pheme_history/sentiment.npy',mmap_mode='r')
 labels=np.load('data/pheme_multitask_hourly/action_labels.npz');stages={'train':0,'validation':1,'test':2};stage={t['thread_id']:stages[t['split']] for t in threads};record_stage=np.array([stage.get(r['thread_id'],-1) for r in records]);times=np.array([r['timestamp'] for r in records]);last_train=max(s['feature_cutoff_exclusive'] for t in graphs if t['split']=='train' for s in t['macro_snapshots']);train_mask=(record_stage<=0)&(times<last_train)
 text,emotion,resources=text_resources(records,embeddings,teacher,train_mask,out,a.text_epochs);words=[tokens(r['text']) for r in records];corpora={s:HistoricalCorpus(records,embeddings,teacher,5,.23,record_stage<=code) for s,code in stages.items()};record_labels=np.full(len(records),-1,dtype=np.int8);source_index=[]
 for i,t in enumerate(threads):
  source=corpora[t['split']].path_index[t['source']['archive_path']];source_index.append(source);record_labels[source]=0;begin,end=labels['reaction_offsets'][i:i+2]
  for r,label in zip(t['reactions'],labels['label'][begin:end]):
   if label>=0:record_labels[corpora[t['split']].path_index[r['archive_path']]]=label
 source_index=np.array(source_index);source_times=times[source_index];source_stage=np.array([stages[t['split']] for t in threads]);topic=[];freq=resources['frequency'];vocab=resources['vocab'];word_matrix=resources['word_matrix']
 for src in source_index:
  ws=keywords(words[int(src)]);phrases=[(w,) for w in ws]+list(zip(ws,ws[1:]));ranked=[]
  for phrase in phrases:
   vectors=np.stack([word_matrix[vocab.get(w,1)]+text[src]/max(np.linalg.norm(text[src]),1e-8) for w in phrase]);vector=sif_phrase(vectors,[freq.get(w,0.) for w in phrase]);weight=float(np.linalg.norm(vector))/max(sum(freq.get(w,0.) for w in phrase),1e-8);ranked.append((weight,vector))
  ranked=sorted(ranked,key=lambda x:-x[0])[:10];vec=np.stack([v for _,v in ranked]) if ranked else np.empty((0,16));topic.append(best_keyword_cluster(vec,text[src],seed=42))
 topic=np.asarray(topic,dtype=np.float32);topic/=np.maximum(np.linalg.norm(topic,axis=1,keepdims=True),1e-8);seen_cache={};result=[];max_history_time=-float('inf');lookups=0
 for ti,(thread,gthread) in enumerate(zip(threads,graphs)):
  split=gthread['split'];corpus=corpora[split];src=int(source_index[ti]);source_uid=thread['source']['uid'];source_time=times[src];new={'thread_id':gthread['thread_id'],'event':gthread['event'],'split':split,'rows':gthread['rows'],'snapshots':[],'macro_snapshots':[]}
  for micro,macro in zip(gthread['snapshots'],gthread['macro_snapshots']):
   cutoff=micro['feature_cutoff_exclusive'];known=set(macro['known_participants_before']);past_chain=[]
   for r in thread['reactions']:
    j=corpus.path_index[r['archive_path']]
    if times[j]<cutoff and j!=src:past_chain.append(j)
   past_chain=np.asarray(past_chain,dtype=np.int64);available=(source_stage<=stages[split])&(source_times<cutoff);available[ti]=True;score=.6*(topic@topic[ti])+.4*np.exp(-.1*np.abs(source_times-source_time)/86400);score[~available]=-np.inf
   other=[ti]+[int(j) for j in np.argsort(-score,kind='stable') if j!=ti and available[j]][:4];topic_users=[set() for _ in other]
   for branch,snapshot in [('snapshots',micro),('macro_snapshots',macro)]:
    nodes=snapshot['node_ids'];n=len(nodes);social=snapshot['social_positions']==2;adj=snapshot['positions']==2;incoming=social.sum(0);outgoing=social.sum(1)
    native_activity=np.zeros((n,2),np.float32);native_drive=np.zeros((n,n,2),np.float32);dpm_factors=np.zeros((n,2,2),np.float32);social_interactions=np.zeros((n,n),np.float32);user_comments=np.zeros((n,len(other)),np.float32)
    last_stance={source_uid:0}
    for j in past_chain:
     if record_labels[j]>=0:last_stance[records[int(j)]['uid']]=int(record_labels[j])
    class_pop=np.bincount(record_labels[past_chain][record_labels[past_chain]>=0],minlength=2)/max(len(known),1)
    previous_class=np.bincount(record_labels[past_chain][(record_labels[past_chain]>=0)&(times[past_chain]>=cutoff-3600)],minlength=2)
    class_total=np.bincount(record_labels[past_chain][record_labels[past_chain]>=0],minlength=2)
    pr=np.ones(n)/max(n,1);transition=social/np.maximum(outgoing[:,None],1)
    for _ in range(30):pr=.15/max(n,1)+.85*(transition.T@pr+pr[outgoing==0].sum()/max(n,1))
    leader=pr>1/max(n,1);leader_count=social@leader;ics=np.zeros((n,3),np.float32);base=np.zeros((n,32),np.float32);fractions=np.zeros((n,3),np.float32);effect_input=np.zeros((n,3,13),np.float32);signal=np.zeros((n,8),np.float32);observed=np.zeros(n,bool);user_topics=np.zeros((n,len(other)),np.float32)
    for k,uid in enumerate(nodes):
     rows=corpus.eligible(uid,cutoff,src);lookups+=1
     if len(rows):
      visible_max=float(times[rows].max());assert visible_max<cutoff or (visible_max==source_time==cutoff and rows[-1]==src);max_history_time=max(max_history_time,visible_max)
     month=rows[times[rows]>=cutoff-30*86400];incoming_rows=corpus.incoming.get(uid,np.empty(0,dtype=np.int64));incoming_rows=incoming_rows[(times[incoming_rows]<cutoff)&(times[incoming_rows]>=cutoff-30*86400)];key=(split,uid,len(rows),int(rows[-1]) if len(rows) else -1,src,len(month),int(month[0]) if len(month) else -1,len(incoming_rows),int(incoming_rows[-1]) if len(incoming_rows) else -1)
     if key not in seen_cache:
      own_keywords=Counter(w for j in np.r_[month,incoming_rows] for w in keywords(words[int(j)]));tags=set(w for w,_ in own_keywords.most_common(8));source_tags=set(keywords(words[src]));jac=len(tags&source_tags)/max(len(tags|source_tags),1);ret=float(corpus.ret[rows].sum());ori=len(rows)-ret;user_text=np.asarray(text[rows]).mean(0) if len(rows) else np.zeros(16);user_emotion=np.asarray(emotion[rows]).mean(0) if len(rows) else np.zeros(2);cos=float(np.asarray(embeddings[rows]).mean(0)@embeddings[src]) if len(rows) else 0.
      seen_cache[key]=(jac,ret,ori,user_text,user_emotion,cos)
     jac,ret,ori,ut,ue,cos=seen_cache[key];observed[k]=len(rows)>0;rate=float(corpus.ret[month].mean()) if len(month) else 0.;friends=np.flatnonzero(social[k]|social[:,k]);friend_uids={nodes[int(j)] for j in friends};frows=np.array([int(j) for j in past_chain if records[int(j)]['uid'] in friend_uids and record_labels[int(j)]>=0],dtype=np.int64)
     stance=np.bincount(record_labels[frows],minlength=2).astype(float) if len(frows) else np.ones(2);fractions[k,:2]=stance/stance.sum();native_labels=record_labels[rows];counts=np.bincount(native_labels[native_labels>=0],minlength=2);friend_score=len(friend_uids&known)/max(len(friend_uids),1);popular=len(past_chain)/max(len(known),1);external=.4*leader_count[k]/max(n,1)+.6*friend_score
     delta=sentiment_transition(float(emotion[rows[-1]].sum()),float(emotion[rows[-2]].sum())) if len(rows)>=2 else 0.
     ics[k]=jac,rate,external;base[k,:16]=[np.log1p(len(rows)),np.log1p(ret),np.log1p(ori),ret/max(len(rows),1),np.log1p(outgoing[k]),np.log1p(incoming[k]),pr[k],friend_score,popular,cos,jac,ue[0],ue[1],delta,np.log1p((cutoff-source_time)/3600),float('#' in records[src]['text'])];base[k,16:]=ut
     native_activity[k]=.5*counts[:2]
     all_incoming=corpus.incoming.get(uid,np.empty(0,dtype=np.int64));all_incoming=all_incoming[times[all_incoming]<cutoff];global_friends=corpus.neighbors(uid,cutoff,src)[1];friend_influence=len(all_incoming)/max(len(global_friends),1)
     for strategy in range(2):
      selected=rows[(native_labels==strategy)&corpus.ret[rows]]
      for j,friend in enumerate(nodes):native_drive[k,j,strategy]=sum(corpus.parent[int(q)]==friend for q in selected)/max(len(selected),1)
      attr=(incoming[k]+outgoing[k])/3  # missing gender=0, two observed degree proxies
      dpm_factors[k,strategy,0]=attr*native_activity[k,strategy]
      pop=previous_class[strategy]/max(class_total[strategy],1)*np.exp(-.1*(cutoff-source_time)/3600)
      dpm_factors[k,strategy,1]=friend_influence*pop*cos
     for strategy in range(3):
      p=float(fractions[k,strategy]);effect_input[k,strategy]=[0,0,np.log1p(ori),np.log1p(outgoing[k]),np.log1p(incoming[k]),np.log1p((cutoff-source_time)/3600),float('#' in records[src]['text']),0,float('@' in records[src]['text']),len(past_chain)*p,len(past_chain)*p,float(ue[0]+ue[1]),np.exp(-min((cutoff-source_time)/3600,80))]
     for j in rows:
      parent=corpus.parent[int(j)]
      if parent in nodes:social_interactions[k,nodes.index(parent)]+=1
      if records[int(j)]['thread_id'] in [threads[o]['thread_id'] for o in other]:
       topic_column=[threads[o]['thread_id'] for o in other].index(records[int(j)]['thread_id']);user_topics[k,topic_column]+=1;user_comments[k,topic_column]+=int(corpus.ret[j])
     for j in [src,*past_chain.tolist()]:
      if records[int(j)]['uid']==uid and record_labels[int(j)] in (0,1):
       age=int(np.floor(max(0,cutoff-times[j])/3600))
       if age<8:signal[k,7-age]=1 if record_labels[int(j)]==0 else -1
    # Method's topic entities are all observed source posts, never future topics.
    m=len(other);x=np.zeros((n+m,32),np.float32);x[:n]=base
    for j,o in enumerate(other):x[n+j,16:]=topic[o];x[n+j,8]=1;x[n+j,14]=np.log1p(max(0,(cutoff-source_times[o])/3600))
    relation=np.zeros((n+m,n+m),np.float32);relation[:n,:n]=adj
    for i,j in zip(*np.nonzero(user_topics)):relation[i,n+j]=relation[n+j,i]=1.
    # The source author is known even if equal-timestamp filtering excludes a
    # copied reaction; history evidence supplies this interaction explicitly.
    if source_uid in nodes:relation[nodes.index(source_uid),n]=relation[n,nodes.index(source_uid)]=1.
    views=np.zeros((3,n+m,n+m),np.float32);contribution=np.array([sum(records[int(j)]['uid']==uid for j in past_chain) for uid in nodes],dtype=float);views[1,:n,:n]=adj*(1+np.log1p(social_interactions))*(1+contribution[None,:]/max(float(contribution.max()),1.));edge_features=np.zeros((4,n+m,n+m),np.float32)
    for i,j in zip(*np.nonzero(user_topics)):
     rs=.6*float(base[i,16:]@topic[other[j]])/max(np.linalg.norm(base[i,16:]),1e-8)+.4*np.exp(-.1*abs(source_times[other[j]]-source_time)/86400);ee=float(base[i,11]+base[i,12]);engagement=user_topics[i,j]+.5*user_comments[i,j];views[0,i,n+j]=views[0,n+j,i]=(.5*rs+.5*ee)*engagement
     edge_features[:,i,n+j]=edge_features[:,n+j,i]=[rs,ee,user_topics[i,j],user_comments[i,j]]
    for i,o in enumerate(other):
     for j,p in enumerate(other):
      if i!=j:views[2,n+i,n+j]=.6*float(topic[o]@topic[p])+.4*np.exp(-.1*abs(source_times[o]-source_times[p])/86400)
    # Native game gives edge-specific competitive influence. The opposite
    # direction has the complementary sigmoid; social/semantic zeros stay zero.
    perception=outgoing/max(float(outgoing.max()),1.)
    effect=np.stack([influence_kgraph(native_activity[:,j,None],perception[:,None],class_pop[j],native_drive[:,:,j]) for j in range(2)],-1)
    edge_game=binary_game(torch.tensor(effect),torch.tensor(fractions[:,:2,None].transpose(0,2,1))).numpy()
    friend_stance=np.array([last_stance.get(uid,-1) for uid in nodes]);mutual=np.where(friend_stance[None,:]==0,edge_game[:,:,0],np.where(friend_stance[None,:]==1,edge_game[:,:,1],.5))
    game=np.stack((edge_game[:,:,0].mean(1),edge_game[:,:,1].mean(1)),-1)
    weighted=relation.copy();weighted[:n,:n]*=mutual
    triples=[]
    for i,j in zip(*np.nonzero(adj)):triples.append((i,0 if social[i,j] else 3,j))
    for i,j in zip(*np.nonzero(user_topics)):triples.append((i,1 if nodes[i]==threads[other[j]]['source']['uid'] else 2,n+j))
    new[branch].append({'x':x,'base':base,'ics':ics,'adjacency':weighted,'views':views,'edge_features':edge_features,'social':social,'user_count':n,'node_ids':nodes,'topic_ids':[threads[o]['thread_id'] for o in other],'fractions':fractions,'effect_input':effect_input,'dpm_factors':dpm_factors,'history_signal':signal,'observed':observed,'game':game,'triples':np.asarray(triples,dtype=np.int64).reshape(-1,3),'sample_nodes':snapshot['sample_nodes'],'target':snapshot['target'],'window':snapshot['window'],'cutoff':snapshot['cutoff'],'end':snapshot['end'],'history_summary':macro['history_summary'],'popularity_target':macro['popularity_target']})
   # No prefix cache contains future labels. Reuse only repeated computations.
  result.append(new)
  if ti%100==0:print(json.dumps({'phase':'features','sources':ti,'seconds':round(time.monotonic()-started,1),'cache_entries':len(seen_cache)}),flush=True)
 torch.save(result,out/'graphs.pt');manifest={'dataset_sha256':graph_hash,'baseline_graphs_sha256':sha(out/'graphs.pt'),'feature_input_dim':32,'native_feature_fields':'ICS 3 original factors; RAM 5 user+4 message+4 repost factors with missing demographics0; K relation triples; DPM five historical slices; TRS three graph views','train_text_records':int(train_mask.sum()),'text_resources_fit_on_train_only':True,'source_texts_can_be_used_only_after_their_publication':True,'future_candidate_state_update_users_never_used':True,'raw_targets_and_duplicates_unchanged':True,'maximum_observed_history_timestamp':max_history_time,'preparation_source_sha256':sha(__file__),'seconds':time.monotonic()-started,'sha256':{p:sha(p) for p in ['data/pheme_history/records.jsonl','data/pheme_history/embeddings.npy','data/pheme_history/sentiment.npy']}}
 (out/'completion.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='data/pheme_paper_baselines');p.add_argument('--text-epochs',type=int,default=10);build(p.parse_args())
