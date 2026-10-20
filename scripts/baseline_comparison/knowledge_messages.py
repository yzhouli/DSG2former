"""Complete time-visible PHEME message entities for native TranSparse graphs.

A graph vertex is a post ID. Every raw archived observation still contributes
its own relation evidence; duplicate supervised action rows remain untouched.
"""
import json
from pathlib import Path
import numpy as np


def add_message_entities(graphs):
 records=list(map(json.loads,Path('data/pheme_history/records.jsonl').read_text().splitlines()));raw={t['thread_id']:t for t in map(json.loads,Path('data/pheme_multitask_hourly/threads.jsonl').read_text().splitlines())};index={r['archive_path']:i for i,r in enumerate(records)}
 data=np.load('data/pheme_paper_baselines/text_record_features.npz');text=data['text'];emotion=data['emotion'];counts={'visible_raw_observations':0,'new_message_vertices':0,'future_messages_used':0}
 for t in graphs:
  thread=raw[t['thread_id']];source=thread['source'];source_index=index[source['archive_path']]
  for branch in ['snapshots','macro_snapshots']:
   for s in t[branch]:
    cutoff=float(s['cutoff']);feature_cutoff=cutoff if s['window']==0 else float(np.nextafter(cutoff,np.inf));visible=[source]+[r for r in thread['reactions'] if r['timestamp']<feature_cutoff]
    users=s['node_ids'];nu=s['user_count'];entities=['user:'+u for u in users]+['message:'+tid for tid in s['topic_ids']];lookup={u:i for i,u in enumerate(entities)};new=[]
    for r in visible:
     entity='message:'+r['id']
     if entity not in lookup:lookup[entity]=len(entities);entities.append(entity);new.append(r)
    x=np.zeros((len(entities),32),np.float32);x[:len(s['x'])]=s['x']
    for r in new:
     ri=index[r['archive_path']];position=lookup['message:'+r['id']];v=np.zeros(32,np.float32);v[8]=1;v[9]=float(text[ri]@text[source_index]/max(np.linalg.norm(text[ri])*np.linalg.norm(text[source_index]),1e-8));v[11:13]=emotion[ri];v[14]=np.log1p(max(0,(feature_cutoff-r['timestamp'])/3600));v[16:]=text[ri];x[position]=v
    adjacency=np.zeros((len(entities),len(entities)),np.float32);adjacency[:len(s['x']),:len(s['x'])]=s['adjacency'];adjacency[:nu,nu]=0;adjacency[nu,:nu]=0
    triples=[tuple(row) for row in s['triples'] if not (row[1] in (1,2) and row[2]==nu)]
    for r in visible:
     ri=index[r['archive_path']];assert r['timestamp']<feature_cutoff or (r is source and r['timestamp']==cutoff)
     if r['uid'] not in users:raise AssertionError('Visible chain participant missing from causal user graph')
     u=lookup['user:'+r['uid']];message=lookup['message:'+r['id']];ret=r.get('reply_to_id') is not None or (r.get('id')!=t['thread_id'] and r is not source);relation=2 if ret else 1;triples.append((u,relation,message));adjacency[u,message]+=1;adjacency[message,u]+=1
     if r.get('reply_to_id') is not None:
      parent=lookup.get('message:'+r['reply_to_id'])
      if parent is not None and parent!=message:triples.append((message,4,parent));adjacency[message,parent]+=1;adjacency[parent,message]+=1
    s['k_x']=x;s['k_adjacency']=adjacency;s['k_triples']=np.asarray(triples,dtype=np.int64).reshape(-1,3);s['k_entity_ids']=entities
    counts['visible_raw_observations']+=len(visible);counts['new_message_vertices']+=len(new)
 return counts
