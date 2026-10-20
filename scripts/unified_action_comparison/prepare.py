"""Reuse audited native modules; attach explicit causal semantic relations."""
import argparse
import inspect
import linecache
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.unified_action_comparison.common import ROOT,SPEC,setup,sha,write


def features(name):
    from scripts.baseline_comparison import prepare_features as native
    from scripts.weibo_comparison.prepare_features import baseline_features,paired,branch_views
    source=inspect.getsource(native.build)
    marker="views=np.zeros((3,n+m,n+m),np.float32);"
    assert source.count(marker)==1
    source=source.replace(marker,
        "semantic_info=snapshot.get('semantic_information_users',[])\n"
        "    for semantic_i in semantic_info:relation[semantic_i,n]=relation[n,semantic_i]=1.\n"
        "    "+marker)
    marker="new[branch].append({'x':x"
    assert source.count(marker)==1
    source=source.replace(marker,
        "for semantic_i in semantic_info:\n"
        "     triples.extend([(int(semantic_i),3,n),(n,3,int(semantic_i))]);views[0,semantic_i,n]=views[0,n,semantic_i]=1.\n"
        "    "+marker)
    source=source.replace("for micro,macro in zip(gthread['snapshots'],gthread['macro_snapshots']):",
        "for micro,macro,use_micro,use_macro in paired(gthread):")
    source=source.replace("for branch,snapshot in [('snapshots',micro),('macro_snapshots',macro)]:",
        "for branch,snapshot in branch_views(micro,macro,use_micro,use_macro):")
    filename='<unified-explicit-user-information-relations>'
    linecache.cache[filename]=(len(source),None,source.splitlines(keepends=True),filename)
    native.paired=paired;native.branch_views=branch_views
    exec(compile(source,filename,'exec'),vars(native))
    # The legacy English adapter requires explicit missingness fields; older
    # complete-text PHEME records intentionally use the equivalent default.
    original=native.text_resources
    def text_resources(records,embeddings,teacher,train_mask,out,epochs):
        for r in records:r.setdefault('text_available',bool(r.get('text','').strip()))
        available=np.array([r['text_available'] for r in records])
        text,emotion,resource=original(records,embeddings,teacher,train_mask&available,out,epochs)
        text[~available]=0;emotion[~available]=0
        np.savez_compressed(out/'text_record_features.npz',text=text,emotion=emotion)
        return text,emotion,resource
    native.text_resources=text_resources
    if SPEC[name]['language']=='en':native.build(SimpleNamespace(output='data/pheme_paper_baselines',text_epochs=10))
    else:baseline_features()


def improved():
    from scripts.weibo_comparison import prepare_features as m
    source=inspect.getsource(m.improved_features)
    assert source.count('for w in range(72):')==1
    source=source.replace('for w in range(72):','for w in sorted(macro_by_window):')
    assert source.count("adj[:n,n]=1;adj[n,:n]=1;")==1
    source=source.replace("adj[:n,n]=1;adj[n,:n]=1;",'')
    exec(compile(source,'<unified-variable-horizon-explicit-relations>','exec'),vars(m))
    m.improved_features()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['features','improved','resources','trgcn']);p.add_argument('--dataset',choices=list(SPEC),required=True)
    a=p.parse_args();torch.set_num_threads(1);setup(ROOT/SPEC[a.dataset]['root'])
    if a.mode=='features':features(a.dataset)
    elif a.mode=='improved':improved()
    elif a.mode=='resources':
        # Internal per-observation Twitter IDs need original parent TweetID
        # resolution. The adapter is also valid for direct-ID PHEME/Weibo.
        from scripts.baseline_comparison import knowledge_messages as messages
        from scripts.weibo_comparison import prepare_features as adapter
        def bounded(graphs):
            source=inspect.getsource(messages.add_message_entities)
            source=source.replace("visible=[source]+[r for r in thread['reactions'] if r['timestamp']<feature_cutoff]",
                "visible=[source]+[r for r in thread['reactions'] if r['timestamp']<feature_cutoff and r['uid'] in s['node_ids']][-32:]")
            def parent(record,visible):
                if 'tweet_id' not in record:return record['reply_to_id']
                matches=[r for r in visible if r.get('tweet_id')==record.get('reply_to_id')
                         and r['uid']==record.get('reply_to_uid') and r['timestamp']<=record['timestamp']
                         and r['archive_path']!=record['archive_path']]
                return max(matches,key=lambda r:(r['timestamp'],r['archive_path']))['id'] if matches else record['reply_to_id']
            source=source.replace("parent=lookup.get('message:'+r['reply_to_id'])","parent=lookup.get('message:'+unified_parent(r,visible))")
            messages.unified_parent=parent
            exec(compile(source,'<unified-causal-message-identities>','exec'),vars(messages))
            counts=messages.add_message_entities(graphs)
            # The native message adapter reconstructs observed post edges;
            # retain the additional current-information semantic edges too.
            for t in graphs:
                for branch in ['snapshots','macro_snapshots']:
                    for s in t[branch]:
                        n=s['user_count']
                        for i,relation,j in s['triples']:
                            if relation==3 and ((i<n and j==n) or (i==n and j<n)):
                                s['k_adjacency'][i,j]=max(1.,s['k_adjacency'][i,j])
            return counts
        adapter.bounded_messages=bounded;adapter.resources()
    else:
        if SPEC[a.dataset]['language']=='zh':
            from scripts.weibo_comparison.baselines import prepare_trgcn
            prepare_trgcn()
        else:
            from scripts.twitter_comparison.prepare import trgcn
            trgcn()
    write(Path('data')/('unified_'+a.mode+'_provenance.json'),dict(complete=True,dataset=a.dataset,
        protocol='unified_action_v1',adapter_sha256=sha(__file__),graph_manifest_sha256=sha('data/pheme_multitask_hourly/dataset_manifest.json'),
        user_information_semantic_edges_explicit=True,original_architecture_unchanged=True))


if __name__=='__main__':main()
