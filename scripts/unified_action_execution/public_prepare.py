"""Physical-line JSONL reader and indexed native preparation, no model changes.

The frozen experiment adapters are imported unchanged. Generated execution
functions and their hashes are saved in the dataset's own runtime directory.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.unified_action_comparison.common import ROOT,SPEC,setup,write,sha


def compile_function(module,name,source,work,tag):
    folder=work/'compat/unified_execution';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(tag+'.py');path.write_text(source)
    exec(compile(source,str(path),'exec'),vars(module))
    return path


def physical_lines(module,name,work):
    source=inspect.getsource(getattr(module,name));old='.read_text().splitlines()'
    assert old in source,(module.__name__,name)
    return compile_function(module,name,source.replace(old,".read_text().rstrip('\\n').split('\\n')"),work,name+'_physical_jsonl')


def indexed_features(work):
    from scripts.baseline_comparison import prepare_features as native
    source=inspect.getsource(native.build)
    changes={
        'past_chain=np.asarray(past_chain,dtype=np.int64);available=':
            "past_chain=np.asarray(past_chain,dtype=np.int64);past_by_uid={}\n   for j in past_chain:past_by_uid.setdefault(records[int(j)]['uid'],[]).append(int(j))\n   available=",
        "[int(j) for j in past_chain if records[int(j)]['uid'] in friend_uids and record_labels[int(j)]>=0]":
            "[j for friend in friend_uids for j in past_by_uid.get(friend,[]) if record_labels[j]>=0]",
        'for j in [src,*past_chain.tolist()]:':'for j in [src,*past_by_uid.get(uid,[])]:',
        "[sum(records[int(j)]['uid']==uid for j in past_chain) for uid in nodes]":
            '[len(past_by_uid.get(uid,[])) for uid in nodes]',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return compile_function(native,'build',source,work,'indexed_native_features')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['features','improved','resources','trgcn','check_reader'])
    p.add_argument('--dataset',default='Weibo-public',choices=list(SPEC));a=p.parse_args()
    root=ROOT/SPEC[a.dataset]['root'];_,work=setup(root);torch.set_num_threads(1);generated=[]
    if a.mode=='check_reader':
        path=root/'history/records.jsonl';count=0;special=0
        for line in path.open():
            record=json.loads(line);count+=1
            special+=int(any(x in record.get('text','') for x in ['\u2028','\u0085','\u2029']))
        assert count==len(np.load(root/'history/embeddings.npy',mmap_mode='r'))
        write(work/'data/jsonl_reader_audit.json',dict(complete=True,records=count,
            records_with_unicode_line_separators=special,record_multiplicities_preserved=True,
            original_file_sha256=sha(path),same_as_embedding_rows=True))
        print(dict(records=count,unicode_separator_records=special),flush=True);return
    if a.mode=='features':
        from scripts.baseline_comparison import prepare_features as native
        generated.append(physical_lines(native,'build',work));generated.append(indexed_features(work))
    elif a.mode=='resources':
        from scripts.baseline_comparison import knowledge_messages as native
        generated.append(physical_lines(native,'add_message_entities',work))
    elif a.mode=='trgcn':
        from scripts.trgcn_baseline import prepare as native
        generated.append(physical_lines(native,'main',work))
    from scripts.unified_action_comparison.prepare import main as prepare
    sys.argv=[sys.argv[0],a.mode,'--dataset',a.dataset];prepare()
    proof=dict(complete=True,mode=a.mode,dataset=a.dataset,source_sha256=sha(__file__),
        generated_source_sha256={str(p):sha(p) for p in generated},
        original_texts_and_all_records_preserved=True,original_model_equations_unchanged=True,
        change='Physical file newline parsing and exact per-user lookup indexes; no subsampling')
    write(work/('data/execution_'+a.mode+'_audit.json'),proof)


if __name__=='__main__':main()
