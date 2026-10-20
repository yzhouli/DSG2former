"""Recover original Twitter15/16 actions without inventing user/text links."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = ROOT / 'tmp/twitter15_16_source_audit/downloads'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def tuples(line):
    return [ast.literal_eval(x.strip()) for x in line.split('->')]


def recover(year):
    out = ROOT / f'data/twitter{year}_np50_information'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'raw_audit.json').exists():
        assert json.loads((out / 'raw_audit.json').read_text())['threads_sha256'] == sha(out / 'threads.jsonl')
        return
    raw = zipfile.ZipFile(ARCHIVES / 'rumdetect2017.zip')
    processed = zipfile.ZipFile(ARCHIVES / 'PLAN_processed_data.zip')
    prefix = f'rumor_detection_acl2017/twitter{year}/'
    labels = dict((tid, label) for label, tid in (l.split(':') for l in raw.read(prefix + 'label.txt').decode().splitlines()))
    source_text = dict(l.split('\t', 1) for l in raw.read(prefix + 'source_tweets.txt').decode().splitlines() if '\t' in l)
    variant = 'baseline' if year == 15 else 'structure_baseline'
    published = {}
    for name in processed.namelist():
        if name.startswith(f'twitter15_16/twitter{year}/split_data/{variant}/split_0/') and name.endswith('.json') and 'small' not in name:
            for line in processed.read(name).decode().splitlines():
                if line.strip():
                    row = json.loads(line)
                    assert str(row['id_']) not in published
                    published[str(row['id_'])] = row
    threads, coverage, excluded = [], Counter(), []
    for sid, label in labels.items():
        if label != 'false': continue
        path = prefix + 'tree/' + sid + '.txt'
        edges = [tuples(l) for l in raw.read(path).decode().splitlines() if l.strip()]
        roots = [child for parent, child in edges if str(parent[0]) == 'ROOT']
        if not roots or any(str(x[1]) != sid for x in roots):
            excluded.append(dict(source_id=sid, reason='Original ROOT tweet ID differs from labeled source; source author cannot be established', root_tuples=roots, archive_edge_lines=len(edges)))
            continue
        root = roots[0]
        assert len({str(x[0]) for x in roots}) == 1
        timestamp = ((int(sid) >> 22) + 1288834974657) / 1000
        texts = {sid: source_text[sid]}
        provenance = {sid: 'original_source_text'}
        row = published.get(sid)
        if row is not None and 'tweet_ids' in row:
            for tid, text in zip(row['tweet_ids'], row['tweets']):
                tid = str(tid)
                if tid in texts and provenance[tid] != 'original_source_text': assert texts[tid] == text
                texts[tid] = text
                provenance[tid] = 'PLAN_tweet_id'
        elif row is not None:
            # The author's notebook takes the earliest original occurrence of
            # each tweet ID, then floor(minutes/10), clipped to [0,99]. Only a
            # singleton original ID AND singleton released text in that bin
            # permits an identity assignment. Ambiguous bins stay unresolved.
            earliest = {}
            for parent, child in edges:
                for uid, tid, minutes in [parent, child]:
                    tid = str(tid)
                    if str(uid) != 'ROOT' and tid != sid:
                        earliest[tid] = min(earliest.get(tid, float('inf')), float(minutes))
            bins, textbins = defaultdict(list), defaultdict(list)
            for tid, minutes in earliest.items(): bins[max(0, min(99, int(minutes // 10)))].append(tid)
            assert len(row['tweets']) == len(row['time_delay'])
            # Index zero is explicitly the triggering source in the notebook.
            texts[sid] = row['tweets'][0]
            provenance[sid] = 'PLAN_known_source_index'
            for text, b in zip(row['tweets'][1:], row['time_delay'][1:]): textbins[int(b)].append(text)
            for b, values in textbins.items():
                if len(values) == 1 and len(bins[b]) == 1:
                    tid = bins[b][0]
                    texts[tid] = values[0]
                    provenance[tid] = 'unique_original_tweet_id_and_unique_PLAN_text_in_time_bin'
                    coverage['uniquely_recovered_reply_ids'] += 1
        def record(uid, tid, seconds, ordinal, parent=None):
            tid = str(tid)
            # An original propagation tree may reuse the source tweet ID for
            # many distinct retweet users. Keep a separate record ID per edge.
            rid = sid if ordinal is None else f'{sid}:edge:{ordinal}'
            return dict(id=rid, tweet_id=tid, uid=str(uid), timestamp=seconds,
                        text=texts.get(tid, ''), text_available=tid in texts,
                        text_provenance=provenance.get(tid, 'unavailable'),
                        reply_to_uid=None if parent is None else str(parent[0]),
                        reply_to_id=None if parent is None else str(parent[1]),
                        is_source_retweet=ordinal is not None and tid == sid,
                        archive_path=f'twitter{year}/{sid}/' + ('source' if ordinal is None else f'edge_{ordinal}'))
        source = record(root[0], sid, timestamp, None)
        reactions = []
        for ordinal, (parent, child) in enumerate(edges):
            if str(parent[0]) == 'ROOT':
                coverage['root_trigger_lines'] += 1
                continue
            uid, tid, delay = child
            r = record(uid, tid, timestamp + 60 * float(delay), ordinal, parent)
            r['original_delay_minutes'] = float(delay)
            r['original_edge_ordinal'] = ordinal
            reactions.append(r)
            coverage['raw_reaction_records'] += 1
            coverage['text_available_records'] += r['text_available']
            coverage['source_retweet_records'] += r['is_source_retweet']
            coverage['nonpositive_delay_records'] += float(delay) <= 0
            if 0 < float(delay) <= 72 * 60:
                coverage['within72h_records'] += 1
                coverage['within72h_text_available_records'] += r['text_available']
        coverage['false_sources'] += 1
        coverage['sources_with_PLAN_release'] += row is not None
        assert len(reactions) + sum(str(p[0]) == 'ROOT' for p, c in edges) == len(edges)
        threads.append(dict(thread_id=sid, event=f'Twitter{year}', source=source, reactions=reactions,
                            source_collection_label='false', source_veracity='false', raw_tree=path))
    threads.sort(key=lambda t: (t['source']['timestamp'], t['thread_id']))
    n = len(threads); a, b = int(n * .8), int(n * .9)
    assert 0 < a < b < n
    with (out / 'threads.jsonl').open('w') as f:
        for i, thread in enumerate(threads):
            thread['split'] = 'train' if i < a else 'validation' if i < b else 'test'
            f.write(json.dumps(thread, ensure_ascii=False) + '\n')
    manifest = dict(complete=True, dataset=f'Twitter{year}', coverage=dict(coverage),
                    sources=n, splits=dict(train=a, validation=b-a, test=n-b),
                    excluded_sources=excluded,
                    original_records_deduplicated=False, source_false_labels_from_original=True,
                    unavailable_text='Retained with empty text and masked semantic features; not weak-labeled as an action',
                    identity='Real original uid/time retained; no anonymous or sampled user identity substitutions',
                    published_teacher_data_preprocessed=True,
                    reply_recovery='ID-linked text for Twitter15; strict singleton time-bin identity recovery for Twitter16',
                    retweets='Reuse of source tweet ID with a different edge is a real separate action, not a source-copy to discard',
                    timestamp='Source Snowflake timestamp plus original relative delay in minutes',
                    horizon_hours=72, window_seconds=3600,
                    threads_sha256=sha(out / 'threads.jsonl'),
                    input_sha256={p.name: sha(p) for p in [ARCHIVES/'rumdetect2017.zip', ARCHIVES/'PLAN_processed_data.zip']})
    write(out / 'raw_audit.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--year', type=int, choices=[15,16], required=True)
    recover(p.parse_args().year)
