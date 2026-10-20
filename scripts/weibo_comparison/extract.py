"""Lossless, provenance-indexed rumor-only adapters for the two uploaded Weibo corpora."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import re
from zipfile import ZipFile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def timestamp(value):
    if isinstance(value, (int, float)):
        return float(value)
    for fmt in ['%a %b %d %H:%M:%S %z %Y', '%Y-%m-%d %H:%M:%S']:
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
            return dt.timestamp()
        except ValueError:
            pass
    raise ValueError('Unrecognized timestamp: ' + str(value))


def mid_from_bid(bid):
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    parts = []
    for end in range(len(bid), 0, -4):
        value = 0
        for ch in bid[max(0, end-4):end]:
            value = value*62 + alphabet.index(ch)
        parts.append(str(value).zfill(7))
    return str(int(''.join(reversed(parts))))


def text(value):
    return html.unescape(re.sub(r'<[^>]*>', '', value or ''))


def comment_rows(items, prefix):
    for i, row in enumerate(items):
        path = prefix + '/' + str(i)
        yield path, row
        # Embedded status is a copy of the root, not a new action. Nested
        # comments are distinct raw archive records, including repeated IDs.
        nested = row.get('comments', [])
        if isinstance(nested, list):
            yield from comment_rows(nested, path + '/comments')


def v1(root):
    for p in sorted((root/'SocialNet-Weibo-V1/fake_news').glob('*/new.json')):
        original = json.loads(p.read_text())
        comments = json.loads(p.with_name('comments.json').read_text())
        bid = p.parent.name
        source_id = mid_from_bid(bid)
        source_uid = str(original['mid'])  # new.json misnames author ID as mid.
        embedded = [r.get('status', {}) for _, r in comment_rows(comments, 'comments')]
        matches = [s for s in embedded if str(s.get('idstr') or s.get('id')) == source_id
                   and str(s.get('user', {}).get('idstr') or s.get('user', {}).get('id')) == source_uid]
        exact = matches[0] if matches else None
        source = dict(id=source_id, uid=source_uid,
                      timestamp=timestamp(exact['created_at'] if exact else original['time']),
                      text=text(original.get('context')), language='zh',
                      reply_to_id=None, reply_to_uid=None, archive_path=str(p),
                      time_resolution='second' if exact else 'minute',
                      source_id_origin='base62 decoded directory BID',
                      source_uid_origin='new.json mid; checked against matching embedded status when available')
        reactions = []
        for pointer, row in comment_rows(comments, 'comments'):
            user = row.get('user') or {}
            uid = user.get('idstr') or user.get('id')
            if uid is None or 'created_at' not in row:
                raise ValueError('Missing action identity/time: ' + str(p) + '/' + pointer)
            reply = row.get('reply_comment') or {}
            parent_id = reply.get('idstr') or reply.get('id') or source_id
            parent_uid = (reply.get('user') or {}).get('idstr') or (reply.get('user') or {}).get('id') or source_uid
            reactions.append(dict(id=str(row.get('idstr') or row['id']), uid=str(uid),
                timestamp=timestamp(row['created_at']), text=text(row.get('text')), language='zh',
                reply_to_id=str(parent_id), reply_to_uid=str(parent_uid),
                archive_path=str(p.with_name('comments.json'))+'#/'+pointer))
        yield dict(thread_id=source_id, event='weibo_v1', source_collection_label='fake_news',
                   verified_veracity='false_by_folder', source=source, reactions=reactions,
                   original_bid=bid, embedded_source_matches=len(matches))


def public(archive):
    with ZipFile(archive) as z:
        labels = []
        for line in z.read('weibo-twitter-public/Weibo.txt').decode().splitlines():
            parts = line.split('\t')
            labels.append((parts[0].split(':')[1], int(parts[1].split(':')[1])))
        for eid, label in labels:
            if label != 1:
                continue
            name = 'weibo-twitter-public/Weibo/' + eid + '.json'
            rows = json.loads(z.read(name))
            if not rows or str(rows[0]['id']) != eid:
                raise ValueError('First post is not labeled source: '+eid)
            records = []
            by_id = {str(r['id']): str(r['uid']) for r in rows}
            for i, r in enumerate(rows):
                parent = str(r['parent']) if r.get('parent') is not None else None
                records.append(dict(id=str(r['id']), uid=str(r['uid']), timestamp=timestamp(r['t']),
                    text=text(r.get('original_text') or r.get('text')), language='zh',
                    reply_to_id=parent, reply_to_uid=by_id.get(parent),
                    archive_path=str(archive)+'!'+name+'#/'+str(i)))
            yield dict(thread_id=eid, event='weibo_public', source_collection_label='rumor_label_1',
                       verified_veracity=None, source=records[0], reactions=records[1:])


def extract(dataset, source, out):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a new empty destination: '+str(out))
    # Sort compact source metadata on disk, not all large propagation threads.
    spool = out/'spool'; spool.mkdir()
    entries = []
    counters = Counter()
    hashes = {}
    for i, t in enumerate(v1(source) if dataset == 'weibo_v1' else public(source)):
        t['reactions'].sort(key=lambda r: (r['timestamp'], r['archive_path']))
        f = spool/(str(i)+'.json')
        f.write_text(json.dumps(t, ensure_ascii=False))
        entries.append((t['source']['timestamp'], t['thread_id'], f))
        counters['sources'] += 1
        counters['raw_reactions'] += len(t['reactions'])
        counters['empty_sources'] += not bool(t['reactions'])
        counters['reactions_before_source'] += sum(r['timestamp'] < t['source']['timestamp'] for r in t['reactions'])
        if i % 100 == 0:
            print(json.dumps(dict(dataset=dataset, scanned=i)), flush=True)
    entries.sort()
    if len(set(t[1] for t in entries)) != len(entries):
        raise ValueError('Duplicate source IDs must be resolved without discarding sources')
    n = len(entries); split_stats = {}; spans = []; users = set(); duplicate_ids = 0
    with (out/'threads.jsonl').open('w') as f, (out/'source_statistics.csv').open('w') as st:
        st.write('thread_id,split,raw_reactions,distinct_reaction_users,hourly_windows\n')
        for i, (_, _, path) in enumerate(entries):
            t = json.loads(path.read_text())
            t['split'] = 'train' if i < int(n*.8) else 'validation' if i < int(n*.9) else 'test'
            start = t['source']['timestamp']
            max_time = max([start]+[r['timestamp'] for r in t['reactions']])
            import math
            hours = max(1, math.ceil((max_time-start)/3600))
            spans.append(hours)
            stat = split_stats.setdefault(t['split'], Counter())
            stat.update(sources=1, raw_reactions=len(t['reactions']), hourly_windows=hours,
                        empty_sources=int(not t['reactions']))
            users.update(r['uid'] for r in [t['source']]+t['reactions'])
            duplicate_ids += len(t['reactions'])-len(set(r['id'] for r in t['reactions']))
            f.write(json.dumps(t, ensure_ascii=False)+'\n')
            st.write(','.join(map(str,[t['thread_id'], t['split'], len(t['reactions']), len(set(r['uid'] for r in t['reactions'])), hours]))+'\n')
            hashes[str(path)] = sha(path)
            path.unlink()
    spool.rmdir()
    report = dict(dataset=dataset, source=str(source), statistics=dict(counters),
        splits={s:dict(c) for s,c in split_stats.items()}, distinct_users=len(users),
        duplicate_reaction_ids_retained=duplicate_ids, deduplicate=False,
        hourly_windows=sum(spans), maximum_hourly_windows=max(spans),
        source_time_min=entries[0][0], source_time_max=entries[-1][0],
        source_sha256=sha(source) if source.is_file() else None,
        raw_thread_hashes=hashes, threads_sha256=sha(out/'threads.jsonl'),
        split_policy='Source publication time ordered whole information 80:10:10',
        selected='fake_news only' if dataset=='weibo_v1' else 'Weibo label=1 only; Twitter excluded',
        raw_history_scope='Selected rumor source posts and all their original reactions; no deduplication')
    (out/'raw_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='raw_thread_hashes'},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=['weibo_v1','weibo_public'], required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); extract(a.dataset,a.source,a.output)
