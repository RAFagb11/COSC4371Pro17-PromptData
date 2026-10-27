"""Independent output checks; brute-force check against 600 deterministically sampled rows."""
import csv, hashlib, importlib.util, itertools, json, random, re, sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit_split',root/'audit_split.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=json.loads((root/'audit.json').read_text()); rr=[]
for source in a['manifest']:
 p=root/'data_cache'/(source['dataset']+'_'+Path(source['file']).name)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
 sp='train' if '/train-' in source['file'] else 'test'
 for i,r in enumerate(pq.read_table(p).to_pylist()):rr.append(dict(row_id=f"{source['dataset']}:{sp}:{i}",dataset=source['dataset'],**r))
with (root/'split.csv').open() as f:split=list(csv.DictReader(f))
with (root/'duplicate_pairs.csv').open() as f:pairs=list(csv.DictReader(f))
assert len(split)==len(rr)==10958
byid={r['row_id']:r for r in rr};spid={r['row_id']:r['split'] for r in split}
assert len(spid)==len(rr)
for r in split:assert int(r['label'])==byid[r['row_id']]['label']
lookup={tuple(sorted((p['row_id_a'],p['row_id_b']))) for p in pairs}
for ds in m.SOURCES:
 tc=Counter(r['text'] for r in rr if r['dataset']==ds)
 assert sum(n*(n-1)//2 for n in tc.values())==a['duplicates']['within_dataset'][ds]['exact']['pairs']
sample=random.Random(42).sample(rr,600);expected=0
for x,y in itertools.combinations(sample,2):
 tx=set(re.findall(r'\w+',m.norm(x['text']))) or {m.norm(x['text'])}
 ty=set(re.findall(r'\w+',m.norm(y['text']))) or {m.norm(y['text'])}
 match=5*len(tx&ty)>=4*len(tx|ty)
 actual=tuple(sorted((x['row_id'],y['row_id']))) in lookup
 assert match==actual
 expected+=match
for p in pairs:assert spid[p['row_id_a']]==spid[p['row_id_b']]
for sp in ['test','validation']:
 texts={byid[r['row_id']]['text'] for r in split if r['split']==sp}
 assert not any(byid[r['row_id']]['text'] in texts for r in split if r['split']=='train')
result=dict(rows_checked=len(rr),sha256_files_checked=len(a['manifest']),sample_rows=600,brute_force_pairs_checked=179700,brute_force_matching_pairs=expected,status='PASS')
(root/'verification.json').write_text(json.dumps(result,indent=2));print(result)
