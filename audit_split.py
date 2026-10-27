"""Pinned full-data audit and duplicate-group split. Run: python audit_split.py"""
import csv, hashlib, json, math, platform, random, re, unicodedata, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
SEED=42
EXPECTED_SHA256={'safeguard_train-00000-of-00001.parquet': '0f1879d20f98e8b47589453cd7dceb71f4a94e4abec8b0cf809188d1aeca3b3e', 'safeguard_test-00000-of-00001.parquet': '99981a32fa354d9f541aea7214c15fed258c67639dcb3da5751968b043d8f742', 'deepset_train-00000-of-00001-9564e8b05b4757ab.parquet': '2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04', 'deepset_test-00000-of-00001-701d16158af87368.parquet': '39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e'}
SOURCES={
 'safeguard':('xTRam1/safe-guard-prompt-injection','a3a877d608f37b7d20d9945671902df895ecdb46',['data/train-00000-of-00001.parquet','data/test-00000-of-00001.parquet']),
 'deepset':('deepset/prompt-injections','4f61ecb038e9c3fb77e21034b22511b523772cdd',['data/train-00000-of-00001-9564e8b05b4757ab.parquet','data/test-00000-of-00001-701d16158af87368.parquet'])}
def save_csv(name,records,fields):
 with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
def norm(t):return ' '.join(unicodedata.normalize('NFKC',t).casefold().split())
def counts(rr):
 c=Counter(r['label'] for r in rr);n=len(rr)
 return dict(rows=n,negative=c[0],positive=c[1],negative_pct=round(100*c[0]/n,6) if n else None,positive_pct=round(100*c[1]/n,6) if n else None)
def pair_summary(pp):
 return {k:dict(pairs=sum(p[2]==k for p in pp),participating_rows=len({i for p in pp if p[2]==k for i in p[:2]})) for k in ['exact','normalized_exact','near']}
def main():
 cache=ROOT/'data_cache';cache.mkdir(exist_ok=True)
 rows=[];manifest=[];schemas={}
 for ds,(repo,rev,files) in SOURCES.items():
  for file in files:
   p=cache/(ds+'_'+Path(file).name);url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}'
   if not p.exists():
    print('Downloading',ds,file,flush=True)
    with urllib.request.urlopen(url,timeout=120) as response:p.write_bytes(response.read())
   data=p.read_bytes()
   assert hashlib.sha256(data).hexdigest()==EXPECTED_SHA256[p.name], f'Unexpected file contents: {p}'
   table=pq.read_table(p)
   schemas.setdefault(ds,[]).append(dict(file=file,schema=str(table.schema),rows=table.num_rows))
   manifest.append(dict(dataset=ds,repo=repo,revision=rev,file=file,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
   sp='train' if '/train-' in file else 'test'
   for i,r in enumerate(table.to_pylist()):rows.append(dict(row_id=f'{ds}:{sp}:{i}',dataset=ds,publisher_split=sp,**r))
 report=dict(seed=SEED,python=platform.python_version(),pyarrow=pyarrow.__version__,manifest=manifest,schemas=schemas,audit={})
 for ds in SOURCES:
  rr=[r for r in rows if r['dataset']==ds];tc=Counter(r['text'] for r in rr if isinstance(r['text'],str))
  report['audit'][ds]={**counts(rr),'columns':2,'fields':['text','label'],
   'publisher_partitions':{s:counts([r for r in rr if r['publisher_split']==s]) for s in ['train','test']},
   'missing_label':sum(r['label'] is None for r in rr),'invalid_label':sum(r['label'] not in (0,1) for r in rr),
   'missing_text_null':sum(r['text'] is None for r in rr),'blank_text':sum(isinstance(r['text'],str) and not r['text'].strip() for r in rr),
   'exact_excess_rows':sum(n-1 for n in tc.values()),'exact_duplicate_groups':sum(n>1 for n in tc.values()),'exact_pairs':sum(n*(n-1)//2 for n in tc.values()),
   'entity_ids':'unavailable: only text and label; no user, host, session, source or campaign ID',
   'time_range':'unavailable: no observation timestamp; repository dates are not collection dates'}
 assert all(isinstance(r['text'],str) and r['text'].strip() and r['label'] in (0,1) for r in rows),'Invalid rows require explicit handling'
 n=len(rows);parent=list(range(n))
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 normalized=[norm(r['text']) for r in rows]
 tokens=[set(re.findall(r'\w+',t)) or {t} for t in normalized]
 freq=Counter(t for ss in tokens for t in ss)
 postings=defaultdict(list);pairs=[];candidate_count=0
 # Complete prefix-filtered Jaccard join. Threshold=4/5, no approximate sampling.
 # A matching pair must overlap in prefixes n-ceil(4*n/5)+1 under one global token order.
 for k,i in enumerate(sorted(range(n),key=lambda i:(len(tokens[i]),i))):
  a=tokens[i];la=len(a);prefix=sorted(a,key=lambda t:(freq[t],t))[:la-(4*la+4)//5+1];possible=set()
  for t in prefix:possible.update(j for j in postings[t] if 5*len(tokens[j])>=4*la)
  for j in sorted(possible):
   candidate_count+=1;b=tokens[j];inter=len(a&b);total=la+len(b)-inter
   if 5*inter>=4*total:
    kind='exact' if rows[i]['text']==rows[j]['text'] else ('normalized_exact' if normalized[i]==normalized[j] else 'near')
    union(i,j);pairs.append((min(i,j),max(i,j),kind,inter/total))
  for t in prefix:postings[t].append(i)
  if k%2000==0:print('Audited',k,'rows;',len(pairs),'matching pairs',flush=True)
 for values in ([r['text'] for r in rows],normalized):
  first={}
  for i,v in enumerate(values):
   if v in first:union(i,first[v])
   else:first[v]=i
 groups=defaultdict(list)
 for i in range(n):groups[find(i)].append(i)
 forced={g for g,ids in groups.items() if any(rows[i]['dataset']=='deepset' or rows[i]['publisher_split']=='test' for i in ids)}
 eligible=sorted(set(groups)-forced);random.Random(SEED).shuffle(eligible)
 val=set(eligible[:math.ceil(.2*len(eligible))])
 for g,ids in groups.items():
  sp='test' if g in forced else ('validation' if g in val else 'train')
  for i in ids:rows[i].update(split=sp,group_id=f'group:{g}')
 report['split_counts']=[dict(dataset=ds,split=sp,**counts([r for r in rows if r['dataset']==ds and r['split']==sp])) for ds in SOURCES for sp in ['train','validation','test']]
 report['groups']=dict(total=len(groups),multirow=sum(len(v)>1 for v in groups.values()),largest=max(map(len,groups.values())),forced_test=len(forced),eligible_train_validation=len(eligible),validation=len(val),train=len(eligible)-len(val),conflicting_label_groups=sum(len({rows[i]['label'] for i in ids})>1 for ids in groups.values()))
 report['movement']=dict(removed_rows=0,safeguard_publisher_train_moved_to_test=sum(r['dataset']=='safeguard' and r['publisher_split']=='train' and r['split']=='test' for r in rows),safeguard_publisher_train_assigned_validation=sum(r['dataset']=='safeguard' and r['split']=='validation' for r in rows),deepset_publisher_train_assigned_external_test=sum(r['dataset']=='deepset' and r['publisher_split']=='train' for r in rows))
 moved=[r for r in rows if r['dataset']=='safeguard' and r['publisher_split']=='train' and r['split']=='test']
 by_group={f'group:{g}':ids for g,ids in groups.items()}
 def links(r,ds):return any(rows[i]['dataset']==ds and (ds=='deepset' or rows[i]['publisher_split']=='test') for i in by_group[r['group_id']])
 report['movement'].update(linked_to_safeguard_publisher_test=sum(links(r,'safeguard') for r in moved),linked_to_deepset_external_test=sum(links(r,'deepset') for r in moved),linked_to_both_test_sources=sum(links(r,'safeguard') and links(r,'deepset') for r in moved))
 report['duplicates']=dict(candidate_pairs_verified=candidate_count,overall=pair_summary(pairs),within_dataset={ds:pair_summary([p for p in pairs if rows[p[0]]['dataset']==rows[p[1]]['dataset']==ds]) for ds in SOURCES},across_datasets=pair_summary([p for p in pairs if rows[p[0]]['dataset']!=rows[p[1]]['dataset']]),within_split={sp:pair_summary([p for p in pairs if rows[p[0]]['split']==rows[p[1]]['split']==sp]) for sp in ['train','validation','test']},across_splits=pair_summary([p for p in pairs if rows[p[0]]['split']!=rows[p[1]]['split']]),publisher_train_test_within_dataset={ds:pair_summary([p for p in pairs if rows[p[0]]['dataset']==rows[p[1]]['dataset']==ds and rows[p[0]]['publisher_split']!=rows[p[1]]['publisher_split']]) for ds in SOURCES})
 def overlap(sp,normalize=False):
  key=norm if normalize else lambda t:t;others={key(r['text']) for r in rows if r['split']==sp}
  return sum(key(r['text']) in others for r in rows if r['split']=='train')
 report['leakage']=dict(train_rows_exactly_matching_any_test=overlap('test'),train_rows_exactly_matching_validation=overlap('validation'),train_rows_normalized_matching_any_test=overlap('test',True),train_rows_normalized_matching_validation=overlap('validation',True),groups_crossing_splits=sum(len({rows[i]['split'] for i in ids})>1 for ids in groups.values()))
 assert all(v==0 for v in report['leakage'].values())
 assert all(v['pairs']==0 for v in report['duplicates']['across_splits'].values())
 assert len({r['row_id'] for r in rows})==n
 for filename,fields in [('split.csv',['row_id','dataset','split','label']),('row_metadata.csv',['row_id','dataset','publisher_split','split','group_id'])]:save_csv(filename,[{f:r[f] for f in fields} for r in rows],fields)
 save_csv('duplicate_pairs.csv',[dict(row_id_a=rows[i]['row_id'],row_id_b=rows[j]['row_id'],kind=kind,jaccard=sim,dataset_a=rows[i]['dataset'],dataset_b=rows[j]['dataset'],split_a=rows[i]['split'],split_b=rows[j]['split']) for i,j,kind,sim in sorted(pairs)],['row_id_a','row_id_b','kind','jaccard','dataset_a','dataset_b','split_a','split_b'])
 save_csv('split_counts.csv',report['split_counts'],['dataset','split','rows','negative','positive','negative_pct','positive_pct'])
 (ROOT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k not in ['manifest','schemas']},indent=2))
if __name__=='__main__':main()
