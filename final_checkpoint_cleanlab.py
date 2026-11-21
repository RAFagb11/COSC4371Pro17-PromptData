"""COSC 4371 Project 17: baseline vs CleanLab, same held-out test rows.

Run inside the repository directory AFTER `python audit_split.py`.
Requirements: `pip install -r requirements_baseline.txt` and `pip install cleanlab==2.7.1`.

Outputs:
  scores.csv                 Final-checkpoint schema (backs up previous scores.csv)
  cleanlab_flags.csv         All training rows with flag and OOF confidence
  cleanlab_summary.json      Training counts, thresholds, metrics, timings
  comparison_deepset_pr.png  PR curves for two values on the same external rows
  error_sample_to_review.csv  10 FP + 10 FN sample, text included for manual categorization

The error sample requires HUMAN review. It is NOT an errors.csv until the
sampled prompts have been assigned meaningful, non-empty categories.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from baseline import load_records, choose_threshold, metrics

ROOT = Path(__file__).resolve().parent
SEED = 42
FPR_LIMIT = 0.05


def model_from_config(cfg):
    v = cfg['representation']
    c = cfg['classifier']
    v_kwargs = {k: v[k] for k in (
        'lowercase','strip_accents','analyzer','token_pattern','min_df','max_df',
        'max_features','sublinear_tf','norm','use_idf','smooth_idf')}
    v_kwargs['ngram_range'] = tuple(v['ngram_range'])
    clf_kwargs = {k: c[k] for k in (
        'C','penalty','solver','class_weight','max_iter','random_state')}
    return Pipeline([
        ('tfidf', TfidfVectorizer(**v_kwargs)),
        ('classifier', LogisticRegression(**clf_kwargs)),
    ])


def main():
    required = ['split.csv', 'row_metadata.csv', 'detector_config.json']
    for f in required:
        if not (ROOT / f).exists():
            raise FileNotFoundError(f'Missing {f}: run audit_split.py in this folder')

    split = pd.read_csv(ROOT/'split.csv', dtype={'row_id':str, 'dataset':str, 'split':str})
    meta = pd.read_csv(ROOT/'row_metadata.csv', usecols=['row_id','group_id'], dtype=str)
    data = load_records()
    d = split.merge(data, on=['row_id','dataset'], validate='one_to_one').merge(meta,on='row_id',validate='one_to_one')
    assert len(d)==10958 and d.row_id.is_unique and (d.label == d.label_loaded).all()
    train = d.query("dataset == 'safeguard' and split == 'train'").sort_values('row_id').reset_index(drop=True)
    valid = d.query("dataset == 'safeguard' and split == 'validation'").sort_values('row_id').reset_index(drop=True)
    tests = {name: d.query("dataset == @name and split == 'test'").sort_values('row_id').reset_index(drop=True)
             for name in ['safeguard','deepset']}
    assert (len(train),len(valid),len(tests['safeguard']),len(tests['deepset'])) == (6156,1537,2603,662)
    cfg = json.loads((ROOT/'detector_config.json').read_text(encoding='utf-8'))
    estimator = model_from_config(cfg)
    X = train.text.to_numpy(dtype=str)
    y = train.label.to_numpy(dtype=int)
    groups = train.group_id.to_numpy()
    cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    oof = np.empty((len(train),2),dtype=float)
    durations=[]
    seen=np.zeros(len(train),dtype=bool)
    for k,(idx_fit,idx_hold) in enumerate(cv.split(X,y,groups),1):
        start=time.perf_counter()
        fold_model=clone(estimator)
        fold_model.fit(X[idx_fit], y[idx_fit])
        oof[idx_hold] = fold_model.predict_proba(X[idx_hold])
        assert fold_model.classes_.tolist()==[0,1]
        seen[idx_hold]=True
        elapsed=time.perf_counter()-start
        durations.append(elapsed)
        print(f'CleanLab out-of-fold predictions: fold {k}/5 done ({elapsed:.2f}s)')
    assert seen.all() and np.isfinite(oof).all()

    # CleanLab flags are suspected mislabeled examples, not verified label errors.
    mask = np.asarray(find_label_issues(labels=y,pred_probs=oof,
                                       filter_by='prune_by_noise_rate',verbose=False),dtype=bool)
    assert mask.shape==(len(train),)
    train_flags=train[['row_id','label','text']].copy()
    train_flags['flag_cleanlab'] = mask
    train_flags['out_of_fold_probability_attack'] = oof[:,1]
    train_flags['out_of_fold_given_label_confidence'] = oof[np.arange(len(y)), y]
    train_flags.to_csv(ROOT/'cleanlab_flags.csv',index=False)
    print(f'CleanLab flagged {int(mask.sum())} / {len(mask)} Safe-Guard TRAIN rows; retaining {int((~mask).sum())}.')
    assert len(set(y[~mask]))==2, 'CleanLab removed an entire class; review flags before training.'

    all_scores=[]
    result={
        'seed': SEED,
        'training_data': 'safeguard train only, unchanged split.csv',
        'cv': '5-fold StratifiedGroupKFold, grouped by audit row_metadata.csv group_id',
        'flagging': 'cleanlab.filter.find_label_issues, filter_by=prune_by_noise_rate',
        'flagged_safeguard_training_rows':int(mask.sum()),
        'starting_training_rows':len(train),
        'remaining_training_rows':int((~mask).sum()),
        'cv_fit_seconds':durations,
        'values':{}
    }
    for value, keep in [('baseline',np.ones(len(train),dtype=bool)),('cleanlab',~mask)]:
        model=clone(estimator)
        start=time.perf_counter()
        model.fit(X[keep],y[keep])
        training_seconds=time.perf_counter()-start
        val_prob = model.predict_proba(valid.text.astype(str).tolist())[:,1]
        threshold,val_fpr,val_recall=choose_threshold(valid.label.to_numpy(dtype=int),val_prob,FPR_LIMIT)
        result['values'][value]={'training_rows':int(keep.sum()),'training_seconds':training_seconds,
                                 'threshold':threshold, 'validation_fpr':val_fpr,
                                 'validation_recall':val_recall,'test_metrics':{}}
        print(f'{value}: trained {int(keep.sum())} rows in {training_seconds:.2f}s; threshold={threshold:.17g}; validation FPR={val_fpr:.4f}')
        for name,test in tests.items():
            probs=model.predict_proba(test.text.astype(str).tolist())[:,1]
            met=metrics(test.label.to_numpy(dtype=int),probs,threshold)
            result['values'][value]['test_metrics'][f'{name}_test']=met
            print(f'  {name}_test: AUROC={met["AUROC"]:.4f}, recall={met["recall"]:.4f}, AP={met["AP"]:.4f}')
            all_scores.append(pd.DataFrame({
                'row_id':test.row_id.to_numpy(),
                'value':value,
                'column':f'{name}_test',
                'label':test.label.to_numpy(dtype=int),
                'score':probs,
                'threshold':threshold,
            }))

    old=ROOT/'scores.csv'
    if old.exists():
        # Never overwrite the original midsemester scores without a backup.
        backup=ROOT/'scores_midsemester_backup.csv'
        if not backup.exists() and 'test_set' in pd.read_csv(old,nrows=0).columns:
            shutil.copy2(old,backup)
            print('Backed up old scores.csv as scores_midsemester_backup.csv')
    scores=pd.concat(all_scores,ignore_index=True)
    scores.to_csv(ROOT/'scores.csv',index=False,float_format='%.17g')
    (ROOT/'cleanlab_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')

    # PR curves meet the figure requirement for two variable values in one column.
    fig,ax=plt.subplots(figsize=(6.7,4.8))
    for value in ['baseline','cleanlab']:
        s=scores[(scores['value']==value)&(scores['column']=='deepset_test')]
        precision,recall,_=precision_recall_curve(s.label,s.score)
        ax.plot(recall,precision,label=f'{value} (AP={result["values"][value]["test_metrics"]["deepset_test"]["AP"]:.3f})')
    ax.set(xlabel='Recall',ylabel='Precision',title='deepset external test: precision–recall curves')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(ROOT/'comparison_deepset_pr.png',dpi=160)
    plt.close(fig)

    # Sample only one value/column for analysis. User must assign categories manually.
    source=scores[(scores.value=='baseline')&(scores.column=='deepset_test')].copy()
    source['predicted_label']=(source.score>=source.threshold).astype(int)
    fp=source[(source.label==0)&(source.predicted_label==1)].sample(n=10,random_state=SEED)
    fn=source[(source.label==1)&(source.predicted_label==0)].sample(n=10,random_state=SEED)
    sample=pd.concat([fp,fn],ignore_index=True)
    text_by_id=d.set_index('row_id').text.to_dict()
    sample['error_type']=np.where(sample.label==0,'false positive','false negative')
    sample['prompt_text']=sample.row_id.map(text_by_id)
    sample['category']=''  # DO NOT automatically invent or assign manual categories.
    sample.to_csv(ROOT/'error_sample_to_review.csv',index=False)
    print('\nFinished. Files: scores.csv, cleanlab_flags.csv, cleanlab_summary.json, comparison_deepset_pr.png, error_sample_to_review.csv')
    print('Next: run the course checker. Before final submission, manually categorize error_sample_to_review.csv and save errors.csv (row_id,value,column,category).')

if __name__=='__main__':
    main()
