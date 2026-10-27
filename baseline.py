"""COSC 4371 Project 17 variant: one fixed prompt-injection baseline.

Run from the same directory as Section 2's audit_split.py, split.csv,
row_metadata.csv, and data_cache/ (run audit_split.py first if needed).

Outputs:
  scores.csv                      one row per test example, course-checker format
  baseline_results.json           training time, validation-selected threshold, metrics
  baseline_confusion_deepset.png  Section 4 figure, external deepset test set
  checker_output.txt              if check_checkpoint.py is in this directory

The model, representation and hyperparameters match detector_config.json.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent
SEED = 42
MAX_VALIDATION_FPR = 0.05


def load_records():
    """Reconstruct row_id exactly as it was constructed by audit_split.py."""
    source_files = {
        'safeguard': {
            'train': 'safeguard_train-00000-of-00001.parquet',
            'test': 'safeguard_test-00000-of-00001.parquet',
        },
        'deepset': {
            'train': 'deepset_train-00000-of-00001-9564e8b05b4757ab.parquet',
            'test': 'deepset_test-00000-of-00001-701d16158af87368.parquet',
        },
    }
    all_records = []
    for dataset, files in source_files.items():
        for pub_split, filename in files.items():
            path = ROOT / 'data_cache' / filename
            if not path.exists():
                raise FileNotFoundError(
                    f'Missing {path}. First run python audit_split.py in this same folder.'
                )
            table = pq.read_table(path, columns=['text', 'label'])
            for i, entry in enumerate(table.to_pylist()):
                all_records.append({
                    'row_id': f'{dataset}:{pub_split}:{i}',
                    'dataset': dataset,
                    'text': entry['text'],
                    'label_loaded': int(entry['label']),
                })
    return pd.DataFrame(all_records)


def choose_threshold(y_valid, p_valid, fpr_limit):
    """Lowest observed score cutoff with empirical validation FPR <= 5%.

    Only validation labels and scores are used, never test labels or scores.
    The smallest qualifying cutoff maximizes recall among valid cutoffs.
    """
    fpr, tpr, thresholds = roc_curve(y_valid, p_valid, drop_intermediate=False)
    admissible = (fpr <= fpr_limit) & np.isfinite(thresholds)
    if not admissible.any():
        raise RuntimeError('No finite decision threshold meets the FPR requirement')
    threshold = float(np.min(thresholds[admissible]))
    val_pred = (p_valid >= threshold).astype(int)
    val_fpr = float(np.mean(val_pred[np.asarray(y_valid) == 0]))
    val_recall = float(np.mean(val_pred[np.asarray(y_valid) == 1]))
    if val_fpr > fpr_limit + 1e-12:
        raise AssertionError('Validation FPR limit violated unexpectedly')
    return threshold, val_fpr, val_recall


def metrics(y, prob, cutoff):
    pred = (prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'rows': int(len(y)), 'positive_rows': int(np.sum(y)),
        'positive_pct': float(100 * np.mean(y)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'false_positive_rate': float(fp / (fp + tn)) if fp + tn else None,
        'AP': float(average_precision_score(y, prob)),
        'AUROC': float(roc_auc_score(y, prob)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'random_AP_reference': float(np.mean(y)),
    }


def main():
    split_path = ROOT / 'split.csv'
    meta_path = ROOT / 'row_metadata.csv'
    for required in [split_path, meta_path, ROOT / 'detector_config.json']:
        if not required.exists():
            raise FileNotFoundError(f'{required} missing: use Section 2 output folder')

    split = pd.read_csv(split_path, dtype={'row_id':str, 'dataset':str, 'split':str})
    meta = pd.read_csv(meta_path, dtype={'row_id':str})
    records = load_records()
    assert len(records) == len(split) == len(meta) == 10958, 'Unexpected dataset/split size'
    assert records['row_id'].is_unique and split['row_id'].is_unique
    assert set(records.row_id) == set(split.row_id) == set(meta.row_id)
    assert meta[['row_id','split']].merge(split[['row_id','split']], on='row_id',
           suffixes=('_metadata','_split')).eval('split_metadata == split_split').all()
    d = split.merge(records, on=['row_id','dataset'], validate='one_to_one')
    assert (d.label == d.label_loaded).all(), 'Dataset labels differ from split.csv'
    assert (d.loc[d.split.isin(['train', 'validation']), 'dataset'] == 'safeguard').all(), \
        'Expected Safe-Guard-only training/validation'
    assert len(d[d.split=='test']) == 3265, 'Unexpected test-set size'

    training = d[(d.dataset=='safeguard') & (d.split=='train')]
    validating = d[(d.dataset=='safeguard') & (d.split=='validation')]
    testing = d[d.split=='test'].copy()

    config = json.loads((ROOT/'detector_config.json').read_text(encoding='utf8'))
    assert config['classifier']['class'] == 'sklearn.linear_model.LogisticRegression'
    assert config['representation']['class'] == 'sklearn.feature_extraction.text.TfidfVectorizer'
    vcfg = config['representation']
    ccfg = config['classifier']
    tfidf = TfidfVectorizer(
        lowercase=vcfg['lowercase'], strip_accents=vcfg['strip_accents'],
        analyzer=vcfg['analyzer'], token_pattern=vcfg['token_pattern'],
        ngram_range=tuple(vcfg['ngram_range']), min_df=vcfg['min_df'],
        max_df=vcfg['max_df'], max_features=vcfg['max_features'],
        sublinear_tf=vcfg['sublinear_tf'], norm=vcfg['norm'],
        use_idf=vcfg['use_idf'], smooth_idf=vcfg['smooth_idf'],
    )
    model = LogisticRegression(
        C=ccfg['C'], penalty=ccfg['penalty'], solver=ccfg['solver'],
        class_weight=ccfg['class_weight'], max_iter=ccfg['max_iter'],
        random_state=ccfg['random_state'],
    )
    print('Training baseline: TF-IDF + LogisticRegression ...', flush=True)
    start = time.perf_counter()
    X_train = tfidf.fit_transform(training.text.tolist())
    model.fit(X_train, training.label.to_numpy())
    elapsed = time.perf_counter() - start
    X_val = tfidf.transform(validating.text.tolist())
    val_prob = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
    cutoff, val_fpr, val_recall = choose_threshold(
        validating.label.to_numpy(), val_prob, MAX_VALIDATION_FPR)
    print(f'Training time: {elapsed:.2f}s')
    print(f'Validation threshold: {cutoff:.17g} (FPR={val_fpr:.4f}; recall={val_recall:.4f})')

    scores = []
    per_test = {}
    for dataset in ['safeguard','deepset']:
        group = testing[testing.dataset == dataset]
        prob = model.predict_proba(tfidf.transform(group.text.tolist()))[:,list(model.classes_).index(1)]
        y = group.label.to_numpy(dtype=int)
        test_set = f'{dataset}_test'
        per_test[test_set] = metrics(y, prob, cutoff)
        for rid, label, p in zip(group.row_id, y, prob):
            scores.append({'row_id':rid, 'test_set':test_set, 'label':int(label), 'score':format(float(p),'.17g')})
        print(f"{test_set}: rows={len(group)}, AUROC={per_test[test_set]['AUROC']:.4f}, "
              f"AP={per_test[test_set]['AP']:.4f}, recall={per_test[test_set]['recall']:.4f}")

    with (ROOT/'scores.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['row_id','test_set','label','score'])
        writer.writeheader(); writer.writerows(scores)
    results = {
        'project': 'COSC4371 Project 17 variant; unfiltered baseline',
        'method': 'no filtering', 'seed':SEED, 'python':platform.python_version(),
        'scikit_learn': __import__('sklearn').__version__,
        'platform':platform.platform(), 'processor':platform.processor(),
        'train_rows':int(len(training)), 'validation_rows':int(len(validating)),
        'training_seconds':round(elapsed,3),
        'threshold_rule': 'lowest observed probability threshold with validation FPR <= 0.05',
        'threshold':cutoff, 'threshold_cli':format(cutoff,'.17g'),
        'validation_fpr':val_fpr, 'validation_recall':val_recall,
        'test_metrics':per_test,
    }
    (ROOT/'baseline_results.json').write_text(json.dumps(results,indent=2),encoding='utf8')

    ext = per_test['deepset_test']
    matrix = np.array([[ext['tn'],ext['fp']],[ext['fn'],ext['tp']]])
    fig, ax = plt.subplots(figsize=(5.8,4.8))
    ax.imshow(matrix, cmap='Blues')
    for (r,c), value in np.ndenumerate(matrix):
        ax.text(c,r,str(value),ha='center',va='center', fontsize=20,
                color='white' if value > matrix.max()*0.5 else 'black')
    ax.set_xticks([0,1],['Predicted benign','Predicted injection'])
    ax.set_yticks([0,1],['Actual benign','Actual injection'])
    ax.set_title(f"Baseline confusion matrix: deepset test\n"
                 f"n={ext['rows']}, positive={ext['positive_pct']:.2f}%")
    ax.set_xlabel('Prediction'); ax.set_ylabel('True label')
    fig.tight_layout()
    fig.savefig(ROOT/'baseline_confusion_deepset.png', dpi=180)
    plt.close(fig)
    print('Wrote scores.csv, baseline_results.json, baseline_confusion_deepset.png')

    checker = ROOT/'check_checkpoint.py'
    if checker.exists():
        args = [sys.executable,str(checker),'--split',str(split_path),
                '--scores',str(ROOT/'scores.csv'), '--threshold', format(cutoff,'.17g')]
        print('Running course checker ...', flush=True)
        checker_result = subprocess.run(args,text=True,capture_output=True)
        checker_text = checker_result.stdout + checker_result.stderr
        (ROOT/'checker_output.txt').write_text(checker_text,encoding='utf8')
        print(checker_text)
        if checker_result.returncode:
            raise RuntimeError(f'Course checker failed (exit code {checker_result.returncode})')
    else:
        print('Put check_checkpoint.py beside baseline.py, then run the checker using the '
              f'threshold {cutoff:.17g}.')

if __name__ == '__main__':
    main()
