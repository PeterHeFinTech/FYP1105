"""Train on real/synthetic data and evaluate on the same held-out real rows."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score, confusion_matrix

from evaluation.benchmark import normalize, features, NUM, CAT

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    frame = normalize(pd.read_csv(path, dtype=str, keep_default_na=False))
    if not frame.loanapproved.isin(['0', '1']).all():
        raise ValueError(f'{path}: loanapproved must contain only 0 and 1')
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('synthetic', nargs='+', type=Path)
    p.add_argument('--out', type=Path, default=ROOT/'outputs/tstr')
    p.add_argument('--rows', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.rows < 2:
        p.error('--rows must be at least 2')
    real = read(ROOT/'data/prepared/train_real.csv')
    test = read(ROOT/'data/prepared/test_real.csv')
    sources = [('real', real)] + [(f'{i}_{f.parent.name}_{f.stem}', read(f)) for i, f in enumerate(args.synthetic, 1)]
    n = min(args.rows, *(len(frame) for _, frame in sources))
    if test.loanapproved.nunique() != 2:
        raise ValueError('Real test set must contain both outcomes')
    args.out.mkdir(parents=True, exist_ok=False)
    cats = [c for c in CAT if c != 'loanapproved'] + [c+'__status' for c in NUM]
    xt = features(test).drop(columns='loanapproved')
    yt = test.loanapproved.astype(int)
    results = []
    for name, frame in sources:
        frame = frame.sample(n=n, random_state=args.seed)
        x = features(frame).drop(columns='loanapproved')
        y = frame.loanapproved.astype(int)
        if y.nunique() != 2:
            raise ValueError(f'{name}: training sample must contain both outcomes')
        estimators = {
            'logistic': LogisticRegression(max_iter=3000, random_state=args.seed),
            'svm': SVC(C=1.0, kernel='rbf', gamma='scale', cache_size=1024),
            'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=4, random_state=args.seed),
        }
        for model_name, estimator in estimators.items():
            prep = ColumnTransformer([
                ('num', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), NUM),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
            ])
            model = make_pipeline(prep, estimator)
            print(f'{name}: {model_name}, {n} training rows', flush=True)
            model.fit(x, y)
            pred = model.predict(xt)
            scores = model.decision_function(xt) if hasattr(estimator, 'decision_function') else model.predict_proba(xt)[:, 1]
            tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
            results.append(dict(source=name, model=model_name, train_rows=n, test_rows=len(test),
                accuracy=accuracy_score(yt,pred), balanced_accuracy=balanced_accuracy_score(yt,pred),
                denied_recall=recall_score(yt,pred,pos_label=0,zero_division=0), auc=roc_auc_score(yt,scores),
                denied_correct=tn, denied_as_approved=fp, approved_as_denied=fn, approved_correct=tp))
            pd.DataFrame(results).to_csv(args.out/'results.csv', index=False)
    result = pd.DataFrame(results)
    base = result[result.source == 'real'].set_index('model')
    for metric in ['accuracy','balanced_accuracy','denied_recall','auc']:
        result[metric+'_minus_real'] = result[metric] - result.model.map(base[metric])
    result.to_csv(args.out/'results.csv', index=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(seed=args.seed, train_rows_per_source=n,
        synthetic_paths=[str(f.resolve()) for f in args.synthetic],
        real_train=str(ROOT/'data/prepared/train_real.csv'), real_test=str(ROOT/'data/prepared/test_real.csv'),
        note='Fixed hyperparameters; preprocessing fitted separately on each training source. Approval is excluded from inputs. Synthetic generators must not have seen the real test set. One seed only; no confidence intervals.'),indent=2))
    print(result.to_string(index=False))
    print(f'Saved {args.out / "results.csv"}')


if __name__ == '__main__':
    main()
