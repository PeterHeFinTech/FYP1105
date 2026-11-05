"""Compare any synthetic HMDA CSV against the fixed full real reference."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.exceptions import ConvergenceWarning
from hmda import NUM, CAT

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT/'data/raw/hmda_2024_delaware.csv'


def numeric(s):
    x = pd.to_numeric(s, errors='coerce')
    return x.where(np.isfinite(x))


def normalize(frame, reference=False):
    frame = frame.copy()
    if reference:
        frame = frame[frame.action_taken.astype(str).isin(['1', '2', '3'])].copy()
        frame['loanapproved'] = frame.action_taken.isin(['1', '2']).astype(int).astype(str)
    missing = set(NUM+CAT)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('CSV has no usable rows')
    frame = frame[NUM+CAT].astype(str).replace('', '__EMPTY__')
    for c in NUM:
        vals = numeric(frame[c]);mask = vals.notna()
        frame.loc[mask,c] = vals[mask].map(lambda v: format(v, '.17g'))
    return frame.reset_index(drop=True)


def tv(a, b):
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    keys = pa.index.union(pb.index)
    return float(abs(pa.reindex(keys,fill_value=0)-pb.reindex(keys,fill_value=0)).sum()/2)


def describe(real, synth, min_group):
    marginal, tail, dependency, groups = [], [], [], []
    for c in NUM:
        a,b = numeric(real[c]).dropna(),numeric(synth[c]).dropna()
        iqr = a.quantile(.75)-a.quantile(.25) if len(a) else np.nan
        w = wasserstein_distance(a,b) if len(a) and len(b) else np.nan
        marginal.append(dict(field=c,kind='numeric',KS=ks_2samp(a,b).statistic if len(a) and len(b) else np.nan,
            Wasserstein=w,Wasserstein_over_real_IQR=w/iqr if iqr>0 else np.nan,
            missing_rate_gap=abs(1-len(a)/len(real)-(1-len(b)/len(synth))),
            real_valid=len(a),synthetic_valid=len(b)))
        for q in [.01,.99]:
            threshold=a.quantile(q) if len(a) else np.nan
            pa=((a<threshold) if q<.5 else (a>threshold)).mean()
            pb=((b<threshold) if q<.5 else (b>threshold)).mean()
            tail.append(dict(field=c,quantile=q,threshold=threshold,real_fraction=pa,
                             synthetic_fraction=pb,absolute_gap=abs(pa-pb)))
        for q in [0,.001,.01,.5,.99,.999,1]:
            tail.append(dict(field=c,quantile=q,real_value=a.quantile(q) if len(a) else np.nan,
                             synthetic_value=b.quantile(q) if len(b) else np.nan))
    for c in CAT:
        marginal.append(dict(field=c,kind='categorical',TV=tv(real[c],synth[c]),
            unseen_category_fraction=float((~synth[c].isin(real[c].unique())).mean())))
    for c,d in itertools.combinations(NUM,2):
        a=real[[c,d]].apply(numeric).corr(method='spearman').iloc[0,1]
        b=synth[[c,d]].apply(numeric).corr(method='spearman').iloc[0,1]
        dependency.append(dict(field1=c,field2=d,metric='spearman_absolute_gap',value=abs(a-b),real=a,synthetic=b))
    for c,d in itertools.combinations(CAT,2):
        a=pd.Series(list(zip(real[c],real[d])));b=pd.Series(list(zip(synth[c],synth[d])))
        dependency.append(dict(field1=c,field2=d,metric='joint_category_TV',value=tv(a,b)))
    # Real-defined income quartiles: fixed boundaries used for both tables.
    cuts=np.unique(numeric(real.income).dropna().quantile([.25,.5,.75]).to_numpy())
    r,s=real.copy(),synth.copy()
    for f in [r,s]:
        vals=numeric(f.income)
        f['income_group']=pd.cut(vals,[-np.inf,*cuts,np.inf],labels=False).map(lambda v: 'missing' if pd.isna(v) else f'Q{int(v)+1}')
    for keys in [['lei'],['income_group'],['lei','income_group']]:
        def aggregate(f):
            return f.assign(approved=f.loanapproved.eq('1').astype(float)).groupby(keys).approved.agg(['size','mean'])
        a,b=aggregate(r),aggregate(s)
        for key in a.index.union(b.index):
            na=int(a.loc[key,'size']) if key in a.index else 0
            nb=int(b.loc[key,'size']) if key in b.index else 0
            pa=float(a.loc[key,'mean']) if na else np.nan
            pb=float(b.loc[key,'mean']) if nb else np.nan
            groups.append(dict(group_type='+'.join(keys),group=str(key),real_rows=na,synthetic_rows=nb,
                real_approval=pa,synthetic_approval=pb,absolute_gap=abs(pa-pb),
                eligible=na>=min_group and nb>=min_group))
    # Numeric conditional fidelity complements categorical joint tables.
    for c in NUM:
        for d in ['lei','loanapproved']:
            for value, a in real.groupby(d):
                av=numeric(a[c]).dropna();bv=numeric(synth.loc[synth[d]==value,c]).dropna()
                dependency.append(dict(field1=c,field2=d,group=value,metric='conditional_KS',
                    value=ks_2samp(av,bv).statistic if min(len(av),len(bv))>=min_group else np.nan,
                    real_rows=len(av),synthetic_rows=len(bv)))
    return dict(marginals=pd.DataFrame(marginal),tails=pd.DataFrame(tail),
                dependencies=pd.DataFrame(dependency),conditional_approval=pd.DataFrame(groups))


def features(frame):
    out=frame[CAT].copy()
    for c in NUM:
        vals=numeric(frame[c])
        out[c]=np.sign(vals)*np.log1p(abs(vals))
        out[c+'__status']=np.where(vals.notna(),'__VALUE__',frame[c])
    return out


def detect(a,b,seed,trees):
    frame=pd.concat([a,b],ignore_index=True)
    labels=np.r_[np.zeros(len(a),dtype=int),np.ones(len(b),dtype=int)]
    hashes=pd.util.hash_pandas_object(frame,index=False).to_numpy()
    try:
        train,test=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed).split(frame,labels,groups=hashes))
    except ValueError:
        return [dict(model=name,status='insufficient_unique_rows',auc=np.nan) for name in ['logistic','random_forest']]
    if min(len(np.unique(labels[train])),len(np.unique(labels[test])))<2:
        return [dict(model=name,status='insufficient_classes_after_group_split',auc=np.nan) for name in ['logistic','random_forest']]
    x=features(frame);cats=CAT+[c+'__status' for c in NUM]
    rows=[]
    for name,estimator in [('logistic',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed)),
        ('random_forest',RandomForestClassifier(n_estimators=trees,min_samples_leaf=5,class_weight='balanced',n_jobs=4,random_state=seed))]:
        prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RobustScaler()),NUM),
                                ('cat',OneHotEncoder(handle_unknown='ignore'),cats)])
        model=make_pipeline(prep,estimator)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(x.iloc[train],labels[train])
        score=roc_auc_score(labels[test],model.predict_proba(x.iloc[test])[:,1])
        rows.append(dict(model=name,auc=float(score),status='ok',train_rows=len(train),test_rows=len(test),
                         train_synthetic_fraction=float(labels[train].mean()),test_synthetic_fraction=float(labels[test].mean()),
                         warnings='; '.join(str(w.message) for w in caught)))
    return rows


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,list):return [clean_json(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def evaluate_file(path,out,args):
    real=normalize(pd.read_csv(REAL,dtype=str,keep_default_na=False),reference=True)
    synth=normalize(pd.read_csv(path,dtype=str,keep_default_na=False))
    out.mkdir(parents=True,exist_ok=False)
    tables=describe(real,synth,args.min_group)
    for name,table in tables.items():table.to_csv(out/f'{name}.csv',index=False)
    # Equal-size samples and real-vs-real calibration. No generator is trained.
    n=min(args.detector_rows,len(synth),len(real)//2)
    detectors=[]
    for repeat in range(args.repeats):
        seed=args.seed+repeat
        rr=real.sample(n=2*n,random_state=seed).reset_index(drop=True)
        ss=synth.sample(n=n,random_state=seed).reset_index(drop=True)
        for comparison,a,b in [('real_vs_synthetic',rr.iloc[:n],ss),('real_vs_real',rr.iloc[:n],rr.iloc[n:])]:
            if n<20:
                rows=[dict(model=m,auc=np.nan,status='fewer_than_20_rows_per_source') for m in ['logistic','random_forest']]
            else:rows=detect(a,b,seed,args.trees)
            for row in rows:detectors.append(dict(comparison=comparison,seed=seed,rows_per_source=n,**row))
        print(f'Detector repeat {repeat+1}/{args.repeats} complete: {path.name}',flush=True)
    det=pd.DataFrame(detectors);det.to_csv(out/'detectors.csv',index=False)
    agg=det.groupby(['comparison','model']).auc.agg(['mean','std','count']).reset_index()
    agg.to_csv(out/'detector_summary.csv',index=False)
    rh=pd.util.hash_pandas_object(real,index=False);sh=pd.util.hash_pandas_object(synth,index=False)
    matched=real.sample(n=min(len(real),len(synth)),random_state=args.seed)
    group_summary=[]
    for kind,g in tables['conditional_approval'].groupby('group_type'):
        valid=g[g.eligible]
        group_summary.append(dict(group_type=kind,eligible_groups=len(valid),
            real_row_coverage=float(valid.real_rows.sum()/len(real)),
            macro_approval_gap=float(valid.absolute_gap.mean()),
            weighted_approval_gap=float(np.average(valid.absolute_gap,weights=valid.real_rows)) if len(valid) else None))
    summary=dict(reference_path=str(REAL),synthetic_path=str(path.resolve()),
        reference_sha256=hashlib.sha256(REAL.read_bytes()).hexdigest(),synthetic_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_scope='Full fixed HMDA CSV, actions 1/2/3, all institutions. Includes generator training rows.',
        real_rows=len(real),synthetic_rows=len(synth),real_approval_rate=float(real.loanapproved.eq('1').mean()),
        synthetic_approval_rate=float(synth.loanapproved.eq('1').mean()),
        approval_rate_absolute_gap=float(abs(real.loanapproved.eq('1').mean()-synth.loanapproved.eq('1').mean())),
        synthetic_duplicate_fraction=float(sh.duplicated().mean()),real_duplicate_fraction=float(rh.duplicated().mean()),
        size_matched_real_duplicate_fraction=float(matched.duplicated().mean()),
        exact_full_reference_match_fraction=float(sh.isin(set(rh)).mean()),
        conditional_groups=group_summary,detectors=agg.to_dict('records'),
        settings=dict(seed=args.seed,repeats=args.repeats,detector_rows=args.detector_rows,trees=args.trees,min_group=args.min_group),
        limitations=['Full-reference similarity, not held-out generator generalization or privacy proof.',
            'Detector AUC near 0.5 means no distinction found by that detector; near 1 means distinguishable.',
            'Detector test folds are disjoint by canonical exact-row hash; preprocessing fitted only on detector training folds.',
            'Repeated-seed standard deviations measure split/sample variability, not formal confidence intervals.',
            'Group scores omit groups with fewer than min_group valid observations in either dataset; coverage is reported.',
            'No scalar overall quality score or universal pass threshold; no prediction utility or membership attack is implemented.'])
    summary=clean_json(summary)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=['# Synthetic HMDA evaluation',f'Input: `{path.resolve()}`',f'Reference: `{REAL}`',
        '**Full-reference similarity: the reference includes generator training records.**',
        f'Real rows: {len(real):,}; synthetic rows: {len(synth):,}.',
        f'Approval-rate gap: {summary["approval_rate_absolute_gap"]*100:.2f} percentage points.',
        f'Synthetic duplicates: {summary["synthetic_duplicate_fraction"]*100:.2f}%.',
        '## Detector AUC (0.5: indistinguishable to this detector; 1: distinguishable)',
        '| Comparison | Detector | Mean AUC | Seed SD |','|---|---|---:|---:|']
    for row in agg.itertuples():lines.append(f'| {row.comparison} | {row.model} | {row.mean:.4f} | {row.std:.4f} |')
    lines+=['## Detailed files','- `marginals.csv`: numerical KS/Wasserstein, missingness, category TV.',
        '- `dependencies.csv`: numerical correlations, categorical joint distributions, conditional numerical KS.',
        '- `conditional_approval.csv`: approval within banks, income groups and bank/income groups.',
        '- `tails.csv`: extreme quantiles and tail frequencies.',
        '- `detectors.csv`: every detector run, sample sizes and convergence warnings.',
        '## Limits',*['- '+v for v in summary['limitations']]]
    (out/'report.md').write_text('\n'.join(lines))
    print(f'Report: {out / "report.md"}',flush=True)
    return dict(file=str(path),report=str(out),approval_gap=summary['approval_rate_absolute_gap'],
                duplicate_fraction=summary['synthetic_duplicate_fraction'],
                **{f'{row.model}_auc':row.mean for row in agg[agg.comparison=='real_vs_synthetic'].itertuples()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('synthetic',nargs='+',type=Path,help='One or more synthetic CSV files; required fields are the shared 13-column schema')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/benchmark')
    p.add_argument('--detector-rows',type=int,default=5000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--trees',type=int,default=150)
    p.add_argument('--min-group',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    for name in ['detector_rows','repeats','trees','min_group']:
        if getattr(args,name)<1:p.error(f'{name} must be positive')
    if not REAL.is_file():p.error(f'Fixed reference CSV missing: {REAL}')
    for path in args.synthetic:
        if not path.is_file():p.error(f'Synthetic CSV missing: {path}')
    if args.out.exists():p.error(f'Output already exists; choose a new --out: {args.out}')
    args.out.mkdir(parents=True)
    results=[]
    for i,path in enumerate(args.synthetic,1):
        results.append(evaluate_file(path,args.out/f'{i:02d}_{path.parent.name}_{path.stem}',args))
    pd.DataFrame(results).to_csv(args.out/'comparison.csv',index=False)
    print(f'Comparison table: {args.out / "comparison.csv"}')


if __name__=='__main__':main()
