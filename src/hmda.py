"""Prepare, train, sample and evaluate an HMDA TabDDPM generator."""
import argparse
import csv
import hashlib
import json
import pickle
import random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import ks_2samp, wasserstein_distance
from model import make_model

NUM = ['income', 'loan_amount', 'property_value', 'loan_term']
CAT = ['lei', 'applicant_age', 'applicant_sex', 'debt_to_income_ratio',
       'loan_purpose', 'loan_type', 'occupancy_type', 'lien_status', 'loanapproved']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Check the active Python environment.')
    return name


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    required = set(NUM + CAT) - {'loanapproved'} | {'action_taken'}
    if required - set(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')
    df = df[df.action_taken.isin(['1', '2', '3'])].copy()
    # Select the study population; no outcome-based ranking of institutions.
    institutions = df.lei.value_counts().head(args.top_institutions).index.tolist()
    df = df[df.lei.isin(institutions)].copy()
    df['loanapproved'] = df.action_taken.isin(['1', '2']).astype(int).astype(str)
    df = df[NUM + CAT].replace('', '__EMPTY__').reset_index(drop=True)
    # Keep exactly identical selected rows together to avoid exact-row leakage.
    keys = pd.util.hash_pandas_object(df, index=False).astype(str)
    unique_keys = keys.drop_duplicates().to_numpy()
    train_keys, test_keys = train_test_split(unique_keys, test_size=0.2, random_state=args.seed)
    train = df[keys.isin(train_keys)].copy()
    test = df[keys.isin(test_keys)].copy()
    train.to_csv(out / 'train_real.csv', index=False)
    test.to_csv(out / 'test_real.csv', index=False)
    metadata = {'numerical': NUM, 'categorical': CAT, 'categories': {}, 'medians': {},
                'bounds': {}, 'seed': args.seed, 'institutions': institutions}
    numerical = []
    encoded_cat = train[CAT].copy()
    for col in NUM:
        values = pd.to_numeric(train[col], errors='coerce')
        values = values.where(np.isfinite(values))
        if not values.notna().any():
            raise ValueError(f'{col} has no finite training values')
        # Numeric values use Gaussian diffusion; NA/Exempt/empty tokens get
        # separate categorical coordinates and are restored after sampling.
        status = col + '__status'
        encoded_cat[status] = np.where(values.notna(), '__VALUE__', train[col])
        metadata['medians'][col] = float(values.median())
        metadata['bounds'][col] = [float(values.min()), float(values.max())]
        numerical.append(values.fillna(values.median()).to_numpy())
    transformer = QuantileTransformer(n_quantiles=min(1000, len(train)),
        output_distribution='normal', random_state=args.seed, subsample=None)
    x_num = transformer.fit_transform(np.column_stack(numerical)).astype('float32')
    metadata['encoded_categorical'] = list(encoded_cat.columns)
    codes = []
    for col in encoded_cat:
        categories = sorted(encoded_cat[col].unique().tolist())
        metadata['categories'][col] = categories
        codes.append(encoded_cat[col].map({v:i for i,v in enumerate(categories)}).to_numpy())
    x = np.column_stack([x_num] + codes).astype('float32')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite encoded training data')
    np.save(out / 'train.npy', x)
    with (out / 'transformer.pkl').open('wb') as f:
        pickle.dump(transformer, f)
    metadata.update({'train_rows':len(train), 'test_rows':len(test),
        'raw_sha256':hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
        'split':'80/20 random split of unique selected rows; duplicate rows stay together'})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Prepared {len(train):,} train / {len(test):,} test rows; {len(institutions)} institutions.')


def load_meta(directory):
    return json.loads((Path(directory) / 'metadata.json').read_text())


def train(args):
    seed_all(args.seed)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    m = load_meta(data)
    device = device_for(args.device)
    x = torch.from_numpy(np.load(data / 'train.npy')).to(device)
    counts = [len(m['categories'][c]) for c in m['encoded_categorical']]
    model = make_model(len(NUM), counts, args.timesteps, args.hidden, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    config = {'timesteps':args.timesteps, 'hidden':args.hidden, 'seed':args.seed,
              'iterations':args.iterations, 'batch_size':args.batch_size, 'lr':args.lr}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    log = (out / 'loss.csv').open('w', buffering=1)
    writer = csv.writer(log)
    writer.writerow(['step','categorical_loss','numerical_loss','total_loss'])
    print(f'Training on {device}; rows={len(x)}, categorical coordinates={len(counts)}', flush=True)
    model.train()
    for step in range(1, args.iterations + 1):
        batch = x[torch.randint(len(x), (args.batch_size,), device=device)]
        optimizer.zero_grad(set_to_none=True)
        cat, num = model.mixed_loss(batch, {})
        loss = cat + num
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}; stopping')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = args.lr * (1 - step / args.iterations)
        if step == 1 or step % 100 == 0:
            writer.writerow([step,cat.item(),num.item(),loss.item()])
            print(f'{step}/{args.iterations} loss={loss.item():.5f}', flush=True)
        if step % args.save_every == 0 or step == args.iterations:
            checkpoint = {'state_dict':model.state_dict(), 'config':config, 'step':step,
                          'metadata':m}
            temp = out / 'checkpoint.tmp'
            torch.save(checkpoint, temp)
            temp.replace(out / 'checkpoint.pt')
    log.close()


def decode(x, metadata, transformer):
    n = len(NUM)
    numeric = transformer.inverse_transform(x[:, :n].astype(np.float64))
    # Restore fitted extremes at float32-rounded Gaussian endpoints.
    endpoints = transformer.transform(np.array([
        [metadata['bounds'][c][0] for c in NUM],
        [metadata['bounds'][c][1] for c in NUM]], dtype=np.float64)).astype(np.float32)
    for i, c in enumerate(NUM):
        numeric[x[:, i] <= endpoints[0, i], i] = metadata['bounds'][c][0]
        numeric[x[:, i] >= endpoints[1, i], i] = metadata['bounds'][c][1]
    result = pd.DataFrame()
    for i,c in enumerate(metadata['encoded_categorical']):
        ids = x[:, n+i].astype(int)
        categories = metadata['categories'][c]
        if ((ids < 0) | (ids >= len(categories))).any():
            raise ValueError('Invalid categorical sample')
        result[c] = np.array(categories)[ids]
    for i,c in enumerate(NUM):
        lo, hi = metadata['bounds'][c]
        v = np.clip(numeric[:, i], lo, hi)
        # HMDA's public amount/value are midpoint-binned in $10,000 bands.
        if c in ['loan_amount','property_value']:
            v = np.rint((v - 5000) / 10000) * 10000 + 5000
        else:
            v = np.rint(v)
        v = np.clip(v, lo, hi)
        result[c] = [str(int(t)) if float(t).is_integer() else str(t) for t in v]
        status = result.pop(c + '__status')
        mask = status != '__VALUE__'
        result.loc[mask, c] = status[mask]
    return result[NUM + CAT].replace('__EMPTY__','')


def sample(args):
    seed_all(args.seed)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    device = device_for(args.device)
    # Only load checkpoints/transformers created by this project and trusted by you.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    m, config = checkpoint['metadata'], checkpoint['config']
    if m != load_meta(args.data):
        raise ValueError('Prepared data metadata does not match checkpoint')
    counts = [len(m['categories'][c]) for c in m['encoded_categorical']]
    model = make_model(len(NUM), counts, config['timesteps'], config['hidden'], device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with (Path(args.data) / 'transformer.pkl').open('rb') as f:
        transformer = pickle.load(f)
    chunks = []
    with torch.no_grad():
        for start in range(0, args.rows, args.batch_size):
            # A dummy class is required by upstream; our denoiser ignores it.
            x, _ = model.sample(min(args.batch_size,args.rows-start), torch.ones(1,device=device))
            x = x.numpy()
            if not np.isfinite(x).all():
                raise RuntimeError('Nonfinite sample; inspect training before proceeding')
            chunks.append(decode(x, m, transformer))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(chunks, ignore_index=True).to_csv(out, index=False)
    print(f'Wrote {args.rows:,} synthetic rows to {out}')


def evaluate(args):
    data = Path(args.data)
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False).replace('', '__EMPTY__')
    real, train_df, synth = read(data/'test_real.csv'), read(data/'train_real.csv'), read(args.synthetic)
    if list(synth.columns) != NUM+CAT or synth.empty:
        raise ValueError('Synthetic table must be nonempty with the expected schema')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for c in NUM:
        a = pd.to_numeric(real[c], errors='coerce').dropna()
        b = pd.to_numeric(synth[c], errors='coerce').dropna()
        records.append({'field':c,'metric':'KS','value':float(ks_2samp(a,b).statistic) if len(a) and len(b) else None})
        records.append({'field':c,'metric':'Wasserstein_raw_units','value':float(wasserstein_distance(a,b)) if len(a) and len(b) else None})
        records.append({'field':c,'metric':'missing_rate_absolute_gap','value':abs((1-len(a)/len(real))-(1-len(b)/len(synth)))})
    for c in CAT:
        a,b=real[c].value_counts(normalize=True),synth[c].value_counts(normalize=True)
        idx=a.index.union(b.index)
        records.append({'field':c,'metric':'total_variation','value':float(abs(a.reindex(idx,fill_value=0)-b.reindex(idx,fill_value=0)).sum()/2)})
    pd.DataFrame(records).to_csv(out/'marginals.csv',index=False)
    # Thresholds come only from training; retain held-out extremes beyond its range.
    tails = []
    quantiles = []
    for c in NUM:
        reference = pd.to_numeric(train_df[c], errors='coerce').dropna()
        low, high = reference.quantile([0.01, 0.99])
        for source, frame in [('real_train', train_df), ('real_test', real), ('synthetic', synth)]:
            values = pd.to_numeric(frame[c], errors='coerce').dropna()
            tails.append({'field': c, 'source': source, 'finite_rows': len(values),
                'lower_threshold_train_p01': low, 'upper_threshold_train_p99': high,
                'below_p01_count': int((values < low).sum()),
                'above_p99_count': int((values > high).sum()),
                'below_p01_fraction': float((values < low).mean()),
                'above_p99_fraction': float((values > high).mean()),
                'outside_training_range_count': int(((values < reference.min()) | (values > reference.max())).sum())})
            for q in [0, .001, .01, .5, .99, .999, 1]:
                quantiles.append({'field': c, 'source': source, 'quantile': q,
                                  'value': values.quantile(q)})
    pd.DataFrame(tails).to_csv(out/'tail_coverage.csv', index=False)
    pd.DataFrame(quantiles).to_csv(out/'quantiles.csv', index=False)
    stats=[]
    for source,df in [('real_test',real),('synthetic',synth)]:
        for lei,g in df.groupby('lei'):
            stats.append({'source':source,'lei':lei,'rows':len(g),'share':len(g)/len(df),
                          'approval_rate':(g.loanapproved=='1').mean()})
    pd.DataFrame(stats).to_csv(out/'institution_rates.csv',index=False)
    # Canonicalize numeric strings for meaningful exact-row comparisons.
    def canon(df):
        df=df.copy()
        for c in NUM:
            n=pd.to_numeric(df[c],errors='coerce');mask=n.notna()
            df.loc[mask,c]=n[mask].map(lambda x:format(x,'.8g'))
        return pd.util.hash_pandas_object(df,index=False)
    hashes=canon(synth)
    corr_a=real[NUM].apply(pd.to_numeric,errors='coerce').corr(method='spearman')
    corr_b=synth[NUM].apply(pd.to_numeric,errors='coerce').corr(method='spearman')
    (corr_a-corr_b).to_csv(out/'spearman_difference.csv')
    m=load_meta(data)
    invalid={c:int((~synth[c].isin(m['categories'][c])).sum()) for c in CAT}
    report={'real_test_rows':len(real),'synthetic_rows':len(synth),
            'real_test_approval_rate':float((real.loanapproved=='1').mean()),
            'synthetic_approval_rate':float((synth.loanapproved=='1').mean()),
            'exact_training_match_fraction':float(hashes.isin(set(canon(train_df))).mean()),
            'synthetic_duplicate_fraction':float(hashes.duplicated().mean()),
            'invalid_categorical_counts':invalid,
            'notes':'Descriptive fidelity checks, not proof of joint-distribution fidelity or privacy. No classifier is trained.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--csv',default=str(ROOT / 'data/raw/hmda_2024_delaware.csv'));a.add_argument('--out',default=str(ROOT / 'data/prepared'));a.add_argument('--top-institutions',type=int,default=10);a.add_argument('--seed',type=int,default=42);a.set_defaults(func=prepare)
    a=sub.add_parser('train');a.add_argument('--data',default=str(ROOT / 'data/prepared'));a.add_argument('--out',default=str(ROOT / 'runs/tabddpm/first'));a.add_argument('--iterations',type=int,default=10000);a.add_argument('--timesteps',type=int,default=1000);a.add_argument('--hidden',type=int,default=256);a.add_argument('--batch-size',type=int,default=512);a.add_argument('--lr',type=float,default=0.001);a.add_argument('--save-every',type=int,default=1000);a.add_argument('--seed',type=int,default=42);a.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');a.set_defaults(func=train)
    a=sub.add_parser('sample');a.add_argument('--data',default=str(ROOT / 'data/prepared'));a.add_argument('--checkpoint',default=str(ROOT / 'runs/tabddpm/first/checkpoint.pt'));a.add_argument('--out',default=str(ROOT / 'outputs/tabddpm/synthetic.csv'));a.add_argument('--rows',type=int,default=10000);a.add_argument('--batch-size',type=int,default=512);a.add_argument('--seed',type=int,default=123);a.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');a.set_defaults(func=sample)
    a=sub.add_parser('evaluate');a.add_argument('--data',default=str(ROOT / 'data/prepared'));a.add_argument('--synthetic',default=str(ROOT / 'outputs/tabddpm/synthetic.csv'));a.add_argument('--out',default=str(ROOT / 'outputs/tabddpm/evaluation'));a.set_defaults(func=evaluate)
    args=p.parse_args()
    for name in ['iterations','timesteps','hidden','batch_size','save_every','rows','top_institutions']:
        if hasattr(args,name) and getattr(args,name)<=0:p.error(f'{name} must be positive')
    if getattr(args,'timesteps',1000)<2:p.error('timesteps must be >=2')
    args.func(args)


if __name__=='__main__':
    main()
