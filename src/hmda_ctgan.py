"""CTGAN baseline sharing HMDA splits, schema, missing tokens and evaluation."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from hmda import NUM, CAT, load_meta, seed_all, device_for


def fingerprint(data):
    return {name: hashlib.sha256((Path(data)/name).read_bytes()).hexdigest()
            for name in ['metadata.json', 'train_real.csv']}


def encode(frame, metadata):
    result = frame[CAT].copy().replace('', '__EMPTY__')
    for c in NUM:
        values = pd.to_numeric(frame[c], errors='coerce')
        values = values.where(np.isfinite(values))
        result[c + '__status'] = np.where(values.notna(), '__VALUE__',
                                          frame[c].replace('', '__EMPTY__'))
        result[c] = values.fillna(metadata['medians'][c]).astype(float)
    return result[NUM + metadata['encoded_categorical']]


def decode(frame, metadata):
    result = frame.copy()
    diagnostics = {}
    for c in metadata['encoded_categorical']:
        result[c] = result[c].astype(str)
        if not result[c].isin(metadata['categories'][c]).all():
            raise ValueError(f'Unknown generated category: {c}')
    for c in NUM:
        values = pd.to_numeric(result[c], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite generated values: {c}')
        lo, hi = metadata['bounds'][c]
        active = result[c + '__status'].eq('__VALUE__').to_numpy()
        diagnostics[c] = {'numeric_rows': int(active.sum()),
            'below_training_min': int(((values < lo) & active).sum()),
            'above_training_max': int(((values > hi) & active).sum())}
        # Match TabDDPM's output policy; report clipping so collapse is visible.
        values = np.clip(values, lo, hi)
        if c in ['loan_amount', 'property_value']:
            values = np.rint((values - 5000) / 10000) * 10000 + 5000
        else:
            values = np.rint(values)
        values = np.clip(values, lo, hi)
        result[c] = [str(int(v)) if float(v).is_integer() else str(v) for v in values]
        status = result.pop(c + '__status')
        result.loc[~active, c] = status[~active]
    return result[NUM + CAT].replace('__EMPTY__', ''), diagnostics


def train(args):
    from ctgan import CTGAN
    seed_all(args.seed)
    device = device_for(args.device)
    m = load_meta(args.data)
    if m['numerical'] != NUM or m['categorical'] != CAT:
        raise ValueError('Prepared schema does not match this implementation')
    frame = pd.read_csv(Path(args.data)/'train_real.csv', dtype=str, keep_default_na=False)
    encoded = encode(frame, m)
    for c in m['encoded_categorical']:
        if not encoded[c].isin(m['categories'][c]).all():
            raise ValueError(f'Training data does not match metadata: {c}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {'epochs': args.epochs, 'batch_size': args.batch_size, 'pac': args.pac,
              'seed': args.seed, 'device': device, 'ctgan': version('ctgan'),
              'torch': torch.__version__, 'training_rows': len(frame),
              'preprocessing': 'CTGAN native mixture normalization; median plus categorical missing status'}
    (out/'config.json').write_text(json.dumps(config, indent=2))
    model = CTGAN(epochs=args.epochs, batch_size=args.batch_size, pac=args.pac,
                  verbose=True, enable_gpu=(device == 'cuda'))
    model.set_random_state(args.seed)
    print(f'CTGAN: {len(frame):,} training rows, {args.epochs} epochs, device={device}', flush=True)
    model.fit(encoded, discrete_columns=m['encoded_categorical'])
    model.loss_values.to_csv(out/'loss.csv', index=False)
    # Trusted local artifact only; portable CPU serialization, device selected at sampling.
    model.set_device(torch.device('cpu'))
    artifact = {'model': model, 'metadata': m, 'fingerprint': fingerprint(args.data),
                'config': config}
    with (out/'checkpoint.tmp').open('wb') as f:
        pickle.dump(artifact, f)
    (out/'checkpoint.tmp').replace(out/'checkpoint.pkl')
    print(f'Saved {out / "checkpoint.pkl"}')


def sample(args):
    seed_all(args.seed)
    out = Path(args.out)
    raw_path = out.with_suffix('.raw.csv')
    report_path = out.with_suffix('.diagnostics.json')
    for p in [out, raw_path, report_path]:
        if p.exists():
            raise FileExistsError(p)
    # Only open checkpoints you trust: pickle can execute code.
    with Path(args.checkpoint).open('rb') as f:
        artifact = pickle.load(f)
    if artifact['fingerprint'] != fingerprint(args.data):
        raise ValueError('Prepared training data differs from checkpoint')
    model = artifact['model']
    model.set_device(torch.device(device_for(args.device)))
    model.set_random_state(args.seed)
    raw = model.sample(args.rows)
    result, diagnostics = decode(raw, artifact['metadata'])
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    result.to_csv(out, index=False)
    report = {'rows': len(result), 'seed': args.seed, 'range_adjustments': diagnostics,
              'raw_duplicate_fraction': float(raw.duplicated().mean()),
              'decoded_duplicate_fraction': float(result.duplicated().mean()),
              'note': 'Same clipping and rounding policy as TabDDPM; raw output retained for diagnosis.'}
    report_path.write_text(json.dumps(report, indent=2))
    print(f'Wrote {out}; raw values and decoding diagnostics saved alongside it.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', default=str(ROOT / 'data/prepared'))
    t.add_argument('--out', default=str(ROOT / 'runs/ctgan/first'))
    t.add_argument('--epochs', type=int, default=300)
    t.add_argument('--batch-size', type=int, default=500)
    t.add_argument('--pac', type=int, default=10)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    t.set_defaults(func=train)
    s = sub.add_parser('sample')
    s.add_argument('--data', default=str(ROOT / 'data/prepared'))
    s.add_argument('--checkpoint', default=str(ROOT / 'runs/ctgan/first/checkpoint.pkl'))
    s.add_argument('--out', default=str(ROOT / 'outputs/ctgan/synthetic.csv'))
    s.add_argument('--rows', type=int, default=10000)
    s.add_argument('--seed', type=int, default=123)
    s.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    s.set_defaults(func=sample)
    args = parser.parse_args()
    for name in ['epochs', 'batch_size', 'pac', 'rows']:
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.command == 'train' and (args.batch_size % 2 or args.batch_size % args.pac):
        parser.error('batch size must be even and divisible by pac')
    args.func(args)


if __name__ == '__main__':
    main()
