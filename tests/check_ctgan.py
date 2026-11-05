"""Temporary one-epoch integration check on fabricated data, not an HMDA run."""
import argparse
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace as A
import numpy as np
import pandas as pd
from hmda import NUM, CAT, prepare, evaluate, load_meta
from hmda_ctgan import train, sample, encode, decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        rng = np.random.default_rng(42)
        n = 160
        frame = pd.DataFrame({c: rng.choice(['1', '2'], n) for c in CAT if c != 'loanapproved'})
        frame['income'] = rng.integers(-20, 500, n).astype(str)
        frame.loc[0, 'income'] = '1000000'
        frame.loc[1, 'income'] = 'NA'
        frame.loc[2, 'income'] = ''
        frame['loan_amount'] = (rng.integers(1, 100, n)*10000+5000).astype(str)
        frame['property_value'] = (rng.integers(1, 200, n)*10000+5000).astype(str)
        frame.loc[3, 'property_value'] = 'Exempt'
        frame['loan_term'] = rng.choice(['180', '360'], n)
        frame['action_taken'] = rng.choice(['1', '2', '3'], n)
        frame.to_csv(root/'input.csv', index=False)
        data = root/'prepared'
        prepare(A(csv=str(root/'input.csv'), out=str(data), top_institutions=2, seed=42))
        real = pd.read_csv(data/'train_real.csv', dtype=str, keep_default_na=False)
        restored, _ = decode(encode(real, load_meta(data)), load_meta(data))
        pd.testing.assert_frame_equal(restored.reset_index(drop=True),
            real.replace('__EMPTY__', '').reset_index(drop=True))
        run = root/'run'
        train(A(seed=42, device=args.device, data=str(data), out=str(run), epochs=1, batch_size=100, pac=10))
        options = dict(seed=123, device=args.device, data=str(data), checkpoint=str(run/'checkpoint.pkl'), rows=32)
        sample(A(**options, out=str(root/'synthetic.csv')))
        sample(A(**options, out=str(root/'repeat.csv')))
        assert (root/'synthetic.csv').read_bytes() == (root/'repeat.csv').read_bytes()
        evaluate(A(data=str(data), synthetic=str(root/'synthetic.csv'), out=str(root/'evaluation')))
        report = json.loads((root/'evaluation/summary.json').read_text())
        assert report['synthetic_rows'] == 32
        assert not any(report['invalid_categorical_counts'].values())
        try:
            sample(A(**options, out=str(root/'synthetic.csv')))
            raise AssertionError('Existing outputs should be protected')
        except FileExistsError:
            pass
        print('PASS: encode/decode, one-epoch fixture fit, checkpoint reload, seeded sampling, shared evaluation, overwrite protection.')


if __name__ == '__main__':
    main()
