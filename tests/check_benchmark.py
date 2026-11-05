"""Check evaluator identity, marginal-vs-joint sensitivity and detector signal."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from evaluation.benchmark import normalize,describe,detect,NUM,CAT

rng=np.random.default_rng(7)
n=400
frame=pd.DataFrame({c:rng.choice(['1','2'],n) for c in CAT})
frame['loanapproved']=rng.choice(['0','1'],n)
for c in NUM:frame[c]=np.arange(n).astype(str)
real=normalize(frame)
identical=describe(real,real,10)
assert identical['marginals'].KS.dropna().eq(0).all()
assert identical['marginals'].TV.dropna().eq(0).all()
assert identical['dependencies'].value.dropna().eq(0).all()
shuffled=real.copy();shuffled['loan_amount']=rng.permutation(shuffled.loan_amount)
res=describe(real,shuffled,10)
assert res['marginals'].KS.dropna().eq(0).all()
assert res['dependencies'].query("field1 == 'income' and field2 == 'loan_amount'").value.iloc[0]>.7
shifted=real.copy();shifted['income']='100000000'
for row in detect(real,shifted,42,30):
 assert row['status']=='ok' and row['auc']>.95,row
bad=frame.drop(columns=['lei'])
try:
 normalize(bad)
 raise AssertionError('Missing schema should fail')
except ValueError:pass
empty=real.copy();empty['income']='NA'
res=describe(real,empty,10)
assert res['marginals'].query("field == 'income'").KS.isna().all()
print('PASS: identical distributions, shuffled dependence, strong detector signal, missing schema and all-missing numeric field.')
