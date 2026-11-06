"""Separate outcome generators; preserve training prevalence and existing decoding."""
from pathlib import Path
import json
import pickle
import subprocess
import sys
import pandas as pd
import torch
from ctgan import CTGAN
from hmda import load_meta, seed_all
from hmda_ctgan import encode, decode, fingerprint

root=Path(__file__).resolve().parent
data=root/'data/prepared'
out=root/'outputs/ctgan_by_outcome'
run=root/'runs/ctgan_by_outcome'
out.mkdir(exist_ok=False,parents=True)
run.mkdir(exist_ok=False,parents=True)
m=load_meta(data)
frame=pd.read_csv(data/'train_real.csv',dtype=str,keep_default_na=False)
assert set(frame.loanapproved)=={'0','1'}
counts=frame.loanapproved.value_counts()
n0=round(10000*counts['0']/len(frame))
parts=[]
diagnostics={}
for label,n in [('0',n0),('1',10000-n0)]:
    seed_all(42)
    encoded=encode(frame.loc[frame.loanapproved==label],m).drop(columns='loanapproved')
    categorical=[c for c in m['encoded_categorical'] if c!='loanapproved']
    print(f'Training outcome={label}, rows={len(encoded)}, epochs=300',flush=True)
    model=CTGAN(epochs=300,batch_size=500,pac=10,verbose=True,enable_gpu=True)
    model.set_random_state(42)
    model.fit(encoded,discrete_columns=categorical)
    model.loss_values.to_csv(run/f'loss_{label}.csv',index=False)
    model.set_random_state(123)
    raw=model.sample(n)
    raw['loanapproved']=label
    raw.to_csv(out/f'raw_{label}.csv',index=False)
    decoded,diagnostics[label]=decode(raw,m)
    parts.append(decoded)
    model.set_device(torch.device('cpu'))
    with (run/f'model_{label}.pkl').open('wb') as f:
        pickle.dump({'model':model,'metadata':m,'fingerprint':fingerprint(data)},f)
combined=pd.concat(parts,ignore_index=True).sample(frac=1,random_state=123)
combined.to_csv(out/'synthetic.csv',index=False)
(out/'protocol.json').write_text(json.dumps({'epochs':300,'training_seed':42,'sampling_seed':123,'training_counts':counts.to_dict(),'synthetic_denied':n0,'diagnostics':diagnostics,'fingerprint':fingerprint(data),'note':'Exploratory experiment after inspecting test results; final confirmation requires a fresh holdout and multiple seeds.'},indent=2))
subprocess.run([sys.executable,'-m','evaluation.tstr',str(root/'outputs/ctgan/synthetic.csv'),str(out/'synthetic.csv'),'--rows','10000','--out',str(out/'tstr')],check=True,cwd=root)
print('COMPLETE',flush=True)
