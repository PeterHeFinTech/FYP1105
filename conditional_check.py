import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

out = Path('outputs/conditional_check')
out.mkdir(parents=True, exist_ok=True)
r = pd.read_csv('data/prepared/test_real.csv', keep_default_na=False)
s = pd.read_csv('outputs/ctgan/synthetic.csv', keep_default_na=False)
rows=[]
for label in [0,1]:
    a=r[r.loanapproved.astype(str)==str(label)]
    b=s[s.loanapproved.astype(str)==str(label)]
    print('\n', 'DENIED' if label==0 else 'APPROVED',len(a),len(b))
    for col in ['income','loan_amount','property_value']:
        x=pd.to_numeric(a[col],errors='coerce'); y=pd.to_numeric(b[col],errors='coerce')
        row=dict(outcome=label,feature=col,real_missing=x.isna().mean(),synthetic_missing=y.isna().mean(),real_median=x.median(),synthetic_median=y.median(),real_p90=x.quantile(.9),synthetic_p90=y.quantile(.9),KS=ks_2samp(x.dropna(),y.dropna()).statistic)
        rows.append(row); print(row)
    for col in ['debt_to_income_ratio','lei']:
        p=a[col].value_counts(normalize=True); q=b[col].value_counts(normalize=True)
        d=pd.concat([p.rename('real'),q.rename('synthetic')],axis=1).fillna(0)
        d['gap']=d.synthetic-d.real
        d.sort_values('gap',key=abs,ascending=False).to_csv(out/f'{label}_{col}.csv')
        print(col,'total variation',abs(d.gap).sum()/2)
        print(d.loc[d.gap.abs().nlargest(5).index].to_string())
    print('income-loan Spearman',a[['income','loan_amount']].apply(pd.to_numeric,errors='coerce').corr(method='spearman').iloc[0,1], b[['income','loan_amount']].apply(pd.to_numeric,errors='coerce').corr(method='spearman').iloc[0,1])
pd.DataFrame(rows).to_csv(out/'numeric.csv',index=False)
