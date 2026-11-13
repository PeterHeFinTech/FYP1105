from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,recall_score,roc_auc_score
from two_party_trial import SHARED,A,B,NUM,LABEL

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=Path('outputs/two_party_trial_v2'))
parser.add_argument('--screen-outcomes',action='store_true')
args=parser.parse_args()
root=args.run
if args.screen_outcomes:
    report=json.loads((root/'report.json').read_text())
    SHARED,A,B=(report[key] for key in ['shared','party_a_private','party_b_private'])
a=pd.read_csv(root/'party_a_train.csv',dtype=str,keep_default_na=False).set_index('alignment_id')
b=pd.read_csv(root/'party_b_train.csv',dtype=str,keep_default_na=False).set_index('alignment_id')
assert a.index.equals(b.index)
assert a[SHARED+[LABEL]].equals(b[SHARED+[LABEL]])
real=a.join(b[B],validate='one_to_one')
test=pd.read_csv(root/'heldout.csv',dtype=str,keep_default_na=False)
synthetic=pd.read_csv(root/'synthetic.csv',dtype=str,keep_default_na=False)
cols=SHARED+A+B
if args.screen_outcomes:
    # Conservative allowlist of application and property fields. Keep the full
    # schema in generation, and screen post-decision fields only in TSTR.
    application_fields={'lei','loan_type','loan_purpose','lien_status','loan_amount','income','debt_to_income_ratio','loan_term','preapproval','reverse_mortgage','open-end_line_of_credit','business_or_commercial_purpose'}
    cols=[c for c in cols if c in set(SHARED+B)|application_fields or c.startswith(('applicant_','co-applicant_')) and 'credit_score_type' not in c]
    NUM=[c for c in cols if c in set(NUM)|{'loan_term','multifamily_affordable_units','ffiec_msa_md_median_family_income'} or c.startswith('tract_')]
cats=[c for c in cols if c not in NUM]+[c+'__status' for c in NUM]
def features(f):
    x=f[cols].copy()
    for c in NUM:
        v=pd.to_numeric(x[c],errors='coerce')
        x[c+'__status']=x[c].where(v.isna(),'VALUE')
        x[c]=v
    return x
def labels(f):
    assert f[LABEL].isin(['1','2','3']).all()
    return (f[LABEL]!='3').astype(int)
xt=features(test);yt=labels(test)
rows=[]
for source,f in [('Real baseline',real),('Two-party split',synthetic)]:
    f=f.sample(n=10000,random_state=42)
    for name,est in [('Logistic',LogisticRegression(max_iter=3000,random_state=42)),('SVM',SVC(cache_size=1024)),('RF',RandomForestClassifier(n_estimators=200,min_samples_leaf=5,n_jobs=4,random_state=42))]:
        prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),NUM),('cat',OneHotEncoder(handle_unknown='ignore'),cats)])
        model=make_pipeline(prep,est)
        print(source,name,flush=True)
        model.fit(features(f),labels(f));pred=model.predict(xt)
        score=model.decision_function(xt) if hasattr(est,'decision_function') else model.predict_proba(xt)[:,1]
        rows.append(dict(Source=source,Model=name,Acc=accuracy_score(yt,pred),Bal_Acc=balanced_accuracy_score(yt,pred),Denied_Rec=recall_score(yt,pred,pos_label=0),AUC=roc_auc_score(yt,score)))
        pd.DataFrame(rows).to_csv(root/'tstr_binary.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False),flush=True)
(root/'tstr_protocol.json').write_text(json.dumps({'train_rows_per_source':10000,'test_rows':len(test),'seed':42,'approved_actions':['1','2'],'denied_actions':['3'],'features':cols,'note':'Central evaluator joins aligned real training partitions for baseline only. Includes potentially post-decision fields; measures synthetic utility, not prospective underwriting.'},indent=2))
