"""Aligned two-party conditional GAN simulation. Intermediate activations are shared.
Not a secure deployment or a CTGAN implementation. Raw tables remain in client objects.
"""
import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

SHARED=['derived_race','derived_ethnicity','applicant_sex','applicant_age','income','loan_amount']
A=['loan_type','loan_purpose','interest_rate','total_points_and_fees','total_loan_costs','discount_points','origination_charges']
B=['census_tract','property_value']
NUM=['income','loan_amount','interest_rate','total_points_and_fees','total_loan_costs','discount_points','origination_charges','property_value']
LABEL='action_taken'

class Codec:
    def __init__(self,frame,cols):
        self.cols=cols;self.spec=[];self.width=0
        for c in cols:
            x=pd.to_numeric(frame[c],errors='coerce') if c in NUM else None
            if x is not None:
                v=np.sign(x)*np.log1p(abs(x));mu=v.mean();sd=max(float(v.std()),.01)
                states=sorted(set(np.where(x.notna(),'VALUE',frame[c])))
                self.spec.append((c,self.width,True,float(mu),sd,states));self.width+=1+len(states)
            else:
                states=sorted(frame[c].unique());self.spec.append((c,self.width,False,0,1,states));self.width+=len(states)
    def encode(self,frame):
        out=np.zeros((len(frame),self.width),dtype='float32')
        for c,k,num,mu,sd,states in self.spec:
            if num:
                x=pd.to_numeric(frame[c],errors='coerce');v=np.sign(x)*np.log1p(abs(x))
                out[:,k]=((v-mu)/sd).fillna(0);tokens=np.where(x.notna(),'VALUE',frame[c]);k+=1
            else: tokens=frame[c]
            ix=pd.Categorical(tokens,categories=states).codes.astype('int64')
            assert (ix>=0).all()
            out[np.arange(len(frame)),k+ix]=1
        return torch.tensor(out)
    def activate(self,x):
        pieces=[]
        for c,k,num,mu,sd,states in self.spec:
            if num: pieces.append(x[:,k:k+1]);k+=1
            pieces.append(F.gumbel_softmax(x[:,k:k+len(states)],tau=.5,hard=True))
        return torch.cat(pieces,1)
    def decode(self,x):
        out={}
        for c,k,num,mu,sd,states in self.spec:
            vals=x[:,k]*sd+mu if num else None
            if num:k+=1
            tokens=np.array(states)[x[:,k:k+len(states)].argmax(1)].astype(object)
            if num:
                values=np.sign(vals)*np.expm1(np.minimum(abs(vals),30))
                if c in ['loan_amount','property_value']:values=np.round((values-5000)/10000)*10000+5000
                elif c=='income': values=np.round(values)
                tokens=np.where(tokens=='VALUE',values.astype(str),tokens)
            out[c]=tokens
        return pd.DataFrame(out)

def net(a,b):return nn.Sequential(nn.Linear(a,128),nn.LeakyReLU(.2),nn.Linear(128,b))

def main():
    global SHARED,A,B,NUM
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',default='outputs/two_party_trial');p.add_argument('--full-schema',action='store_true');args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    raw=pd.read_csv('data/raw/hmda_2024_delaware.csv',dtype=str,keep_default_na=False)
    if args.full_schema:
        SHARED=SHARED+['activity_year']
        property_fields={'state_code','county_code','census_tract','derived_msa-md','property_value','derived_dwelling_category','construction_method','occupancy_type','manufactured_home_secured_property_type','manufactured_home_land_property_interest','total_units','multifamily_affordable_units','ffiec_msa_md_median_family_income'}
        B=[c for c in raw if c in property_fields or c.startswith('tract_')]
        A=[c for c in raw if c not in set(SHARED+B+[LABEL])]
        NUM=NUM+['loan_to_value_ratio','rate_spread','lender_credits','loan_term','prepayment_penalty_term','intro_rate_period','multifamily_affordable_units','ffiec_msa_md_median_family_income']+[c for c in B if c.startswith('tract_')]
        NUM=[c for c in NUM if pd.to_numeric(raw[c],errors='coerce').notna().sum()>1]
        assert set(SHARED+A+B+[LABEL])==set(raw.columns)
    cols=SHARED+A+B
    counts=raw[LABEL].value_counts();eligible=counts[counts>=5].index
    raw=raw[raw[LABEL].isin(eligible)].copy()
    train,test=train_test_split(raw,test_size=.2,random_state=args.seed,stratify=raw[LABEL])
    # Row indices are simulation alignment keys, not evidence of repeated-person linkage.
    train[SHARED+A+[LABEL]].to_csv(out/'party_a_train.csv',index_label='alignment_id')
    train[SHARED+B+[LABEL]].to_csv(out/'party_b_train.csv',index_label='alignment_id')
    test[cols+[LABEL]].to_csv(out/'heldout.csv',index=False)
    codecs=[Codec(train,SHARED),Codec(train,A),Codec(train,B)]
    real=[c.encode(train).to(device) for c in codecs]
    labels=sorted(train[LABEL].unique());y=torch.tensor(pd.Categorical(train[LABEL],categories=labels).codes.astype('int64'),device=device)
    widths=[c.width for c in codecs];nc=len(labels)
    G=nn.Sequential(nn.Linear(64+nc,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,sum(widths))).to(device)
    EA=net(widths[0]+widths[1],64).to(device);EB=net(widths[0]+widths[2],64).to(device)
    D=net(128+nc,1).to(device)
    optg=torch.optim.Adam(G.parameters(),lr=2e-4,betas=(.5,.9))
    optd=torch.optim.Adam(list(EA.parameters())+list(EB.parameters())+list(D.parameters()),lr=2e-4,betas=(.5,.9))
    def gen(yb):
        x=G(torch.cat([torch.randn(len(yb),64,device=device),F.one_hot(yb,nc).float()],1))
        return [c.activate(v) for c,v in zip(codecs,x.split(widths,1))]
    def score(parts,yb):
        # Client encoders send activations. Central discriminator never receives raw columns.
        return D(torch.cat([EA(torch.cat(parts[:2],1)),EB(torch.cat([parts[0],parts[2]],1)),F.one_hot(yb,nc).float()],1))
    history=[]
    for epoch in range(args.epochs):
        for ix in torch.randperm(len(train),device=device).split(256):
            yy=y[ix];rr=[r[ix] for r in real];ff=gen(yy)
            optd.zero_grad();dl=F.softplus(-score(rr,yy)).mean()+F.softplus(score([v.detach() for v in ff],yy)).mean();dl.backward();optd.step()
            optg.zero_grad();ff=gen(yy);gl=F.softplus(-score(ff,yy)).mean();gl.backward();optg.step()
        history.append({'epoch':epoch+1,'generator_loss':gl.item(),'discriminator_loss':dl.item()})
        if (epoch+1)%10==0:print(history[-1],flush=True)
    pd.DataFrame(history).to_csv(out/'loss.csv',index=False)
    # Match training class proportions, while explicitly conditioning every generated row.
    probs=np.array([(y==i).sum().item() for i in range(nc)],dtype=float);probs/=probs.sum()
    sample_y=np.random.choice(nc,10000,p=probs)
    parts=[]
    with torch.no_grad():
        for ys in np.array_split(sample_y,40):
            generated=gen(torch.tensor(ys,device=device));df=pd.concat([c.decode(v.cpu().numpy()) for c,v in zip(codecs,generated)],axis=1);df[LABEL]=np.array(labels)[ys];parts.append(df)
    syn=pd.concat(parts,ignore_index=True);syn.to_csv(out/'synthetic.csv',index=False)
    metrics=[]
    for label in labels:
        r=test[test[LABEL]==label];s=syn[syn[LABEL]==label]
        if len(s)==0:continue
        for c in cols:
            if c in NUM:
                a=pd.to_numeric(r[c],errors='coerce');b=pd.to_numeric(s[c],errors='coerce')
                metrics.append(dict(label=label,field=c,metric='KS',value=float(ks_2samp(a.dropna(),b.dropna()).statistic) if a.notna().any() and b.notna().any() else None))
                metrics.append(dict(label=label,field=c,metric='missing_rate_gap',value=float(b.isna().mean()-a.isna().mean())))
            else:
                pq=pd.concat([r[c].value_counts(normalize=True),s[c].value_counts(normalize=True)],axis=1).fillna(0)
                metrics.append(dict(label=label,field=c,metric='TV',value=float(abs(pq.iloc[:,0]-pq.iloc[:,1]).sum()/2)))
    pd.DataFrame(metrics).to_csv(out/'conditional_metrics.csv',index=False)
    matches=pd.MultiIndex.from_frame(syn[cols+[LABEL]]).isin(pd.MultiIndex.from_frame(train[cols+[LABEL]]))
    report={'architecture':'conditional split GAN with aligned client encoders and joint discriminator','shared':SHARED,'party_a_private':A,'party_b_private':B,'condition':LABEL,'labels':labels,'excluded_action_counts':counts[~counts.index.isin(eligible)].to_dict(),'train_rows':len(train),'test_rows':len(test),'synthetic_rows':len(syn),'duplicate_fraction':float(syn.duplicated().mean()),'exact_training_match_fraction':float(matches.mean()),'seed':args.seed,'epochs':args.epochs,'device':device,'limitations':['Single-process split-learning simulation; intermediate activations and gradients may leak information.','Local preprocessing uses training data only; schemas/categories exposed by design.','No differential privacy, secure aggregation, or deployment isolation.','Different schema and split from earlier binary experiments; metrics are not directly comparable.','Interest/fee availability may reveal outcome; do not treat downstream prediction as prospective underwriting.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    torch.save({'G':G.state_dict(),'EA':EA.state_dict(),'EB':EB.state_dict(),'D':D.state_dict(),'codecs':codecs,'labels':labels,'report':report},out/'checkpoint.pt')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
