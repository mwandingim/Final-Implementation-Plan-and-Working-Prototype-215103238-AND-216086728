from pathlib import Path
import os, json
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE=str(Path(__file__).resolve().parents[1])
DATA=f'{BASE}/02_data/raw/t21_vendor_vpn_iam.csv'
OUT=f'{BASE}/08_outputs/predictive_adversarial'
MODEL_DIR=f'{BASE}/04_models/predictive'
os.makedirs(OUT, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
SEED=210821

df=pd.read_csv(DATA)
df['timestamp']=pd.to_datetime(df['timestamp'], utc=True)
df=df.sort_values('timestamp').reset_index(drop=True)
df['target']=(df['event_label']!='NORMAL').astype(int)

# Build only prior-history features: no current/future labels are used to construct them.
rows=[]
for account, g in df.groupby('account_id', sort=False):
    g=g.sort_values('timestamp')
    prior=deque()
    for _, r in g.iterrows():
        ts=r['timestamp']
        while prior and (ts-prior[0][0]).total_seconds()>86400:
            prior.popleft()
        n=len(prior)
        nonnormal=sum(x[1] for x in prior)
        failures=sum(x[2] for x in prior)
        offhours=sum(x[3] for x in prior)
        newdev=sum(x[4] for x in prior)
        durations=[x[5] for x in prior]
        prev_ts=prior[-1][0] if prior else pd.NaT
        rows.append({
            'event_id':r['event_id'], 'timestamp':ts, 'vendor_id':r['vendor_id'], 'account_id':r['account_id'],
            'role':r['role'], 'access_zone':r['access_zone'], 'auth_result':r['auth_result'],
            'MFA_status':r['MFA_status'], 'off_hours':r['off_hours'], 'new_device':r['new_device'],
            'prior_sessions_24h':n, 'prior_non_normal_24h':nonnormal,
            'prior_failure_24h':failures, 'prior_offhours_24h':offhours,
            'prior_new_device_24h':newdev,
            'prior_non_normal_rate_24h':(nonnormal/n if n else 0.0),
            'prior_mean_duration_24h':(np.mean(durations) if durations else 0.0),
            'hours_since_previous':((ts-prev_ts).total_seconds()/3600 if pd.notna(prev_ts) else 999.0),
            'target':int(r['target'])
        })
        prior.append((ts,int(r['target']),int(r['auth_result']=='FAILURE'),int(r['off_hours']=='YES'),int(r['new_device']=='YES'),float(r['session_duration_sec'])))

m=pd.DataFrame(rows)
cat=['vendor_id','account_id','role','access_zone','auth_result','MFA_status','off_hours','new_device']
num=['prior_sessions_24h','prior_non_normal_24h','prior_failure_24h','prior_offhours_24h','prior_new_device_24h','prior_non_normal_rate_24h','prior_mean_duration_24h','hours_since_previous']
features=cat+num

# Chronological 70/30 split.
split=int(len(m)*0.70)
train=m.iloc[:split].copy(); test=m.iloc[split:].copy()
X_train,y_train=train[features],train['target']; X_test,y_test=test[features],test['target']
pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat)], remainder='passthrough')
model=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=-1)
pipe=Pipeline([('preprocess',pre),('model',model)])
pipe.fit(X_train,y_train)
prob=pipe.predict_proba(X_test)[:,1]; pred=(prob>=0.50).astype(int)
cm=confusion_matrix(y_test,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
metrics={
 'train_rows':len(train),'test_rows':len(test),'positive_rate_train':float(y_train.mean()),'positive_rate_test':float(y_test.mean()),
 'accuracy':float(accuracy_score(y_test,pred)),'precision':float(precision_score(y_test,pred,zero_division=0)),
 'recall':float(recall_score(y_test,pred,zero_division=0)),'f1_score':float(f1_score(y_test,pred,zero_division=0)),
 'false_negative_rate':float(fn/(fn+tp)) if (fn+tp) else 0.0,'threshold':0.50,'true_negative':int(tn),'false_positive':int(fp),'false_negative':int(fn),'true_positive':int(tp),
 'random_seed':SEED,'forecast_definition':'risk of the current vendor session using only information available before the session plus session-start context'
}
pd.DataFrame([metrics]).to_csv(f'{OUT}/predictive_model_metrics.csv',index=False)

rank=test[['event_id','timestamp','vendor_id','account_id','role','access_zone','target']].copy(); rank['risk_probability']=prob; rank['predicted_label']=np.where(pred==1,'HIGH_RISK','LOW_RISK'); rank=rank.sort_values('risk_probability',ascending=False)
rank.to_csv(f'{OUT}/predictive_test_scored.csv',index=False)
rank.head(15).to_csv(f'{OUT}/top_15_predictive_risk_sessions.csv',index=False)

# Feature importance.
names=pipe.named_steps['preprocess'].get_feature_names_out(); imp=pipe.named_steps['model'].feature_importances_
fi=pd.DataFrame({'feature':names,'importance':imp}).sort_values('importance',ascending=False).head(15)
fi.to_csv(f'{OUT}/predictive_feature_importance.csv',index=False)
fig,ax=plt.subplots(figsize=(8,6)); q=fi.sort_values('importance'); ax.barh(q['feature'],q['importance']); ax.set_xlabel('Importance'); ax.set_title('Predictive Risk Model — Top Features'); fig.tight_layout(); fig.savefig(f'{OUT}/predictive_feature_importance.png',dpi=180); plt.close(fig)

# Adversarial stress tests on the first 3 highest-risk held-out sessions.
base_rows=test.loc[rank.head(3).index].copy()
# Re-align to rank rows by event id.
base_rows=test.set_index('event_id').loc[rank.head(3)['event_id']].copy().reset_index()

def score(frame): return pipe.predict_proba(frame[features])[:,1]

cases=[]
for _, r in base_rows.iterrows():
    base=float(score(pd.DataFrame([r]))[0])
    # A: mask temporal/device novelty signals.
    a=r.copy(); a['off_hours']='NO'; a['new_device']='NO'; a['MFA_status']='YES'
    pa=float(score(pd.DataFrame([a]))[0])
    cases.append({'event_id':r['event_id'],'scenario':'A_feature_masking','description':'Set off-hours, new-device and MFA indicators to normal-looking values while retaining other fields.','original_probability':base,'manipulated_probability':pa,'delta':pa-base})
    # B: replace contextual access indicators with a common low-risk context.
    b=r.copy(); b['access_zone']='YARD-A'; b['role']='MaintenanceVendor'; b['auth_result']='SUCCESS'
    pb=float(score(pd.DataFrame([b]))[0])
    cases.append({'event_id':r['event_id'],'scenario':'B_context_substitution','description':'Replace access context and authentication state with a common benign-looking context.','original_probability':base,'manipulated_probability':pb,'delta':pb-base})
    # C: suppress prior-history warning signals.
    c=r.copy();
    for col in ['prior_sessions_24h','prior_non_normal_24h','prior_failure_24h','prior_offhours_24h','prior_new_device_24h','prior_non_normal_rate_24h','prior_mean_duration_24h']:
        c[col]=0
    c['hours_since_previous']=999.0
    pc=float(score(pd.DataFrame([c]))[0])
    cases.append({'event_id':r['event_id'],'scenario':'C_history_dilution','description':'Suppress prior 24-hour behavioural history to test dependence on historical context.','original_probability':base,'manipulated_probability':pc,'delta':pc-base})

adv=pd.DataFrame(cases); adv['risk_class_original']=np.where(adv.original_probability>=0.50,'HIGH_RISK','LOW_RISK'); adv['risk_class_manipulated']=np.where(adv.manipulated_probability>=0.50,'HIGH_RISK','LOW_RISK')
adv.to_csv(f'{OUT}/adversarial_stress_test_results.csv',index=False)

# Summary figure.
summary=adv.groupby('scenario')[['original_probability','manipulated_probability']].mean().reset_index()
fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(len(summary)); w=.35; ax.bar(x-w/2,summary.original_probability,w,label='Original'); ax.bar(x+w/2,summary.manipulated_probability,w,label='Manipulated'); ax.set_xticks(x,summary.scenario); ax.set_ylabel('Mean risk probability'); ax.set_title('Adversarial Stress-Test Risk Changes'); ax.legend(); fig.tight_layout(); fig.savefig(f'{OUT}/adversarial_risk_comparison.png',dpi=180); plt.close(fig)

# QA and metadata.
qa=pd.DataFrame([
 {'check':'Chronological split','status':'PASS','detail':f'{len(train)} train / {len(test)} test rows; test follows training period'},
 {'check':'Prior-history leakage control','status':'PASS','detail':'Rolling features use only records earlier than the scored session'},
 {'check':'Adversarial cases','status':'PASS','detail':'3 synthetic feature-manipulation cases evaluated'},
 {'check':'Risk threshold','status':'PASS','detail':'0.50 probability threshold used consistently'},
 {'check':'Synthetic-only testing','status':'PASS','detail':'No live system or operational endpoint was tested'}
])
qa.to_csv(f'{OUT}/predictive_adversarial_qa.csv',index=False)
metadata={'algorithm':'RandomForestClassifier','n_estimators':300,'max_depth':10,'min_samples_leaf':2,'class_weight':'balanced','random_seed':SEED,'split':'chronological 70/30','threshold':0.50,'forecast':'current-session risk from prior 24h account history and session-start context','adversarial_cases':['feature_masking','context_substitution','history_dilution']}
with open(f'{MODEL_DIR}/predictive_model_metadata.json','w') as f: json.dump(metadata,f,indent=2)
import joblib; joblib.dump(pipe,f'{MODEL_DIR}/predictive_risk_model.joblib')
print(pd.DataFrame([metrics]).T)
print('\nTop 15:'); print(rank.head(15)[['event_id','vendor_id','account_id','risk_probability','target']].to_string(index=False))
print('\nAdversarial:'); print(adv.to_string(index=False))
