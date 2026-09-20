from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

BASE=str(Path(__file__).resolve().parents[1])
DATA=f'{BASE}/02_data/raw/t21_vendor_vpn_iam.csv'
OUT=f'{BASE}/08_outputs/supervised_model'
MODEL_DIR=f'{BASE}/04_models/supervised'
os.makedirs(OUT, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

# Load vendor VPN/IAM sessions. The target is the supplied binary event label.
df=pd.read_csv(DATA)
df['timestamp']=pd.to_datetime(df['timestamp'], utc=True)
df['hour']=df['timestamp'].dt.hour
df['day_of_week']=df['timestamp'].dt.dayofweek
df['session_duration_min']=df['session_duration_sec']/60.0

df['target']=(df['event_label']!='NORMAL').astype(int)

# Exclude identifiers and fields that directly reveal the label/incident outcome.
features=[
    'vendor_id','account_id','auth_result','MFA_status','session_duration_sec',
    'role','access_zone','off_hours','new_device','hour','day_of_week','session_duration_min'
]
X=df[features].copy(); y=df['target']

categorical=['vendor_id','account_id','auth_result','MFA_status','role','access_zone','off_hours','new_device']
numeric=['session_duration_sec','hour','day_of_week','session_duration_min']
pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),categorical)], remainder='passthrough')
model=RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=2,
    class_weight='balanced', random_state=210821, n_jobs=-1
)
pipe=Pipeline([('preprocess',pre),('model',model)])

X_train,X_test,y_train,y_test,idx_train,idx_test=train_test_split(
    X,y,df.index,test_size=0.30,stratify=y,random_state=210821
)
pipe.fit(X_train,y_train)
proba=pipe.predict_proba(X_test)[:,1]
pred=(proba>=0.50).astype(int)
cm=confusion_matrix(y_test,pred,labels=[0,1])
tn,fp,fn,tp=cm.ravel()
metrics={
    'train_rows':int(len(X_train)), 'test_rows':int(len(X_test)),
    'positive_rate_train':float(y_train.mean()), 'positive_rate_test':float(y_test.mean()),
    'accuracy':float(accuracy_score(y_test,pred)),
    'precision':float(precision_score(y_test,pred,zero_division=0)),
    'recall':float(recall_score(y_test,pred,zero_division=0)),
    'f1_score':float(f1_score(y_test,pred,zero_division=0)),
    'false_negative_rate':float(fn/(fn+tp)) if (fn+tp) else 0.0,
    'threshold':0.50, 'true_negative':int(tn),'false_positive':int(fp),
    'false_negative':int(fn),'true_positive':int(tp),
    'random_seed':210821
}
pd.DataFrame([metrics]).to_csv(f'{OUT}/supervised_model_metrics.csv',index=False)
with open(f'{OUT}/classification_report.txt','w') as f: f.write(classification_report(y_test,pred,target_names=['NORMAL','NON_NORMAL'],zero_division=0))

# Confusion matrix figure.
fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm)
ax.set_xticks([0,1],['Predicted NORMAL','Predicted NON-NORMAL']); ax.set_yticks([0,1],['Actual NORMAL','Actual NON-NORMAL'])
for i in range(2):
    for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
ax.set_title('Supervised Model Confusion Matrix'); fig.tight_layout(); fig.savefig(f'{OUT}/confusion_matrix.png',dpi=180); plt.close(fig)

# Feature importance after preprocessing.
feature_names=pipe.named_steps['preprocess'].get_feature_names_out()
imp=pipe.named_steps['model'].feature_importances_
fi=pd.DataFrame({'feature':feature_names,'importance':imp}).sort_values('importance',ascending=False).head(15)
fi.to_csv(f'{OUT}/top_feature_importance.csv',index=False)
fig,ax=plt.subplots(figsize=(8,6)); fi2=fi.sort_values('importance'); ax.barh(fi2['feature'],fi2['importance']); ax.set_xlabel('Importance'); ax.set_title('Top Supervised-Model Features'); fig.tight_layout(); fig.savefig(f'{OUT}/feature_importance.png',dpi=180); plt.close(fig)

# Held-out investigation ranking: the 30% test partition is retained as the unseen scoring set.
invest=df.loc[idx_test,['event_id','timestamp','vendor_id','account_id','device_id','auth_result','MFA_status','role','access_zone','off_hours','new_device','event_label','incident_id']].copy()
invest['risk_probability']=proba
invest['predicted_label']=np.where(pred==1,'NON_NORMAL','NORMAL')
invest['actual_binary']=y_test.values
invest=invent_sorted=invest.sort_values('risk_probability',ascending=False)
invent_sorted.head(15).to_csv(f'{OUT}/top_15_high_risk_sessions.csv',index=False)
invent_sorted.to_csv(f'{OUT}/held_out_investigation_scored.csv',index=False)

# Save model metadata for reproducibility.
metadata={
    'algorithm':'RandomForestClassifier', 'n_estimators':300, 'max_depth':10,
    'min_samples_leaf':2, 'class_weight':'balanced', 'random_seed':210821,
    'split':'70/30 stratified', 'target':'event_label != NORMAL',
    'leakage_controls':['event_label','incident_id','event_id','timestamp excluded as raw field'],
    'threshold':0.50,
    'source_dataset':os.path.basename(DATA)
}
with open(f'{MODEL_DIR}/model_metadata.json','w') as f: json.dump(metadata,f,indent=2)
print('Supervised model completed')
print(pd.DataFrame([metrics]).T)
print('\nTop 15 high-risk held-out sessions:')
print(invent_sorted.head(15)[['event_id','vendor_id','account_id','risk_probability','actual_binary','predicted_label']].to_string(index=False))
