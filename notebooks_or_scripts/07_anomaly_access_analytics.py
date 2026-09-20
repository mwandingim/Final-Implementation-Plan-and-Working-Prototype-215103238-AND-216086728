from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'02_data/raw/t21_vendor_vpn_iam.csv'
OUT = ROOT/'08_outputs/anomaly_access'
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT/'04_models/anomaly'
MODEL.mkdir(parents=True, exist_ok=True)

SEED = 210821
CONTAMINATION = 0.05
N_ESTIMATORS = 300

# Load access evidence. Labels are retained only for post-hoc evaluation.
df = pd.read_csv(DATA)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df['hour'] = df['timestamp'].dt.hour
df['session_duration_min'] = df['session_duration_sec'] / 60.0

df['auth_failed'] = (df['auth_result'].eq('FAIL')).astype(int)
df['mfa_missing'] = (df['MFA_status'].eq('NO')).astype(int)
df['off_hours_flag'] = (df['off_hours'].eq('YES')).astype(int)
df['new_device_flag'] = (df['new_device'].eq('YES')).astype(int)
# Cyclical time features avoid treating 23:00 and 00:00 as far apart.
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

features = [
    'session_duration_min', 'auth_failed', 'mfa_missing',
    'off_hours_flag', 'new_device_flag', 'hour_sin', 'hour_cos',
    'role', 'access_zone'
]
X = df[features].copy()
cat = ['role', 'access_zone']
num = [c for c in features if c not in cat]
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat)], remainder='passthrough')
pipe = Pipeline([
    ('preprocess', pre),
    ('model', IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=SEED,
        n_jobs=-1,
        max_features=1.0,
    ))
])
pipe.fit(X)
joblib.dump(pipe, MODEL/'isolation_forest_vendor_access.joblib')

# sklearn decision_function: lower = more abnormal; convert to positive risk ranking.
df['anomaly_score'] = -pipe.decision_function(X)
df['anomaly_flag'] = (pipe.predict(X) == -1).astype(int)
df['anomaly_rank'] = df['anomaly_score'].rank(method='first', ascending=False).astype(int)
df = df.sort_values(['anomaly_score', 'timestamp'], ascending=[False, True]).reset_index(drop=True)

df['known_non_normal'] = (df['event_label'] != 'NORMAL').astype(int)

# Post-hoc evaluation only; labels were not used for fitting.
y = df['known_non_normal']
pred = df['anomaly_flag']
metrics = pd.DataFrame([{
    'n_sessions': len(df),
    'flagged_anomalies': int(pred.sum()),
    'flag_rate_pct': round(pred.mean()*100, 2),
    'posthoc_precision': round(precision_score(y, pred, zero_division=0), 4),
    'posthoc_recall': round(recall_score(y, pred, zero_division=0), 4),
    'posthoc_f1': round(f1_score(y, pred, zero_division=0), 4),
    'contamination_parameter': CONTAMINATION,
    'n_estimators': N_ESTIMATORS,
    'random_seed': SEED,
}])
metrics.to_csv(OUT/'anomaly_model_metrics.csv', index=False)

# Top anomalous sessions.
top15 = df[['event_id','timestamp','vendor_id','account_id','device_id','auth_result','MFA_status',
            'session_duration_sec','role','access_zone','off_hours','new_device','event_label',
            'incident_id','anomaly_score','anomaly_flag']].head(15)
top15.to_csv(OUT/'top_15_anomalous_sessions.csv', index=False)

# Vendor-level anomaly summary.
vendor_summary = (df.groupby('vendor_id')
    .agg(sessions=('event_id','count'), anomalies=('anomaly_flag','sum'),
         anomaly_rate=('anomaly_flag','mean'), mean_score=('anomaly_score','mean'),
         max_score=('anomaly_score','max'))
    .reset_index())
vendor_summary['anomaly_rate_pct'] = vendor_summary['anomaly_rate']*100
vendor_summary = vendor_summary.sort_values(['anomaly_rate','max_score'], ascending=False)
vendor_summary.to_csv(OUT/'vendor_anomaly_summary.csv', index=False)

# Compare anomaly ranking with supervised model where the scored investigation file exists.
sup_path = ROOT/'08_outputs/supervised_model/held_out_investigation_scored.csv'
comparison = None
if sup_path.exists():
    sup = pd.read_csv(sup_path)
    # Match by event_id where possible. Investigation scoring uses the same VPN event identifiers.
    keep = [c for c in ['event_id','risk_probability','model_predicted_label','risk_rank'] if c in sup.columns]
    comparison = df[['event_id','anomaly_score','anomaly_rank','event_label']].merge(sup[keep], on='event_id', how='inner')
    comparison.to_csv(OUT/'supervised_vs_anomaly_comparison.csv', index=False)
    if len(comparison) > 1 and 'risk_probability' in comparison:
        spearman = comparison[['anomaly_rank','risk_probability']].corr(method='spearman').iloc[0,1]
    else:
        spearman = np.nan
else:
    spearman = np.nan

# Figure 1: anomaly score distribution by known label (evaluation only).
plt.figure(figsize=(9,5))
for label in ['NORMAL','SUSPICIOUS']:
    vals = df.loc[df['event_label'].eq(label), 'anomaly_score']
    plt.hist(vals, bins=30, alpha=0.55, label=label)
plt.xlabel('Isolation Forest anomaly score (higher = more abnormal)')
plt.ylabel('Sessions')
plt.title('Vendor VPN/IAM Anomaly Score Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'A1_anomaly_score_distribution.png', dpi=180)
plt.close()

# Figure 2: top vendors by anomaly rate.
v = vendor_summary.head(10).sort_values('anomaly_rate_pct')
plt.figure(figsize=(9,5))
plt.barh(v['vendor_id'], v['anomaly_rate_pct'])
plt.xlabel('Flagged anomaly rate (%)')
plt.ylabel('Vendor')
plt.title('Top Vendors by Anomalous Session Rate')
plt.tight_layout()
plt.savefig(OUT/'A2_vendor_anomaly_rate.png', dpi=180)
plt.close()

# Figure 3: top anomalous sessions.
t = top15.sort_values('anomaly_score')
plt.figure(figsize=(9,6))
plt.barh(t['event_id'], t['anomaly_score'])
plt.xlabel('Anomaly score (higher = more abnormal)')
plt.ylabel('Session event')
plt.title('Top 15 Anomalous Vendor VPN/IAM Sessions')
plt.tight_layout()
plt.savefig(OUT/'A3_top_anomalous_sessions.png', dpi=180)
plt.close()

summary = {
    'sessions_analysed': int(len(df)),
    'flagged_anomalies': int(pred.sum()),
    'flag_rate_pct': round(pred.mean()*100,2),
    'contamination': CONTAMINATION,
    'n_estimators': N_ESTIMATORS,
    'random_seed': SEED,
    'posthoc_precision': round(precision_score(y,pred,zero_division=0),4),
    'posthoc_recall': round(recall_score(y,pred,zero_division=0),4),
    'posthoc_f1': round(f1_score(y,pred,zero_division=0),4),
    'rank_spearman_vs_supervised_probability': None if pd.isna(spearman) else round(float(spearman),4),
}
(OUT/'anomaly_summary.json').write_text(json.dumps(summary, indent=2))

print('ANOMALY / ACCESS ANALYTICS')
print('Sessions analysed:', len(df))
print('Anomalies flagged:', int(pred.sum()), f"({pred.mean()*100:.2f}%)")
print('Post-hoc precision:', f"{precision_score(y,pred,zero_division=0):.4f}")
print('Post-hoc recall:', f"{recall_score(y,pred,zero_division=0):.4f}")
print('Post-hoc F1:', f"{f1_score(y,pred,zero_division=0):.4f}")
print('\nTop 10 anomalous sessions:')
print(top15[['event_id','vendor_id','account_id','anomaly_score','event_label','off_hours','new_device']].head(10).to_string(index=False))
print('\nTop vendors by anomaly rate:')
print(vendor_summary[['vendor_id','sessions','anomalies','anomaly_rate_pct']].head(10).to_string(index=False))
