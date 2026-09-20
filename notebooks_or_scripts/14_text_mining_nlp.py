from pathlib import Path
import re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'02_data/raw/t21_maintenance_notes.csv'
OUT=BASE/'08_outputs/text_mining'
OUT.mkdir(parents=True, exist_ok=True)
MODEL=BASE/'04_models/text_mining'
MODEL.mkdir(parents=True, exist_ok=True)
SEED=210821

df=pd.read_csv(DATA)
df['timestamp']=pd.to_datetime(df['timestamp'], utc=True)
df['risk_label']=(df['event_label'].isin(['SUSPICIOUS','INCIDENT'])).astype(int)
X=df['free_text'].fillna('').astype(str)
y=df['risk_label']

# 70/30 stratified split, fixed seed
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,stratify=y,random_state=SEED)
pipe=Pipeline([
 ('tfidf',TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=1,sublinear_tf=True,stop_words='english')),
 ('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED))
])
pipe.fit(X_train,y_train)
pred=pipe.predict(X_test)
proba=pipe.predict_proba(X_test)[:,1]
cm=confusion_matrix(y_test,pred,labels=[0,1])
tn,fp,fn,tp=cm.ravel()
metrics={
 'rows':len(df),'normal':int((y==0).sum()),'risk':int((y==1).sum()),
 'train_rows':len(X_train),'test_rows':len(X_test),'random_seed':SEED,
 'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred,zero_division=0),
 'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0),
 'false_negative_rate':fn/(fn+tp) if (fn+tp) else 0.0,
 'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)
}
pd.DataFrame([metrics]).to_csv(OUT/'nlp_model_metrics.csv',index=False)
with open(OUT/'nlp_classification_report.txt','w') as f:f.write(classification_report(y_test,pred,target_names=['NORMAL','RISK'],zero_division=0))
pd.DataFrame(cm,index=['Actual NORMAL','Actual RISK'],columns=['Pred NORMAL','Pred RISK']).to_csv(OUT/'nlp_confusion_matrix.csv')

# Score all notes and rank
all_prob=pipe.predict_proba(X)[:,1]
out=df[['note_id','timestamp','asset_id','vendor_id','category','priority','event_label','incident_id','free_text']].copy()
out['risk_probability']=all_prob
out['predicted_risk']=(all_prob>=0.50).astype(int)
out.sort_values(['risk_probability','priority'],ascending=[False,True]).head(15).to_csv(OUT/'top_15_text_risk_notes.csv',index=False)
out.sort_values('risk_probability',ascending=False).to_csv(OUT/'all_text_scores.csv',index=False)

# Feature weights
vec=pipe.named_steps['tfidf']; clf=pipe.named_steps['clf']; names=vec.get_feature_names_out(); coef=clf.coef_[0]
terms=pd.DataFrame({'term':names,'coefficient':coef})
terms['abs_coefficient']=terms.coefficient.abs()
terms.sort_values('coefficient',ascending=False).head(20).to_csv(OUT/'top_risk_terms.csv',index=False)
terms.sort_values('coefficient').head(20).to_csv(OUT/'top_normal_terms.csv',index=False)

# Simple keyword/entity extraction for operational use
entities=[]
for text in df.free_text:
    entities.extend(re.findall(r'\b(?:V\d{3}|CRANE\d{2}|PLC\d{2}|YARD-\d{2}|NOTE-\d{5})\b', text.upper()))
entity_counts=pd.Series(entities).value_counts().rename_axis('entity').reset_index(name='mentions') if entities else pd.DataFrame(columns=['entity','mentions'])
entity_counts.to_csv(OUT/'text_entities.csv',index=False)

# Category/risk summary
cat=df.groupby(['category','event_label']).size().unstack(fill_value=0).reset_index()
cat.to_csv(OUT/'text_category_summary.csv',index=False)

# Figures
plt.figure(figsize=(8,5)); plt.bar(['NORMAL','RISK'],[(y==0).sum(),(y==1).sum()]); plt.title('Maintenance Note Risk-Class Distribution'); plt.ylabel('Notes'); plt.tight_layout(); plt.savefig(OUT/'NLP_1_class_distribution.png',dpi=180); plt.close()

top=terms.sort_values('coefficient',ascending=False).head(12).sort_values('coefficient')
plt.figure(figsize=(8,6)); plt.barh(top.term,top.coefficient); plt.title('Top TF-IDF Terms Associated with Risk'); plt.xlabel('Logistic Regression coefficient'); plt.tight_layout(); plt.savefig(OUT/'NLP_2_top_risk_terms.png',dpi=180); plt.close()

ranked=out.nlargest(15,'risk_probability').sort_values('risk_probability')
plt.figure(figsize=(8,6)); plt.barh(ranked.note_id,ranked.risk_probability); plt.title('Top 15 Text-Risk Notes'); plt.xlabel('Predicted risk probability'); plt.tight_layout(); plt.savefig(OUT/'NLP_3_top_text_risk_notes.png',dpi=180); plt.close()

# Operational findings
high=out[out.risk_probability>=0.5]
findings={
 'corpus_rows':int(len(df)),
 'unique_text_templates':int(df.free_text.nunique()),
 'risk_notes':int((y==1).sum()),
 'model_flagged_notes':int(len(high)),
 'incident_notes':int((df.event_label=='INCIDENT').sum()),
 'top_categories':df[df.risk_label==1]['category'].value_counts().to_dict(),
 'note':'The corpus contains 304 synthetic maintenance/incident notes and 11 unique text templates. Results therefore demonstrate reproducible text classification and triage within this synthetic corpus, but should not be interpreted as general NLP performance on unseen real-world prose.'
}
json.dump(findings,open(OUT/'nlp_summary.json','w'),indent=2,default=str)

# Save pipeline via joblib
import joblib
joblib.dump(pipe,MODEL/'tfidf_logistic_text_risk.joblib')

print('=== T21 SECTION 10 TEXT MINING / NLP ===')
print(f'Corpus: {len(df)} notes | unique templates: {df.free_text.nunique()}')
print(f'Train/test: {len(X_train)}/{len(X_test)} | seed={SEED}')
for k in ['accuracy','precision','recall','f1','false_negative_rate']: print(f'{k}: {metrics[k]:.4f}')
print(f'Confusion matrix TN={tn}, FP={fp}, FN={fn}, TP={tp}')
print(f'Flagged at 0.50: {len(high)}')
print('Top risk terms:', ', '.join(terms.sort_values('coefficient',ascending=False).head(10).term))
print('Outputs:', OUT)
