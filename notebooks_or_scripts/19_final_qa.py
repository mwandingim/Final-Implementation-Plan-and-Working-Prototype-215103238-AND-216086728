from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'02_data/raw'
OUT = ROOT/'08_outputs'
SCRIPTS = ROOT/'03_notebooks_or_scripts'
PROTO = ROOT/'07_dashboard_or_prototype'/'t21_analyst_prototype.html'

results=[]
def test(tid, component, condition, expected, actual, passed):
    results.append({'test_id':tid,'component':component,'input_condition':condition,'expected_result':expected,'actual_result_status':actual,'status':'PASS' if passed else 'FAIL'})

# T01 Dataset integrity
try:
    csvs=sorted(DATA.glob('*.csv'))
    frames={p.name:pd.read_csv(p) for p in csvs}
    event_files=[p for p in csvs if p.name not in {'t21_asset_inventory.csv','t21_scenario_catalogue.csv'}]
    total=9728
    total_actual=sum(len(frames[p.name]) for p in event_files if p.name!='t21_integrated_security_events.csv')
    integ=len(frames['t21_integrated_security_events.csv'])
    dup=sum(df.duplicated().sum() for df in frames.values())
    passed=(len(csvs)==10 and integ==9728 and total_actual>=9728 and dup==0)
    test('T01','Dataset integrity','All T21 CSV source files',f'10 CSV files; 9,728 integrated events; no duplicate rows',f'{len(csvs)} CSV files; {integ:,} integrated events; {dup} duplicate rows',passed)
except Exception as e:
    test('T01','Dataset integrity','All T21 CSV source files','Validation succeeds',str(e),False)

# T02 Baseline outputs
baseline=['B2_source_summary.csv','B2_summary_table.csv','B2_vendor_individual_baseline.csv']
figs=['B2_1_source_volume.png','B2_2_vendor_baseline.png','B2_3_hourly_login_baseline.png']
exists=[(OUT/'baseline_eda'/x).exists() for x in baseline+figs]
test('T02','Baseline and EDA','Baseline tables and visual evidence', 'Required tables and 3 figures exist', f'{sum(exists)}/{len(exists)} required outputs exist', all(exists))

# T03 ML outputs
ml=[OUT/'supervised_model'/'supervised_model_metrics.csv', OUT/'supervised_model'/'confusion_matrix.png', OUT/'supervised_model'/'top_15_high_risk_sessions.csv', OUT/'anomaly_access'/'anomaly_model_metrics.csv', OUT/'predictive_adversarial'/'predictive_model_metrics.csv']
exists=[p.exists() for p in ml]
test('T03','Security models','Supervised, anomaly and predictive outputs', 'Metrics and ranking/evaluation artefacts exist', f'{sum(exists)}/{len(exists)} required model outputs exist', all(exists))

# T04 Investigation and intelligence
required=[OUT/'incident_investigation'/'incident_correlated_timeline.csv',OUT/'incident_investigation'/'incident_response_actions.csv',OUT/'security_intelligence'/'priority_intelligence_requirements.csv',OUT/'security_intelligence'/'operational_intelligence_product.csv']
exists=[p.exists() for p in required]
timeline_rows=len(pd.read_csv(required[0])) if required[0].exists() else 0
passed=all(exists) and timeline_rows>0
test('T04','Investigation and intelligence','Correlated timeline and intelligence products', 'Timeline and intelligence evidence generated', f'{sum(exists)}/{len(exists)} files exist; timeline has {timeline_rows} rows', passed)

# T05 Prototype
passed=PROTO.exists() and PROTO.stat().st_size>10000
content=PROTO.read_text(errors='ignore') if PROTO.exists() else ''
markers=['Incident','Simulation','Intelligence','NLP','Predictive']
marker_count=sum(m in content for m in markers)
test('T05','Analyst prototype','Prototype HTML', 'Prototype loads and contains major analytical workflow sections', f'File size {PROTO.stat().st_size if PROTO.exists() else 0:,} bytes; {marker_count}/5 workflow markers found', passed and marker_count>=4)

qa=pd.DataFrame(results)
out=OUT/'final_qa'
out.mkdir(parents=True,exist_ok=True)
qa.to_csv(out/'final_qa_test_log.csv',index=False)
summary={'total_tests':len(qa),'passed':int((qa.status=='PASS').sum()),'failed':int((qa.status=='FAIL').sum()),'overall_status':'PASS' if (qa.status=='PASS').all() else 'FAIL'}
(out/'final_qa_summary.json').write_text(json.dumps(summary,indent=2))
qa.to_markdown(out/'final_qa_report.md',index=False)
print(qa.to_string(index=False))
print('\nSUMMARY:',summary)
