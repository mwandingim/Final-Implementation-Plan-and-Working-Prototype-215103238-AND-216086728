from pathlib import Path
import json, html
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '07_dashboard_or_prototype'
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / '08_outputs'

FILES = {
    'supervised': BASE/'supervised_model/top_15_high_risk_sessions.csv',
    'anomaly': BASE/'anomaly_access/top_15_anomalous_sessions.csv',
    'predictive': BASE/'predictive_adversarial/top_15_predictive_risk_sessions.csv',
    'timeline': BASE/'incident_investigation/incident_correlated_timeline.csv',
    'intelligence': BASE/'security_intelligence/operational_indicators.csv',
    'simulation': BASE/'simulation/control_simulation_summary.csv',
    'nlp': BASE/'text_mining/top_15_text_risk_notes.csv',
    'baseline': BASE/'baseline_eda/B2_summary_table.csv',
}

def records(path):
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient='records', date_format='iso'))

data = {k: records(v) for k,v in FILES.items()}

html_doc = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>T21 Maritime Port Security Analytics — Analyst Prototype</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f4f6f8;color:#17202a}.wrap{max-width:1400px;margin:auto;padding:24px}.header{background:#102a43;color:white;padding:24px;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.card{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 5px #0001}.metric{font-size:28px;font-weight:700}.section{margin-top:18px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.toolbar input{padding:9px;border:1px solid #ccd6dd;border-radius:6px}.toolbar button{padding:9px 14px;border:0;border-radius:6px;background:#1677c8;color:white;cursor:pointer}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{padding:8px;border-bottom:1px solid #e4e8eb;text-align:left;vertical-align:top}th{background:#eaf0f4;position:sticky;top:0}.tablewrap{max-height:360px;overflow:auto;border-radius:8px}.note{font-size:13px;color:#52616b}.badge{display:inline-block;padding:3px 7px;border-radius:10px;background:#e8eef3}.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}.footer{margin-top:24px;font-size:12px;color:#607080}@media(max-width:900px){.grid,.cols{grid-template-columns:1fr 1fr}}@media(max-width:600px){.grid,.cols{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="header"><h1>T21 Maritime Port Security Analytics</h1><p>Analyst-facing prototype — synthetic data only</p><span class="badge">Refreshable snapshot</span> <span class="badge">Decision support, not automated response</span></div>
<div class="grid"><div class="card"><div class="note">Vendor sessions analysed</div><div class="metric">1,800</div></div><div class="card"><div class="note">Anomalous sessions</div><div class="metric">90</div></div><div class="card"><div class="note">Incident timeline events</div><div class="metric">37</div></div><div class="card"><div class="note">Maintenance notes scored</div><div class="metric">304</div></div></div>
<div class="section card"><h2>1. Load / Refresh and Baseline</h2><p class="note">The prototype loads the generated analytical snapshots from the reproducible project outputs. Use the browser refresh control to reload the latest generated snapshot.</p><div id="baseline"></div></div>
<div class="section card"><h2>2. High-Risk Access Sessions</h2><div class="toolbar"><input id="sessionFilter" placeholder="Filter vendor/account/device"><button onclick="renderSessions()">Apply filter</button><button onclick="location.reload()">Refresh</button></div><div class="cols"><div><h3>Supervised model</h3><div id="supervised"></div></div><div><h3>Unsupervised anomaly model</h3><div id="anomaly"></div></div></div></div>
<div class="section card"><h2>3. Predictive Risk</h2><div id="predictive"></div></div>
<div class="section card"><h2>4. Incident Timeline</h2><div class="toolbar"><input id="incidentFilter" placeholder="Filter incident/source/action"><button onclick="renderTimeline()">Apply filter</button></div><div id="timeline"></div></div>
<div class="section card"><h2>5. Security Intelligence</h2><div id="intelligence"></div></div>
<div class="section card"><h2>6. Simulation and Control Evaluation</h2><div id="simulation"></div></div>
<div class="section card"><h2>7. Text Mining / NLP</h2><div id="nlp"></div></div>
<div class="footer">T21 synthetic environment. Results are analytical evidence for the capstone prototype and do not represent a live maritime terminal or real organisation.</div>
</div>
<script>
const DATA = __DATA__;
function esc(x){return String(x??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));}
function table(id, rows, cols){let h='<div class="tablewrap"><table><thead><tr>'+cols.map(c=>'<th>'+esc(c)+'</th>').join('')+'</tr></thead><tbody>'; for(const r of rows) h+='<tr>'+cols.map(c=>'<td>'+esc(r[c])+'</td>').join('')+'</tr>'; return h+'</tbody></table></div>';}
function renderBaseline(){document.getElementById('baseline').innerHTML=table('b',DATA.baseline,['Measure','Result','Interpretation']);}
function renderSessions(){const q=document.getElementById('sessionFilter').value.toLowerCase(); const filt=a=>!q||JSON.stringify(a).toLowerCase().includes(q); document.getElementById('supervised').innerHTML=table('s',DATA.supervised.filter(filt).slice(0,15),['event_id','timestamp','vendor_id','account_id','risk_probability','predicted_label']); document.getElementById('anomaly').innerHTML=table('a',DATA.anomaly.filter(filt).slice(0,15),['event_id','timestamp','vendor_id','account_id','anomaly_score','anomaly_flag']);}
function renderPredictive(){document.getElementById('predictive').innerHTML=table('p',DATA.predictive,['event_id','timestamp','vendor_id','account_id','risk_probability','predicted_label']);}
function renderTimeline(){const q=document.getElementById('incidentFilter').value.toLowerCase(); const rows=DATA.timeline.filter(r=>!q||JSON.stringify(r).toLowerCase().includes(q)); document.getElementById('timeline').innerHTML=table('t',rows,['timestamp_utc','evidence_source','observed_action','analytical_interpretation','incident_id']);}
function renderAll(){renderBaseline();renderSessions();renderPredictive();renderTimeline();document.getElementById('intelligence').innerHTML=table('i',DATA.intelligence,['indicator_type','indicator','context','source_incident','confidence']);document.getElementById('simulation').innerHTML=table('m',DATA.simulation,['control','mean_risk_index','p95_risk_index','mean_continuity_hours','risk_reduction_vs_baseline_pct']);document.getElementById('nlp').innerHTML=table('n',DATA.nlp,['note_id','asset_id','vendor_id','category','priority','risk_probability','predicted_risk','free_text']);}
renderAll();
</script></body></html>'''.replace('__DATA__', json.dumps(data))

(OUT/'t21_analyst_prototype.html').write_text(html_doc, encoding='utf-8')
(OUT/'README.md').write_text('''# T21 Analyst Prototype\n\nOpen `t21_analyst_prototype.html` in a modern web browser. The prototype is a static, self-contained analyst-facing dashboard generated from the reproducible outputs in `08_outputs/`.\n\nImplemented workflow: load/refresh -> baseline -> supervised high-risk sessions -> anomaly sessions -> predictive risk -> incident timeline -> intelligence -> simulation -> NLP.\n\nAll data are synthetic. The prototype provides decision support and does not execute containment or other operational actions.\n''', encoding='utf-8')
print(OUT/'t21_analyst_prototype.html')
