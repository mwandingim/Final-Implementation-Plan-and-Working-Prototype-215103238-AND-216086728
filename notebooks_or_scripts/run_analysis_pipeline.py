from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    '00_generate_t21_dataset.py',
    '01_validate_dataset.py',
    '03_baseline_eda.py',
    '05_supervised_model.py',
    '07_anomaly_access_analytics.py',
    '08_incident_investigation.py',
    '10_security_intelligence.py',
    '12_simulation.py',
    '14_text_mining_nlp.py',
    '16_predictive_adversarial.py',
    '18_prototype.py',
    '19_final_qa.py',
]
for name in SCRIPTS:
    print(f'\n=== Running {name} ===')
    subprocess.run([sys.executable, str(ROOT / '03_notebooks_or_scripts' / name)], cwd=ROOT, check=True)
print('\nPipeline completed successfully.')
