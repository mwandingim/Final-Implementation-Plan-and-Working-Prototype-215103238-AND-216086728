"""T21 Maritime Port Security Analytics - Security-control simulation.

Monte Carlo what-if analysis using the four synthetic incident scenarios. Control
parameters are explicit modelling assumptions, not measured effectiveness claims.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '02_data' / 'raw'
OUT = ROOT / '08_outputs' / 'simulation'
OUT.mkdir(parents=True, exist_ok=True)

SEED = 210821
N = 10000
rng = np.random.default_rng(SEED)

catalogue = pd.read_csv(DATA / 't21_scenario_catalogue.csv')
incident_ids = ['INC-T21-001','INC-T21-002','INC-T21-003','INC-T21-004']
scenario_ids = ['S01','S02','S03','S04']

# Scenario impact weights are analytical assumptions for the simulation scale.
scenario = pd.DataFrame({
    'scenario_id': scenario_ids,
    'scenario_name': [
        'Compromised vendor account',
        'Unauthorized OT command sequence',
        'Remote administration and data transfer',
        'Cargo/manifest manipulation'],
    'baseline_lambda_30d': [1.0, 1.0, 1.0, 1.0],
    'impact_weight': [8.0, 10.0, 9.0, 7.0],
    'continuity_hours': [2.0, 4.0, 3.0, 2.5],
})

# Explicit assumptions: occurrence reduction and impact reduction under each control.
# MFA/session restriction mainly reduces remote-account misuse; segmentation limits OT reach;
# enhanced monitoring primarily reduces impact through earlier detection/response.
controls = pd.DataFrame({
    'control': ['Baseline', 'MFA + session restrictions', 'Network segmentation', 'Enhanced monitoring'],
    'occurrence_reduction': [0.00, 0.65, 0.45, 0.20],
    'impact_reduction': [0.00, 0.25, 0.50, 0.55],
    'continuity_multiplier': [1.00, 1.10, 1.25, 1.05],
})

# Scenario-specific modifiers to reflect where each control is most relevant.
occ_mult = {
    'Baseline': [0,0,0,0],
    'MFA + session restrictions': [1.00, 0.75, 1.00, 0.85],
    'Network segmentation': [0.55, 1.00, 1.00, 0.80],
    'Enhanced monitoring': [1.00, 1.00, 1.00, 1.00],
}
impact_mult = {
    'Baseline': [0,0,0,0],
    'MFA + session restrictions': [0.90, 0.80, 0.95, 0.90],
    'Network segmentation': [0.80, 0.45, 0.55, 0.75],
    'Enhanced monitoring': [0.55, 0.55, 0.45, 0.60],
}

records = []
for _, c in controls.iterrows():
    name = c['control']
    occ = []
    impacts = []
    continuity = []
    for j, row in scenario.iterrows():
        base_lam = row.baseline_lambda_30d
        if name == 'Baseline':
            lam = base_lam
            impact_factor = 1.0
        elif name == 'MFA + session restrictions':
            lam = base_lam * (1 - c.occurrence_reduction * occ_mult[name][j])
            impact_factor = 1 - c.impact_reduction * impact_mult[name][j]
        elif name == 'Network segmentation':
            lam = base_lam * (1 - c.occurrence_reduction * occ_mult[name][j])
            impact_factor = 1 - c.impact_reduction * impact_mult[name][j]
        else:
            lam = base_lam * (1 - c.occurrence_reduction)
            impact_factor = 1 - c.impact_reduction * impact_mult[name][j]
        counts = rng.poisson(lam, N)
        risk = counts * row.impact_weight * impact_factor
        hours = counts * row.continuity_hours * c.continuity_multiplier
        occ.append(counts)
        impacts.append(risk)
        continuity.append(hours)
    total_risk = np.sum(np.vstack(impacts), axis=0)
    total_hours = np.sum(np.vstack(continuity), axis=0)
    total_events = np.sum(np.vstack(occ), axis=0)
    records.append({
        'control': name,
        'mean_events_30d': total_events.mean(),
        'p95_events_30d': np.percentile(total_events,95),
        'mean_risk_index': total_risk.mean(),
        'p95_risk_index': np.percentile(total_risk,95),
        'mean_continuity_hours': total_hours.mean(),
        'p95_continuity_hours': np.percentile(total_hours,95),
        'risk_reduction_vs_baseline_pct': np.nan,
    })

summary = pd.DataFrame(records)
base_risk = summary.loc[summary.control=='Baseline','mean_risk_index'].iloc[0]
summary['risk_reduction_vs_baseline_pct'] = (1-summary['mean_risk_index']/base_risk)*100
summary.to_csv(OUT/'control_simulation_summary.csv', index=False)

# Scenario-level expected results
scenario_rows=[]
for name in controls.control:
    c = controls.loc[controls.control==name].iloc[0]
    for j, row in scenario.iterrows():
        if name=='Baseline':
            lam=row.baseline_lambda_30d; impact_factor=1
        elif name=='MFA + session restrictions':
            lam=row.baseline_lambda_30d*(1-c.occurrence_reduction*occ_mult[name][j]); impact_factor=1-c.impact_reduction*impact_mult[name][j]
        elif name=='Network segmentation':
            lam=row.baseline_lambda_30d*(1-c.occurrence_reduction*occ_mult[name][j]); impact_factor=1-c.impact_reduction*impact_mult[name][j]
        else:
            lam=row.baseline_lambda_30d*(1-c.occurrence_reduction); impact_factor=1-c.impact_reduction*impact_mult[name][j]
        scenario_rows.append({'control':name,'scenario_id':row.scenario_id,'scenario_name':row.scenario_name,'expected_events_30d':lam,'impact_factor':impact_factor,'expected_risk_index':lam*row.impact_weight*impact_factor})
scenario_detail=pd.DataFrame(scenario_rows)
scenario_detail.to_csv(OUT/'scenario_control_comparison.csv', index=False)

# Sensitivity: vary control effectiveness +/-20% and compare mean risk.
sens=[]
for name in controls.control[1:]:
    c=controls.loc[controls.control==name].iloc[0]
    for delta in [-0.20,0,0.20]:
        adj_occ=np.clip(c.occurrence_reduction*(1+delta),0,0.95)
        adj_imp=np.clip(c.impact_reduction*(1+delta),0,0.95)
        exp=0
        for j,row in scenario.iterrows():
            if name=='MFA + session restrictions':
                lam=row.baseline_lambda_30d*(1-adj_occ*occ_mult[name][j]); imp=1-adj_imp*impact_mult[name][j]
            elif name=='Network segmentation':
                lam=row.baseline_lambda_30d*(1-adj_occ*occ_mult[name][j]); imp=1-adj_imp*impact_mult[name][j]
            else:
                lam=row.baseline_lambda_30d*(1-adj_occ); imp=1-adj_imp*impact_mult[name][j]
            exp += lam*row.impact_weight*imp
        sens.append({'control':name,'effectiveness_adjustment':f'{int(delta*100):+d}%','expected_risk_index':exp,'risk_reduction_vs_baseline_pct':(1-exp/base_risk)*100})
pd.DataFrame(sens).to_csv(OUT/'simulation_sensitivity.csv',index=False)

# Figures
plt.figure(figsize=(9,5))
plt.bar(summary.control, summary.mean_risk_index)
plt.ylabel('Mean 30-day risk index')
plt.title('T21 Security-Control Simulation: Expected Risk')
plt.xticks(rotation=20, ha='right')
plt.tight_layout(); plt.savefig(OUT/'simulation_risk_comparison.png', dpi=180); plt.close()

plt.figure(figsize=(9,5))
plt.bar(summary.control, summary.mean_continuity_hours)
plt.ylabel('Mean continuity impact (hours)')
plt.title('T21 Security-Control Simulation: Continuity Impact')
plt.xticks(rotation=20, ha='right')
plt.tight_layout(); plt.savefig(OUT/'simulation_continuity_comparison.png', dpi=180); plt.close()

plt.figure(figsize=(9,5))
for name in controls.control[1:]:
    s=pd.DataFrame([x for x in sens if x['control']==name])
    plt.plot(s['effectiveness_adjustment'], s['expected_risk_index'], marker='o', label=name)
plt.ylabel('Expected 30-day risk index')
plt.xlabel('Control-effectiveness sensitivity')
plt.title('T21 Simulation Sensitivity Analysis')
plt.legend()
plt.tight_layout(); plt.savefig(OUT/'simulation_sensitivity.png', dpi=180); plt.close()

assumptions = pd.DataFrame({
    'assumption': [
        'Baseline incident frequency',
        'Risk weights',
        'MFA/session restriction effectiveness',
        'Network segmentation effectiveness',
        'Enhanced monitoring effectiveness',
        'Continuity multipliers',
        'Simulation iterations'],
    'value': [
        'One observed synthetic incident per scenario in the 30-day dataset; Poisson lambda=1.0',
        'S01=8, S02=10, S03=9, S04=7; analytical scale only',
        '65% occurrence reduction; 25% impact reduction before scenario modifiers',
        '45% occurrence reduction; 50% impact reduction before scenario modifiers',
        '20% occurrence reduction; 55% impact reduction before scenario modifiers',
        'MFA=1.10, segmentation=1.25, monitoring=1.05',
        '10,000 Monte Carlo iterations with seed 210821'],
})
assumptions.to_csv(OUT/'simulation_assumptions.csv',index=False)

print('Simulation complete')
print(summary.round(3).to_string(index=False))
print('\nScenario comparison:')
print(scenario_detail.round(3).to_string(index=False))
print('\nSensitivity:')
print(pd.DataFrame(sens).round(3).to_string(index=False))
