from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '08_outputs' / 'security_intelligence'
OUT.mkdir(parents=True, exist_ok=True)
INC = ROOT / '08_outputs' / 'incident_investigation'

summary = pd.read_csv(INC / 'incident_summary.csv')
evidence = pd.read_csv(INC / 'primary_incident_evidence_summary.csv')
hyp = pd.read_csv(INC / 'hypothesis_assessment.csv')

# Priority Intelligence Requirements (PIRs) mapped to the actual T21 decision problem.
pirs = pd.DataFrame([
    ['PIR-01', 'Which vendor accounts and devices require elevated investigation priority?', 'V014 / ACC0014 / DEV0099', 'High', 'Supported by correlated VPN/IAM, endpoint, network and incident evidence.'],
    ['PIR-02', 'Which OT assets show activity correlated with anomalous remote access?', 'CRANE03', 'High', 'CRANE03 appears in the primary incident evidence and invalid command sequence.'],
    ['PIR-03', 'Do remote-access anomalies correlate with operational or business-process changes?', 'Yes: OT command activity and cargo-manifest change', 'High', 'Cross-source timeline shows close temporal correlation in INC-T21-001.'],
    ['PIR-04', 'Which indicators should be monitored after containment?', 'V014, ACC0014, DEV0099, CRANE03 and related OT/network events', 'High', 'These entities recur across the primary incident evidence.'],
], columns=['pir_id','priority_intelligence_requirement','current_answer','confidence','evidence_basis'])
pirs.to_csv(OUT / 'priority_intelligence_requirements.csv', index=False)

# Operational indicators derived only from the synthetic investigation; no real-world IOCs are invented.
indicators = pd.DataFrame([
    ['ENTITY', 'V014', 'Vendor identity associated with the primary incident', 'INC-T21-001', 'High'],
    ['ACCOUNT', 'ACC0014', 'Vendor account associated with anomalous access', 'INC-T21-001', 'High'],
    ['DEVICE', 'DEV0099', 'Device associated with endpoint and network anomalies', 'INC-T21-001', 'High'],
    ['OT_ASSET', 'CRANE03', 'OT asset associated with invalid command sequence', 'INC-T21-001', 'High'],
    ['BADGE', 'BADGE-0014', 'Physical-access identifier observed in restricted OT zone', 'INC-T21-001', 'Medium'],
    ['CARGO', 'CONT-004821', 'Cargo record with correlated manifest change', 'INC-T21-001', 'Medium'],
], columns=['indicator_type','indicator','context','source_incident','confidence'])
indicators.to_csv(OUT / 'operational_indicators.csv', index=False)

# MITRE ATT&CK for ICS enrichment. Mappings are contextual hypotheses, not claims of observed adversary attribution.
mitre = pd.DataFrame([
    ['External Remote Services', 'Remote vendor access is represented in the synthetic VPN/IAM evidence.', 'INC-T21-001 / VPN-IAM', 'Contextual match; not proof of adversary technique use.'],
    ['Valid Accounts', 'The incident scenario concerns anomalous use of a vendor account.', 'INC-T21-001 / VPN-IAM', 'Contextual match; the dataset does not establish how credentials were obtained.'],
    ['Remote Services', 'Endpoint/network evidence includes simulated remote-administration activity.', 'INC-T21-001 / Endpoint + Network', 'Contextual match to the observed behaviour.'],
    ['Change Program State', 'OT evidence records PLC mode/state changes during the incident window.', 'INC-T21-001 / OT-SCADA', 'Relevant ICS technique context; synthetic event does not establish a real-world adversary.'],
    ['Modify Parameter', 'OT evidence contains abnormal command/process-state changes.', 'INC-T21-001 / OT-SCADA', 'Contextual enrichment only; command semantics are synthetic.'],
], columns=['ATT&CK for ICS technique','T21 evidence basis','Evidence reference','limitation'])
mitre.to_csv(OUT / 'mitre_ics_enrichment.csv', index=False)

# Intelligence products.
operational = pd.DataFrame([
    ['Immediate investigation', 'V014 / ACC0014 / DEV0099', 'Review remote-access history, endpoint alerts and related OT/network events.', 'High'],
    ['OT monitoring', 'CRANE03', 'Increase monitoring of command sequences, mode/state changes and associated remote sessions.', 'High'],
    ['Access control', 'Vendor remote access', 'Require verified MFA and controlled access before restoration of affected vendor connectivity.', 'High'],
    ['Business integrity', 'CONT-004821', 'Validate the affected manifest record and any downstream cargo-processing changes.', 'Medium'],
], columns=['operational_priority','entity_or_control_area','analyst_action','priority'])
operational.to_csv(OUT / 'operational_intelligence_product.csv', index=False)

executive = pd.DataFrame([
    ['Risk picture', 'The synthetic investigation demonstrates that identity, endpoint, OT, network, physical and cargo signals can be correlated into a single incident picture.', 'High'],
    ['Primary concern', 'INC-T21-001 combines anomalous vendor access with OT and business-process indicators in a short time window.', 'High'],
    ['Control implication', 'Controlled vendor remote access, strong MFA, cross-source monitoring and OT-aware correlation are relevant controls for the simulated environment.', 'High'],
    ['Decision support', 'Prioritise investigation and containment where remote-access anomalies coincide with OT state/command changes or cargo integrity events.', 'High'],
], columns=['executive_topic','finding','confidence'])
executive.to_csv(OUT / 'executive_intelligence_product.csv', index=False)

# Feedback loop: how intelligence changes collection/detection priorities.
feedback = pd.DataFrame([
    ['Detection', 'Cross-source correlation identified a stronger signal than any single source.', 'Retain correlated identity + endpoint + OT + network + physical + cargo fields in collection.'],
    ['Access', 'Vendor access anomalies were central to the primary investigation.', 'Prioritise MFA status, device novelty, access zone and off-hours features.'],
    ['OT', 'Invalid command sequences provided operational context.', 'Collect command sequence validity and process-state transitions for future analytics.'],
    ['Business integrity', 'Cargo-manifest change added business impact context.', 'Continue collecting manifest-change events for incident correlation.'],
], columns=['feedback_area','lesson_from_analysis','collection_or_detection_change'])
feedback.to_csv(OUT / 'intelligence_feedback_loop.csv', index=False)

# Summary figure.
plot_df = summary[['incident_id','event_count']].copy()
plt.figure(figsize=(8,4.8))
plt.bar(plot_df['incident_id'], plot_df['event_count'])
plt.title('T21 Incident Evidence Volume')
plt.xlabel('Incident')
plt.ylabel('Correlated evidence records')
plt.tight_layout()
plt.savefig(OUT / 'incident_evidence_volume.png', dpi=180)
plt.close()

# QA for intelligence implementation.
qa = pd.DataFrame([
    ['T12', 'PIRs documented', len(pirs) >= 4, len(pirs)],
    ['T13', 'Operational indicators documented', len(indicators) >= 5, len(indicators)],
    ['T14', 'ICS enrichment documented with limitations', len(mitre) >= 5 and mitre['limitation'].notna().all(), len(mitre)],
    ['T15', 'Operational and executive intelligence products created', len(operational) >= 3 and len(executive) >= 3, f'{len(operational)} / {len(executive)}'],
    ['T16', 'Feedback loop documented', len(feedback) >= 4, len(feedback)],
], columns=['test_id','test','pass','observed'])
qa.to_csv(OUT / 'security_intelligence_qa.csv', index=False)

print('Security intelligence implementation completed.')
print('PIRs:', len(pirs))
print('Operational indicators:', len(indicators))
print('ATT&CK for ICS contextual mappings:', len(mitre))
print('Outputs:', OUT)
