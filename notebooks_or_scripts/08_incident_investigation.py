from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '02_data' / 'raw'
OUT = ROOT / '08_outputs' / 'incident_investigation'
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    'VPN/IAM': 't21_vendor_vpn_iam.csv',
    'Endpoint': 't21_endpoint_alerts.csv',
    'OT/SCADA': 't21_ot_scada_crane.csv',
    'Network': 't21_network_flows.csv',
    'Cargo': 't21_cargo_manifest.csv',
    'Physical access': 't21_physical_access.csv',
    'Maintenance notes': 't21_maintenance_notes.csv',
}

frames = []
for source, fname in SOURCES.items():
    df = pd.read_csv(RAW / fname)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['evidence_source'] = source
    frames.append(df)

all_events = pd.concat(frames, ignore_index=True, sort=False)
all_incident = all_events[all_events['incident_id'].notna()].copy()

# Incident catalogue derived from the labelled synthetic scenario package.
scenario = pd.read_csv(RAW / 't21_scenario_catalogue.csv')
scenario_map = {
    'INC-T21-001': 'S01 – Compromised vendor account',
    'INC-T21-002': 'S02 – Unauthorized OT command sequence',
    'INC-T21-003': 'S03 – Remote administration and data transfer',
    'INC-T21-004': 'S04 – Cargo/manifest manipulation',
}

summary_rows = []
for inc, g in all_incident.groupby('incident_id'):
    summary_rows.append({
        'incident_id': inc,
        'scenario': scenario_map.get(inc, 'Unmapped'),
        'start_utc': g['timestamp'].min().isoformat(),
        'end_utc': g['timestamp'].max().isoformat(),
        'event_count': len(g),
        'sources': ', '.join(sorted(g['evidence_source'].unique())),
        'incident_label_events': int((g['event_label'] == 'INCIDENT').sum()),
        'suspicious_events': int((g['event_label'] == 'SUSPICIOUS').sum()),
    })
incident_summary = pd.DataFrame(summary_rows).sort_values('incident_id')
incident_summary.to_csv(OUT / 'incident_summary.csv', index=False)

# Primary investigation: INC-T21-001 (S01).
primary_id = 'INC-T21-001'
g = all_incident[all_incident['incident_id'] == primary_id].copy().sort_values('timestamp')

# Build a compact evidence timeline with one row per event.
def observed_action(row):
    s = row['evidence_source']
    if s == 'VPN/IAM':
        return f"{row.get('auth_result', '')} vendor session; MFA={row.get('MFA_status', '')}; zone={row.get('access_zone', '')}; new_device={row.get('new_device', '')}"
    if s == 'Endpoint':
        return f"{row.get('alert_type', '')}; process={row.get('process_name', '')}; severity={row.get('severity', '')}"
    if s == 'OT/SCADA':
        return f"{row.get('command_type', '')}; asset={row.get('asset_id', '')}; state={row.get('process_state', '')}; sequence_valid={row.get('command_sequence_valid', '')}"
    if s == 'Network':
        return f"{row.get('alert_type', '')}; device={row.get('device_id', '')}; bytes_out={row.get('bytes_out', '')}; dst_port={row.get('dst_port', '')}"
    if s == 'Cargo':
        return f"{row.get('action', '')}; container={row.get('container_id', '')}; manifest_change={row.get('manifest_change', '')}; destination={row.get('destination_code', '')}"
    if s == 'Physical access':
        return f"{row.get('access_result', '')} physical access; badge={row.get('badge_id', '')}; zone={row.get('zone', '')}; direction={row.get('direction', '')}"
    if s == 'Maintenance notes':
        return str(row.get('free_text', ''))
    return ''

def interpretation(row):
    s = row['evidence_source']
    label = row.get('event_label', '')
    if s == 'VPN/IAM':
        return 'Remote access evidence is inconsistent with the normal baseline because the session is linked to off-hours/new-device or failed-MFA indicators.'
    if s == 'Endpoint':
        return 'Endpoint telemetry provides corroborating evidence of unusual remote/administrative tooling on the affected device.'
    if s == 'OT/SCADA':
        return 'OT evidence shows invalid command sequencing and abnormal/degraded process states on the affected asset.'
    if s == 'Network':
        return 'Network evidence shows unusual OT/remote-administration connections and/or large outbound transfers from the affected device.'
    if s == 'Cargo':
        return 'Cargo evidence records a manifest change during the correlated incident window.'
    if s == 'Physical access':
        return 'Physical access evidence places a vendor badge in the restricted OT zone during the incident window.'
    if s == 'Maintenance notes':
        return 'The contemporaneous note explicitly records cross-source correlation and initiation of incident review.'
    return ''

timeline = pd.DataFrame({
    'event_id': g['event_id'].astype(str),
    'timestamp_utc': g['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'evidence_source': g['evidence_source'],
    'observed_action': g.apply(observed_action, axis=1),
    'analytical_interpretation': g.apply(interpretation, axis=1),
    'event_label': g['event_label'],
    'incident_id': g['incident_id'],
})
timeline.to_csv(OUT / 'incident_correlated_timeline.csv', index=False)

# Evidence summary for primary incident.
key_evidence = [
    ['VPN/IAM', 'V014 / ACC0014 / DEV0099', 'Off-hours sessions, new device, failed/absent MFA indicators and OT-zone access', 'Supports unauthorized use of a vendor account.'],
    ['Endpoint', 'DEV0099', 'Remote administration / PLC / credential-helper simulated tools and critical alerts', 'Corroborates unusual activity on the same device.'],
    ['OT/SCADA', 'CRANE03', 'PLC mode change, unauthorized command, emergency stop; invalid sequence', 'Shows operational/OT impact during the correlated window.'],
    ['Network', 'DEV0099', 'Unusual OT connections, remote administration activity and large outbound transfer alerts', 'Corroborates network-level abnormality.'],
    ['Physical access', 'BADGE-0014', 'Vendor badge granted entry to OT-ZONE at 22:49 UTC', 'Provides physical-context evidence during the incident window.'],
    ['Cargo', 'CONT-004821', 'Manifest destination change at 22:51 UTC', 'Shows business-process integrity impact.'],
    ['Maintenance note', 'CRANE03 / V014', 'Cross-source proximity noted; incident review initiated', 'Documents analyst escalation and corroboration.'],
]
evidence_df = pd.DataFrame(key_evidence, columns=['source','entity','observed_evidence','interpretation'])
evidence_df.to_csv(OUT / 'primary_incident_evidence_summary.csv', index=False)

# Hypothesis assessment (analytical assessment; ground-truth scenario retained separately).
hypotheses = pd.DataFrame([
    ['H1', 'Legitimate maintenance activity', 'Partially supported by the presence of a vendor session, but weakened by new-device/off-hours/failed-MFA indicators and invalid OT commands.', 'Low'],
    ['H2', 'Equipment fault or operator error', 'Could explain individual OT alarms, but does not explain the correlated identity, endpoint, network, physical-access and cargo evidence as well.', 'Low–Moderate'],
    ['H3', 'Cyber-enabled vendor-account compromise or unauthorized use', 'Supported by correlated anomalous vendor access, endpoint remote tooling, invalid OT commands, network anomalies, restricted-zone access and cargo change.', 'High'],
], columns=['hypothesis_id','hypothesis','evidence_assessment','confidence'])
hypotheses.to_csv(OUT / 'hypothesis_assessment.csv', index=False)

# Response actions are analytical recommendations for the synthetic scenario.
response = pd.DataFrame([
    ['Containment', 'Temporarily suspend the affected vendor account ACC0014 and isolate DEV0099 from OT-connected services.', 'Limit further unauthorized access while preserving evidence.'],
    ['Containment', 'Restrict remote vendor access to the affected OT zone and require verified MFA before reactivation.', 'Reduce recurrence risk while maintaining controlled vendor support.'],
    ['Eradication', 'Review and remove unauthorized remote/administrative tooling represented in endpoint alerts; rotate affected access credentials in the simulation.', 'Remove the simulated persistence/access path.'],
    ['Recovery', 'Validate CRANE03 operating state and review the affected cargo manifest record before returning affected functions to normal operation.', 'Restore safe and trusted operational state.'],
    ['Monitoring', 'Increase monitoring of V014/ACC0014/DEV0099 and correlated OT/network events after recovery.', 'Detect recurrence or residual abnormal behaviour.'],
    ['Lessons learned', 'Tune cross-source correlation rules to combine identity, endpoint, OT, network, physical and cargo signals.', 'Improve earlier detection of multi-stage events.'],
], columns=['phase','response_action','purpose'])
response.to_csv(OUT / 'incident_response_actions.csv', index=False)

# Timeline visual: event count by source for the primary incident.
source_counts = g['evidence_source'].value_counts().sort_values(ascending=True)
plt.figure(figsize=(8,5))
source_counts.plot(kind='barh')
plt.title('INC-T21-001 Correlated Evidence by Source')
plt.xlabel('Number of events')
plt.ylabel('Evidence source')
plt.tight_layout()
plt.savefig(OUT / 'incident_evidence_by_source.png', dpi=180)
plt.close()

# Key timeline visual: event density over time.
plt.figure(figsize=(10,4.5))
plt.scatter(g['timestamp'], range(len(g)), s=20)
plt.title('INC-T21-001 Correlated Event Timeline')
plt.xlabel('UTC timestamp')
plt.ylabel('Chronological event index')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(OUT / 'incident_correlated_timeline.png', dpi=180)
plt.close()

# QA checks.
qa = pd.DataFrame([
    ['T07', 'Incident sources included', len(SOURCES) >= 6, len(SOURCES)],
    ['T08', 'Primary timeline contains correlated evidence', len(timeline) >= 20, len(timeline)],
    ['T09', 'Primary incident spans multiple source types', timeline['evidence_source'].nunique() >= 6, timeline['evidence_source'].nunique()],
    ['T10', 'Hypothesis assessment contains three hypotheses', len(hypotheses) == 3, len(hypotheses)],
    ['T11', 'Response actions cover containment, eradication and recovery', set(['Containment','Eradication','Recovery']).issubset(set(response['phase'])), ', '.join(sorted(response['phase'].unique()))],
], columns=['test_id','test','pass','observed'])
qa.to_csv(OUT / 'incident_investigation_qa.csv', index=False)

print('Incident investigation completed.')
print(incident_summary.to_string(index=False))
print('\nPrimary incident:', primary_id)
print('Timeline rows:', len(timeline))
print('Sources:', timeline['evidence_source'].nunique())
print('Outputs:', OUT)
