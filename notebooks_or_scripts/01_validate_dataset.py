"""T21 Maritime Port Dataset - Step 1 validation and quality checks.

Purpose:
    Validate the synthetic multi-source T21 dataset before baseline analytics.

Outputs:
    08_outputs/validation/t21_dataset_validation_summary.csv
    08_outputs/validation/t21_column_quality_summary.csv
    08_outputs/validation/t21_validation_report.md

The script is deterministic and uses only the local project data files.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "02_data" / "raw"
OUT_DIR = BASE / "08_outputs" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "asset_inventory": "t21_asset_inventory.csv",
    "cargo_manifest": "t21_cargo_manifest.csv",
    "endpoint_alerts": "t21_endpoint_alerts.csv",
    "integrated_security_events": "t21_integrated_security_events.csv",
    "maintenance_notes": "t21_maintenance_notes.csv",
    "network_flows": "t21_network_flows.csv",
    "ot_scada_crane": "t21_ot_scada_crane.csv",
    "physical_access": "t21_physical_access.csv",
    "scenario_catalogue": "t21_scenario_catalogue.csv",
    "vendor_vpn_iam": "t21_vendor_vpn_iam.csv",
}

EXPECTED_EVENT_DATASETS = {
    "cargo_manifest", "endpoint_alerts", "maintenance_notes", "network_flows",
    "ot_scada_crane", "physical_access", "vendor_vpn_iam"
}

rows = []
column_rows = []
frames = {}

for name, filename in DATASETS.items():
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    frames[name] = df
    timestamp_ok = None
    date_min = None
    date_max = None
    invalid_timestamp = 0
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        invalid_timestamp = int(parsed.isna().sum())
        timestamp_ok = invalid_timestamp == 0
        if parsed.notna().any():
            date_min = parsed.min().isoformat()
            date_max = parsed.max().isoformat()

    duplicate_rows = int(df.duplicated().sum())
    duplicate_event_ids = None
    unique_event_ids = None
    if "event_id" in df.columns:
        duplicate_event_ids = int(df["event_id"].duplicated().sum())
        unique_event_ids = int(df["event_id"].nunique(dropna=True))

    missing_cells = int(df.isna().sum().sum())
    missing_columns = int((df.isna().sum() > 0).sum())

    label_counts = ""
    if "event_label" in df.columns:
        label_counts = "; ".join(f"{k}={v}" for k, v in df["event_label"].value_counts(dropna=False).to_dict().items())

    rows.append({
        "dataset": name,
        "file": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "date_min_utc": date_min,
        "date_max_utc": date_max,
        "missing_cells": missing_cells,
        "columns_with_missing": missing_columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_event_ids": duplicate_event_ids,
        "unique_event_ids": unique_event_ids,
        "invalid_timestamps": invalid_timestamp,
        "timestamp_validation": "PASS" if timestamp_ok is True else ("N/A" if timestamp_ok is None else "FAIL"),
        "label_distribution": label_counts,
    })

    for col in df.columns:
        column_rows.append({
            "dataset": name,
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_percent": round(float(df[col].isna().mean() * 100), 3),
            "unique_values": int(df[col].nunique(dropna=True)),
        })

summary = pd.DataFrame(rows)
column_quality = pd.DataFrame(column_rows)
summary.to_csv(OUT_DIR / "t21_dataset_validation_summary.csv", index=False)
column_quality.to_csv(OUT_DIR / "t21_column_quality_summary.csv", index=False)

# Cross-source and package-level checks.
primary_event_rows = int(summary[summary.dataset.isin(EXPECTED_EVENT_DATASETS)]["rows"].sum())
primary_event_expected = 9728
all_primary_plus_assets = primary_event_rows + len(frames["asset_inventory"])
text_records = len(frames["maintenance_notes"])

# Check the integrated index is internally unique and its binary label mapping is consistent.
integrated = frames["integrated_security_events"]
label_binary_ok = True
if {"event_label", "event_label_binary"}.issubset(integrated.columns):
    expected_binary = integrated["event_label"].isin(["SUSPICIOUS", "INCIDENT"]).astype(int)
    label_binary_ok = bool((expected_binary == integrated["event_label_binary"]).all())

# Check expected project-wide date range.
all_dates = []
for df in frames.values():
    if "timestamp" in df.columns:
        all_dates.extend(pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna().tolist())
project_min = min(all_dates).isoformat() if all_dates else None
project_max = max(all_dates).isoformat() if all_dates else None

critical_failures = []
for r in rows:
    # incident_id is intentionally blank for normal events; other missing values are failures.
    allowed_missing = 0
    if r["dataset"] != "scenario_catalogue":
        df = frames[r["dataset"]]
        if "incident_id" in df.columns:
            allowed_missing = int(df["incident_id"].isna().sum())
    unexpected_missing = r["missing_cells"] - allowed_missing
    if unexpected_missing > 0:
        critical_failures.append(f"{r['dataset']}: {unexpected_missing} unexpected missing cells")
    if r["duplicate_rows"] > 0:
        critical_failures.append(f"{r['dataset']}: duplicate rows")
    if r["invalid_timestamps"] > 0:
        critical_failures.append(f"{r['dataset']}: invalid timestamps")
    if r["duplicate_event_ids"] not in (None, 0):
        critical_failures.append(f"{r['dataset']}: duplicate event IDs")
if primary_event_rows != primary_event_expected:
    critical_failures.append(f"Primary event total expected {primary_event_expected}, found {primary_event_rows}")
if text_records < 200:
    critical_failures.append(f"Text records below required minimum: {text_records}")
if not label_binary_ok:
    critical_failures.append("Integrated event binary label mapping inconsistent")

status = "PASS" if not critical_failures else "REVIEW"

report = []
report.append("# T21 Dataset Validation Report")
report.append("")
report.append("## 1. Purpose")
report.append("This validation establishes a reproducible quality baseline before exploratory analysis and modelling. It checks row counts, schema size, timestamps, missing values, duplicate rows/event identifiers, labels and project-wide coverage.")
report.append("")
report.append("## 2. Project-level results")
report.append(f"- Validation status: **{status}**")
report.append(f"- Primary event/transaction records: **{primary_event_rows:,}**")
report.append(f"- Primary records including asset inventory: **{all_primary_plus_assets:,}**")
report.append(f"- Unstructured maintenance-note records: **{text_records:,}**")
report.append(f"- Overall event date range: **{project_min} to {project_max}**")
report.append(f"- Integrated event binary-label mapping: **{'PASS' if label_binary_ok else 'FAIL'}**")
report.append("")
report.append("## 3. Dataset checks")
report.append("")
report.append(summary.to_markdown(index=False))
report.append("")
report.append("## 4. Quality interpretation")
report.append("The dataset is synthetic and deterministic. A clean result here means the files are structurally ready for analysis; it does not mean the synthetic data are free from modelling bias or that the simulated patterns represent real port operations.")
report.append("")
report.append("## 5. Issues requiring attention")
if critical_failures:
    report.extend([f"- {x}" for x in critical_failures])
else:
    report.append("- No structural validation failures were detected in Step 1.")
report.append("")
report.append("## 6. Reproducibility")
report.append("Run `python 03_notebooks_or_scripts/01_validate_dataset.py` from the project root. The script writes the CSV quality summaries and this report under `08_outputs/validation/`.")

(OUT_DIR / "t21_validation_report.md").write_text("\n".join(report), encoding="utf-8")

print("T21 DATASET VALIDATION")
print("=" * 80)
print(summary[["dataset", "rows", "columns", "missing_cells", "duplicate_rows", "duplicate_event_ids", "invalid_timestamps"]].to_string(index=False))
print("-" * 80)
print(f"Primary event records: {primary_event_rows:,}")
print(f"Text records: {text_records:,}")
print(f"Overall date range: {project_min} -> {project_max}")
print(f"Integrated binary-label mapping: {'PASS' if label_binary_ok else 'FAIL'}")
print(f"VALIDATION STATUS: {status}")
