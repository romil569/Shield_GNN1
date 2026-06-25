"""Audit downloaded Elliptic++ CSV files before graph construction."""

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset_new"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def classify_file(path, columns):
    """Infer the semantic role of an Elliptic++ CSV from its name and columns."""
    lower_name = path.name.lower()
    lower_columns = {str(col).lower() for col in columns}
    traits = {
        "looks_like_transaction_features": "txs_features" in lower_name or ("txid" in lower_columns and "time step" in lower_columns),
        "looks_like_transaction_labels": "txs_classes" in lower_name or ("txid" in lower_columns and "class" in lower_columns and "feature" not in lower_name),
        "looks_like_transaction_edge_list": "txs_edgelist" in lower_name or {"txid1", "txid2"}.issubset(lower_columns),
        "looks_like_wallet_features": "wallets_features.csv" == lower_name or (
            "address" in lower_columns
            and "time step" in lower_columns
            and "class" not in lower_columns
            and "combined" not in lower_name
        ),
        "looks_like_wallet_labels": "wallets_classes" in lower_name or ("address" in lower_columns and "class" in lower_columns and "feature" not in lower_name),
        "looks_like_wallet_wallet_edge_list": "addraddr" in lower_name,
        "looks_like_wallet_transaction_edge_list": "addrtx" in lower_name or {"input_address", "txid"}.issubset(lower_columns),
        "looks_like_transaction_wallet_edge_list": "txaddr" in lower_name or {"txid", "output_address"}.issubset(lower_columns),
        "has_timestamps_or_timesteps": any("time" in col for col in lower_columns),
        "has_labels": "class" in lower_columns,
    }
    numeric_feature_columns = [
        col for col in columns
        if str(col).lower() not in {"txid", "address", "class", "time step"}
    ]
    traits["candidate_feature_dimension"] = len(numeric_feature_columns)
    return traits


def count_rows(path):
    """Count CSV rows without loading the whole file into memory."""
    total = 0
    for chunk in pd.read_csv(path, chunksize=200_000):
        total += len(chunk)
    return total


def select_detected_file(rows, trait_name):
    """Return a unique detected file for a trait, or None when missing/ambiguous."""
    matches = [row["filename"] for row in rows if row.get(trait_name)]
    if len(matches) == 1:
        return matches[0]
    return None


def write_markdown(audit_rows, detected):
    """Write a compact Markdown audit report."""
    lines = [
        "# Elliptic++ Dataset Audit",
        "",
        f"Dataset folder: `{DATASET_DIR}`",
        "",
        "## Detected Files",
        "",
        "| File | Shape | Feature Dim | Labels | Timesteps | Role Hints |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    role_keys = [
        "looks_like_transaction_features",
        "looks_like_transaction_labels",
        "looks_like_transaction_edge_list",
        "looks_like_wallet_features",
        "looks_like_wallet_labels",
        "looks_like_wallet_wallet_edge_list",
        "looks_like_wallet_transaction_edge_list",
        "looks_like_transaction_wallet_edge_list",
    ]
    for row in audit_rows:
        roles = [key.replace("looks_like_", "") for key in role_keys if row[key]]
        lines.append(
            f"| `{row['filename']}` | {row['shape']} | {row['candidate_feature_dimension']} | "
            f"{row['has_labels']} | {row['has_timestamps_or_timesteps']} | {', '.join(roles) or '-'} |"
        )

    lines.extend(["", "## Summary", ""])
    for key, value in detected.items():
        lines.append(f"- {key}: `{value}`" if value else f"- {key}: MISSING OR AMBIGUOUS")
    return "\n".join(lines) + "\n"


def main():
    """Audit dataset_new and save machine-readable reports."""
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(DATASET_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {DATASET_DIR}")

    audit_rows = []
    for path in csv_files:
        sample = pd.read_csv(path, nrows=5)
        rows = count_rows(path)
        traits = classify_file(path, sample.columns)
        row = {
            "filename": path.name,
            "path": str(path),
            "shape": [int(rows), int(len(sample.columns))],
            "columns": [str(col) for col in sample.columns],
            "sample_rows": sample.astype(object).where(pd.notnull(sample), None).to_dict(orient="records"),
            **traits,
        }
        audit_rows.append(row)

    detected = {
        "detected_transaction_feature_file": select_detected_file(audit_rows, "looks_like_transaction_features"),
        "detected_transaction_label_file": select_detected_file(audit_rows, "looks_like_transaction_labels"),
        "detected_transaction_edge_file": select_detected_file(audit_rows, "looks_like_transaction_edge_list"),
        "detected_wallet_feature_file": select_detected_file(audit_rows, "looks_like_wallet_features"),
        "detected_wallet_label_file": select_detected_file(audit_rows, "looks_like_wallet_labels"),
        "detected_wallet_wallet_edge_file": select_detected_file(audit_rows, "looks_like_wallet_wallet_edge_list"),
        "detected_wallet_to_transaction_edge_file": select_detected_file(audit_rows, "looks_like_wallet_transaction_edge_list"),
        "detected_transaction_to_wallet_edge_file": select_detected_file(audit_rows, "looks_like_transaction_wallet_edge_list"),
    }
    required = [
        "detected_transaction_feature_file",
        "detected_transaction_label_file",
        "detected_transaction_edge_file",
        "detected_wallet_feature_file",
        "detected_wallet_label_file",
        "detected_wallet_to_transaction_edge_file",
        "detected_transaction_to_wallet_edge_file",
    ]
    missing_or_ambiguous = [key for key in required if not detected[key]]
    detected["missing_or_ambiguous_files"] = missing_or_ambiguous

    (TABLES_DIR / "ellipticpp_dataset_audit.json").write_text(
        json.dumps({"files": audit_rows, "detected": detected}, indent=2),
        encoding="utf-8",
    )
    (TABLES_DIR / "ellipticpp_dataset_audit.md").write_text(
        write_markdown(audit_rows, detected),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "filename": row["filename"],
                "shape_rows": row["shape"][0],
                "shape_columns": row["shape"][1],
                "candidate_feature_dimension": row["candidate_feature_dimension"],
                **{key: row[key] for key in row if key.startswith("looks_like_")},
            }
            for row in audit_rows
        ]
    ).to_csv(TABLES_DIR / "ellipticpp_detected_files.csv", index=False)

    print("Elliptic++ dataset audit complete.")
    for key, value in detected.items():
        if key != "missing_or_ambiguous_files":
            print(f"{key}: {value if value else 'MISSING OR AMBIGUOUS'}")
    if missing_or_ambiguous:
        print("Missing or ambiguous required files:")
        for key in missing_or_ambiguous:
            print(f"- {key}")
        sys.exit(1)
    print("All required files were detected. Wallet-wallet edge file is optional and was "
          f"{'found' if detected['detected_wallet_wallet_edge_file'] else 'not found'}.")


if __name__ == "__main__":
    main()
