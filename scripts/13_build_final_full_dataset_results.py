"""Build final full-dataset model comparison tables."""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


MODEL_SPECS = [
    ("GCN", "clean_static", "baseline_gcn_full_100ep_metrics.json"),
    ("GraphSAGE", "clean_static", "baseline_graphsage_full_100ep_metrics.json"),
    ("GAT", "clean_static", "baseline_gat_full_100ep_metrics.json"),
    ("TimeAwareGCN", "clean_temporal", "temporal_time_aware_gcn_metrics.json"),
    ("TemporalGCN-GRU", "clean_temporal", "temporal_gcn_gru_full_100ep_memsafe_metrics.json"),
    ("SHIELD-GNN Clean", "clean_shield", "shield_clean_full_100ep_memsafe_metrics.json"),
    (
        "SHIELD-GNN Attack+Defense",
        "attack_defense_shield",
        "shield_attack_defense_full_100ep_memsafe_metrics.json",
    ),
]


def load_metrics(path):
    """Load a metrics JSON if present and valid."""
    if not path.is_file():
        return None, "missing metrics file"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"corrupted metrics JSON: {exc}"


def row_from_metrics(model_name, experiment_type, filename):
    """Create one comparison row."""
    metrics_path = TABLES_DIR / filename
    metrics, error = load_metrics(metrics_path)
    base = {
        "model": model_name,
        "experiment_type": experiment_type,
        "artifact_scope": "full",
        "epochs_requested": None,
        "early_stopped_epoch": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "loss": None,
        "num_nodes": None,
        "illicit_count": None,
        "licit_count": None,
        "checkpoint_path": None,
        "metrics_path": str(metrics_path),
        "notes": "",
    }
    if error:
        base["notes"] = f"missing: {error}"
        return base

    test = metrics.get("test_metrics")
    if not isinstance(test, dict):
        base["checkpoint_path"] = metrics.get("checkpoint_path")
        base["notes"] = "failed: missing test_metrics in metrics JSON"
        return base

    args = metrics.get("args", {})
    epochs_requested = metrics.get("epochs_requested") or args.get("epochs") or 100
    best_epoch = metrics.get("best_epoch")
    notes = "completed"
    if best_epoch and epochs_requested and int(best_epoch) < int(epochs_requested):
        notes = "completed with early stopping"

    base.update(
        {
            "epochs_requested": epochs_requested,
            "early_stopped_epoch": best_epoch,
            "accuracy": test.get("accuracy"),
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "f1": test.get("f1"),
            "roc_auc": test.get("roc_auc"),
            "loss": test.get("loss"),
            "num_nodes": test.get("num_nodes"),
            "illicit_count": test.get("illicit_count"),
            "licit_count": test.get("licit_count"),
            "checkpoint_path": metrics.get("checkpoint_path"),
            "notes": notes,
        }
    )
    return base


def metric_leader(df, metric):
    """Return the best row for a numeric metric."""
    valid = df[pd.notna(df[metric])].copy()
    if valid.empty:
        return None
    return valid.sort_values(metric, ascending=False).iloc[0].to_dict()


def write_paper_summary(df, path):
    """Write a short paper-ready summary."""
    best_f1 = metric_leader(df, "f1")
    best_auc = metric_leader(df, "roc_auc")
    best_recall = metric_leader(df, "recall")
    shield_rows = df[df["model"].str.contains("SHIELD-GNN", regex=False)]
    shield_missing = shield_rows[~shield_rows["notes"].str.startswith("completed")]

    lines = ["# Final Full-Dataset Results Summary", ""]
    if best_f1:
        lines.append(f"- Best model by F1: {best_f1['model']} (F1={best_f1['f1']:.4f}).")
    if best_auc:
        lines.append(f"- Best model by ROC-AUC: {best_auc['model']} (ROC-AUC={best_auc['roc_auc']:.4f}).")
    if best_recall:
        lines.append(f"- Best model by recall: {best_recall['model']} (recall={best_recall['recall']:.4f}).")
    lines.extend(
        [
            "",
            "The full Elliptic dataset remains highly imbalanced: the test split contains 408 illicit nodes and 8,433 licit nodes. Accuracy is therefore less informative than F1, recall, and ROC-AUC for fraud detection.",
            "",
            "In this run, SHIELD-GNN uses the full graph context plus memory-safe temporal processing. Its attack+defense variant improves slightly over the clean SHIELD-GNN run on F1, recall, and ROC-AUC, but the strongest F1 and ROC-AUC are still from the static GCN baseline. This suggests the current SHIELD configuration is robustly executable but needs further tuning before it outperforms the simpler baseline.",
        ]
    )
    if not shield_missing.empty:
        missing_models = ", ".join(shield_missing["model"].tolist())
        lines.append("")
        lines.append(f"Warning: SHIELD-GNN result missing or failed for: {missing_models}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df):
    """Render a simple GitHub-flavored Markdown table without optional dependencies."""
    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def main():
    """Build final comparison artifacts."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_from_metrics(*spec) for spec in MODEL_SPECS]
    df = pd.DataFrame(rows)

    csv_path = TABLES_DIR / "final_model_comparison_full_dataset.csv"
    md_path = TABLES_DIR / "final_model_comparison_full_dataset.md"
    json_path = TABLES_DIR / "final_model_comparison_full_dataset.json"
    summary_path = TABLES_DIR / "final_results_paper_summary.md"

    df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_paper_summary(df, summary_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved paper summary: {summary_path}")
    print(df[["model", "f1", "roc_auc", "recall", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
