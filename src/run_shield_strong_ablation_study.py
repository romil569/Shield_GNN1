"""Run stronger SHIELD-GNN stress ablations on full Elliptic."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_robustness_table import (
    build_model,
    compute_metrics,
    load_attack_graph,
    load_clean_graph,
    load_defense_graph,
    tensors_from_graph,
)
from src.training_utils import get_device


CLEAN_DIR = "data\\processed\\full\\graph_artifacts"
STRONG_ATTACK_DIR = "data\\processed\\full\\attack_artifacts\\targeted_evasion_strong_test_illicit_fn0.10_fe0.15_fp0.10_full"
DEFENSE_DIRS = {
    "strong_defended_p90": "data\\processed\\full\\defense_artifacts\\strong_hybrid_pruned_p90_full",
    "strong_defended_p95": "data\\processed\\full\\defense_artifacts\\strong_hybrid_pruned_p95_full",
    "strong_defended_p975": "data\\processed\\full\\defense_artifacts\\strong_hybrid_pruned_p975_full",
}
TABLES_DIR = PROJECT_ROOT / "results" / "tables" / "ablations_strong"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots" / "ablations_strong"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "model_checkpoints" / "ablations_strong"


VARIANTS = [
    {
        "ablation_name": "shield_strong_full",
        "removed_component": "none",
        "use_attack": True,
        "use_defense": True,
        "defense_dir": DEFENSE_DIRS["strong_defended_p95"],
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.01,
    },
    {
        "ablation_name": "shield_strong_clean_only",
        "removed_component": "attack_and_defense",
        "use_attack": False,
        "use_defense": False,
        "alpha": 0.0,
        "beta": 0.1,
        "gamma": 0.0,
    },
    {
        "ablation_name": "shield_strong_no_defense",
        "removed_component": "defense_branch",
        "use_attack": True,
        "use_defense": False,
        "alpha": 0.0,
        "beta": 0.1,
        "gamma": 0.0,
        "flags": ["--disable-defense-branch"],
    },
    {
        "ablation_name": "shield_strong_no_temporal",
        "removed_component": "temporal_consistency",
        "use_attack": True,
        "use_defense": True,
        "defense_dir": DEFENSE_DIRS["strong_defended_p95"],
        "alpha": 1.0,
        "beta": 0.0,
        "gamma": 0.01,
        "flags": ["--disable-temporal-consistency"],
    },
    {
        "ablation_name": "shield_strong_no_pruning_reg",
        "removed_component": "pruning_regularization",
        "use_attack": True,
        "use_defense": True,
        "defense_dir": DEFENSE_DIRS["strong_defended_p95"],
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.0,
        "flags": ["--disable-pruning-regularization"],
    },
    {
        "ablation_name": "shield_strong_defense_only_no_consistency",
        "removed_component": "attack_branch_and_temporal_consistency",
        "use_attack": False,
        "use_defense": True,
        "defense_dir": DEFENSE_DIRS["strong_defended_p95"],
        "alpha": 1.0,
        "beta": 0.0,
        "gamma": 0.01,
        "flags": ["--disable-attack-branch", "--disable-temporal-consistency"],
    },
    {
        "ablation_name": "shield_strong_attack_only",
        "removed_component": "defense_branch_and_temporal_consistency",
        "use_attack": True,
        "use_defense": False,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
        "flags": ["--disable-defense-branch", "--disable-temporal-consistency"],
    },
]


def parse_args():
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def dataframe_to_markdown(df):
    """Render DataFrame as Markdown."""
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def train_cmd(args, variant, hidden=None):
    """Build train command for one strong variant."""
    cmd = [
        sys.executable,
        "src\\train_shield_gnn.py",
        "--artifact-dir", CLEAN_DIR,
        "--epochs", str(args.epochs),
        "--hidden-channels", str(hidden if hidden is not None else args.hidden_channels),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--device", args.device,
        "--save-name", variant["ablation_name"],
        "--ablation-name", variant["ablation_name"],
        "--output-subdir", "ablations_strong",
        "--alpha-defended", str(variant["alpha"]),
        "--beta-consistency", str(variant["beta"]),
        "--gamma-pruning", str(variant["gamma"]),
    ]
    if variant["use_attack"]:
        cmd.extend(["--use-attack", "--attack-dir", STRONG_ATTACK_DIR])
    if variant["use_defense"]:
        cmd.extend(["--use-defense", "--defense-dir", variant["defense_dir"]])
    cmd.extend(variant.get("flags", []))
    return cmd


def run_command(cmd, retry_cmd):
    """Run training and retry once with hidden=16 on CUDA OOM."""
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode == 0:
        return "completed", "completed"
    text = completed.stderr or completed.stdout or ""
    print(text)
    if "out of memory" in text.lower():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        retry = subprocess.run(retry_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
        if retry.stdout:
            print(retry.stdout)
        if retry.returncode == 0:
            return "completed", "completed with hidden=16 retry"
        return "failed", (retry.stderr or retry.stdout or "retry failed").splitlines()[0]
    return "failed", text.splitlines()[0] if text else "command failed"


def train_variants(args):
    """Train all strong variants."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for variant in VARIANTS:
        checkpoint = CHECKPOINT_DIR / f"{variant['ablation_name']}_best.pt"
        if checkpoint.is_file():
            statuses.append({**variant, "status": "existing", "notes": "checkpoint already exists"})
            continue
        if args.skip_training:
            statuses.append({**variant, "status": "skipped", "notes": "skip-training requested"})
            continue
        print(f"Training {variant['ablation_name']}")
        status, notes = run_command(train_cmd(args, variant), train_cmd(args, variant, hidden=16))
        statuses.append({**variant, "status": status, "notes": notes})
    return statuses


def load_graphs():
    """Load clean, strong attacked, and strong defended graphs."""
    graphs = [
        load_clean_graph(CLEAN_DIR, "undirected"),
        load_attack_graph(STRONG_ATTACK_DIR),
    ]
    graphs[0]["name"] = "clean"
    graphs[1]["name"] = "strong_attacked"
    for name, path in DEFENSE_DIRS.items():
        graph = load_defense_graph(path)
        graph["name"] = name
        graphs.append(graph)
    return graphs


def evaluate_variant(variant, graphs, device):
    """Evaluate one checkpoint on all strong graphs."""
    checkpoint_path = CHECKPOINT_DIR / f"{variant['ablation_name']}_best.pt"
    rows = []
    if not checkpoint_path.is_file():
        for graph in graphs:
            rows.append(base_row(variant, checkpoint_path, graph["name"], "missing", "checkpoint missing"))
        return rows
    checkpoint = torch.load(checkpoint_path, map_location=device)
    spec = {"kind": "shield", "model_name": "shield_gnn"}
    for graph in graphs:
        row = base_row(variant, checkpoint_path, graph["name"], "failed", "")
        try:
            tensors = tensors_from_graph(graph, device, include_edge_weight=True)
            model = build_model(spec, checkpoint, graph, device)
            model.eval()
            with torch.no_grad():
                logits = model(tensors["x"], tensors["edge_index"], tensors["timesteps"], tensors["edge_weight"])
            row.update(compute_metrics(logits, tensors["y"], tensors["test_mask"]))
            row["status"] = "completed"
            row["notes"] = "evaluated"
        except Exception as exc:
            row["notes"] = str(exc).splitlines()[0]
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
    return rows


def base_row(variant, checkpoint_path, graph_name, status, notes):
    """Create base evaluation row."""
    return {
        "ablation_name": variant["ablation_name"],
        "removed_component": variant["removed_component"],
        "checkpoint_path": str(checkpoint_path),
        "evaluation_graph": graph_name,
        "status": status,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "loss": None,
        "num_test_nodes": None,
        "illicit_count": None,
        "licit_count": None,
        "notes": notes,
    }


def build_summary(long_df):
    """Build wide strong-ablation summary."""
    rows = []
    defense_graphs = ["strong_defended_p90", "strong_defended_p95", "strong_defended_p975"]
    for name, group in long_df.groupby("ablation_name", sort=False):
        first = group.iloc[0]
        values = {}
        for graph in ["clean", "strong_attacked", *defense_graphs]:
            row = group[(group["evaluation_graph"] == graph) & (group["status"] == "completed")]
            values[f"{graph}_f1"] = row.iloc[0]["f1"] if not row.empty else None
            values[f"{graph}_auc"] = row.iloc[0]["roc_auc"] if not row.empty else None
        attacked_f1 = values["strong_attacked_f1"]
        attacked_auc = values["strong_attacked_auc"]
        best_def_f1 = max([values[f"{g}_f1"] for g in defense_graphs if values[f"{g}_f1"] is not None], default=None)
        best_def_auc = max([values[f"{g}_auc"] for g in defense_graphs if values[f"{g}_auc"] is not None], default=None)
        row = {
            "ablation_name": name,
            "removed_component": first["removed_component"],
            "clean_f1": values["clean_f1"],
            "strong_attacked_f1": attacked_f1,
            "strong_defended_p90_f1": values["strong_defended_p90_f1"],
            "strong_defended_p95_f1": values["strong_defended_p95_f1"],
            "strong_defended_p975_f1": values["strong_defended_p975_f1"],
            "clean_auc": values["clean_auc"],
            "strong_attacked_auc": attacked_auc,
            "strong_defended_p90_auc": values["strong_defended_p90_auc"],
            "strong_defended_p95_auc": values["strong_defended_p95_auc"],
            "strong_defended_p975_auc": values["strong_defended_p975_auc"],
            "f1_drop_clean_to_strong_attack": values["clean_f1"] - attacked_f1 if values["clean_f1"] is not None and attacked_f1 is not None else None,
            "best_f1_recovery_attack_to_defense": best_def_f1 - attacked_f1 if best_def_f1 is not None and attacked_f1 is not None else None,
            "auc_drop_clean_to_strong_attack": values["clean_auc"] - attacked_auc if values["clean_auc"] is not None and attacked_auc is not None else None,
            "best_auc_recovery_attack_to_defense": best_def_auc - attacked_auc if best_def_auc is not None and attacked_auc is not None else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_reports(long_df, summary_df):
    """Save strong ablation tables and plots."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(TABLES_DIR / "shield_strong_component_ablation_long.csv", index=False)
    (TABLES_DIR / "shield_strong_component_ablation_long.md").write_text(dataframe_to_markdown(long_df), encoding="utf-8")
    summary_df.to_csv(TABLES_DIR / "shield_strong_component_ablation_summary.csv", index=False)
    (TABLES_DIR / "shield_strong_component_ablation_summary.md").write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    (TABLES_DIR / "shield_strong_component_ablation_summary.json").write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    best = summary_df.sort_values("strong_defended_p95_f1", ascending=False, na_position="last").head(1)
    lines = ["# Strong SHIELD-GNN Ablation Summary", ""]
    if not best.empty:
        lines.append(
            f"- Best p95 defended F1: {best.iloc[0]['ablation_name']} "
            f"({best.iloc[0]['strong_defended_p95_f1']:.4f})."
        )
    lines.append("- Strong attack/defense tables evaluate clean, strong attacked, and p90/p95/p97.5 defended graphs.")
    (TABLES_DIR / "shield_strong_ablation_paper_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not summary_df.empty:
        plot_df = summary_df.sort_values("strong_defended_p95_f1", ascending=True)
        plt.figure(figsize=(9, 4))
        plt.barh(plot_df["ablation_name"], plot_df["strong_defended_p95_f1"])
        plt.xlabel("Strong p95 Defended F1")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shield_strong_ablation_p95_defended_f1.png", dpi=150)
        plt.close()


def main():
    """Run strong ablation study."""
    args = parse_args()
    statuses = train_variants(args)
    device = get_device(args.device)
    graphs = load_graphs()
    rows = []
    for variant in VARIANTS:
        rows.extend(evaluate_variant(variant, graphs, device))
    long_df = pd.DataFrame(rows)
    summary_df = build_summary(long_df)
    save_reports(long_df, summary_df)
    (TABLES_DIR / "shield_strong_training_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    print(f"Saved strong ablation outputs under {TABLES_DIR}")


if __name__ == "__main__":
    main()
