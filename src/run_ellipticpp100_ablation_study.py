"""Run Elliptic++ Train200K 100-epoch SHIELD-GNN ablation study."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / "results" / "plots" / ".matplotlib-cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_robustness_table import build_model, compute_metrics, tensors_from_graph
from src.training_utils import get_device, load_graph_artifacts


STUDY_NAME = "Elliptic++ 100-epoch ablation"
OUTPUT_SUBDIR = "ablations_ellipticpp100"
ARTIFACT_DIR = "data\\processed\\ellipticpp_train200k\\graph_artifacts"
ATTACK_DIR = "data\\processed\\ellipticpp_train200k\\attack_artifacts\\targeted_evasion_test_illicit_full"
DEFENSE_DIR = "data\\processed\\ellipticpp_train200k\\defense_artifacts\\targeted_evasion_hybrid_pruned_p95_full"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "model_checkpoints" / OUTPUT_SUBDIR
TABLES_DIR = PROJECT_ROOT / "results" / "tables" / OUTPUT_SUBDIR
PLOTS_DIR = PROJECT_ROOT / "results" / "plots" / OUTPUT_SUBDIR
PAPER_TABLES_DIR = PROJECT_ROOT / "paper" / "tables" / "ellipticpp100_ablation"


VARIANTS = [
    {
        "ablation_name": "ellipticpp100_shield_full",
        "removed_component": "none",
        "use_attack": True,
        "use_defense": True,
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.01,
    },
    {
        "ablation_name": "ellipticpp100_clean_only",
        "removed_component": "attack_and_defense",
        "use_attack": False,
        "use_defense": False,
        "alpha": 0.0,
        "beta": 0.1,
        "gamma": 0.0,
    },
    {
        "ablation_name": "ellipticpp100_no_defense_branch",
        "removed_component": "defense_branch",
        "use_attack": True,
        "use_defense": False,
        "alpha": 0.0,
        "beta": 0.1,
        "gamma": 0.0,
        "flags": ["--disable-defense-branch"],
    },
    {
        "ablation_name": "ellipticpp100_no_attack_branch",
        "removed_component": "attack_branch",
        "use_attack": False,
        "use_defense": True,
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.01,
        "flags": ["--disable-attack-branch"],
    },
    {
        "ablation_name": "ellipticpp100_no_temporal_consistency",
        "removed_component": "temporal_consistency",
        "use_attack": True,
        "use_defense": True,
        "alpha": 1.0,
        "beta": 0.0,
        "gamma": 0.01,
        "flags": ["--disable-temporal-consistency"],
    },
    {
        "ablation_name": "ellipticpp100_no_pruning_regularization",
        "removed_component": "pruning_regularization",
        "use_attack": True,
        "use_defense": True,
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.0,
        "flags": ["--disable-pruning-regularization"],
    },
    {
        "ablation_name": "ellipticpp100_no_node_type_features",
        "removed_component": "node_type_features",
        "use_attack": True,
        "use_defense": True,
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.01,
        "flags": ["--disable-node-type-features"],
    },
]


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=STUDY_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Skip completed checkpoints and continue missing variants.")
    return parser.parse_args()


def ensure_dirs():
    """Create output directories."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def dataframe_to_markdown(df):
    """Render a DataFrame to Markdown without tabulate."""
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


def resolve(path):
    """Resolve project-relative paths."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_cmd(args, variant, hidden_channels=None):
    """Build train_shield_gnn command for one variant."""
    hidden = hidden_channels if hidden_channels is not None else args.hidden_channels
    cmd = [
        sys.executable,
        "-u",
        "src\\train_shield_gnn.py",
        "--artifact-dir",
        ARTIFACT_DIR,
        "--epochs",
        str(args.epochs),
        "--hidden-channels",
        str(hidden),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--patience",
        str(args.patience),
        "--device",
        args.device,
        "--save-name",
        variant["ablation_name"],
        "--ablation-name",
        variant["ablation_name"],
        "--output-subdir",
        OUTPUT_SUBDIR,
        "--alpha-defended",
        str(variant["alpha"]),
        "--beta-consistency",
        str(variant["beta"]),
        "--gamma-pruning",
        str(variant["gamma"]),
    ]
    if variant["use_attack"]:
        cmd.extend(["--use-attack", "--attack-dir", ATTACK_DIR])
    if variant["use_defense"]:
        cmd.extend(["--use-defense", "--defense-dir", DEFENSE_DIR])
    cmd.extend(variant.get("flags", []))
    return cmd


def run_command(cmd, retry_cmd):
    """Run command and retry with smaller hidden channels on CUDA OOM."""
    print("Running:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.returncode == 0:
        return "completed", "completed"
    text = completed.stderr or completed.stdout or ""
    print(text, flush=True)
    if "out of memory" in text.lower() or "cuda error" in text.lower() and "memory" in text.lower():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("CUDA memory failure detected; retrying with hidden_channels=16.", flush=True)
        retry = subprocess.run(retry_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
        if retry.stdout:
            print(retry.stdout, flush=True)
        if retry.returncode == 0:
            return "completed", "completed with hidden_channels=16 retry"
        retry_text = retry.stderr or retry.stdout or "retry failed"
        print(retry_text, flush=True)
        return "failed", retry_text.splitlines()[0] if retry_text else "retry failed"
    return "failed", text.splitlines()[0] if text else "command failed"


def train_variants(args):
    """Train or resume all variants."""
    statuses = []
    for variant in VARIANTS:
        checkpoint = CHECKPOINT_DIR / f"{variant['ablation_name']}_best.pt"
        if checkpoint.is_file():
            print(f"Skipping existing checkpoint: {checkpoint}", flush=True)
            statuses.append({**variant, "status": "existing", "notes": "checkpoint already exists"})
            continue
        status, notes = run_command(train_cmd(args, variant), train_cmd(args, variant, hidden_channels=16))
        statuses.append({**variant, "status": status, "notes": notes})
    (TABLES_DIR / "ellipticpp100_training_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    return statuses


def load_clean_graph():
    """Load clean Elliptic++ Train200K graph."""
    artifacts = load_graph_artifacts(resolve(ARTIFACT_DIR))
    return {
        "name": "clean",
        "path": str(resolve(ARTIFACT_DIR)),
        "X": artifacts["X"],
        "y": artifacts["y"],
        "edge_index": artifacts["edge_index_undirected"],
        "timesteps": artifacts["timesteps"],
        "test_mask": artifacts["test_mask"],
        "edge_weight": None,
    }


def load_attack_graph():
    """Load attacked Elliptic++ graph."""
    path = resolve(ATTACK_DIR)
    return {
        "name": "attacked",
        "path": str(path),
        "X": np.load(path / "attacked_X.npy"),
        "y": np.load(path / "attacked_y.npy"),
        "edge_index": np.load(path / "attacked_edge_index.npy"),
        "timesteps": np.load(path / "attacked_timesteps.npy"),
        "test_mask": np.load(path / "attacked_test_mask.npy"),
        "edge_weight": None,
    }


def load_defense_graph():
    """Load defended Elliptic++ graph."""
    path = resolve(DEFENSE_DIR)
    return {
        "name": "defended",
        "path": str(path),
        "X": np.load(path / "defended_X.npy"),
        "y": np.load(path / "defended_y.npy"),
        "edge_index": np.load(path / "defended_edge_index.npy"),
        "timesteps": np.load(path / "defended_timesteps.npy"),
        "test_mask": np.load(path / "defended_test_mask.npy"),
        "edge_weight": np.load(path / "defended_edge_weight.npy"),
    }


def node_type_slice():
    """Return feature slice for Elliptic++ node-type one-hot columns."""
    schema = json.loads((resolve(ARTIFACT_DIR) / "feature_schema.json").read_text(encoding="utf-8"))
    start = int(schema.get("transaction_feature_dim", 0)) + int(schema.get("wallet_feature_dim", 0))
    return slice(start, start + 2)


def maybe_mask_node_type(graph, variant):
    """Zero node-type one-hot features for the no-node-type variant."""
    if variant["ablation_name"] != "ellipticpp100_no_node_type_features":
        return graph
    graph = dict(graph)
    graph["X"] = graph["X"].copy()
    slc = node_type_slice()
    graph["X"][:, slc] = 0.0
    return graph


def evaluate_variant(variant, graphs, device):
    """Evaluate one ablation checkpoint on clean/attacked/defended graphs."""
    checkpoint_path = CHECKPOINT_DIR / f"{variant['ablation_name']}_best.pt"
    rows = []
    if not checkpoint_path.is_file():
        for graph in graphs:
            rows.append(base_row(variant, checkpoint_path, graph, "missing", "checkpoint missing"))
        return rows
    checkpoint = torch.load(checkpoint_path, map_location=device)
    spec = {"kind": "shield", "model_name": "shield_gnn"}
    for graph in graphs:
        graph_eval = maybe_mask_node_type(graph, variant)
        row = base_row(variant, checkpoint_path, graph_eval, "failed", "")
        try:
            tensors = tensors_from_graph(graph_eval, device, include_edge_weight=True)
            model = build_model(spec, checkpoint, graph_eval, device)
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


def base_row(variant, checkpoint_path, graph, status, notes):
    """Create base long-format evaluation row."""
    return {
        "study": STUDY_NAME,
        "dataset": "ellipticpp_train200k",
        "ablation_name": variant["ablation_name"],
        "removed_component": variant["removed_component"],
        "checkpoint_path": str(checkpoint_path),
        "evaluation_graph": graph["name"],
        "graph_path": graph["path"],
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
    """Build wide summary with drop/recovery metrics."""
    rows = []
    for name, group in long_df.groupby("ablation_name", sort=False):
        first = group.iloc[0]
        values = {}
        for graph in ["clean", "attacked", "defended"]:
            row = group[(group["evaluation_graph"] == graph) & (group["status"] == "completed")]
            values[f"{graph}_f1"] = row.iloc[0]["f1"] if not row.empty else None
            values[f"{graph}_auc"] = row.iloc[0]["roc_auc"] if not row.empty else None
        clean_f1 = values["clean_f1"]
        attacked_f1 = values["attacked_f1"]
        defended_f1 = values["defended_f1"]
        clean_auc = values["clean_auc"]
        attacked_auc = values["attacked_auc"]
        defended_auc = values["defended_auc"]
        if defended_f1 is not None and attacked_f1 is not None:
            conclusion = "defense helped" if defended_f1 > attacked_f1 else "defense did not improve F1"
        else:
            conclusion = "missing evaluation"
        rows.append(
            {
                "study": STUDY_NAME,
                "dataset": "ellipticpp_train200k",
                "ablation_name": name,
                "removed_component": first["removed_component"],
                "clean_f1": clean_f1,
                "attacked_f1": attacked_f1,
                "defended_f1": defended_f1,
                "clean_auc": clean_auc,
                "attacked_auc": attacked_auc,
                "defended_auc": defended_auc,
                "f1_drop_clean_to_attack": clean_f1 - attacked_f1 if clean_f1 is not None and attacked_f1 is not None else None,
                "f1_recovery_attack_to_defense": defended_f1 - attacked_f1 if defended_f1 is not None and attacked_f1 is not None else None,
                "auc_drop_clean_to_attack": clean_auc - attacked_auc if clean_auc is not None and attacked_auc is not None else None,
                "auc_recovery_attack_to_defense": defended_auc - attacked_auc if defended_auc is not None and attacked_auc is not None else None,
                "importance_conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)


def load_existing_metric(path):
    """Load an existing metrics JSON if present."""
    path = PROJECT_ROOT / path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("test_metrics", {})


def build_baseline_comparison(summary_df):
    """Compare existing Elliptic++ baselines with SHIELD ablation variants."""
    rows = []
    baseline_specs = [
        ("GCN", "results\\tables\\baseline_gcn_ellipticpp_train200k_100ep_metrics.json"),
        ("GraphSAGE", "results\\tables\\baseline_graphsage_ellipticpp_train200k_100ep_metrics.json"),
        ("SHIELD-GNN Clean Existing", "results\\tables\\shield_clean_ellipticpp_train200k_100ep_memsafe_metrics.json"),
        ("SHIELD-GNN Attack+Defense Existing", "results\\tables\\shield_attack_defense_ellipticpp_train200k_100ep_memsafe_metrics.json"),
    ]
    for model, path in baseline_specs:
        metrics = load_existing_metric(path)
        if metrics:
            rows.append(
                {
                    "model_or_variant": model,
                    "source": path,
                    "clean_f1": metrics.get("f1"),
                    "clean_auc": metrics.get("roc_auc"),
                    "attacked_f1": None,
                    "defended_f1": None,
                    "notes": "existing clean test metrics only",
                }
            )
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "model_or_variant": row["ablation_name"],
                "source": row["checkpoint_path"] if "checkpoint_path" in row else "ellipticpp100 ablation",
                "clean_f1": row["clean_f1"],
                "clean_auc": row["clean_auc"],
                "attacked_f1": row["attacked_f1"],
                "defended_f1": row["defended_f1"],
                "notes": row["importance_conclusion"],
            }
        )
    return pd.DataFrame(rows)


def save_latex_tables(summary_df, comparison_df):
    """Save paper LaTeX tables."""
    component_cols = ["ablation_name", "removed_component", "clean_f1", "attacked_f1", "defended_f1", "clean_auc", "attacked_auc", "defended_auc"]
    baseline_cols = ["model_or_variant", "clean_f1", "clean_auc", "attacked_f1", "defended_f1", "notes"]
    (PAPER_TABLES_DIR / "table_ellipticpp100_component_ablation.tex").write_text(
        summary_df[component_cols].to_latex(index=False, float_format="%.4f"),
        encoding="utf-8",
    )
    (PAPER_TABLES_DIR / "table_ellipticpp100_baseline_comparison.tex").write_text(
        comparison_df[baseline_cols].to_latex(index=False, float_format="%.4f"),
        encoding="utf-8",
    )


def save_plots(summary_df, comparison_df):
    """Save required plots."""
    completed = summary_df[pd.notna(summary_df["defended_f1"])].copy()
    if not completed.empty:
        by_f1 = completed.sort_values("defended_f1", ascending=True)
        plt.figure(figsize=(9, 4.5))
        plt.barh(by_f1["ablation_name"], by_f1["defended_f1"])
        plt.xlabel("Defended F1")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ellipticpp100_defended_f1_by_ablation.png", dpi=150)
        plt.close()

        by_auc = completed.sort_values("defended_auc", ascending=True)
        plt.figure(figsize=(9, 4.5))
        plt.barh(by_auc["ablation_name"], by_auc["defended_auc"])
        plt.xlabel("Defended ROC-AUC")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ellipticpp100_defended_auc_by_ablation.png", dpi=150)
        plt.close()

        x = np.arange(len(completed))
        width = 0.25
        plt.figure(figsize=(11, 5))
        plt.bar(x - width, completed["clean_f1"], width, label="clean")
        plt.bar(x, completed["attacked_f1"], width, label="attacked")
        plt.bar(x + width, completed["defended_f1"], width, label="defended")
        plt.xticks(x, completed["ablation_name"], rotation=45, ha="right")
        plt.ylabel("F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ellipticpp100_clean_attacked_defended_f1_grouped.png", dpi=150)
        plt.close()

        plt.figure(figsize=(9, 4.5))
        drop_df = completed.sort_values("f1_drop_clean_to_attack", ascending=True)
        plt.barh(drop_df["ablation_name"], drop_df["f1_drop_clean_to_attack"])
        plt.xlabel("F1 Drop: Clean to Attacked")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ellipticpp100_component_importance_drop.png", dpi=150)
        plt.close()

    comp = comparison_df[pd.notna(comparison_df["clean_f1"])].copy()
    if not comp.empty:
        comp = comp.sort_values("clean_f1", ascending=True)
        plt.figure(figsize=(10, max(4, 0.28 * len(comp))))
        plt.barh(comp["model_or_variant"], comp["clean_f1"])
        plt.xlabel("Clean F1")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ellipticpp100_baseline_vs_shield_f1.png", dpi=150)
        plt.close()


def write_paper_summary(summary_df, comparison_df):
    """Write paper-ready summary with honest conclusions."""
    lines = ["# Elliptic++ 100-Epoch Ablation Summary", ""]
    completed = summary_df[pd.notna(summary_df["defended_f1"])].copy()
    if completed.empty:
        lines.append("No completed Elliptic++ 100-epoch ablation checkpoints were available.")
    else:
        best_def = completed.sort_values("defended_f1", ascending=False).iloc[0]
        full = completed[completed["ablation_name"] == "ellipticpp100_shield_full"]
        lines.append(
            f"- Best defended F1 variant: {best_def['ablation_name']} "
            f"(F1={best_def['defended_f1']:.4f}, ROC-AUC={best_def['defended_auc']:.4f})."
        )
        if not full.empty:
            full_row = full.iloc[0]
            lines.append(
                f"- Full SHIELD defended F1: {full_row['defended_f1']:.4f}; "
                f"attacked F1: {full_row['attacked_f1']:.4f}."
            )
            if best_def["ablation_name"] != "ellipticpp100_shield_full":
                lines.append("- Full SHIELD did not beat every ablation; report this honestly.")
            else:
                lines.append("- Full SHIELD is the strongest defended-F1 ablation in this run.")
    lines.append("- Baseline comparison uses existing Elliptic++ 100-epoch baseline metrics when available.")
    (TABLES_DIR / "ellipticpp100_ablation_paper_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(long_df, summary_df):
    """Save all required output files."""
    comparison_df = build_baseline_comparison(summary_df)
    long_df.to_csv(TABLES_DIR / "ellipticpp100_component_ablation_long.csv", index=False)
    (TABLES_DIR / "ellipticpp100_component_ablation_long.md").write_text(dataframe_to_markdown(long_df), encoding="utf-8")
    (TABLES_DIR / "ellipticpp100_component_ablation_long.json").write_text(json.dumps(long_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    summary_df.to_csv(TABLES_DIR / "ellipticpp100_component_ablation_summary.csv", index=False)
    (TABLES_DIR / "ellipticpp100_component_ablation_summary.md").write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    (TABLES_DIR / "ellipticpp100_component_ablation_summary.json").write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    comparison_df.to_csv(TABLES_DIR / "ellipticpp100_baseline_vs_ablation_comparison.csv", index=False)
    (TABLES_DIR / "ellipticpp100_baseline_vs_ablation_comparison.md").write_text(dataframe_to_markdown(comparison_df), encoding="utf-8")
    save_latex_tables(summary_df, comparison_df)
    save_plots(summary_df, comparison_df)
    write_paper_summary(summary_df, comparison_df)


def main():
    """Run the Elliptic++ 100-epoch ablation study."""
    args = parse_args()
    ensure_dirs()
    print(STUDY_NAME, flush=True)
    print(f"Settings: epochs={args.epochs}, hidden={args.hidden_channels}, layers={args.num_layers}, dropout={args.dropout}, lr={args.lr}, patience={args.patience}, device={args.device}", flush=True)
    train_variants(args)
    device = get_device(args.device)
    graphs = [load_clean_graph(), load_attack_graph(), load_defense_graph()]
    rows = []
    for variant in VARIANTS:
        rows.extend(evaluate_variant(variant, graphs, device))
    long_df = pd.DataFrame(rows)
    summary_df = build_summary(long_df)
    save_outputs(long_df, summary_df)
    print(f"Saved Elliptic++ 100-epoch ablation outputs under {TABLES_DIR}", flush=True)


if __name__ == "__main__":
    main()
