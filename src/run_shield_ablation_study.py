"""Run SHIELD-GNN component, pruning-mode, and loss-weight ablations."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


TABLES_DIR = PROJECT_ROOT / "results" / "tables" / "ablations"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots" / "ablations"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "model_checkpoints" / "ablations"


DATASETS = {
    "full_elliptic": {
        "artifact_dir": "data\\processed\\full\\graph_artifacts",
        "attack_dir": "data\\processed\\full\\attack_artifacts\\targeted_evasion_test_illicit_fn0.05_fe0.08_fp0.10_full",
        "defense_dir": "data\\processed\\full\\defense_artifacts\\targeted_evasion_hybrid_pruned_p95_full",
        "defense_root": "data\\processed\\full\\defense_artifacts",
    },
    "ellipticpp_train200k": {
        "artifact_dir": "data\\processed\\ellipticpp_train200k\\graph_artifacts",
        "attack_dir": "data\\processed\\ellipticpp_train200k\\attack_artifacts\\targeted_evasion_test_illicit_full",
        "defense_dir": "data\\processed\\ellipticpp_train200k\\defense_artifacts\\targeted_evasion_hybrid_pruned_p95_full",
        "defense_root": "data\\processed\\ellipticpp_train200k\\defense_artifacts",
    },
}


def parse_args():
    """Parse ablation runner args."""
    parser = argparse.ArgumentParser(description="Run SHIELD-GNN ablation studies.")
    parser.add_argument("--dataset", choices=["full_elliptic", "ellipticpp_train200k", "both"], default="full_elliptic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--skip-training", action="store_true", help="Only evaluate existing ablation checkpoints.")
    parser.add_argument("--max-variants", type=int, default=None, help="Optional cap for quick partial runs.")
    return parser.parse_args()


def dataframe_to_markdown(df):
    """Render a DataFrame as Markdown without optional tabulate."""
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


def run_command(cmd, retry_cmd=None):
    """Run a command and optionally retry a memory-safe command on CUDA OOM."""
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode == 0:
        return "completed", "completed"
    stderr = completed.stderr or completed.stdout or ""
    print(stderr)
    if retry_cmd and ("out of memory" in stderr.lower() or "cuda" in stderr.lower() and "memory" in stderr.lower()):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Retrying with memory-safe settings:", " ".join(retry_cmd))
        retry = subprocess.run(retry_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
        if retry.stdout:
            print(retry.stdout)
        if retry.returncode == 0:
            return "completed", "completed after memory-safe retry"
        return "failed", (retry.stderr or retry.stdout or "retry failed").splitlines()[0]
    return "failed", stderr.splitlines()[0] if stderr else "command failed"


def base_train_cmd(args, config, variant, hidden=None):
    """Build a train_shield_gnn command for one ablation variant."""
    name = variant["ablation_name"]
    cmd = [
        sys.executable,
        "src\\train_shield_gnn.py",
        "--artifact-dir", config["artifact_dir"],
        "--epochs", str(args.epochs),
        "--hidden-channels", str(hidden if hidden is not None else args.hidden_channels),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--device", args.device,
        "--save-name", name,
        "--ablation-name", name,
        "--output-subdir", "ablations",
        "--alpha-defended", str(variant.get("alpha_defended", 1.0)),
        "--beta-consistency", str(variant.get("beta_consistency", 0.1)),
        "--gamma-pruning", str(variant.get("gamma_pruning", 0.01)),
    ]
    if variant.get("use_attack"):
        cmd.extend(["--use-attack", "--attack-dir", variant.get("attack_dir", config["attack_dir"])])
    if variant.get("use_defense"):
        cmd.extend(["--use-defense", "--defense-dir", variant.get("defense_dir", config["defense_dir"])])
    for flag in variant.get("flags", []):
        cmd.append(flag)
    return cmd


def component_variants(dataset):
    """Return requested component variants for a dataset."""
    if dataset == "full_elliptic":
        return [
            {"ablation_name": "shield_full_targeted", "removed_component": "none", "use_attack": True, "use_defense": True, "alpha_defended": 1.0, "beta_consistency": 0.1, "gamma_pruning": 0.01},
            {"ablation_name": "shield_no_defense_branch", "removed_component": "defense_branch", "use_attack": True, "use_defense": False, "alpha_defended": 0.0, "beta_consistency": 0.1, "gamma_pruning": 0.0, "flags": ["--disable-defense-branch"]},
            {"ablation_name": "shield_no_attack_branch", "removed_component": "attack_branch", "use_attack": False, "use_defense": True, "alpha_defended": 1.0, "beta_consistency": 0.1, "gamma_pruning": 0.01, "flags": ["--disable-attack-branch"]},
            {"ablation_name": "shield_no_temporal_consistency", "removed_component": "temporal_consistency", "use_attack": True, "use_defense": True, "alpha_defended": 1.0, "beta_consistency": 0.0, "gamma_pruning": 0.01, "flags": ["--disable-temporal-consistency"]},
            {"ablation_name": "shield_no_pruning_regularization", "removed_component": "pruning_regularization", "use_attack": True, "use_defense": True, "alpha_defended": 1.0, "beta_consistency": 0.1, "gamma_pruning": 0.0, "flags": ["--disable-pruning-regularization"]},
            {"ablation_name": "shield_no_attack_no_defense_clean_only", "removed_component": "attack_and_defense", "use_attack": False, "use_defense": False, "alpha_defended": 0.0, "beta_consistency": 0.1, "gamma_pruning": 0.0},
            {"ablation_name": "shield_defense_only_no_consistency", "removed_component": "attack_branch_and_temporal_consistency", "use_attack": False, "use_defense": True, "alpha_defended": 1.0, "beta_consistency": 0.0, "gamma_pruning": 0.01, "flags": ["--disable-temporal-consistency"]},
            {"ablation_name": "shield_attack_only_no_defense_no_consistency", "removed_component": "defense_branch_and_temporal_consistency", "use_attack": True, "use_defense": False, "alpha_defended": 0.0, "beta_consistency": 0.0, "gamma_pruning": 0.0, "flags": ["--disable-defense-branch", "--disable-temporal-consistency"]},
        ]
    return [
        {"ablation_name": "ellipticpp_shield_full", "removed_component": "none", "use_attack": True, "use_defense": True, "alpha_defended": 1.0, "beta_consistency": 0.1, "gamma_pruning": 0.01, "skip_reason": "skipped: Elliptic++ multi-branch ablation training is too heavy for this run; graph has over 1M clean nodes and over 1.05M attacked/defended nodes"},
        {"ablation_name": "ellipticpp_shield_no_defense_branch", "removed_component": "defense_branch", "use_attack": True, "use_defense": False, "alpha_defended": 0.0, "beta_consistency": 0.1, "gamma_pruning": 0.0, "flags": ["--disable-defense-branch"], "skip_reason": "skipped: Elliptic++ multi-branch ablation training is too heavy for this run; graph has over 1M clean nodes and over 1.05M attacked/defended nodes"},
        {"ablation_name": "ellipticpp_shield_no_temporal_consistency", "removed_component": "temporal_consistency", "use_attack": True, "use_defense": True, "alpha_defended": 1.0, "beta_consistency": 0.0, "gamma_pruning": 0.01, "flags": ["--disable-temporal-consistency"], "skip_reason": "skipped: Elliptic++ multi-branch ablation training is too heavy for this run; graph has over 1M clean nodes and over 1.05M attacked/defended nodes"},
        {"ablation_name": "ellipticpp_shield_no_node_type_features", "removed_component": "node_type_features", "use_attack": True, "use_defense": True, "alpha_defended": 1.0, "beta_consistency": 0.1, "gamma_pruning": 0.01, "flags": ["--disable-node-type-features"], "skip_reason": "skipped: Elliptic++ node-type ablation requires retraining the 1M-node Train200K graph with masked type features"},
        {"ablation_name": "ellipticpp_shield_clean_only", "removed_component": "attack_and_defense", "use_attack": False, "use_defense": False, "alpha_defended": 0.0, "beta_consistency": 0.1, "gamma_pruning": 0.0, "skip_reason": "skipped: Elliptic++ clean-only ablation would duplicate the completed non-ablation clean SHIELD run unless explicitly rerun under ablation naming"},
    ]


def loss_weight_variants():
    """Return minimum requested loss-weight grid."""
    values = [
        ("shield_loss_ablation_a1_b01_g001", 1.0, 0.1, 0.01),
        ("shield_loss_ablation_a1_b00_g001", 1.0, 0.0, 0.01),
        ("shield_loss_ablation_a1_b01_g000", 1.0, 0.1, 0.0),
        ("shield_loss_ablation_a05_b01_g001", 0.5, 0.1, 0.01),
        ("shield_loss_ablation_a1_b03_g001", 1.0, 0.3, 0.01),
        ("shield_loss_ablation_a1_b01_g005", 1.0, 0.1, 0.05),
    ]
    return [
        {
            "ablation_name": name,
            "removed_component": "loss_weight_grid",
            "use_attack": True,
            "use_defense": True,
            "alpha_defended": alpha,
            "beta_consistency": beta,
            "gamma_pruning": gamma,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
        for name, alpha, beta, gamma in values
    ]


def ensure_pruning_artifacts(config):
    """Generate hard, soft, and hybrid defense artifacts for pruning ablation."""
    variants = []
    for mode in ["hard", "soft", "hybrid"]:
        output_name = f"ablation_targeted_{mode}_pruned_p95_full"
        defense_dir = Path(PROJECT_ROOT / config["defense_root"] / output_name)
        if not defense_dir.is_dir():
            cmd = [
                sys.executable,
                "src\\run_edge_pruning_defense.py",
                "--input-type", "attack",
                "--attack-dir", config["attack_dir"],
                "--mode", mode,
                "--threshold-mode", "percentile",
                "--score-percentile", "95",
                "--prune-ratio", "0.05",
                "--min-prune-edges", "5000",
                "--preserve-original-test-mask",
                "--output-root", config["defense_root"],
                "--output-name", output_name,
                "--save-artifacts",
            ]
            status, notes = run_command(cmd)
            if status != "completed":
                print(f"Skipping {mode} pruning training because artifact generation failed: {notes}")
                continue
        variants.append(
            {
                "ablation_name": f"shield_{mode}_pruning",
                "removed_component": f"pruning_mode_{mode}",
                "pruning_mode": mode,
                "use_attack": True,
                "use_defense": True,
                "defense_dir": str(Path(config["defense_root"]) / output_name),
                "alpha_defended": 1.0,
                "beta_consistency": 0.1,
                "gamma_pruning": 0.01,
            }
        )
    return variants


def train_variants(args, dataset, config, variants):
    """Train all variants, respecting unique save names."""
    results = {}
    for variant in variants[: args.max_variants] if args.max_variants else variants:
        checkpoint = CHECKPOINT_DIR / f"{variant['ablation_name']}_best.pt"
        if checkpoint.is_file():
            results[variant["ablation_name"]] = {"status": "existing", "notes": "checkpoint already exists"}
            continue
        if args.skip_training:
            results[variant["ablation_name"]] = {"status": "skipped", "notes": "skip-training requested"}
            continue
        cmd = base_train_cmd(args, config, variant)
        retry = base_train_cmd(args, config, variant, hidden=16)
        status, notes = run_command(cmd, retry_cmd=retry)
        results[variant["ablation_name"]] = {"status": status, "notes": notes}
    return results


def load_checkpoint_metrics(name):
    """Load best epoch from a train metrics file when available."""
    path = TABLES_DIR / f"{name}_metrics.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_checkpoint(dataset, config, variant, device):
    """Evaluate one ablation checkpoint on clean, attacked, and defended graphs."""
    checkpoint_path = CHECKPOINT_DIR / f"{variant['ablation_name']}_best.pt"
    graphs = [
        load_clean_graph(config["artifact_dir"], "undirected"),
        load_attack_graph(config["attack_dir"]),
        load_defense_graph(config["defense_dir"]),
    ]
    spec = {
        "model": variant["ablation_name"],
        "kind": "shield",
        "model_name": "shield_gnn",
        "checkpoint": checkpoint_path,
    }
    rows = []
    if not checkpoint_path.is_file():
        for graph in graphs:
            rows.append(
                {
                    "dataset": dataset,
                    "ablation_name": variant["ablation_name"],
                    "removed_component": variant.get("removed_component", ""),
                    "checkpoint_path": str(checkpoint_path),
                    "evaluation_graph": graph["name"],
                    "status": "missing",
                    "notes": variant.get("skip_reason", "checkpoint missing"),
                }
            )
        return rows
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for graph in graphs:
        base = {
            "dataset": dataset,
            "ablation_name": variant["ablation_name"],
            "removed_component": variant.get("removed_component", ""),
            "checkpoint_path": str(checkpoint_path),
            "evaluation_graph": graph["name"],
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "loss": None,
            "num_test_nodes": None,
            "illicit_count": None,
            "licit_count": None,
            "status": "failed",
            "notes": "",
        }
        try:
            tensors = tensors_from_graph(graph, device, include_edge_weight=True)
            model = build_model(spec, checkpoint, graph, device)
            model.eval()
            with torch.no_grad():
                logits = model(tensors["x"], tensors["edge_index"], tensors["timesteps"], tensors["edge_weight"])
            metrics = compute_metrics(logits, tensors["y"], tensors["test_mask"])
            base.update(metrics)
            base["status"] = "completed"
            base["notes"] = "evaluated"
        except Exception as exc:
            base["notes"] = str(exc).splitlines()[0]
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(base)
    return rows


def build_summary(long_df):
    """Build component summary table with drop/recovery metrics."""
    rows = []
    for (dataset, ablation_name), group in long_df.groupby(["dataset", "ablation_name"], sort=False):
        row = group.iloc[0]
        values = {}
        for graph_name in ["clean", "attacked", "defended"]:
            graph_rows = group[(group["evaluation_graph"] == graph_name) & (group["status"] == "completed")]
            values[f"{graph_name}_f1"] = graph_rows.iloc[0]["f1"] if not graph_rows.empty else None
            values[f"{graph_name}_auc"] = graph_rows.iloc[0]["roc_auc"] if not graph_rows.empty else None
        clean_f1, attacked_f1, defended_f1 = values["clean_f1"], values["attacked_f1"], values["defended_f1"]
        clean_auc, attacked_auc, defended_auc = values["clean_auc"], values["attacked_auc"], values["defended_auc"]
        if group["status"].eq("missing").all():
            notes = "; ".join(sorted(set(str(note) for note in group["notes"].dropna())))
            conclusion = notes if notes else "missing checkpoint"
        else:
            conclusion = "completed"
        if attacked_f1 is not None and defended_f1 is not None:
            conclusion = "defense helped" if defended_f1 > attacked_f1 else "defense did not improve F1"
        rows.append(
            {
                "dataset": dataset,
                "ablation_name": ablation_name,
                "removed_component": row["removed_component"],
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


def save_all_tables(long_df, summary_df, variants):
    """Save requested ablation tables and simple paper summary."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(TABLES_DIR / "shield_component_ablation_long.csv", index=False)
    (TABLES_DIR / "shield_component_ablation_long.md").write_text(dataframe_to_markdown(long_df), encoding="utf-8")
    (TABLES_DIR / "shield_component_ablation_long.json").write_text(json.dumps(long_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    summary_df.to_csv(TABLES_DIR / "shield_component_ablation_summary.csv", index=False)
    (TABLES_DIR / "shield_component_ablation_summary.md").write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    (TABLES_DIR / "shield_component_ablation_summary.json").write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2), encoding="utf-8")

    pruning_names = {v["ablation_name"]: v for v in variants if "pruning_mode" in v}
    pruning_rows = []
    for name, variant in pruning_names.items():
        row = summary_df[summary_df["ablation_name"] == name]
        if row.empty:
            continue
        summary_path = PROJECT_ROOT / variant["defense_dir"] / "pruning_summary.json"
        pruning_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        r = row.iloc[0].to_dict()
        pruning_rows.append(
            {
                "pruning_mode": variant["pruning_mode"],
                "pruned_edges": pruning_summary.get("pruned_edges"),
                "prune_percentage": pruning_summary.get("prune_ratio"),
                "clean_f1": r.get("clean_f1"),
                "attacked_f1": r.get("attacked_f1"),
                "defended_f1": r.get("defended_f1"),
                "clean_auc": r.get("clean_auc"),
                "attacked_auc": r.get("attacked_auc"),
                "defended_auc": r.get("defended_auc"),
            }
        )
    pruning_df = pd.DataFrame(pruning_rows)
    pruning_df.to_csv(TABLES_DIR / "shield_pruning_mode_ablation.csv", index=False)
    (TABLES_DIR / "shield_pruning_mode_ablation.md").write_text(dataframe_to_markdown(pruning_df), encoding="utf-8")

    loss_rows = []
    for variant in [v for v in variants if v.get("removed_component") == "loss_weight_grid"]:
        row = summary_df[summary_df["ablation_name"] == variant["ablation_name"]]
        metrics = load_checkpoint_metrics(variant["ablation_name"])
        if row.empty:
            continue
        r = row.iloc[0].to_dict()
        loss_rows.append(
            {
                "alpha_defended": variant["alpha"],
                "beta_consistency": variant["beta"],
                "gamma_pruning": variant["gamma"],
                "clean_f1": r.get("clean_f1"),
                "attacked_f1": r.get("attacked_f1"),
                "defended_f1": r.get("defended_f1"),
                "clean_auc": r.get("clean_auc"),
                "attacked_auc": r.get("attacked_auc"),
                "defended_auc": r.get("defended_auc"),
                "best_epoch": metrics.get("best_epoch"),
                "notes": r.get("importance_conclusion"),
            }
        )
    loss_df = pd.DataFrame(loss_rows)
    loss_df.to_csv(TABLES_DIR / "shield_loss_weight_ablation.csv", index=False)
    (TABLES_DIR / "shield_loss_weight_ablation.md").write_text(dataframe_to_markdown(loss_df), encoding="utf-8")
    save_plots(summary_df, pruning_df, loss_df)

    best = summary_df.sort_values("defended_f1", ascending=False, na_position="last").head(1)
    largest_f1_drop = summary_df.sort_values("f1_drop_clean_to_attack", ascending=False, na_position="last").head(1)
    lines = ["# SHIELD-GNN Ablation Paper Summary", ""]
    if not best.empty and pd.notna(best.iloc[0].get("defended_f1")):
        lines.append(f"- Best SHIELD variant by defended F1: {best.iloc[0]['ablation_name']} (F1={best.iloc[0]['defended_f1']:.4f}).")
    if not largest_f1_drop.empty and pd.notna(largest_f1_drop.iloc[0].get("f1_drop_clean_to_attack")):
        lines.append(f"- Largest clean-to-attack F1 drop: {largest_f1_drop.iloc[0]['ablation_name']} ({largest_f1_drop.iloc[0]['f1_drop_clean_to_attack']:.4f}).")
    lines.append("- Component importance should be interpreted from the saved summary table; any component with non-positive recovery or lower defended F1 is an honest limitation rather than a positive finding.")
    (TABLES_DIR / "shield_ablation_paper_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(summary_df, pruning_df, loss_df):
    """Save compact ablation visualizations."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = summary_df[pd.notna(summary_df["defended_f1"])].copy()
    if not completed.empty:
        completed = completed.sort_values("defended_f1", ascending=True)
        plt.figure(figsize=(10, max(4, 0.28 * len(completed))))
        plt.barh(completed["ablation_name"], completed["defended_f1"])
        plt.xlabel("Defended F1")
        plt.ylabel("Ablation")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shield_component_ablation_defended_f1.png", dpi=150)
        plt.close()

        completed_auc = completed.sort_values("defended_auc", ascending=True)
        plt.figure(figsize=(10, max(4, 0.28 * len(completed_auc))))
        plt.barh(completed_auc["ablation_name"], completed_auc["defended_auc"])
        plt.xlabel("Defended ROC-AUC")
        plt.ylabel("Ablation")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shield_component_ablation_defended_auc.png", dpi=150)
        plt.close()

    if not pruning_df.empty and "defended_f1" in pruning_df:
        plt.figure(figsize=(6, 4))
        plt.bar(pruning_df["pruning_mode"], pruning_df["defended_f1"])
        plt.xlabel("Pruning Mode")
        plt.ylabel("Defended F1")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shield_pruning_mode_defended_f1.png", dpi=150)
        plt.close()

    if not loss_df.empty and "defended_f1" in loss_df:
        labels = [
            f"a={row.alpha_defended}, b={row.beta_consistency}, g={row.gamma_pruning}"
            for row in loss_df.itertuples()
        ]
        plt.figure(figsize=(10, max(4, 0.35 * len(loss_df))))
        plt.barh(labels, loss_df["defended_f1"])
        plt.xlabel("Defended F1")
        plt.ylabel("Loss Weights")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shield_loss_weight_defended_f1.png", dpi=150)
        plt.close()


def main():
    """Run all requested ablation groups."""
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ["full_elliptic", "ellipticpp_train200k"] if args.dataset == "both" else [args.dataset]
    device = get_device(args.device)
    all_variants = []
    all_rows = []
    for dataset in datasets:
        config = DATASETS[dataset]
        variants = component_variants(dataset)
        if dataset == "full_elliptic":
            variants += ensure_pruning_artifacts(config)
            variants += loss_weight_variants()
        all_variants.extend(variants)
        train_variants(args, dataset, config, variants)
        for variant in variants:
            all_rows.extend(evaluate_checkpoint(dataset, config, variant, device))
    long_df = pd.DataFrame(all_rows)
    summary_df = build_summary(long_df)
    save_all_tables(long_df, summary_df, all_variants)
    print(f"Saved ablation tables under {TABLES_DIR}")


if __name__ == "__main__":
    main()
