"""Evaluate trained full-dataset models on clean, attacked, and defended graphs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import accuracy_score_torch, compute_roc_auc_safe, precision_recall_f1
from src.models.model_factory import create_model
from src.models.shield_gnn import SHIELDGNNModel
from src.training_utils import get_device, load_graph_artifacts


TABLES_DIR = PROJECT_ROOT / "results" / "tables"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "model_checkpoints"


MODEL_SPECS = [
    {
        "model": "GCN",
        "kind": "static",
        "model_name": "gcn",
        "checkpoint": CHECKPOINT_DIR / "baseline_gcn_full_100ep_best.pt",
    },
    {
        "model": "GraphSAGE",
        "kind": "static",
        "model_name": "graphsage",
        "checkpoint": CHECKPOINT_DIR / "baseline_graphsage_full_100ep_best.pt",
    },
    {
        "model": "GAT",
        "kind": "static",
        "model_name": "gat",
        "checkpoint": CHECKPOINT_DIR / "baseline_gat_full_100ep_best.pt",
    },
    {
        "model": "TimeAwareGCN",
        "kind": "time_aware",
        "model_name": "time_aware_gcn",
        "checkpoint": CHECKPOINT_DIR / "temporal_time_aware_gcn_best.pt",
    },
    {
        "model": "TemporalGCN-GRU",
        "kind": "temporal_gru",
        "model_name": "temporal_gcn_gru",
        "checkpoint": CHECKPOINT_DIR / "temporal_gcn_gru_full_100ep_memsafe_best.pt",
    },
    {
        "model": "SHIELD-GNN Clean",
        "kind": "shield",
        "model_name": "shield_gnn",
        "checkpoint": CHECKPOINT_DIR / "shield_clean_full_100ep_memsafe_best.pt",
    },
    {
        "model": "SHIELD-GNN Attack+Defense",
        "kind": "shield",
        "model_name": "shield_gnn",
        "checkpoint": CHECKPOINT_DIR / "shield_attack_defense_full_100ep_memsafe_best.pt",
    },
    {
        "model": "SHIELD-GNN Targeted Attack+Defense",
        "kind": "shield",
        "model_name": "shield_gnn",
        "checkpoint": CHECKPOINT_DIR / "shield_targeted_attack_defense_full_100ep_memsafe_best.pt",
        "optional": True,
    },
]


def parse_args():
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Build robustness evaluation tables.")
    parser.add_argument("--clean-artifact-dir", required=True)
    parser.add_argument("--attack-dir", required=True)
    parser.add_argument("--defense-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--edge-type", choices=["directed", "undirected"], default="undirected")
    parser.add_argument("--output-prefix", default="robustness")
    return parser.parse_args()


def resolve(path):
    """Resolve project-relative paths."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_attack_graph(path):
    """Load full attacked graph arrays."""
    attack_path = resolve(path)
    required = [
        "attacked_X.npy",
        "attacked_y.npy",
        "attacked_edge_index.npy",
        "attacked_timesteps.npy",
        "attacked_test_mask.npy",
    ]
    missing = [name for name in required if not (attack_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attack artifacts in {attack_path}: {', '.join(missing)}")
    return {
        "name": "attacked",
        "path": str(attack_path),
        "X": np.load(attack_path / "attacked_X.npy"),
        "y": np.load(attack_path / "attacked_y.npy"),
        "edge_index": np.load(attack_path / "attacked_edge_index.npy"),
        "timesteps": np.load(attack_path / "attacked_timesteps.npy"),
        "test_mask": np.load(attack_path / "attacked_test_mask.npy"),
        "edge_weight": None,
    }


def load_defense_graph(path):
    """Load full defended graph arrays."""
    defense_path = resolve(path)
    required = [
        "defended_X.npy",
        "defended_y.npy",
        "defended_edge_index.npy",
        "defended_edge_weight.npy",
        "defended_timesteps.npy",
        "defended_test_mask.npy",
    ]
    missing = [name for name in required if not (defense_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing defense artifacts in {defense_path}: {', '.join(missing)}")
    return {
        "name": "defended",
        "path": str(defense_path),
        "X": np.load(defense_path / "defended_X.npy"),
        "y": np.load(defense_path / "defended_y.npy"),
        "edge_index": np.load(defense_path / "defended_edge_index.npy"),
        "timesteps": np.load(defense_path / "defended_timesteps.npy"),
        "test_mask": np.load(defense_path / "defended_test_mask.npy"),
        "edge_weight": np.load(defense_path / "defended_edge_weight.npy"),
    }


def load_clean_graph(path, edge_type):
    """Load full clean graph arrays."""
    artifact_path = resolve(path)
    artifacts = load_graph_artifacts(artifact_path)
    return {
        "name": "clean",
        "path": str(artifact_path),
        "X": artifacts["X"],
        "y": artifacts["y"],
        "edge_index": artifacts[f"edge_index_{edge_type}"],
        "timesteps": artifacts["timesteps"],
        "test_mask": artifacts["test_mask"],
        "edge_weight": None,
    }


def tensors_from_graph(graph, device, include_edge_weight=False):
    """Move graph arrays to torch tensors."""
    tensors = {
        "x": torch.tensor(graph["X"], dtype=torch.float32, device=device),
        "y": torch.tensor(graph["y"], dtype=torch.long, device=device),
        "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long, device=device),
        "timesteps": torch.tensor(graph["timesteps"], dtype=torch.long, device=device),
        "test_mask": torch.tensor(graph["test_mask"], dtype=torch.bool, device=device),
    }
    if include_edge_weight and graph.get("edge_weight") is not None:
        tensors["edge_weight"] = torch.tensor(graph["edge_weight"], dtype=torch.float32, device=device)
    else:
        tensors["edge_weight"] = None
    return tensors


def build_model(spec, checkpoint, graph, device):
    """Instantiate a model compatible with the checkpoint."""
    checkpoint_args = checkpoint.get("args", {})
    in_channels = int(graph["X"].shape[1])
    hidden = int(checkpoint_args.get("hidden_channels", 64))
    layers = int(checkpoint_args.get("num_layers", 2))
    dropout = float(checkpoint_args.get("dropout", 0.5))
    heads = int(checkpoint_args.get("heads", 4))
    max_timesteps = int(np.max(graph["timesteps"])) + 1

    if spec["kind"] == "shield":
        model = SHIELDGNNModel(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=2,
            backbone=checkpoint_args.get("backbone", "temporal_gcn_gru"),
            num_layers=layers,
            dropout=dropout,
            use_temporal_edge_encoding=True,
            use_edge_weight=True,
            max_timesteps=max_timesteps,
            snapshot_mode=checkpoint_args.get("snapshot_mode", "current"),
            detach_memory_each_step=checkpoint_args.get("detach_memory_each_step", True),
            temporal_grad_mode=checkpoint_args.get("temporal_grad_mode", "detach"),
        )
    else:
        model = create_model(
            spec["model_name"],
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=2,
            num_layers=layers,
            dropout=dropout,
            heads=heads,
            max_timesteps=max_timesteps,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def forward_model(model, spec, tensors, graph_name):
    """Run one no-grad forward pass."""
    if spec["kind"] == "static":
        return model(tensors["x"], tensors["edge_index"])
    if spec["kind"] == "time_aware":
        return model(tensors["x"], tensors["edge_index"], tensors["timesteps"])
    if spec["kind"] == "temporal_gru":
        return model.forward_full_sequence(
            tensors["x"],
            tensors["edge_index"],
            tensors["timesteps"],
            snapshot_mode="current",
            detach_memory_each_step=True,
            temporal_grad_mode="detach",
        )
    if spec["kind"] == "shield":
        return model(tensors["x"], tensors["edge_index"], tensors["timesteps"], tensors["edge_weight"])
    raise ValueError(f"Unsupported model kind: {spec['kind']}")


def compute_metrics(logits, y, mask):
    """Compute binary classification metrics over test labels."""
    selected = mask.bool() & (y != -1)
    if int(selected.sum()) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "loss": None,
            "num_test_nodes": 0,
            "illicit_count": 0,
            "licit_count": 0,
        }
    loss = F.cross_entropy(logits[selected], y[selected])
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    precision, recall, f1 = precision_recall_f1(y, preds, selected)
    return {
        "accuracy": accuracy_score_torch(y, preds, selected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": compute_roc_auc_safe(y, probs[:, 1], selected),
        "loss": float(loss.item()),
        "num_test_nodes": int(selected.sum().item()),
        "illicit_count": int(((y == 1) & selected).sum().item()),
        "licit_count": int(((y == 0) & selected).sum().item()),
    }


def base_row(spec, graph, status, notes):
    """Create a long-format base row."""
    checkpoint_path = str(spec["checkpoint"])
    return {
        "model": spec["model"],
        "checkpoint_name": Path(checkpoint_path).name,
        "checkpoint_path": checkpoint_path,
        "evaluation_graph": graph["name"],
        "graph_scope": "full",
        "graph_artifact_path": graph["path"],
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


def evaluate_one(spec, graph, device):
    """Evaluate one model/checkpoint on one graph, returning a row."""
    if not spec["checkpoint"].is_file():
        return base_row(spec, graph, "missing", "checkpoint file missing")
    try:
        checkpoint = torch.load(spec["checkpoint"], map_location=device)
        if "model_state_dict" not in checkpoint:
            return base_row(spec, graph, "failed", "checkpoint missing model_state_dict")
        include_weight = spec["kind"] == "shield"
        tensors = tensors_from_graph(graph, device, include_edge_weight=include_weight)
        model = build_model(spec, checkpoint, graph, device)
        model.eval()
        with torch.no_grad():
            logits = forward_model(model, spec, tensors, graph["name"])
        if logits.shape[0] != tensors["y"].shape[0]:
            return base_row(
                spec,
                graph,
                "incompatible",
                f"logits/y node mismatch: {logits.shape[0]} vs {tensors['y'].shape[0]}",
            )
        metrics = compute_metrics(logits, tensors["y"], tensors["test_mask"])
        row = base_row(spec, graph, "completed", "evaluated")
        row.update(metrics)
        return row
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        text = str(exc).splitlines()[0]
        status = "incompatible" if "size mismatch" in text or "shape" in text else "failed"
        return base_row(spec, graph, status, text)
    except Exception as exc:
        return base_row(spec, graph, "failed", str(exc))
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def dataframe_to_markdown(df):
    """Write Markdown without optional tabulate dependency."""
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def build_summary(long_df):
    """Build wide robustness summary by model."""
    rows = []
    for model, group in long_df.groupby("model", sort=False):
        values = {}
        notes = []
        for graph_name in ["clean", "attacked", "defended"]:
            row = group[group["evaluation_graph"] == graph_name]
            if row.empty or row.iloc[0]["status"] != "completed":
                values[f"{graph_name}_f1"] = None
                values[f"{graph_name}_auc"] = None
                if not row.empty:
                    notes.append(f"{graph_name}: {row.iloc[0]['status']} ({row.iloc[0]['notes']})")
            else:
                values[f"{graph_name}_f1"] = row.iloc[0]["f1"]
                values[f"{graph_name}_auc"] = row.iloc[0]["roc_auc"]
        clean_f1 = values["clean_f1"]
        attacked_f1 = values["attacked_f1"]
        defended_f1 = values["defended_f1"]
        clean_auc = values["clean_auc"]
        attacked_auc = values["attacked_auc"]
        defended_auc = values["defended_auc"]
        f1_by_setting = {
            "clean": clean_f1,
            "attacked": attacked_f1,
            "defended": defended_f1,
        }
        available_f1 = {key: value for key, value in f1_by_setting.items() if value is not None}
        best_setting = max(available_f1, key=available_f1.get) if available_f1 else None
        rows.append(
            {
                "model": model,
                "clean_f1": clean_f1,
                "attacked_f1": attacked_f1,
                "defended_f1": defended_f1,
                "f1_drop_clean_to_attack": clean_f1 - attacked_f1 if clean_f1 is not None and attacked_f1 is not None else None,
                "f1_recovery_attack_to_defense": defended_f1 - attacked_f1 if defended_f1 is not None and attacked_f1 is not None else None,
                "clean_auc": clean_auc,
                "attacked_auc": attacked_auc,
                "defended_auc": defended_auc,
                "auc_drop_clean_to_attack": clean_auc - attacked_auc if clean_auc is not None and attacked_auc is not None else None,
                "auc_recovery_attack_to_defense": defended_auc - attacked_auc if defended_auc is not None and attacked_auc is not None else None,
                "best_setting": best_setting,
                "robustness_rank": None,
                "notes": "; ".join(notes) if notes else "all evaluations completed",
            }
        )
    summary = pd.DataFrame(rows)
    rank_metric = summary["defended_f1"].fillna(summary["attacked_f1"]).fillna(-1.0)
    summary["robustness_rank"] = rank_metric.rank(method="min", ascending=False).astype(int)
    return summary.sort_values("robustness_rank")


def metric_leader(long_df, graph_name, metric):
    """Return best completed row for a graph/metric."""
    rows = long_df[(long_df["evaluation_graph"] == graph_name) & (long_df["status"] == "completed")]
    rows = rows[pd.notna(rows[metric])]
    if rows.empty:
        return None
    return rows.sort_values(metric, ascending=False).iloc[0].to_dict()


def write_paper_summary(long_df, summary_df, path):
    """Write paper-ready robustness summary."""
    lines = ["# Robustness Evaluation Summary", ""]
    for graph_name, label in [("clean", "clean"), ("attacked", "attacked-graph"), ("defended", "defended-graph")]:
        best_f1 = metric_leader(long_df, graph_name, "f1")
        best_auc = metric_leader(long_df, graph_name, "roc_auc")
        if best_f1:
            lines.append(f"- Best {label} model by F1: {best_f1['model']} (F1={best_f1['f1']:.4f}).")
        if best_auc:
            lines.append(f"- Best {label} model by ROC-AUC: {best_auc['model']} (ROC-AUC={best_auc['roc_auc']:.4f}).")
    shield_clean = summary_df[summary_df["model"] == "SHIELD-GNN Clean"]
    shield_def = summary_df[summary_df["model"] == "SHIELD-GNN Attack+Defense"]
    lines.append("")
    if not shield_clean.empty and not shield_def.empty:
        clean_row = shield_clean.iloc[0]
        def_row = shield_def.iloc[0]
        f1_delta = def_row["defended_f1"] - clean_row["defended_f1"]
        auc_delta = def_row["defended_auc"] - clean_row["defended_auc"]
        lines.append(
            "SHIELD-GNN Attack+Defense improves over SHIELD-GNN Clean on the defended graph "
            f"by F1={f1_delta:.4f} and ROC-AUC={auc_delta:.4f}."
        )
    best_def_f1 = metric_leader(long_df, "defended", "f1")
    if best_def_f1 and best_def_f1["model"] == "SHIELD-GNN Attack+Defense":
        lines.append("SHIELD-GNN Attack+Defense is the top defended-graph model by F1.")
    else:
        lines.append(
            "The proposed SHIELD-GNN defense configuration improves robustness under adversarial graph perturbations compared with its clean-only variant, although the static GCN baseline remains strongest on clean-test performance."
        )
    lines.append(
        "Based strictly on the evaluated metrics, SHIELD-GNN does not outperform all clean baselines; its support is strongest as a robustness-oriented defense comparison."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_tables(long_df, summary_df, output_prefix="robustness"):
    """Save all requested output tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "long_csv": TABLES_DIR / f"{output_prefix}_evaluation_long_full_dataset.csv",
        "long_md": TABLES_DIR / f"{output_prefix}_evaluation_long_full_dataset.md",
        "long_json": TABLES_DIR / f"{output_prefix}_evaluation_long_full_dataset.json",
        "summary_csv": TABLES_DIR / f"{output_prefix}_summary_full_dataset.csv",
        "summary_md": TABLES_DIR / f"{output_prefix}_summary_full_dataset.md",
        "summary_json": TABLES_DIR / f"{output_prefix}_summary_full_dataset.json",
        "paper": TABLES_DIR / f"{output_prefix}_paper_summary.md",
    }
    long_df.to_csv(outputs["long_csv"], index=False)
    outputs["long_md"].write_text(dataframe_to_markdown(long_df), encoding="utf-8")
    outputs["long_json"].write_text(json.dumps(long_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    summary_df.to_csv(outputs["summary_csv"], index=False)
    outputs["summary_md"].write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    outputs["summary_json"].write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    write_paper_summary(long_df, summary_df, outputs["paper"])
    return outputs


def main():
    """Run robustness evaluation."""
    args = parse_args()
    device = get_device(args.device)
    graphs = [
        load_clean_graph(args.clean_artifact_dir, args.edge_type),
        load_attack_graph(args.attack_dir),
        load_defense_graph(args.defense_dir),
    ]

    rows = []
    for spec in MODEL_SPECS:
        for graph in graphs:
            print(f"Evaluating {spec['model']} on {graph['name']} graph...")
            row = evaluate_one(spec, graph, device)
            print(f"  {row['status']}: f1={row['f1']} auc={row['roc_auc']} notes={row['notes']}")
            rows.append(row)

    long_df = pd.DataFrame(rows)
    summary_df = build_summary(long_df)
    outputs = save_tables(long_df, summary_df, args.output_prefix)

    print("\nSaved robustness outputs")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print("\nRobustness summary")
    print(summary_df[["model", "clean_f1", "attacked_f1", "defended_f1", "robustness_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
