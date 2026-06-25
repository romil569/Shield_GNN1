"""Sanity-audit SHIELD-GNN ablation graphs, evaluation paths, and loss components."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_robustness_table import build_model, compute_metrics, tensors_from_graph
from src.shield_losses import pruning_regularization, shield_total_loss, supervised_node_loss
from src.training_utils import get_device, load_graph_artifacts


TABLES_DIR = PROJECT_ROOT / "results" / "tables" / "ablations"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "model_checkpoints" / "ablations"


def resolve(path):
    """Resolve project-relative paths."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def edge_hash(edge_index):
    """Hash an edge_index array by shape, dtype, and bytes."""
    h = hashlib.blake2b(digest_size=16)
    h.update(str(edge_index.shape).encode("utf-8"))
    h.update(str(edge_index.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(edge_index).view(np.uint8))
    return h.hexdigest()


def load_clean_graph(path):
    """Load clean graph for audit/eval."""
    artifacts = load_graph_artifacts(resolve(path))
    return {
        "name": "clean",
        "X": artifacts["X"],
        "y": artifacts["y"],
        "edge_index": artifacts["edge_index_undirected"],
        "timesteps": artifacts["timesteps"],
        "train_mask": artifacts["train_mask"],
        "test_mask": artifacts["test_mask"],
        "edge_weight": None,
    }


def load_attack_graph(path):
    """Load attack graph for audit/eval."""
    path = resolve(path)
    return {
        "name": "attacked",
        "X": np.load(path / "attacked_X.npy"),
        "y": np.load(path / "attacked_y.npy"),
        "edge_index": np.load(path / "attacked_edge_index.npy"),
        "timesteps": np.load(path / "attacked_timesteps.npy"),
        "train_mask": np.load(path / "attacked_train_mask.npy"),
        "test_mask": np.load(path / "attacked_test_mask.npy"),
        "edge_weight": None,
    }


def load_defense_graph(path):
    """Load defended graph for audit/eval."""
    path = resolve(path)
    return {
        "name": "defended",
        "X": np.load(path / "defended_X.npy"),
        "y": np.load(path / "defended_y.npy"),
        "edge_index": np.load(path / "defended_edge_index.npy"),
        "timesteps": np.load(path / "defended_timesteps.npy"),
        "train_mask": np.load(path / "defended_train_mask.npy"),
        "test_mask": np.load(path / "defended_test_mask.npy"),
        "edge_weight": np.load(path / "defended_edge_weight.npy"),
    }


def adjacency_lists(edge_index, num_nodes):
    """Build adjacency lists for neighborhood comparison."""
    adj = [set() for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            adj[u].add(v)
    return adj


def neighborhood_signature(graph, max_nodes=500):
    """Return sampled 1-hop/2-hop counts for test nodes."""
    test_nodes = np.flatnonzero(graph["test_mask"] & (graph["y"] != -1))
    if test_nodes.size > max_nodes:
        rng = np.random.default_rng(42)
        test_nodes = np.sort(rng.choice(test_nodes, size=max_nodes, replace=False))
    adj = adjacency_lists(graph["edge_index"], graph["X"].shape[0])
    one = []
    two = []
    for node in test_nodes:
        one_hop = set(adj[int(node)])
        two_hop = set(one_hop)
        for nbr in one_hop:
            two_hop.update(adj[nbr])
        two_hop.discard(int(node))
        one.append(len(one_hop))
        two.append(len(two_hop))
    return {"nodes": test_nodes, "one_hop": np.asarray(one), "two_hop": np.asarray(two)}


def graph_difference_audit(clean, attack, defense):
    """Compare clean, attacked, and defended graph artifacts."""
    attack_edges = {tuple(edge) for edge in attack["edge_index"].T.tolist()}
    defense_edges = {tuple(edge) for edge in defense["edge_index"].T.tolist()}
    removed = len(attack_edges - defense_edges)
    attack_sig = neighborhood_signature(attack)
    defense_sig = neighborhood_signature(defense)
    same_nodes = np.intersect1d(attack_sig["nodes"], defense_sig["nodes"])
    node_to_attack = {int(n): i for i, n in enumerate(attack_sig["nodes"])}
    node_to_defense = {int(n): i for i, n in enumerate(defense_sig["nodes"])}
    one_changed = 0
    two_changed = 0
    for node in same_nodes:
        ai = node_to_attack[int(node)]
        di = node_to_defense[int(node)]
        one_changed += int(attack_sig["one_hop"][ai] != defense_sig["one_hop"][di])
        two_changed += int(attack_sig["two_hop"][ai] != defense_sig["two_hop"][di])
    return {
        "clean": graph_stats(clean),
        "attacked": graph_stats(attack),
        "defended": graph_stats(defense),
        "edges_removed_by_defense": int(removed),
        "percentage_attacked_edges_removed": float(removed / max(attack["edge_index"].shape[1], 1)),
        "defended_edge_index_differs_from_attacked": edge_hash(attack["edge_index"]) != edge_hash(defense["edge_index"]),
        "sampled_test_nodes_for_neighborhoods": int(same_nodes.size),
        "test_nodes_with_changed_1hop_degree": int(one_changed),
        "test_nodes_with_changed_2hop_reach": int(two_changed),
        "mean_attack_1hop": float(attack_sig["one_hop"].mean()) if attack_sig["one_hop"].size else 0.0,
        "mean_defense_1hop": float(defense_sig["one_hop"].mean()) if defense_sig["one_hop"].size else 0.0,
        "mean_attack_2hop": float(attack_sig["two_hop"].mean()) if attack_sig["two_hop"].size else 0.0,
        "mean_defense_2hop": float(defense_sig["two_hop"].mean()) if defense_sig["two_hop"].size else 0.0,
    }


def graph_stats(graph):
    """Basic graph stats."""
    return {
        "node_count": int(graph["X"].shape[0]),
        "edge_count": int(graph["edge_index"].shape[1]),
        "edge_hash": edge_hash(graph["edge_index"]),
        "test_labeled_nodes": int((graph["test_mask"] & (graph["y"] != -1)).sum()),
    }


def code_audit():
    """Inspect evaluation and training source for expected branch behavior."""
    eval_text = (PROJECT_ROOT / "src" / "evaluate_robustness_table.py").read_text(encoding="utf-8")
    train_text = (PROJECT_ROOT / "src" / "train_shield_gnn.py").read_text(encoding="utf-8")
    return {
        "evaluation_audit": {
            "clean_evaluation_uses_clean_edge_index": (
                "load_clean_graph" in eval_text
                and "artifacts[f\"edge_index_{edge_type}\"]" in eval_text
            ),
            "attacked_evaluation_uses_attacked_edge_index": "attacked_edge_index.npy" in eval_text,
            "defended_evaluation_uses_defended_edge_index": "defended_edge_index.npy" in eval_text,
            "checkpoint_loading_uses_model_state_dict": "model_state_dict" in eval_text and "load_state_dict" in eval_text,
            "separate_graphs_loaded_for_clean_attack_defense": "load_clean_graph" in eval_text and "load_attack_graph" in eval_text and "load_defense_graph" in eval_text,
        },
        "training_branch_audit": {
            "use_attack_loads_attacked_graph": "attack_dir=args.attack_dir if args.use_attack" in train_text,
            "use_defense_loads_defended_graph": "defense_dir=args.defense_dir if args.use_defense" in train_text,
            "disable_defense_branch_removes_defense": "not args.disable_defense_branch" in train_text,
            "disable_attack_branch_removes_attack": "not args.disable_attack_branch" in train_text,
            "disable_temporal_consistency_sets_beta_zero": "beta = 0.0 if args.disable_temporal_consistency" in train_text,
            "disable_pruning_regularization_sets_gamma_zero": "args.disable_pruning_regularization" in train_text,
            "alpha_beta_gamma_used_in_total_loss": "args.alpha_defended" in train_text and "beta_consistency=beta" in train_text and "gamma_pruning=gamma" in train_text,
        },
    }


def model_for_checkpoint(checkpoint, graph, device):
    """Build SHIELD model from checkpoint args."""
    spec = {"kind": "shield", "model_name": "shield_gnn"}
    return build_model(spec, checkpoint, graph, device)


def tensors(graph, device):
    """Move graph to torch tensors including train/test masks."""
    t = tensors_from_graph(graph, device, include_edge_weight=True)
    t["train_mask"] = torch.tensor(graph["train_mask"], dtype=torch.bool, device=device)
    return t


def loss_debug_for_variant(name, clean, attack, defense, device):
    """Compute loss components from one trained checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / f"{name}_best.pt"
    if not checkpoint_path.is_file():
        return {"ablation_name": name, "status": "missing", "notes": "checkpoint missing"}
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_for_checkpoint(checkpoint, clean, device).to(device)
    model.eval()
    clean_t = tensors(clean, device)
    attack_t = tensors(attack, device)
    defense_t = tensors(defense, device)
    args = checkpoint.get("args", {})
    with torch.no_grad():
        clean_logits = model(clean_t["x"], clean_t["edge_index"], clean_t["timesteps"], clean_t["edge_weight"])
        attack_logits = model(attack_t["x"], attack_t["edge_index"], attack_t["timesteps"], attack_t["edge_weight"])
        defense_logits = model(defense_t["x"], defense_t["edge_index"], defense_t["timesteps"], defense_t["edge_weight"])
        clean_ce = supervised_node_loss(clean_logits, clean_t["y"], clean_t["train_mask"])
        attack_ce = supervised_node_loss(attack_logits, attack_t["y"], attack_t["train_mask"])
        defense_ce = supervised_node_loss(defense_logits, defense_t["y"], defense_t["train_mask"])
        same_attack = attack_logits.shape == clean_logits.shape
        same_defense = defense_logits.shape == clean_logits.shape
        total, comps = shield_total_loss(
            clean_logits,
            clean_t["y"],
            clean_t["train_mask"],
            defended_logits=defense_logits if same_defense else None,
            attacked_logits=attack_logits if same_attack else None,
            timesteps=clean_t["timesteps"],
            edge_index=clean_t["edge_index"],
            edge_weight=defense_t["edge_weight"],
            alpha_defended=float(args.get("alpha_defended", 1.0)),
            beta_consistency=0.0 if args.get("disable_temporal_consistency") else float(args.get("beta_consistency", 0.1)),
            gamma_pruning=0.0 if args.get("disable_pruning_regularization") else float(args.get("gamma_pruning", 0.01)),
        )
        total_with_branches = total + attack_ce + float(args.get("alpha_defended", 1.0)) * defense_ce
    return {
        "ablation_name": name,
        "status": "completed",
        "clean_ce_loss": float(clean_ce.item()),
        "attack_ce_loss": float(attack_ce.item()),
        "defended_ce_loss": float(defense_ce.item()),
        "temporal_consistency_loss": float(comps["temporal_consistency_loss"].item()),
        "pruning_regularization_loss": float(pruning_regularization(edge_weight=defense_t["edge_weight"]).item()),
        "shield_total_loss_without_attack_branch_ce": float(total.item()),
        "total_loss_with_branch_ce": float(total_with_branches.item()),
        "attack_logits_same_shape_as_clean": bool(same_attack),
        "defense_logits_same_shape_as_clean": bool(same_defense),
    }


def write_markdown(report):
    """Write human-readable audit."""
    lines = ["# SHIELD-GNN Ablation Sanity Audit", ""]
    graph = report["graph_difference_audit"]
    lines += [
        "## Graph Difference Audit",
        f"- Clean nodes/edges: {graph['clean']['node_count']}/{graph['clean']['edge_count']}",
        f"- Attacked nodes/edges: {graph['attacked']['node_count']}/{graph['attacked']['edge_count']}",
        f"- Defended nodes/edges: {graph['defended']['node_count']}/{graph['defended']['edge_count']}",
        f"- Defense removed edges: {graph['edges_removed_by_defense']} ({graph['percentage_attacked_edges_removed']:.4%})",
        f"- Defended edge index differs from attacked: {graph['defended_edge_index_differs_from_attacked']}",
        f"- Sampled test nodes with changed 1-hop degree: {graph['test_nodes_with_changed_1hop_degree']}/{graph['sampled_test_nodes_for_neighborhoods']}",
        f"- Sampled test nodes with changed 2-hop reach: {graph['test_nodes_with_changed_2hop_reach']}/{graph['sampled_test_nodes_for_neighborhoods']}",
        "",
        "## Evaluation Audit",
    ]
    for key, value in report["code_audit"]["evaluation_audit"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Training Branch Audit")
    for key, value in report["code_audit"]["training_branch_audit"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Interpretation")
    if graph["test_nodes_with_changed_1hop_degree"] == 0 and graph["test_nodes_with_changed_2hop_reach"] == 0:
        lines.append("- The defense changed global edges but did not materially alter sampled test-node neighborhoods; similar attacked/defended F1 can be a real consequence of weak local defense effect.")
    else:
        lines.append("- The defense changes some sampled test-node neighborhoods; similar F1 is less likely to be caused by identical graph evaluation.")
    return "\n".join(lines) + "\n"


def parse_args():
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-artifact-dir", default="data\\processed\\full\\graph_artifacts")
    parser.add_argument("--attack-dir", default="data\\processed\\full\\attack_artifacts\\targeted_evasion_test_illicit_fn0.05_fe0.08_fp0.10_full")
    parser.add_argument("--defense-dir", default="data\\processed\\full\\defense_artifacts\\targeted_evasion_hybrid_pruned_p95_full")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    """Run sanity audit."""
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    clean = load_clean_graph(args.clean_artifact_dir)
    attack = load_attack_graph(args.attack_dir)
    defense = load_defense_graph(args.defense_dir)
    report = {
        "graph_difference_audit": graph_difference_audit(clean, attack, defense),
        "code_audit": code_audit(),
    }
    debug_names = [
        "shield_full_targeted",
        "shield_no_temporal_consistency",
        "shield_no_pruning_regularization",
        "shield_no_defense_branch",
    ]
    debug_rows = [loss_debug_for_variant(name, clean, attack, defense, device) for name in debug_names]
    pd.DataFrame(debug_rows).to_csv(TABLES_DIR / "shield_loss_component_debug.csv", index=False)
    (TABLES_DIR / "shield_ablation_sanity_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (TABLES_DIR / "shield_ablation_sanity_audit.md").write_text(write_markdown(report), encoding="utf-8")
    print("Saved sanity audit and loss component debug outputs.")


if __name__ == "__main__":
    main()
