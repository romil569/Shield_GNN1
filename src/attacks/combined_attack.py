"""Combined adversarial attack pipeline for transaction graphs."""

from src.attacks.attack_utils import calculate_attack_stats, validate_graph_arrays
from src.attacks.fake_edge_injection import inject_fake_edges
from src.attacks.fake_node_injection import inject_fake_nodes
from src.attacks.feature_perturbation import perturb_node_features


def run_combined_attack(
    X,
    y,
    edge_index,
    timesteps=None,
    fake_node_rate=0.02,
    fake_edge_rate=0.03,
    feature_perturb_rate=0.05,
    seed=42,
):
    """Apply fake node, fake edge, and feature perturbation attacks in sequence."""
    X, y, edge_index = validate_graph_arrays(X, y, edge_index)
    node_X, node_y, node_edges, fake_node_indices, node_summary = inject_fake_nodes(
        X,
        y,
        edge_index,
        timesteps=timesteps,
        injection_rate=fake_node_rate,
        seed=seed,
    )
    edge_index_attacked, fake_edges, edge_summary = inject_fake_edges(
        node_X,
        node_y,
        node_edges,
        injection_rate=fake_edge_rate,
        strategy="illicit_to_licit",
        seed=seed + 1,
    )
    attacked_X, perturbed_node_indices, feature_summary = perturb_node_features(
        node_X,
        node_y,
        perturbation_rate=feature_perturb_rate,
        strategy="illicit",
        seed=seed + 2,
    )
    combined_summary = {
        "attack_type": "combined_attack",
        "fake_node_summary": node_summary,
        "fake_edge_summary": edge_summary,
        "feature_perturbation_summary": feature_summary,
        "final_graph_stats": calculate_attack_stats(X, edge_index, attacked_X, edge_index_attacked),
        "fake_nodes_added": int(len(fake_node_indices)),
        "fake_edges_sampled": int(fake_edges.shape[1]),
        "perturbed_nodes": int(len(perturbed_node_indices)),
    }
    return attacked_X, node_y, edge_index_attacked, combined_summary
