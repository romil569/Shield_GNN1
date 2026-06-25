"""Build Elliptic++ Train200K homogeneous graph artifacts."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset_new"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ellipticpp_train200k" / "graph_artifacts"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
ORIGINAL_FULL_DIR = PROJECT_ROOT / "data" / "processed" / "full" / "graph_artifacts"
DATASET_NAME = "EllipticPlusPlus-Train200K"


def map_labels(series):
    """Map Elliptic++ class values to 0 licit, 1 illicit, -1 unknown."""
    numeric = pd.to_numeric(series, errors="coerce")
    labels = np.full(len(series), -1, dtype=np.int64)
    labels[numeric.to_numpy() == 1] = 1
    labels[numeric.to_numpy() == 2] = 0
    return labels


def normalize_features(frame, id_columns):
    """Return finite float32 feature values and feature column names."""
    feature_columns = [col for col in frame.columns if col not in id_columns]
    values = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(0.0).to_numpy(dtype=np.float32)
    return values, [str(col) for col in feature_columns]


def safe_read_csv(name):
    """Read a required CSV by name from dataset_new."""
    path = DATASET_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Required Elliptic++ file missing: {path}")
    return pd.read_csv(path)


def add_edges_from_frame(edge_frame, source_col, target_col, source_map, target_map, edges):
    """Map string/raw IDs to contiguous graph indices and append valid edges."""
    src = edge_frame[source_col].map(source_map)
    dst = edge_frame[target_col].map(target_map)
    valid = src.notna() & dst.notna()
    if valid.any():
        mapped = np.vstack(
            [
                src[valid].to_numpy(dtype=np.int64),
                dst[valid].to_numpy(dtype=np.int64),
            ]
        )
        edges.append(mapped)
    return int(valid.sum()), int((~valid).sum())


def unique_edge_index(edge_index):
    """Deduplicate a [2, E] edge index."""
    if edge_index.shape[1] == 0:
        return edge_index.astype(np.int64)
    pairs = np.unique(edge_index.T, axis=0)
    return pairs.T.astype(np.int64, copy=False)


def make_undirected(edge_index):
    """Create deduplicated undirected edges by adding reverse directions."""
    return unique_edge_index(np.concatenate([edge_index, edge_index[::-1]], axis=1))


def compute_graph_features(num_nodes, directed_edges, timesteps, train_labels):
    """Compute degree and train-label-only neighborhood summary features."""
    src, dst = directed_edges
    in_degree = np.bincount(dst, minlength=num_nodes).astype(np.float32)
    out_degree = np.bincount(src, minlength=num_nodes).astype(np.float32)
    total_degree = in_degree + out_degree
    max_degree = float(total_degree.max()) if total_degree.size and total_degree.max() > 0 else 1.0
    normalized_degree = total_degree / max_degree

    same_time = (timesteps[src] == timesteps[dst]).astype(np.float32) if directed_edges.shape[1] else np.array([], dtype=np.float32)
    same_timestep_degree = np.zeros(num_nodes, dtype=np.float32)
    if same_time.size:
        np.add.at(same_timestep_degree, src, same_time)
        np.add.at(same_timestep_degree, dst, same_time)

    train_known = train_labels != -1
    licit_neighbor = np.zeros(num_nodes, dtype=np.float32)
    illicit_neighbor = np.zeros(num_nodes, dtype=np.float32)
    known_neighbor = np.zeros(num_nodes, dtype=np.float32)
    if directed_edges.shape[1]:
        dst_labels = train_labels[dst]
        dst_known = train_known[dst].astype(np.float32)
        np.add.at(known_neighbor, src, dst_known)
        np.add.at(licit_neighbor, src, (dst_labels == 0).astype(np.float32))
        np.add.at(illicit_neighbor, src, (dst_labels == 1).astype(np.float32))

        src_labels = train_labels[src]
        src_known = train_known[src].astype(np.float32)
        np.add.at(known_neighbor, dst, src_known)
        np.add.at(licit_neighbor, dst, (src_labels == 0).astype(np.float32))
        np.add.at(illicit_neighbor, dst, (src_labels == 1).astype(np.float32))

    denom = np.maximum(total_degree, 1.0)
    unknown_neighbor_ratio = np.clip(1.0 - (known_neighbor / denom), 0.0, 1.0)
    licit_neighbor_ratio = licit_neighbor / denom
    illicit_neighbor_ratio = illicit_neighbor / denom

    graph_values = np.column_stack(
        [
            in_degree,
            out_degree,
            total_degree,
            normalized_degree,
            same_timestep_degree,
            unknown_neighbor_ratio,
            licit_neighbor_ratio,
            illicit_neighbor_ratio,
        ]
    ).astype(np.float32)
    names = [
        "in_degree",
        "out_degree",
        "total_degree",
        "normalized_degree",
        "same_timestep_degree",
        "unknown_neighbor_ratio_train_labels_only",
        "licit_neighbor_ratio_train_labels_only",
        "illicit_neighbor_ratio_train_labels_only",
    ]
    return graph_values, names


def chronological_split(tx_timestep, tx_labels, wallet_labels, tx_count, wallet_count, tx_to_wallet_edges, wallet_to_tx_edges):
    """Create transaction chronological split plus wallet train supervision."""
    train_mask = np.zeros(tx_count + wallet_count, dtype=bool)
    val_mask = np.zeros_like(train_mask)
    test_mask = np.zeros_like(train_mask)

    labeled_tx = np.flatnonzero(tx_labels != -1)
    unique_times = np.sort(np.unique(tx_timestep[labeled_tx]))
    if unique_times.size == 0:
        raise ValueError("No labeled transaction timesteps found for chronological split.")
    train_cut = max(1, int(np.ceil(unique_times.size * 0.70)))
    val_cut = max(train_cut + 1, int(np.ceil(unique_times.size * 0.85)))
    train_times = set(unique_times[:train_cut].tolist())
    val_times = set(unique_times[train_cut:val_cut].tolist())
    test_times = set(unique_times[val_cut:].tolist())

    train_tx = labeled_tx[np.isin(tx_timestep[labeled_tx], list(train_times))]
    val_tx = labeled_tx[np.isin(tx_timestep[labeled_tx], list(val_times))]
    test_tx = labeled_tx[np.isin(tx_timestep[labeled_tx], list(test_times))]
    train_mask[train_tx] = True
    val_mask[val_tx] = True
    test_mask[test_tx] = True

    labeled_wallet = np.flatnonzero(wallet_labels != -1) + tx_count
    touched_by_eval = np.zeros(tx_count + wallet_count, dtype=bool)
    eval_tx = np.zeros(tx_count, dtype=bool)
    eval_tx[val_tx] = True
    eval_tx[test_tx] = True
    for edge_index in [tx_to_wallet_edges, wallet_to_tx_edges]:
        if edge_index.size == 0:
            continue
        src, dst = edge_index
        incident_eval = np.zeros(src.shape[0], dtype=bool)
        tx_src = src < tx_count
        tx_dst = dst < tx_count
        incident_eval[tx_src] |= eval_tx[src[tx_src]]
        incident_eval[tx_dst] |= eval_tx[dst[tx_dst]]
        touched_by_eval[src[incident_eval]] = True
        touched_by_eval[dst[incident_eval]] = True

    strict_wallet_train = labeled_wallet[~touched_by_eval[labeled_wallet]]
    relaxed_wallet_train = labeled_wallet
    train_mask_strict = train_mask.copy()
    train_mask_strict[strict_wallet_train] = True
    train_mask_relaxed = train_mask.copy()
    train_mask_relaxed[relaxed_wallet_train] = True

    if int(train_mask_strict.sum()) >= 200_000:
        train_mask = train_mask_strict.copy()
        split_protocol = "chronological_transaction_split_with_strict_wallet_train"
        leakage_risk = "low"
    else:
        train_mask = train_mask_relaxed.copy()
        split_protocol = "chronological_transaction_split_with_relaxed_wallet_train200k"
        leakage_risk = (
            "moderate: strict wallet exclusion did not reach 200,000 train labels, "
            "so all labeled wallets are train-only while validation/test remain transactions."
        )

    return train_mask, val_mask, test_mask, train_mask_strict, train_mask_relaxed, {
        "split_protocol": split_protocol,
        "leakage_risk": leakage_risk,
        "unique_labeled_transaction_timesteps": int(unique_times.size),
        "train_timesteps": [int(x) for x in sorted(train_times)],
        "val_timesteps": [int(x) for x in sorted(val_times)],
        "test_timesteps": [int(x) for x in sorted(test_times)],
        "strict_wallet_train_labels": int(strict_wallet_train.size),
        "relaxed_wallet_train_labels": int(relaxed_wallet_train.size),
    }


def write_reports(summary, label_rows, split_rows, feature_schema, merge_report):
    """Save tabular and Markdown build reports."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(TABLES_DIR / "ellipticpp_train200k_dataset_summary.csv", index=False)
    (TABLES_DIR / "ellipticpp_train200k_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(label_rows).to_csv(TABLES_DIR / "ellipticpp_train200k_label_distribution.csv", index=False)
    pd.DataFrame(split_rows).to_csv(TABLES_DIR / "ellipticpp_train200k_split_distribution.csv", index=False)
    (TABLES_DIR / "ellipticpp_train200k_feature_schema.md").write_text(
        "# Elliptic++ Train200K Feature Schema\n\n"
        f"- Total feature dimension: {feature_schema['total_feature_dim']}\n"
        f"- Transaction feature block: {feature_schema['transaction_feature_dim']}\n"
        f"- Wallet feature block: {feature_schema['wallet_feature_dim']}\n"
        f"- Node type one-hot dim: 2\n"
        f"- Graph feature dim: {len(feature_schema['graph_feature_names'])}\n"
        f"- Timestep feature dim: {len(feature_schema['timestep_feature_names'])}\n\n"
        "Label-neighbor graph features use train labels only to avoid validation/test leakage.\n",
        encoding="utf-8",
    )
    (TABLES_DIR / "ellipticpp_train200k_merge_report.md").write_text(
        "# Elliptic++ Train200K Merge Report\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in merge_report.items())
        + "\n",
        encoding="utf-8",
    )


def main():
    """Build the Elliptic++ Train200K graph artifacts."""
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")
    if not ORIGINAL_FULL_DIR.is_dir():
        print(f"Warning: original full artifact directory was not found at {ORIGINAL_FULL_DIR}")

    print("Loading Elliptic++ CSV files...")
    tx_features = safe_read_csv("txs_features.csv")
    tx_classes = safe_read_csv("txs_classes.csv")
    tx_edges = safe_read_csv("txs_edgelist.csv")
    wallet_features = safe_read_csv("wallets_features.csv")
    wallet_classes = safe_read_csv("wallets_classes.csv")
    addr_tx = safe_read_csv("AddrTx_edgelist.csv")
    tx_addr = safe_read_csv("TxAddr_edgelist.csv")

    tx_features = tx_features.drop_duplicates("txId").reset_index(drop=True)
    wallet_features = wallet_features.drop_duplicates("address").reset_index(drop=True)
    tx_count = len(tx_features)
    wallet_count = len(wallet_features)
    num_nodes = tx_count + wallet_count

    tx_id_to_idx = pd.Series(np.arange(tx_count, dtype=np.int64), index=tx_features["txId"]).to_dict()
    wallet_id_to_idx = pd.Series(np.arange(wallet_count, dtype=np.int64) + tx_count, index=wallet_features["address"]).to_dict()

    print("Preparing labels and features...")
    tx_class_map = tx_classes.drop_duplicates("txId").set_index("txId")["class"].to_dict()
    wallet_class_map = wallet_classes.drop_duplicates("address").set_index("address")["class"].to_dict()
    tx_labels = map_labels(tx_features["txId"].map(tx_class_map))
    wallet_labels = map_labels(wallet_features["address"].map(wallet_class_map))
    y = np.concatenate([tx_labels, wallet_labels]).astype(np.int64)

    tx_timestep = pd.to_numeric(tx_features["Time step"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    wallet_timestep = pd.to_numeric(wallet_features["Time step"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    timesteps = np.concatenate([tx_timestep, wallet_timestep]).astype(np.int64)

    tx_values, tx_feature_names = normalize_features(tx_features, {"txId", "Time step"})
    wallet_values, wallet_feature_names = normalize_features(wallet_features, {"address", "Time step"})

    print("Building edge index...")
    edge_blocks = []
    valid_tt, missing_tt = add_edges_from_frame(tx_edges, "txId1", "txId2", tx_id_to_idx, tx_id_to_idx, edge_blocks)
    tx_to_wallet_blocks = []
    wallet_to_tx_blocks = []
    valid_at, missing_at = add_edges_from_frame(addr_tx, "input_address", "txId", wallet_id_to_idx, tx_id_to_idx, edge_blocks)
    add_edges_from_frame(addr_tx, "input_address", "txId", wallet_id_to_idx, tx_id_to_idx, wallet_to_tx_blocks)
    valid_ta, missing_ta = add_edges_from_frame(tx_addr, "txId", "output_address", tx_id_to_idx, wallet_id_to_idx, edge_blocks)
    add_edges_from_frame(tx_addr, "txId", "output_address", tx_id_to_idx, wallet_id_to_idx, tx_to_wallet_blocks)
    edge_index_directed = unique_edge_index(np.concatenate(edge_blocks, axis=1) if edge_blocks else np.empty((2, 0), dtype=np.int64))
    edge_index_undirected = make_undirected(edge_index_directed)
    tx_to_wallet_edges = np.concatenate(tx_to_wallet_blocks, axis=1) if tx_to_wallet_blocks else np.empty((2, 0), dtype=np.int64)
    wallet_to_tx_edges = np.concatenate(wallet_to_tx_blocks, axis=1) if wallet_to_tx_blocks else np.empty((2, 0), dtype=np.int64)

    train_mask, val_mask, test_mask, train_mask_strict, train_mask_relaxed, split_report = chronological_split(
        tx_timestep, tx_labels, wallet_labels, tx_count, wallet_count, tx_to_wallet_edges, wallet_to_tx_edges
    )
    train_labels_for_features = np.full_like(y, -1)
    train_labels_for_features[train_mask & (y != -1)] = y[train_mask & (y != -1)]
    graph_features, graph_feature_names = compute_graph_features(num_nodes, edge_index_directed, timesteps, train_labels_for_features)
    max_timestep = max(float(timesteps.max()), 1.0)
    timestep_features = np.column_stack([timesteps.astype(np.float32), timesteps.astype(np.float32) / max_timestep]).astype(np.float32)

    print("Assembling unified feature matrix...")
    X = np.zeros((num_nodes, tx_values.shape[1] + wallet_values.shape[1] + 2 + graph_features.shape[1] + timestep_features.shape[1]), dtype=np.float32)
    cursor = 0
    X[:tx_count, cursor:cursor + tx_values.shape[1]] = tx_values
    cursor += tx_values.shape[1]
    X[tx_count:, cursor:cursor + wallet_values.shape[1]] = wallet_values
    cursor += wallet_values.shape[1]
    X[:tx_count, cursor] = 1.0
    X[tx_count:, cursor + 1] = 1.0
    cursor += 2
    X[:, cursor:cursor + graph_features.shape[1]] = graph_features
    cursor += graph_features.shape[1]
    X[:, cursor:cursor + timestep_features.shape[1]] = timestep_features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    labeled_mask = y != -1
    unknown_mask = y == -1
    transaction_node_mask = np.zeros(num_nodes, dtype=bool)
    wallet_node_mask = np.zeros(num_nodes, dtype=bool)
    transaction_node_mask[:tx_count] = True
    wallet_node_mask[tx_count:] = True
    node_type = np.zeros(num_nodes, dtype=np.int64)
    node_type[tx_count:] = 1

    feature_schema = {
        "dataset_name": DATASET_NAME,
        "total_feature_dim": int(X.shape[1]),
        "transaction_feature_dim": int(tx_values.shape[1]),
        "wallet_feature_dim": int(wallet_values.shape[1]),
        "transaction_feature_names": tx_feature_names,
        "wallet_feature_names": wallet_feature_names,
        "node_type_feature_names": ["is_transaction", "is_wallet"],
        "graph_feature_names": graph_feature_names,
        "timestep_feature_names": ["raw_timestep", "normalized_timestep"],
    }
    merge_report = {
        "dataset_name": DATASET_NAME,
        "source_folder": str(DATASET_DIR),
        "wallet_wallet_edges_available": False,
        "valid_transaction_transaction_edges": valid_tt,
        "missing_transaction_transaction_edges": missing_tt,
        "valid_wallet_to_transaction_edges": valid_at,
        "missing_wallet_to_transaction_edges": missing_at,
        "valid_transaction_to_wallet_edges": valid_ta,
        "missing_transaction_to_wallet_edges": missing_ta,
        **split_report,
        "train200k_requirement_met": bool(int((train_mask & labeled_mask).sum()) >= 200_000),
        "original_full_artifacts_preserved": bool((ORIGINAL_FULL_DIR / "X.npy").is_file()),
    }
    summary = {
        "dataset_name": DATASET_NAME,
        "num_nodes": int(num_nodes),
        "num_transaction_nodes": int(tx_count),
        "num_wallet_nodes": int(wallet_count),
        "num_directed_edges": int(edge_index_directed.shape[1]),
        "num_undirected_edges": int(edge_index_undirected.shape[1]),
        "num_features": int(X.shape[1]),
        "num_timesteps": int(np.unique(timesteps).size),
        "labeled_nodes": int(labeled_mask.sum()),
        "unknown_nodes": int(unknown_mask.sum()),
        "train_labeled_nodes": int((train_mask & labeled_mask).sum()),
        "val_labeled_nodes": int((val_mask & labeled_mask).sum()),
        "test_labeled_nodes": int((test_mask & labeled_mask).sum()),
        "split_protocol": merge_report["split_protocol"],
        "leakage_risk": merge_report["leakage_risk"],
    }
    label_rows = [
        {"scope": "all", "licit": int((y == 0).sum()), "illicit": int((y == 1).sum()), "unknown": int((y == -1).sum())},
        {"scope": "transactions", "licit": int((tx_labels == 0).sum()), "illicit": int((tx_labels == 1).sum()), "unknown": int((tx_labels == -1).sum())},
        {"scope": "wallets", "licit": int((wallet_labels == 0).sum()), "illicit": int((wallet_labels == 1).sum()), "unknown": int((wallet_labels == -1).sum())},
    ]
    split_rows = [
        {"split": "train", "nodes": int(train_mask.sum()), "labeled": int((train_mask & labeled_mask).sum()), "wallet_nodes": int((train_mask & wallet_node_mask).sum())},
        {"split": "validation", "nodes": int(val_mask.sum()), "labeled": int((val_mask & labeled_mask).sum()), "wallet_nodes": int((val_mask & wallet_node_mask).sum())},
        {"split": "test", "nodes": int(test_mask.sum()), "labeled": int((test_mask & labeled_mask).sum()), "wallet_nodes": int((test_mask & wallet_node_mask).sum())},
    ]

    print(f"Saving artifacts to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    np.save(OUTPUT_DIR / "edge_index_directed.npy", edge_index_directed)
    np.save(OUTPUT_DIR / "edge_index_undirected.npy", edge_index_undirected)
    np.save(OUTPUT_DIR / "timesteps.npy", timesteps)
    np.save(OUTPUT_DIR / "train_mask.npy", train_mask)
    np.save(OUTPUT_DIR / "val_mask.npy", val_mask)
    np.save(OUTPUT_DIR / "test_mask.npy", test_mask)
    np.save(OUTPUT_DIR / "labeled_mask.npy", labeled_mask)
    np.save(OUTPUT_DIR / "unknown_mask.npy", unknown_mask)
    np.save(OUTPUT_DIR / "transaction_node_mask.npy", transaction_node_mask)
    np.save(OUTPUT_DIR / "wallet_node_mask.npy", wallet_node_mask)
    np.save(OUTPUT_DIR / "node_type.npy", node_type)
    np.save(OUTPUT_DIR / "train_mask_strict.npy", train_mask_strict)
    np.save(OUTPUT_DIR / "train_mask_train200k_relaxed.npy", train_mask_relaxed)
    (OUTPUT_DIR / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "node_mapping.json").write_text(
        json.dumps(
            {
                "transaction_id_to_node_index": {str(k): int(v) for k, v in tx_id_to_idx.items()},
                "wallet_address_to_node_index": {str(k): int(v) for k, v in wallet_id_to_idx.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "merge_report.json").write_text(json.dumps(merge_report, indent=2), encoding="utf-8")
    write_reports(summary, label_rows, split_rows, feature_schema, merge_report)

    print("Elliptic++ Train200K graph build complete.")
    print(json.dumps(summary, indent=2))
    if not merge_report["train200k_requirement_met"]:
        print("Warning: labeled training nodes are below 200,000; no labels were invented.", file=sys.stderr)


if __name__ == "__main__":
    main()
