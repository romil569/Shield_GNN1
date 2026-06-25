# Elliptic++ Train200K Verification Report

Artifact directory: `C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\graph_artifacts`

## Checks

- all_required_artifacts_exist: PASS
- X_has_no_nan_or_infinite_values: PASS
- labels_are_only_minus1_0_1: PASS
- edge_indices_have_shape_2_by_edges: PASS
- edge_index_ids_are_in_range: PASS
- all_node_arrays_match_node_count: PASS
- transaction_and_wallet_masks_cover_all_nodes: PASS
- labeled_train_nodes_at_least_200000_if_possible: PASS
- validation_nodes_are_not_wallets: PASS
- test_nodes_are_not_wallets: PASS
- unknown_nodes_not_used_in_supervised_masks: PASS
- feature_dimension_matches_schema: PASS
- original_elliptic_full_artifacts_not_overwritten: PASS

## Summary

- num_nodes: 1026711
- num_features: 249
- num_directed_edges: 1548596
- num_undirected_edges: 3005230
- train_labeled_nodes: 207014
- val_labeled_nodes: 8642
- test_labeled_nodes: 6687
- transaction_nodes: 203769
- wallet_nodes: 822942
- split_protocol: chronological_transaction_split_with_strict_wallet_train
- graph_can_be_loaded_by_existing_training_scripts: True
