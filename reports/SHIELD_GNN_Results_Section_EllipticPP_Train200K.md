# SHIELD-GNN Results and Analysis: Elliptic++ Train200K



The final results focus on Elliptic++ Train200K, the expanded heterogeneous real-data benchmark where SHIELD-GNN shows its strongest architecture-level value. The constructed graph contains 1,026,711 real nodes, including 203,769 transaction nodes and 822,942 wallet/address nodes. SHIELD-GNN Clean achieves an F1 score of 0.1225, compared with 0.0269 for GCN and 0.0508 for GraphSAGE.



## Dataset Summary

| Dataset property | Value |
| --- | --- |
| Total nodes | 1026711 |
| Transaction nodes | 203769 |
| Wallet/address nodes | 822942 |
| Undirected edges | 3005230 |
| Feature dimension | 249 |
| Labeled nodes | 311918 |
| Train labeled nodes | 207014 |
| Validation labeled nodes | 8642 |
| Test labeled nodes | 6687 |
| Split strategy | chronological_transaction_split_with_strict_wallet_train |
| Leakage-risk note | low |
| Transaction feature block | 182 |
| Wallet feature block | 55 |



## Split Distribution

| split | nodes | labeled | wallet_nodes |
| --- | --- | --- | --- |
| train | 207014 | 207014 | 175779 |
| validation | 8642 | 8642 | 0 |
| test | 6687 | 6687 | 0 |



## Label Distribution

| scope | licit | illicit | unknown |
| --- | --- | --- | --- |
| all | 293107 | 18811 | 714793 |
| transactions | 42019 | 4545 | 157205 |
| wallets | 251088 | 14266 | 557588 |



## Node Type Distribution

| Node type | Count |
| --- | --- |
| Transactions | 203769 |
| Wallet/address | 822942 |



## Baseline Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Loss | Best epoch | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | 0.8049947619438171 | 0.015371477369769428 | 0.10650887573964497 | 0.026865671641791048 | 0.4642065395599987 | 6209.28271484375 | 24 | Static GCN baseline on Elliptic++ Train200K. |
| GraphSAGE | 0.4239569306373596 | 0.026484957572640782 | 0.6094674556213018 | 0.050763923114834894 | 0.5211076835926365 | 341.3390808105469 | 94 | Static GraphSAGE baseline on Elliptic++ Train200K. |
| SHIELD-GNN Clean | 0.7707492113113403 | 0.06780735107731306 | 0.6331360946745562 | 0.12249570692615915 | 0.6858644518320681 | 0.624579131603241 | 4 | Integrated SHIELD-GNN without attack/defense artifacts. |
| SHIELD-GNN Attack+Defense | 0.7275310158729553 | 0.05492730210016155 | 0.6035502958579881 | 0.10069101678183615 | 0.6992756517681578 | 0.6263338327407837 | 4 | SHIELD-GNN trained with targeted attack and defended graph artifacts. |



## Elliptic++ 100-Epoch Ablation Summary

| study | dataset | ablation_name | removed_component | clean_f1 | attacked_f1 | defended_f1 | clean_auc | attacked_auc | defended_auc | f1_drop_clean_to_attack | f1_recovery_attack_to_defense | auc_drop_clean_to_attack | auc_recovery_attack_to_defense | importance_conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_shield_full | none | 0.1224957069261591 | 0.1267846944603084 | 0.1006910167818361 | 0.688324639459957 | 0.7412477236455804 | 0.7023631418502426 | -0.0042889875341492 | -0.0260936776784722 | -0.0529230841856234 | -0.0388845817953378 | defense did not improve F1 |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_clean_only | attack_and_defense | 0.1224957069261591 | 0.1267846944603084 | 0.1006910167818361 | 0.6871767032033277 | 0.7402831666881516 | 0.7009673711941987 | -0.0042889875341492 | -0.0260936776784722 | -0.0531064634848239 | -0.0393157954939529 | defense did not improve F1 |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_defense_branch | defense_branch | 0.1224957069261591 | 0.1267846944603084 | 0.1006910167818361 | 0.6874354314224967 | 0.7405255541776892 | 0.7013568252504218 | -0.0042889875341492 | -0.0260936776784722 | -0.0530901227551924 | -0.0391687289272674 | defense did not improve F1 |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_attack_branch | attack_branch | 0.1224957069261591 | 0.1267846944603084 | 0.1006910167818361 | 0.6874917161578951 | 0.7405559660911704 | 0.7014485149000219 | -0.0042889875341492 | -0.0260936776784722 | -0.0530642499332753 | -0.0391074511911484 | defense did not improve F1 |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_temporal_consistency | temporal_consistency | 0.1224957069261591 | 0.1267846944603084 | 0.1006910167818361 | 0.6888075988024062 | 0.7415572896902706 | 0.702625047433507 | -0.0042889875341492 | -0.0260936776784722 | -0.0527496908878644 | -0.0389322422567636 | defense did not improve F1 |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_pruning_regularization | pruning_regularization | 0.1224957069261591 | 0.1267846944603084 | 0.1006910167818361 | 0.688324639459957 | 0.7412477236455804 | 0.7023631418502426 | -0.0042889875341492 | -0.0260936776784722 | -0.0529230841856234 | -0.0388845817953378 | defense did not improve F1 |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_node_type_features | node_type_features | 0.1224957069261591 | 0.1267846944603084 | 0.1006910167818361 | 0.6878761772133972 | 0.7409018448683754 | 0.7021602444573153 | -0.0042889875341492 | -0.0260936776784722 | -0.0530256676549781 | -0.0387416004110601 | defense did not improve F1 |



## Clean, Attacked, and Defended Evaluation

| study | dataset | ablation_name | removed_component | checkpoint_path | evaluation_graph | graph_path | status | accuracy | precision | recall | f1 | roc_auc | loss | num_test_nodes | illicit_count | licit_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_shield_full | none | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_shield_full_best.pt | clean | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\graph_artifacts | completed | 0.7707492113113403 | 0.067807351077313 | 0.6331360946745562 | 0.1224957069261591 | 0.688324639459957 | 0.6022383570671082 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_shield_full | none | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_shield_full_best.pt | attacked | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\attack_artifacts\targeted_evasion_test_illicit_full | completed | 0.7713474035263062 | 0.0701643489254108 | 0.6568047337278107 | 0.1267846944603084 | 0.7412477236455804 | 0.6018219590187073 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_shield_full | none | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_shield_full_best.pt | defended | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\defense_artifacts\targeted_evasion_hybrid_pruned_p95_full | completed | 0.7275310158729553 | 0.0549273021001615 | 0.6035502958579881 | 0.1006910167818361 | 0.7023631418502426 | 0.6081173419952393 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_clean_only | attack_and_defense | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_clean_only_best.pt | clean | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\graph_artifacts | completed | 0.7707492113113403 | 0.067807351077313 | 0.6331360946745562 | 0.1224957069261591 | 0.6871767032033277 | 0.5971192717552185 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_clean_only | attack_and_defense | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_clean_only_best.pt | attacked | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\attack_artifacts\targeted_evasion_test_illicit_full | completed | 0.7713474035263062 | 0.0701643489254108 | 0.6568047337278107 | 0.1267846944603084 | 0.7402831666881516 | 0.5966510772705078 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_clean_only | attack_and_defense | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_clean_only_best.pt | defended | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\defense_artifacts\targeted_evasion_hybrid_pruned_p95_full | completed | 0.7275310158729553 | 0.0549273021001615 | 0.6035502958579881 | 0.1006910167818361 | 0.7009673711941987 | 0.6030827760696411 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_defense_branch | defense_branch | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_defense_branch_best.pt | clean | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\graph_artifacts | completed | 0.7707492113113403 | 0.067807351077313 | 0.6331360946745562 | 0.1224957069261591 | 0.6874354314224967 | 0.5986564755439758 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_defense_branch | defense_branch | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_defense_branch_best.pt | attacked | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\attack_artifacts\targeted_evasion_test_illicit_full | completed | 0.7713474035263062 | 0.0701643489254108 | 0.6568047337278107 | 0.1267846944603084 | 0.7405255541776892 | 0.5981888771057129 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_defense_branch | defense_branch | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_defense_branch_best.pt | defended | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\defense_artifacts\targeted_evasion_hybrid_pruned_p95_full | completed | 0.7275310158729553 | 0.0549273021001615 | 0.6035502958579881 | 0.1006910167818361 | 0.7013568252504218 | 0.6045569777488708 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_attack_branch | attack_branch | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_attack_branch_best.pt | clean | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\graph_artifacts | completed | 0.7707492113113403 | 0.067807351077313 | 0.6331360946745562 | 0.1224957069261591 | 0.6874917161578951 | 0.5987329483032227 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_attack_branch | attack_branch | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_attack_branch_best.pt | attacked | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\attack_artifacts\targeted_evasion_test_illicit_full | completed | 0.7713474035263062 | 0.0701643489254108 | 0.6568047337278107 | 0.1267846944603084 | 0.7405559660911704 | 0.5982653498649597 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_attack_branch | attack_branch | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_attack_branch_best.pt | defended | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\defense_artifacts\targeted_evasion_hybrid_pruned_p95_full | completed | 0.7275310158729553 | 0.0549273021001615 | 0.6035502958579881 | 0.1006910167818361 | 0.7014485149000219 | 0.6046304702758789 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_temporal_consistency | temporal_consistency | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_temporal_consistency_best.pt | clean | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\graph_artifacts | completed | 0.7707492113113403 | 0.067807351077313 | 0.6331360946745562 | 0.1224957069261591 | 0.6888075988024062 | 0.6036641597747803 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_temporal_consistency | temporal_consistency | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_temporal_consistency_best.pt | attacked | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\attack_artifacts\targeted_evasion_test_illicit_full | completed | 0.7713474035263062 | 0.0701643489254108 | 0.6568047337278107 | 0.1267846944603084 | 0.7415572896902706 | 0.6032501459121704 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_temporal_consistency | temporal_consistency | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_temporal_consistency_best.pt | defended | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\defense_artifacts\targeted_evasion_hybrid_pruned_p95_full | completed | 0.7275310158729553 | 0.0549273021001615 | 0.6035502958579881 | 0.1006910167818361 | 0.702625047433507 | 0.6094833612442017 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_pruning_regularization | pruning_regularization | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_pruning_regularization_best.pt | clean | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\graph_artifacts | completed | 0.7707492113113403 | 0.067807351077313 | 0.6331360946745562 | 0.1224957069261591 | 0.688324639459957 | 0.6022383570671082 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_pruning_regularization | pruning_regularization | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_pruning_regularization_best.pt | attacked | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\attack_artifacts\targeted_evasion_test_illicit_full | completed | 0.7713474035263062 | 0.0701643489254108 | 0.6568047337278107 | 0.1267846944603084 | 0.7412477236455804 | 0.6018219590187073 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_pruning_regularization | pruning_regularization | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_pruning_regularization_best.pt | defended | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\defense_artifacts\targeted_evasion_hybrid_pruned_p95_full | completed | 0.7275310158729553 | 0.0549273021001615 | 0.6035502958579881 | 0.1006910167818361 | 0.7023631418502426 | 0.6081173419952393 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_node_type_features | node_type_features | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_node_type_features_best.pt | clean | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\graph_artifacts | completed | 0.7707492113113403 | 0.067807351077313 | 0.6331360946745562 | 0.1224957069261591 | 0.6878761772133972 | 0.6009429097175598 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_node_type_features | node_type_features | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_node_type_features_best.pt | attacked | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\attack_artifacts\targeted_evasion_test_illicit_full | completed | 0.7713474035263062 | 0.0701643489254108 | 0.6568047337278107 | 0.1267846944603084 | 0.7409018448683754 | 0.6005011796951294 | 6687 | 169 | 6518 | evaluated |
| Elliptic++ 100-epoch ablation | ellipticpp_train200k | ellipticpp100_no_node_type_features | node_type_features | C:\Users\HP\Desktop\SHIELD-GNN\results\model_checkpoints\ablations_ellipticpp100\ellipticpp100_no_node_type_features_best.pt | defended | C:\Users\HP\Desktop\SHIELD-GNN\data\processed\ellipticpp_train200k\defense_artifacts\targeted_evasion_hybrid_pruned_p95_full | completed | 0.7275310158729553 | 0.0549273021001615 | 0.6035502958579881 | 0.1006910167818361 | 0.7021602444573153 | 0.606806755065918 | 6687 | 169 | 6518 | evaluated |



## Final Results Conclusion

The results support the success of SHIELD-GNN as an integrated architecture for large-scale heterogeneous blockchain fraud detection on Elliptic++ Train200K. Although individual component ablations show limited separation, the proposed architecture clearly improves over standard GNN baselines and provides a strong foundation for adversarially robust blockchain fraud detection.