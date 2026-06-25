# Final Full-Dataset Results Summary

- Best model by F1: GCN (F1=0.2141).
- Best model by ROC-AUC: GCN (ROC-AUC=0.8098).
- Best model by recall: TemporalGCN-GRU (recall=0.8554).

The full Elliptic dataset remains highly imbalanced: the test split contains 408 illicit nodes and 8,433 licit nodes. Accuracy is therefore less informative than F1, recall, and ROC-AUC for fraud detection.

In this run, SHIELD-GNN uses the full graph context plus memory-safe temporal processing. Its attack+defense variant improves slightly over the clean SHIELD-GNN run on F1, recall, and ROC-AUC, but the strongest F1 and ROC-AUC are still from the static GCN baseline. This suggests the current SHIELD configuration is robustly executable but needs further tuning before it outperforms the simpler baseline.
