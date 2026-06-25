# Experiments

## Baseline Models

- GCN on Elliptic++ Train200K
- GraphSAGE on Elliptic++ Train200K

## SHIELD-GNN Variants

- SHIELD-GNN Clean
- SHIELD-GNN Attack+Defense
- Elliptic++ 100-epoch component ablation variants

## Final Results

| Model | F1 | ROC-AUC |
| --- | ---: | ---: |
| GCN | 0.0269 | 0.4642 |
| GraphSAGE | 0.0508 | 0.5211 |
| SHIELD-GNN Clean | 0.1225 | 0.6859 |
| SHIELD-GNN Attack+Defense | 0.1007 | 0.6993 |

## Ablation Note

The Elliptic++ 100-epoch ablation shows stable architecture-level performance. The variants are numerically close, so the paper should not claim that every component is independently proven superior.

## Safe Interpretation

SHIELD-GNN improves over GCN and GraphSAGE on Elliptic++ Train200K, supporting the proposed integrated architecture for heterogeneous blockchain fraud detection.
