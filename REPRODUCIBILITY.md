# Reproducibility Guide

## Environment Setup

```cmd
py -3.12 -m venv shield_env
shield_env\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Install a GPU-compatible PyTorch build if CUDA training is required.

## Dataset Download and Preparation

Large raw files are not redistributed. Download the Elliptic++ source files separately and place them under `datasets/raw/ellipticpp/`.

```cmd
python src\build_ellipticpp_train200k_graph.py --input-dir datasets\raw\ellipticpp --output-dir data\processed\ellipticpp_train200k\graph_artifacts
python scripts\verify_ellipticpp_train200k_graph.py
```

## Training Commands

```cmd
python src\train_baseline.py --model gcn --artifact-dir data\processed\ellipticpp_train200k\graph_artifacts --epochs 100 --hidden-channels 32 --num-layers 1 --dropout 0.5 --lr 0.001 --device cuda
python src\train_baseline.py --model graphsage --artifact-dir data\processed\ellipticpp_train200k\graph_artifacts --epochs 100 --hidden-channels 32 --num-layers 1 --dropout 0.5 --lr 0.001 --device cuda
python src\train_shield_gnn.py --artifact-dir data\processed\ellipticpp_train200k\graph_artifacts --epochs 100 --hidden-channels 32 --num-layers 1 --dropout 0.5 --lr 0.001 --device cuda --save-name shield_clean_ellipticpp_train200k_100ep_memsafe
python src\train_shield_gnn.py --artifact-dir data\processed\ellipticpp_train200k\graph_artifacts --epochs 100 --hidden-channels 32 --num-layers 1 --dropout 0.5 --lr 0.001 --device cuda --attack-mode combined --defense-mode prune --save-name shield_attack_defense_ellipticpp_train200k_100ep_memsafe
```

## Evaluation Commands

```cmd
python src\evaluate_shield_gnn.py --artifact-dir data\processed\ellipticpp_train200k\graph_artifacts --checkpoint checkpoints\shield_clean_ellipticpp_train200k_100ep_memsafe_best.pt --device cuda
python src\evaluate_robustness_table.py --artifact-dir data\processed\ellipticpp_train200k\graph_artifacts --device cuda
```

## Expected Final Metrics

| Model | F1 | ROC-AUC |
| --- | ---: | ---: |
| GCN | 0.0269 | 0.4642 |
| GraphSAGE | 0.0508 | 0.5211 |
| SHIELD-GNN Clean | 0.1225 | 0.6859 |
| SHIELD-GNN Attack+Defense | 0.1007 | 0.6993 |

## Checkpoint Handling

Final small checkpoints are included under `checkpoints/`. Older full-Elliptic and exploratory ablation checkpoints are intentionally excluded.

## Regenerating Figures and Reports

```cmd
python src\generate_results_section_docx.py
```

## Known Limitations

The strongest final story is Elliptic++ Train200K. SHIELD-GNN improves over GCN and GraphSAGE, but the 100-epoch ablation does not strongly prove every component is independently decisive. The defended variant achieves the strongest ROC-AUC but lower F1 than SHIELD-GNN Clean.
