# Inference

A separate production inference CLI was not finalized in the original project. For model inspection and reproducible scoring, use:

```cmd
python src\evaluate_shield_gnn.py --artifact-dir data\processed\ellipticpp_train200k\graph_artifacts --checkpoint checkpoints\shield_clean_ellipticpp_train200k_100ep_memsafe_best.pt --device cuda
```
