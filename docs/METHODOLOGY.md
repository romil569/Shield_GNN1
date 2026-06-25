# Methodology

SHIELD-GNN is designed for adversarially robust blockchain fraud detection on a heterogeneous graph containing transaction nodes and wallet/address nodes.

## Pipeline

1. Build the Elliptic++ Train200K heterogeneous graph.
2. Encode transaction and wallet/address node features.
3. Create temporal graph representations and snapshots.
4. Train baseline GNN models.
5. Simulate adversarial graph perturbations.
6. Apply robust edge pruning.
7. Train SHIELD-GNN with clean, defended, temporal, and pruning objectives.
8. Evaluate clean, attacked, and defended performance.

## Architecture Explanation

The architecture combines graph neural message passing, temporal consistency regularization, attack-aware evaluation, and defense-aware edge pruning. It is evaluated primarily as an integrated model rather than as a set of independently proven modules.

## Loss Objective

```text
L_total = L_clean + alpha L_defended + beta L_temporal + gamma L_prune
```

- `L_clean`: supervised fraud classification on the clean graph.
- `L_defended`: supervised objective on the defended/pruned graph.
- `L_temporal`: temporal consistency regularization.
- `L_prune`: pruning regularization for defense stability.
