# Dataset Documentation

## Elliptic++ Train200K Summary

| Item | Value |
| --- | ---: |
| Total nodes | 1,026,711 |
| Transaction nodes | 203,769 |
| Wallet/address nodes | 822,942 |
| Undirected edges | 3,005,230 |
| Feature dimension | 249 |
| Labeled nodes | 311,918 |

## Split Strategy

The final split uses transaction labels with a strict train/validation/test separation intended to reduce leakage from wallet/address connectivity.

| Split | Labeled nodes |
| --- | ---: |
| Train | 207,014 |
| Validation | 8,642 |
| Test | 6,687 |

## Local Data Layout

```text
datasets/raw/ellipticpp/
data/processed/ellipticpp_train200k/graph_artifacts/
```

Raw and processed arrays are excluded from Git. Small metadata is provided under `datasets/metadata/`.
