# Dataset: Elliptic++ Train200K

This repository is centered on the Elliptic++ Train200K benchmark.

| Item | Value |
| --- | ---: |
| Total nodes | 1,026,711 |
| Transaction nodes | 203,769 |
| Wallet/address nodes | 822,942 |
| Undirected edges | 3,005,230 |
| Feature dimension | 249 |
| Labeled nodes | 311,918 |
| Train labeled nodes | 207,014 |
| Validation labeled nodes | 8,642 |
| Test labeled nodes | 6,687 |

Large raw and processed files are not redistributed. Download the original Elliptic++ files separately and place them in:

```text
datasets/raw/ellipticpp/
```

Expected processed artifact destination:

```text
data/processed/ellipticpp_train200k/graph_artifacts/
```

Build and verify:

```cmd
python src\build_ellipticpp_train200k_graph.py --input-dir datasets\raw\ellipticpp --output-dir data\processed\ellipticpp_train200k\graph_artifacts
python scripts\verify_ellipticpp_train200k_graph.py
```

Small metadata files are included in `datasets/metadata/`.
