"""Generate final experiment command files without running training."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "run_final_training_commands.txt"


def latest_artifact_dir(parent, pattern):
    """Return the latest matching artifact path relative to the project root."""
    matches = sorted(parent.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        return None
    return matches[0].relative_to(PROJECT_ROOT)


def main():
    """Write CMD-compatible experiment commands."""
    attack_dir = latest_artifact_dir(PROJECT_ROOT / "data" / "processed" / "attack_artifacts", "combined_*")
    defense_dir = latest_artifact_dir(PROJECT_ROOT / "data" / "processed" / "defense_artifacts", "*hybrid_pruned")
    attack_arg = str(attack_dir).replace("/", "\\") if attack_dir else r"data\processed\attack_artifacts\combined_latest"
    defense_arg = str(defense_dir).replace("/", "\\") if defense_dir else r"data\processed\defense_artifacts\combined_hybrid_pruned"

    full_artifacts = r"data\processed\full\graph_artifacts"

    commands = rf"""cd C:\Users\HP\Desktop\SHIELD-GNN
shield_env\Scripts\activate

python src\train_baseline.py --model gcn --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 64 --num-layers 2 --dropout 0.5 --lr 0.001 --device cuda --save-name baseline_gcn_full_100ep
python src\train_baseline.py --model graphsage --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 64 --num-layers 2 --dropout 0.5 --lr 0.001 --device cuda --save-name baseline_graphsage_full_100ep
python src\train_baseline.py --model gat --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 32 --num-layers 2 --dropout 0.5 --lr 0.001 --device cuda --save-name baseline_gat_full_100ep

python src\train_temporal.py --model temporal_gcn_gru --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 32 --num-layers 1 --dropout 0.5 --lr 0.001 --device cuda --snapshot-mode current --temporal-grad-mode detach --detach-memory-each-step --save-name temporal_gcn_gru_full_100ep_memsafe
python src\train_temporal.py --model time_aware_gcn --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 64 --num-layers 2 --dropout 0.5 --lr 0.001 --device cuda

python src\run_attack_simulation.py --attack combined --fake-node-rate 0.02 --fake-edge-rate 0.03 --feature-perturb-rate 0.05 --save-artifacts
python src\run_edge_pruning_defense.py --input-type clean --mode hybrid --save-artifacts --output-name clean_hybrid_pruned

python src\train_shield_gnn.py --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 64 --num-layers 2 --dropout 0.5 --lr 0.001 --device cuda --save-name shield_clean_full_100ep
python src\train_shield_gnn.py --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 64 --num-layers 2 --dropout 0.5 --lr 0.001 --device cuda --use-attack --attack-dir {attack_arg} --use-defense --defense-dir {defense_arg} --save-name shield_attack_defense_full_100ep

python src\evaluate_shield_gnn.py --checkpoint results\model_checkpoints\shield_clean_full_100ep_best.pt
python src\evaluate_shield_gnn.py --checkpoint results\model_checkpoints\shield_attack_defense_full_100ep_best.pt --attack-dir {attack_arg} --defense-dir {defense_arg}

REM CPU fallback references:
REM python src\train_baseline.py --model gcn --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 64 --num-layers 2 --dropout 0.5 --lr 0.001 --device cpu --save-name baseline_gcn_full_100ep_cpu
REM python src\train_temporal.py --model temporal_gcn_gru --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 32 --num-layers 1 --dropout 0.5 --lr 0.001 --device cpu --snapshot-mode current --temporal-grad-mode detach --detach-memory-each-step --save-name temporal_gcn_gru_full_100ep_cpu
REM Optional cumulative TemporalGCN-GRU reference:
REM python src\train_temporal.py --model temporal_gcn_gru --artifact-dir {full_artifacts} --epochs 100 --hidden-channels 32 --num-layers 1 --dropout 0.5 --lr 0.001 --device cuda --snapshot-mode cumulative --temporal-grad-mode detach --detach-memory-each-step --save-name temporal_gcn_gru_full_100ep_cumulative_memsafe
"""
    OUTPUT_PATH.write_text(commands, encoding="utf-8")
    print(f"Generated command file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
