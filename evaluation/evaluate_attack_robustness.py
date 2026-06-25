"""Placeholder-safe evaluator for future clean-vs-attacked robustness checks."""

import argparse
from pathlib import Path


def parse_args():
    """Parse robustness evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained models on attacked graphs.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--attack-dir", default=None)
    return parser.parse_args()


def main():
    """Exit safely until trained checkpoints are available."""
    args = parse_args()
    if not args.checkpoint:
        print("No checkpoint provided. Robustness evaluation skipped safely.")
        print("Train a model manually first, then pass --checkpoint and --attack-dir.")
        return
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    if args.attack_dir and not Path(args.attack_dir).is_dir():
        print(f"Attack artifact directory not found: {args.attack_dir}")
        return
    print("Checkpoint found. Full robustness evaluation will be implemented after trained models exist.")


if __name__ == "__main__":
    main()
