import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_result(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    base = Path("analysis")

    early   = load_result(base / "early"   / "evaluation_results.json")
    late    = load_result(base / "late"    / "evaluation_results.json")
    hybrid  = load_result(base / "hybrid"  / "evaluation_results.json")

    out = {
        "dataset": "pamap2",
        "modalities": ["imu_hand", "imu_chest", "imu_ankle", "heart_rate"],
        "results": {
            "early_fusion": {
                "accuracy":  early["test_accuracy"],
                "f1_macro":  early["test_f1_macro"],
                "ece":       early["ece"],
            },
            "late_fusion": {
                "accuracy":  late["test_accuracy"],
                "f1_macro":  late["test_f1_macro"],
                "ece":       late["ece"],
            },
            "hybrid_fusion": {
                "accuracy":  hybrid["test_accuracy"],
                "f1_macro":  hybrid["test_f1_macro"],
                "ece":       hybrid["ece"],
            },
        },
    }

    # Save JSON
    exp_dir = Path("experiments")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "fusion_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved to experiments/fusion_comparison.json")

    # Make comparison plot
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Early", "Late", "Hybrid"]
    accs   = [
        early["test_accuracy"],
        late["test_accuracy"],
        hybrid["test_accuracy"],
    ]
    f1s    = [
        early["test_f1_macro"],
        late["test_f1_macro"],
        hybrid["test_f1_macro"],
    ]

    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x - 0.15, accs, width=0.3, label="Accuracy")
    plt.bar(x + 0.15, f1s,  width=0.3, label="F1-macro")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Fusion Strategy Comparison (PAMAP2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "fusion_comparison.png")
    plt.close()
    print("Saved plot to analysis/fusion_comparison.png")


if __name__ == "__main__":
    main()