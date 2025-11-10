"""
Experiment 3: Confidence Calibration Audit

- Loads a trained checkpoint (early / late / hybrid)
- Evaluates on test set
- Computes: ECE, MCE, NLL, Accuracy
- Saves results into: experiments/uncertainty.json
- Saves reliability diagram: analysis/calibration_<fusion>.png
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- SAFE GLOBALS FOR TORCH LOAD (same as eval.py) ---
import torch.serialization
import typing
import collections
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

torch.serialization.add_safe_globals([
    DictConfig,
    ListConfig,
    ContainerMetadata,
    Metadata,
    AnyNode,
    typing.Any,
    dict,
    list,
    int,
    float,
    str,
    bool,
    tuple,
    type(None),
    collections.defaultdict,
])
# -----------------------------------------------------

from train import MultimodalFusionModule
from data import create_dataloaders
from uncertainty import CalibrationMetrics


def collect_logits_labels(model, dataloader, device="cpu"):
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting logits for calibration"):
            features, labels, mask = batch
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            mask = mask.to(device)

            logits = model(features, mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return all_logits, all_labels


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibration / Uncertainty experiment")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_json", type=str, default="experiments/uncertainty.json",
                        help="Path to JSON file to store calibration results")
    parser.add_argument("--calib_plot", type=str, default=None,
                        help="Path to save reliability diagram (PNG)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    print(f"Loading model from: {ckpt_path}")
    model = MultimodalFusionModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(args.device)

    # get config to rebuild dataloader
    config = model.config
    dataset_cfg = config.dataset

    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        dataset_name=dataset_cfg.name,
        data_dir=dataset_cfg.data_dir,
        modalities=dataset_cfg.modalities,
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
    )

    # 1) collect logits + labels
    logits, labels = collect_logits_labels(model, test_loader, device=args.device)
    probs = F.softmax(logits, dim=1)
    confidences, preds = torch.max(probs, dim=1)

    # 2) compute metrics
    accuracy = (preds == labels).float().mean().item()
    ece = CalibrationMetrics.expected_calibration_error(
        confidences, preds, labels, num_bins=15
    )
    mce = CalibrationMetrics.maximum_calibration_error(
        confidences, preds, labels, num_bins=15
    )
    nll = CalibrationMetrics.negative_log_likelihood(logits, labels)

    print("\n=== Calibration / Uncertainty Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ECE     : {ece:.4f}")
    print(f"MCE     : {mce:.4f}")
    print(f"NLL     : {nll:.4f}")

    # 3) save reliability diagram
    fusion_type = config.model.fusion_type
    if args.calib_plot is None:
        calib_plot_path = Path("analysis") / f"calibration_{fusion_type}.png"
    else:
        calib_plot_path = Path(args.calib_plot)
    calib_plot_path.parent.mkdir(parents=True, exist_ok=True)

    CalibrationMetrics.reliability_diagram(
        confidences.numpy(),
        preds.numpy(),
        labels.numpy(),
        num_bins=15,
        save_path=str(calib_plot_path),
    )
    print(f"Reliability diagram saved to: {calib_plot_path}")

    # 4) update experiments/uncertainty.json
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if out_json.is_file():
        with open(out_json, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # if file is corrupted or empty, start fresh
                data = {}
    else:
        data = {}

    # ensure basic structure
    if "dataset" not in data:
        data["dataset"] = dataset_cfg.name
    if "results" not in data or not isinstance(data["results"], dict):
        data["results"] = {}

    # insert / overwrite for this fusion type
    data["results"][fusion_type] = {
        "accuracy": accuracy,
        "ece": ece,
        "mce": mce,
        "nll": nll,
    }

    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated calibration results in: {out_json}")


if __name__ == "__main__":
    main()
