"""
Uncertainty Quantification for Multimodal Fusion

Implements methods for estimating and calibrating confidence scores:
1. MC Dropout for epistemic uncertainty
2. Calibration metrics (ECE, reliability diagrams)
3. Uncertainty-weighted fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty via variance.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10):
        """
        Args:
            model: The model to estimate uncertainty for
            num_samples: Number of MC dropout samples
        """
        super().__init__()
        self.model = model
        self.num_samples = num_samples
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Returns:
            mean_logits: (batch_size, num_classes) - mean prediction
            uncertainty: (batch_size,) - prediction uncertainty (variance)
        """
        # 1. Remember current mode and enable dropout (train mode)
        was_training = self.model.training
        self.model.train()

        logits_samples = []

        # 2. Run num_samples stochastic forward passes
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(*args, **kwargs)  # (B, C)
                logits_samples.append(logits.unsqueeze(0))  # (1, B, C)

        # Restore original mode
        if not was_training:
            self.model.eval()

        # Stack: (S, B, C)
        logits_samples = torch.cat(logits_samples, dim=0)

        # 3. Mean and variance of predictions
        mean_logits = logits_samples.mean(dim=0)  # (B, C)

        # Use variance of predicted probabilities as uncertainty
        probs_samples = F.softmax(logits_samples, dim=-1)  # (S, B, C)
        var_probs = probs_samples.var(dim=0)               # (B, C)
        uncertainty = var_probs.mean(dim=-1)               # (B,)

        return mean_logits, uncertainty


class CalibrationMetrics:
    """
    Compute calibration metrics for confidence estimates.
    
    Key metrics:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)  
    - Negative Log-Likelihood (NLL)
    """
    
    @staticmethod
    def expected_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ (|bin_accuracy - bin_confidence|) * (bin_size / total_size)
        """
        confidences = confidences.detach().cpu()
        predictions = predictions.detach().cpu()
        labels = labels.detach().cpu()

        N = confidences.numel()
        ece = torch.tensor(0.0)

        # Bin boundaries: [0,1], equal-width bins
        bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)

        for i in range(num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            # Include left edge, exclude right edge (except last bin)
            if i == num_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)

            bin_size = in_bin.sum().item()
            if bin_size == 0:
                continue

            bin_confidences = confidences[in_bin]
            bin_predictions = predictions[in_bin]
            bin_labels = labels[in_bin]

            avg_confidence = bin_confidences.mean()
            accuracy = (bin_predictions == bin_labels).float().mean()

            ece += (bin_size / N) * torch.abs(accuracy - avg_confidence)

        return float(ece.item())
    
    @staticmethod
    def maximum_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE = max_bin |bin_accuracy - bin_confidence|
        """
        confidences = confidences.detach().cpu()
        predictions = predictions.detach().cpu()
        labels = labels.detach().cpu()

        mce = torch.tensor(0.0)

        bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)

        for i in range(num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == num_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)

            bin_size = in_bin.sum().item()
            if bin_size == 0:
                continue

            bin_confidences = confidences[in_bin]
            bin_predictions = predictions[in_bin]
            bin_labels = labels[in_bin]

            avg_confidence = bin_confidences.mean()
            accuracy = (bin_predictions == bin_labels).float().mean()

            gap = torch.abs(accuracy - avg_confidence)
            if gap > mce:
                mce = gap

        return float(mce.item())
    
    @staticmethod
    def negative_log_likelihood(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute average Negative Log-Likelihood (NLL).
        
        NLL = -log P(y_true | x)
        """
        logits = logits.detach()
        labels = labels.detach()
        nll = F.cross_entropy(logits, labels, reduction='mean')
        return float(nll.item())
    
    @staticmethod
    def reliability_diagram(
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
        save_path: str = None
    ) -> None:
        """
        Plot reliability diagram showing calibration.
        """
        import matplotlib.pyplot as plt

        confidences = np.asarray(confidences)
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        N = len(confidences)
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == num_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)

            count = in_bin.sum()
            if count == 0:
                bin_accs.append(0.0)
                bin_confs.append(0.0)
                bin_counts.append(0)
                continue

            bin_conf = confidences[in_bin].mean()
            bin_acc = (predictions[in_bin] == labels[in_bin]).mean()

            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(count)

        bin_accs = np.asarray(bin_accs)
        bin_confs = np.asarray(bin_confs)
        bin_counts = np.asarray(bin_counts)

        # Compute ECE for annotation
        # reuse torch implementation
        ece = CalibrationMetrics.expected_calibration_error(
            torch.from_numpy(confidences),
            torch.from_numpy(predictions),
            torch.from_numpy(labels),
            num_bins=num_bins,
        )

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Bar plot: predicted confidence (x) vs accuracy (height)
        bin_width = 1.0 / num_bins
        ax.bar(
            bin_edges[:-1] + bin_width / 2.0,
            bin_accs,
            width=bin_width,
            edgecolor='black',
            align='center',
            alpha=0.7,
        )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle='--')

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Reliability Diagram (ECE = {ece:.4f})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.grid(True, linestyle='--', alpha=0.3)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)


class UncertaintyWeightedFusion(nn.Module):
    """
    Fuse modalities weighted by inverse uncertainty.
    
    Intuition: More uncertain modalities get lower weight.
    Weight_i ∝ 1 / (uncertainty_i + ε)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        modality_predictions: Dict[str, torch.Tensor],
        modality_uncertainties: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse modality predictions weighted by inverse uncertainty.
        
        Args:
            modality_predictions: Dict of {modality: logits}
                                Each tensor: (batch_size, num_classes)
            modality_uncertainties: Dict of {modality: uncertainty}
                                   Each tensor: (batch_size,)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            fused_logits: (batch_size, num_classes) - weighted fusion
            fusion_weights: (batch_size, num_modalities) - used weights
        """
        # Assume the mask's modality order matches keys order
        modality_names = list(modality_predictions.keys())
        batch_size = next(iter(modality_predictions.values())).shape[0]
        num_modalities = len(modality_names)

        device = next(iter(modality_predictions.values())).device
        modality_mask = modality_mask.to(device=device, dtype=torch.float32)

        # 1. Compute inverse uncertainty weights: w_i = 1 / (σ_i + ε)
        #    Shape for each: (B,)
        weights_list = []
        for i, name in enumerate(modality_names):
            sigma = modality_uncertainties[name].to(device)  # (B,)
            w = 1.0 / (sigma + self.epsilon)                 # (B,)
            # apply availability mask
            mask_i = modality_mask[:, i]                     # (B,)
            w = w * mask_i
            weights_list.append(w.unsqueeze(1))              # (B,1)

        weights = torch.cat(weights_list, dim=1)  # (B, M)

        # 2. Normalize weights to sum to 1
        weight_sums = weights.sum(dim=1, keepdim=True)  # (B,1)
        # Avoid division by zero: if all weights are 0 for a sample, keep them uniform
        zero_mask = (weight_sums == 0.0)
        if zero_mask.any():
            # Set uniform weights for those samples across available modalities
            # For simplicity, just set equal weights across all modalities
            weights[zero_mask.expand_as(weights)] = 1.0
            weight_sums = weights.sum(dim=1, keepdim=True)

        fusion_weights = weights / weight_sums  # (B, M)

        # 3. Fuse predictions: Σ w_i * pred_i
        num_classes = next(iter(modality_predictions.values())).shape[1]
        fused_logits = torch.zeros(
            batch_size, num_classes, device=device,
            dtype=next(iter(modality_predictions.values())).dtype
        )

        for i, name in enumerate(modality_names):
            logits_i = modality_predictions[name].to(device)  # (B,C)
            w_i = fusion_weights[:, i].unsqueeze(1)           # (B,1)
            fused_logits = fused_logits + w_i * logits_i

        return fused_logits, fusion_weights


class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling.
    
    Learns a single temperature parameter T that scales logits:
    P_calibrated = softmax(logits / T)
    
    Reference: Guo et al. "On Calibration of Modern Neural Networks", ICML 2017
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: (batch_size, num_classes) - model outputs
            
        Returns:
            scaled_logits: (batch_size, num_classes) - temperature-scaled logits
        """
        return logits / self.temperature
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> None:
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: (N, num_classes) - validation set logits
            labels: (N,) - validation set labels
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        self.train()
        self.temperature.data.fill_(1.0)

        logits = logits.detach()
        labels = labels.detach()
        device = logits.device
        self.to(device)

        # Use LBFGS as in the original paper, or Adam; we'll do LBFGS.
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter
        )

        def _closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        self.eval()


class EnsembleUncertainty:
    """
    Estimate uncertainty via ensemble of models.
    
    Train multiple models with different initializations/data splits.
    Uncertainty = variance across ensemble predictions.
    """
    
    def __init__(self, models: list):
        """
        Args:
            models: List of trained models (same architecture)
        """
        self.models = models
        self.num_models = len(models)
    
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and uncertainty from ensemble.
        
        Args:
            inputs: Model inputs
            
        Returns:
            mean_predictions: (batch_size, num_classes) - average prediction probs
            uncertainty: (batch_size,) - prediction variance (across ensemble)
        """
        device = inputs.device
        all_probs = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                model.to(device)
                logits = model(inputs)               # (B,C)
                probs = F.softmax(logits, dim=-1)    # (B,C)
                all_probs.append(probs.unsqueeze(0))  # (1,B,C)

        # (M,B,C)
        all_probs = torch.cat(all_probs, dim=0)

        # Mean prediction
        mean_probs = all_probs.mean(dim=0)  # (B,C)

        # Variance across models
        var_probs = all_probs.var(dim=0)    # (B,C)
        uncertainty = var_probs.mean(dim=-1)  # (B,)

        return mean_probs, uncertainty


def compute_calibration_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    num_bins: int = 15
) -> Dict[str, float]:
    """
    Compute all calibration metrics on a dataset.
    
    Args:
        model: Trained model (single-input classification model)
        dataloader: Test/validation dataloader
        device: Device to run on
        
    Returns:
        metrics: Dict with ECE, MCE, NLL, accuracy
    """
    model.eval()
    model.to(device)

    all_confidences = []
    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            # Assumes batch = (inputs, labels)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)                  # (B,C)
            probs = F.softmax(logits, dim=1)        # (B,C)
            confidences, predictions = torch.max(probs, dim=1)

            all_logits.append(logits.cpu())
            all_confidences.append(confidences.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    logits = torch.cat(all_logits)
    confidences = torch.cat(all_confidences)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    # Metrics
    ece = CalibrationMetrics.expected_calibration_error(
        confidences, predictions, labels, num_bins=num_bins
    )
    mce = CalibrationMetrics.maximum_calibration_error(
        confidences, predictions, labels, num_bins=num_bins
    )
    nll = CalibrationMetrics.negative_log_likelihood(logits, labels)
    accuracy = float((predictions == labels).float().mean().item())

    return {
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "accuracy": accuracy,
    }


if __name__ == '__main__':
    # Test calibration metrics
    print("Testing calibration metrics...")
    
    # Generate fake predictions
    num_samples = 1000
    num_classes = 10
    
    # Well-calibrated predictions (random logits)
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    
    # Test ECE
    try:
        ece = CalibrationMetrics.expected_calibration_error(
            confidences, predictions, labels
        )
        print(f"✓ ECE computed: {ece:.4f}")
    except NotImplementedError:
        print("✗ ECE not implemented yet")
    
    # Test reliability diagram
    try:
        CalibrationMetrics.reliability_diagram(
            confidences.numpy(),
            predictions.numpy(),
            labels.numpy(),
            save_path='test_reliability.png'
        )
        print("✓ Reliability diagram created")
    except NotImplementedError:
        print("✗ Reliability diagram not implemented yet")
