"""
Training Script for Multimodal Sensor Fusion

Uses PyTorch Lightning for training with Hydra configuration.
Most infrastructure is provided - students need to integrate their fusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json

from data import create_dataloaders
from fusion import build_fusion_model
from encoders import build_encoder

# --- Add near the top of src/train.py, after imports ---
import torch.serialization
import typing
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata

torch.serialization.add_safe_globals([
    DictConfig,
    ListConfig,
    ContainerMetadata,
    typing.Any,
    dict,
])

class MultimodalFusionModule(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal fusion training.
    
    Handles training loop, validation, and logging.
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra configuration object
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Build encoders for each modality
        self.encoders = nn.ModuleDict()
        modality_output_dims = {}
        
        for modality in config.dataset.modalities:
            encoder_config = config.model.encoders.get(modality, {})
            
            # --- FIX: make heart_rate 1D, others 17D ---
            if modality == "heart_rate":
                input_dim = 1
            else:
                input_dim = encoder_config.get('input_dim', 17)

            output_dim = config.model.output_dim
            
            self.encoders[modality] = build_encoder(
                modality=modality,
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_config=encoder_config
            )
            modality_output_dims[modality] = output_dim
        
        # Build fusion model
        self.fusion_model = build_fusion_model(
            fusion_type=config.model.fusion_type,
            modality_dims=modality_output_dims,
            num_classes=config.dataset.get('num_classes', 18),
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.get('num_heads', 4),
            dropout=config.model.dropout
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.train_metrics = []
        self.val_metrics = []
    
    def forward(self, features, mask=None):
        """Forward pass through encoders and fusion model."""
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in features:
                encoded_features[modality] = encoder(features[modality])
        
        output = self.fusion_model(encoded_features, mask)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        features, labels, mask = batch
        logits = self(features, mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        features, labels, mask = batch
        logits = self(features, mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc, 'preds': preds,
                'labels': labels, 'confidences': confidences}
    
    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        features, labels, mask = batch
        logits = self(features, mask)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)
        self.log('test/acc', acc, on_epoch=True)
        return {'preds': preds, 'labels': labels, 'confidences': confidences}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.config.training.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        if self.config.training.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate / 100
            )
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        elif self.config.training.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1)
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        else:
            return optimizer


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """Main training function."""
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)
    
    pl.seed_everything(config.seed)
    save_dir = Path(config.experiment.save_dir) / config.experiment.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        modalities=config.dataset.modalities,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        modality_dropout=config.training.augmentation.modality_dropout
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\nCreating model...")
    model = MultimodalFusionModule(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename='{epoch}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=config.experiment.save_top_k,
        save_last=True
    )
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=config.training.early_stopping_patience,
        mode='min',
        verbose=True
    )
    logger = TensorBoardLogger(save_dir=save_dir, name='logs')
    
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='auto',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=config.experiment.log_every_n_steps,
        gradient_clip_val=config.training.gradient_clip_norm,
        deterministic=True,
        enable_progress_bar=True
    )
    
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\nTesting best model...")
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded state dict.")
    print("  Missing keys   :", missing)
    print("  Unexpected keys:", unexpected)

    trainer.test(model, test_loader, ckpt_path=None)
    
    results = {
        'best_model_path': str(best_model_path),
        'best_val_loss': float(checkpoint_callback.best_model_score),
        'config': OmegaConf.to_container(config, resolve=True)
    }
    results_file = save_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Results saved to: {results_file}")
    print(f"Best model: {best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()
