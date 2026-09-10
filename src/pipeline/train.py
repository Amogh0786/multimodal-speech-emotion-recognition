import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import hydra
from omegaconf import DictConfig
import wandb
from loguru import logger
from tqdm import tqdm
import os
import numpy as np

from src.data.dataset import create_folds, RAVDESSDataset
from src.data.preprocessor import AudioPreprocessor
from src.data.augmentations import AudioAugmenter
from src.models.wav2vec_classifier import Wav2Vec2Classifier
from src.models.feature_extractors import ResNet18SpectrogramExtractor
from src.utils.metrics import compute_metrics
from src.utils.logger import setup_logger

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()

@hydra.main(version_base="1.3", config_path="../../configs", config_name="base_config")
def main(cfg: DictConfig):
    setup_logger(os.path.join(cfg.logging.log_dir, "train.log"))
    
    if cfg.logging.use_wandb:
        wandb.init(project=cfg.logging.wandb_project, config=dict(cfg))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data Setup
    logger.info("Setting up data pipeline...")
    df = create_folds(cfg.data.raw_dir, n_folds=cfg.data.n_folds)
    
    # Train on Fold 0 for simplicity in this script, or loop over folds
    train_df = df[df['fold'] != 0]
    val_df = df[df['fold'] == 0]
    
    preprocessor = AudioPreprocessor(target_sr=cfg.data.sample_rate, duration=cfg.data.duration)
    augmenter = AudioAugmenter(sample_rate=cfg.data.sample_rate)
    
    train_dataset = RAVDESSDataset(train_df, preprocessor, augmenter)
    val_dataset = RAVDESSDataset(val_df, preprocessor, None)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    # Model Setup
    logger.info(f"Initializing model: {cfg.model.type}")
    if cfg.model.type == "wav2vec2":
        model = Wav2Vec2Classifier(
            model_name=cfg.model.backbone_name,
            n_classes=cfg.data.n_classes,
            freeze_feature_extractor=cfg.model.freeze_feature_extractor,
            dropout=cfg.model.dropout
        ).to(device)
        
        # Differential learning rates
        optimizer = torch.optim.AdamW([
            {'params': model.wav2vec2.parameters(), 'lr': cfg.training.learning_rate_backbone},
            {'params': model.pooling.parameters(), 'lr': cfg.training.learning_rate_head},
            {'params': model.classifier.parameters(), 'lr': cfg.training.learning_rate_head}
        ], weight_decay=cfg.training.weight_decay)
        
    elif cfg.model.type == "resnet18":
        model = ResNet18SpectrogramExtractor(
            n_classes=cfg.data.n_classes,
            dropout=cfg.model.dropout
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate_head, weight_decay=cfg.training.weight_decay)
        
    # Loss fn
    if cfg.training.loss == "label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing_alpha)
    else:
        criterion = FocalLoss(gamma=cfg.training.focal_loss_gamma)
        
    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)
    
    # Training Loop
    logger.info("Starting training...")
    best_val_f1 = 0.0
    
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                if cfg.model.type == "resnet18":
                    outputs = model(inputs, augment=True)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Val]"):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        metrics = compute_metrics(all_preds, all_labels)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, UAR: {metrics['uar']:.4f}")
        
        if cfg.logging.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": metrics['macro_f1'],
                "val_uar": metrics['uar'],
                "lr": scheduler.get_last_lr()[0]
            })
            
        if metrics['macro_f1'] > best_val_f1:
            best_val_f1 = metrics['macro_f1']
            torch.save(model.state_dict(), "best_model.pth")
            logger.info("Saved new best model!")
            
    if cfg.logging.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
