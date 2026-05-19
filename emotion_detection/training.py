#!/usr/bin/env python3
"""
Research-Grade Multi-Label Emotion Classification Training
==========================================================

CRITICAL FIXES FROM PREVIOUS APPROACH:
1. ✅ True multi-label architecture (BCEWithLogitsLoss, not CrossEntropy)
2. ✅ Asymmetric Loss for severe class imbalance
3. ✅ Class-specific threshold optimization
4. ✅ Proper multi-label metrics (per-label F1, Hamming Loss)
5. ✅ Contrastive learning for emotion relationships
6. ✅ Two-stage training (balanced pre-train → full fine-tune)
7. ✅ R-Drop regularization for generalization

TARGET: >80% Macro F1 for research publication

Author: Research-Optimized for Multi-Label Imbalance
Version: 5.0 - Publication-Ready
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import random
import gc
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
warnings.filterwarnings('ignore')

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    DebertaV2Config,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    hamming_loss, accuracy_score, jaccard_score,
    multilabel_confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class Config:
    """Research-optimized configuration for multi-label emotion classification"""
    
    # Model
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 256
    NUM_LABELS = 21  # Will be set from data
    
    # Two-Stage Training Strategy
    USE_TWO_STAGE = True  # Stage 1: Balanced subset, Stage 2: Full data
    STAGE1_EPOCHS = 5  # Pre-train on balanced data
    STAGE2_EPOCHS = 15  # Fine-tune on full data
    
    # Stage 1: Balance by downsampling majority class
    STAGE1_MAX_SAMPLES_PER_LABEL = 3000  # Cap neutral at 3k for balance
    
    # Batch Configuration
    BATCH_SIZE = 16  # Increased for stability
    GRADIENT_ACCUMULATION_STEPS = 2
    VAL_BATCH_SIZE = 32
    
    # Learning Configuration
    LEARNING_RATE = 2e-5  # Standard for DeBERTa
    STAGE2_LEARNING_RATE = 5e-6  # Lower for fine-tuning
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    
    # Regularization
    DROPOUT_RATE = 0.2  # Reduced for multi-label
    HIDDEN_DROPOUT_PROB = 0.2
    ATTENTION_PROBS_DROPOUT_PROB = 0.1
    GRADIENT_CLIP_NORM = 1.0
    USE_R_DROP = True  # Regularized dropout
    R_DROP_ALPHA = 4.0  # Weight for KL divergence
    
    # Loss Configuration (CRITICAL FOR MULTI-LABEL)
    LOSS_TYPE = "asymmetric"  # Options: "bce", "focal", "asymmetric"
    
    # Asymmetric Loss (Best for imbalanced multi-label)
    ASL_GAMMA_NEG = 4  # Focus on hard negatives
    ASL_GAMMA_POS = 1  # Less focus on easy positives
    ASL_CLIP = 0.05  # Clip very small probabilities
    
    # Class weighting (dynamic per-label)
    USE_CLASS_WEIGHTS = True
    MIN_POS_WEIGHT = 0.5
    MAX_POS_WEIGHT = 10.0
    
    # Label smoothing for multi-label
    LABEL_SMOOTHING = 0.05  # Small for multi-label
    
    # Threshold Optimization
    OPTIMIZE_THRESHOLDS = True  # Find best threshold per label
    INITIAL_THRESHOLD = 0.5
    THRESHOLD_SEARCH_STEPS = 20
    
    # Contrastive Learning (emotion relationships)
    USE_CONTRASTIVE = True
    CONTRASTIVE_TEMPERATURE = 0.07
    CONTRASTIVE_WEIGHT = 0.1
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_DELTA = 0.001
    METRIC_FOR_BEST_MODEL = "f1_macro"  # or "f1_micro"
    
    # Paths
    DATA_DIR = Path("/kaggle/input/emdata/processed_emotion_data_v4")
    OUTPUT_DIR = Path("/kaggle/working/")
    
    # Performance
    FP16 = torch.cuda.is_available()
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # Logging
    LOG_INTERVAL = 50  # Log every N steps
    SAVE_INTERVAL = 1  # Save every N epochs


# ============================================================================
# Asymmetric Loss (SOTA for imbalanced multi-label)
# ============================================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    
    Paper: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    - Addresses class imbalance in multi-label settings
    - Better than Focal Loss and BCE for imbalanced data
    """
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_labels] - raw model outputs
            targets: [batch_size, num_labels] - binary labels (0 or 1)
        """
        # Sigmoid activation
        probs = torch.sigmoid(logits)
        
        # Asymmetric Focusing
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # Calculate loss
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        
        # Asymmetric focusing
        loss_pos = loss_pos * ((1 - probs_pos) ** self.gamma_pos)
        loss_neg = loss_neg * (probs_pos ** self.gamma_neg)
        
        loss = -(loss_pos + loss_neg)
        return loss.mean()


class MultilabelFocalLoss(nn.Module):
    """Focal Loss adapted for multi-label classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


# ============================================================================
# R-Drop Regularization
# ============================================================================

class RDropRegularization:
    """
    R-Drop: Regularized Dropout for Neural Networks
    
    Paper: "R-Drop: Regularized Dropout for Neural Networks" (NeurIPS 2021)
    - Pass same input twice with different dropout
    - Minimize KL divergence between predictions
    - Improves generalization significantly
    """
    
    def __init__(self, alpha=4.0):
        self.alpha = alpha
    
    def compute_kl_loss(self, logits1, logits2):
        """Compute symmetric KL divergence"""
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        
        q1 = F.softmax(logits1, dim=-1)
        q2 = F.softmax(logits2, dim=-1)
        
        kl_loss = F.kl_div(p1, q2, reduction='batchmean') + \
                  F.kl_div(p2, q1, reduction='batchmean')
        
        return kl_loss / 2


# ============================================================================
# Contrastive Loss (Learn emotion relationships)
# ============================================================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for emotion relationship learning
    
    Paper: "Supervised Contrastive Learning" (NeurIPS 2020)
    - Learn that similar emotions should have similar representations
    - E.g., joy and excitement should be closer than joy and sadness
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, hidden_dim] - normalized embeddings
            labels: [batch_size, num_labels] - multi-hot labels
        """
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Mask for positive pairs (same labels)
        labels_expanded = labels.unsqueeze(0)  # [1, batch, num_labels]
        labels_transposed = labels.unsqueeze(1)  # [batch, 1, num_labels]
        
        # Positive pairs: samples sharing at least one label
        pos_mask = (labels_expanded * labels_transposed).sum(-1) > 0
        pos_mask.fill_diagonal_(False)
        
        # Negative mask
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        
        # Sum over negatives
        neg_sum = (exp_sim * neg_mask).sum(1, keepdim=True)
        
        # Loss for each positive pair
        loss = 0
        num_pos = pos_mask.sum(1)
        
        for i in range(batch_size):
            if num_pos[i] > 0:
                pos_sim = exp_sim[i][pos_mask[i]]
                loss_i = -torch.log(pos_sim / (pos_sim + neg_sum[i])).mean()
                loss += loss_i
        
        return loss / batch_size if batch_size > 0 else loss


# ============================================================================
# Enhanced Model with Contrastive Learning
# ============================================================================

class MultiLabelEmotionModel(nn.Module):
    """
    Research-grade multi-label emotion classifier
    
    Features:
    - Multi-label classification head (sigmoid, not softmax!)
    - Contrastive learning for emotion relationships
    - Projection head for better representations
    """
    
    def __init__(self, model_name, num_labels, config, pos_weights=None):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        
        # Load DeBERTa config
        deberta_config = DebertaV2Config.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",  # CRITICAL!
            hidden_dropout_prob=config.HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob=config.ATTENTION_PROBS_DROPOUT_PROB,
        )
        
        # Base model
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            config=deberta_config
        )
        
        # Additional dropout
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        # Projection head for contrastive learning
        if config.USE_CONTRASTIVE:
            hidden_size = self.deberta.config.hidden_size
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 256)  # Project to lower dim
            )
        
        # Loss functions
        if config.LOSS_TYPE == "asymmetric":
            self.classification_loss = AsymmetricLoss(
                gamma_neg=config.ASL_GAMMA_NEG,
                gamma_pos=config.ASL_GAMMA_POS,
                clip=config.ASL_CLIP
            )
        elif config.LOSS_TYPE == "focal":
            self.classification_loss = MultilabelFocalLoss()
        else:  # BCE
            self.classification_loss = nn.BCEWithLogitsLoss(
                pos_weight=pos_weights
            )
        
        if config.USE_CONTRASTIVE:
            self.contrastive_loss = SupConLoss(
                temperature=config.CONTRASTIVE_TEMPERATURE
            )
    
    def forward(self, input_ids, attention_mask, labels=None, return_features=False):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, num_labels] - multi-hot encoded
        """
        # Get base model outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get pooled representation (CLS token)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        pooled = hidden_states[:, 0, :]  # CLS token
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Logits for classification
        logits = outputs.logits
        
        result = {'logits': logits}
        
        # Compute losses if labels provided
        if labels is not None:
            # Classification loss
            cls_loss = self.classification_loss(logits, labels)
            result['loss'] = cls_loss
            result['cls_loss'] = cls_loss
            
            # Contrastive loss
            if self.config.USE_CONTRASTIVE:
                features = self.projection(pooled)
                contrastive_loss = self.contrastive_loss(features, labels)
                result['contrastive_loss'] = contrastive_loss
                result['loss'] = cls_loss + self.config.CONTRASTIVE_WEIGHT * contrastive_loss
        
        if return_features:
            result['features'] = pooled
        
        return result


# ============================================================================
# Dataset with Multi-Label Support
# ============================================================================

class MultiLabelEmotionDataset(Dataset):
    """Dataset for multi-label emotion classification"""
    
    def __init__(self, df, tokenizer, mlb, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        
        # Parse emotions (handle both formats)
        if isinstance(row['emotions'], str):
            emotions = row['emotions'].split('|')
        else:
            emotions = row['emotions']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Multi-hot encoding
        labels = self.mlb.transform([emotions])[0]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }


# ============================================================================
# Threshold Optimization
# ============================================================================

class ThresholdOptimizer:
    """
    Optimize classification thresholds per label
    
    Not all labels should use 0.5 threshold!
    Rare labels need lower thresholds.
    """
    
    def __init__(self, num_labels, search_steps=20):
        self.num_labels = num_labels
        self.search_steps = search_steps
        self.optimal_thresholds = [0.5] * num_labels
    
    def optimize(self, probs, labels):
        """
        Find optimal threshold for each label
        
        Args:
            probs: [n_samples, num_labels] - predicted probabilities
            labels: [n_samples, num_labels] - true labels
        """
        print("\n🎯 Optimizing per-label thresholds...")
        
        for label_idx in range(self.num_labels):
            best_f1 = 0
            best_threshold = 0.5
            
            label_probs = probs[:, label_idx]
            label_true = labels[:, label_idx]
            
            # Search thresholds
            for threshold in np.linspace(0.1, 0.9, self.search_steps):
                preds = (label_probs >= threshold).astype(int)
                f1 = f1_score(label_true, preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.optimal_thresholds[label_idx] = best_threshold
            
        print("✅ Threshold optimization complete")
        return self.optimal_thresholds
    
    def apply_thresholds(self, probs):
        """Apply optimized thresholds to predictions"""
        preds = np.zeros_like(probs)
        for i in range(self.num_labels):
            preds[:, i] = (probs[:, i] >= self.optimal_thresholds[i]).astype(int)
        return preds


# ============================================================================
# Trainer
# ============================================================================

class MultiLabelTrainer:
    """Research-grade multi-label emotion trainer"""
    
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Load data
        self.load_data()
        
        # Initialize tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(config.MODEL_NAME)
        
        # R-Drop
        if config.USE_R_DROP:
            self.rdrop = RDropRegularization(alpha=config.R_DROP_ALPHA)
        
        # Threshold optimizer
        if config.OPTIMIZE_THRESHOLDS:
            self.threshold_optimizer = ThresholdOptimizer(
                self.num_labels, 
                config.THRESHOLD_SEARCH_STEPS
            )
        
        # Best metrics tracking
        self.best_metric = 0
        self.best_model_state = None
    
    def setup_logging(self):
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log_file = self.config.OUTPUT_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*100)
        self.logger.info("RESEARCH-GRADE MULTI-LABEL EMOTION TRAINING v5.0")
        self.logger.info("="*100)
    
    def setup_directories(self):
        (self.config.OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
        (self.config.OUTPUT_DIR / "metrics").mkdir(exist_ok=True)
        (self.config.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
    
    def load_data(self):
        """Load preprocessed multi-label data"""
        self.logger.info("📂 Loading preprocessed data...")
        
        # Load metadata
        with open(self.config.DATA_DIR / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.num_labels = metadata['dataset_info']['num_labels']
        self.label_names = metadata['dataset_info']['label_names']
        
        # Load label mapping
        with open(self.config.DATA_DIR / "label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
        
        # Create MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([label_mapping['classes']])
        
        # Load splits
        train_df = pd.read_csv(self.config.DATA_DIR / "csv_splits" / "train.csv")
        val_df = pd.read_csv(self.config.DATA_DIR / "csv_splits" / "val.csv")
        test_df = pd.read_csv(self.config.DATA_DIR / "csv_splits" / "test.csv")
        
        # Parse emotions
        for df in [train_df, val_df, test_df]:
            df['emotions'] = df['emotions'].apply(
                lambda x: x.split('|') if isinstance(x, str) else x
            )
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        self.logger.info(f"✅ Loaded: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
        self.logger.info(f"   Labels: {self.num_labels}")
    
    def create_balanced_subset(self, df, max_samples_per_label):
        """
        Create balanced subset for Stage 1 training
        
        Strategy: Cap majority class (neutral) while keeping all minority samples
        """
        self.logger.info(f"\n📊 Creating balanced subset (max {max_samples_per_label} per label)...")
        
        # Count label occurrences
        label_counts = defaultdict(int)
        label_samples = defaultdict(list)
        
        for idx, row in df.iterrows():
            emotions = row['emotions']
            for emotion in emotions:
                label_samples[emotion].append(idx)
                label_counts[emotion] += 1
        
        # Sample indices to keep
        keep_indices = set()
        
        for label, indices in label_samples.items():
            count = len(indices)
            if count > max_samples_per_label:
                # Downsample
                sampled = random.sample(indices, max_samples_per_label)
                keep_indices.update(sampled)
                self.logger.info(f"   {label:15s}: {count:6,} → {max_samples_per_label:6,}")
            else:
                # Keep all
                keep_indices.update(indices)
                self.logger.info(f"   {label:15s}: {count:6,} (kept all)")
        
        balanced_df = df.loc[list(keep_indices)].reset_index(drop=True)
        self.logger.info(f"\n✅ Balanced subset: {len(balanced_df):,} samples")
        
        return balanced_df
    
    def compute_pos_weights(self, df):
        """
        Compute pos_weight for BCEWithLogitsLoss
        
        pos_weight = (num_negatives / num_positives) per label
        Helps with class imbalance
        """
        label_counts = np.zeros(self.num_labels)
        
        for _, row in df.iterrows():
            emotions = row['emotions']
            labels_binary = self.mlb.transform([emotions])[0]
            label_counts += labels_binary
        
        total_samples = len(df)
        pos_weights = []
        
        for count in label_counts:
            if count > 0:
                neg_count = total_samples - count
                weight = neg_count / count
                # Clip to reasonable range
                weight = np.clip(weight, self.config.MIN_POS_WEIGHT, self.config.MAX_POS_WEIGHT)
            else:
                weight = 1.0
            pos_weights.append(weight)
        
        return torch.FloatTensor(pos_weights).to(device)
    
    def create_data_loaders(self, train_df, val_df):
        """Create data loaders"""
        train_dataset = MultiLabelEmotionDataset(
            train_df, self.tokenizer, self.mlb, self.config.MAX_LENGTH
        )
        val_dataset = MultiLabelEmotionDataset(
            val_df, self.tokenizer, self.mlb, self.config.MAX_LENGTH
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, scaler, epoch):
        """Training epoch with R-Drop"""
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_contrastive_loss = 0
        total_rdrop_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', ncols=120)
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # R-Drop: forward twice with different dropout
            if self.config.USE_R_DROP and model.training:
                if self.config.FP16:
                    with torch.cuda.amp.autocast():
                        # First forward
                        outputs1 = model(input_ids, attention_mask, labels)
                        # Second forward (different dropout)
                        outputs2 = model(input_ids, attention_mask, labels)
                        
                        # R-Drop KL divergence
                        rdrop_loss = self.rdrop.compute_kl_loss(
                            outputs1['logits'], outputs2['logits']
                        )
                        
                        # Combined loss
                        loss = (outputs1['loss'] + outputs2['loss']) / 2 + \
                               self.config.R_DROP_ALPHA * rdrop_loss
                else:
                    outputs1 = model(input_ids, attention_mask, labels)
                    outputs2 = model(input_ids, attention_mask, labels)
                    rdrop_loss = self.rdrop.compute_kl_loss(
                        outputs1['logits'], outputs2['logits']
                    )
                    loss = (outputs1['loss'] + outputs2['loss']) / 2 + \
                           self.config.R_DROP_ALPHA * rdrop_loss
                
                cls_loss = (outputs1.get('cls_loss', 0) + outputs2.get('cls_loss', 0)) / 2
                contrastive_loss = (outputs1.get('contrastive_loss', 0) + outputs2.get('contrastive_loss', 0)) / 2
            else:
                if self.config.FP16:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask, labels)
                        loss = outputs['loss']
                else:
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                
                cls_loss = outputs.get('cls_loss', loss)
                contrastive_loss = outputs.get('contrastive_loss', 0)
                rdrop_loss = torch.tensor(0.0)
            
            # Backward
            loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            if self.config.FP16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                if self.config.FP16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.GRADIENT_CLIP_NORM
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.GRADIENT_CLIP_NORM
                    )
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
            total_cls_loss += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
            if isinstance(contrastive_loss, torch.Tensor):
                total_contrastive_loss += contrastive_loss.item()
            if isinstance(rdrop_loss, torch.Tensor):
                total_rdrop_loss += rdrop_loss.item()
            
            # Progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss/(step+1):.4f}',
                'cls': f'{total_cls_loss/(step+1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.1e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(train_loader) if total_contrastive_loss > 0 else 0
        avg_rdrop_loss = total_rdrop_loss / len(train_loader) if total_rdrop_loss > 0 else 0
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'contrastive_loss': avg_contrastive_loss,
            'rdrop_loss': avg_rdrop_loss
        }
    
    def validate(self, model, val_loader, optimize_thresholds=False):
        """Validation with comprehensive multi-label metrics"""
        model.eval()
        
        all_probs = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', ncols=100):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Get probabilities (sigmoid for multi-label!)
                probs = torch.sigmoid(outputs['logits'])
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate results
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        
        # Optimize thresholds if requested
        if optimize_thresholds and self.config.OPTIMIZE_THRESHOLDS:
            self.threshold_optimizer.optimize(all_probs, all_labels)
            all_preds = self.threshold_optimizer.apply_thresholds(all_probs)
        else:
            # Use fixed threshold (0.5 or optimized)
            if hasattr(self, 'threshold_optimizer') and self.threshold_optimizer.optimal_thresholds[0] != 0.5:
                all_preds = self.threshold_optimizer.apply_thresholds(all_probs)
            else:
                all_preds = (all_probs >= self.config.INITIAL_THRESHOLD).astype(int)
        
        # Compute comprehensive metrics
        metrics = self.compute_multilabel_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics, all_probs, all_preds, all_labels
    
    def compute_multilabel_metrics(self, y_true, y_pred, y_probs):
        """
        Comprehensive multi-label metrics
        
        Critical metrics for multi-label:
        - F1 Micro: Overall performance across all labels
        - F1 Macro: Average F1 per label (good for imbalance)
        - Hamming Loss: Fraction of wrong labels
        - Subset Accuracy: Exact match accuracy
        """
        metrics = {}
        
        # F1 Scores
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
        
        # Hamming Loss (lower is better)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # Subset Accuracy (exact match)
        metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
        
        # Jaccard Score (IoU for sets)
        metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-label F1 scores
        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        metrics['per_label_f1'] = {
            self.label_names[i]: float(per_label_f1[i]) 
            for i in range(len(per_label_f1))
        }
        
        # Per-label precision and recall
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics['per_label_precision'] = {
            self.label_names[i]: float(precision[i]) 
            for i in range(len(precision))
        }
        metrics['per_label_recall'] = {
            self.label_names[i]: float(recall[i]) 
            for i in range(len(recall))
        }
        
        return metrics
    
    def print_metrics(self, metrics, stage="Validation"):
        """Pretty print metrics"""
        print(f"\n{'='*80}")
        print(f"{stage} METRICS")
        print(f"{'='*80}")
        print(f"Loss:              {metrics.get('loss', 0):.4f}")
        print(f"\n📊 Overall Metrics:")
        print(f"   F1 Micro:       {metrics['f1_micro']:.4f} (overall labels)")
        print(f"   F1 Macro:       {metrics['f1_macro']:.4f} (avg per label)")
        print(f"   F1 Weighted:    {metrics['f1_weighted']:.4f}")
        print(f"   F1 Samples:     {metrics['f1_samples']:.4f}")
        print(f"   Hamming Loss:   {metrics['hamming_loss']:.4f}")
        print(f"   Subset Accuracy:{metrics['subset_accuracy']:.4f} (exact match)")
        print(f"   Jaccard Micro:  {metrics['jaccard_micro']:.4f}")
        
        # Top and bottom performing labels
        per_label_f1 = metrics['per_label_f1']
        sorted_labels = sorted(per_label_f1.items(), key=lambda x: x[1])
        
        print(f"\n🔴 Bottom 5 Labels (Need Improvement):")
        for label, f1 in sorted_labels[:5]:
            prec = metrics['per_label_precision'][label]
            rec = metrics['per_label_recall'][label]
            print(f"   {label:15s}: F1={f1:.4f}, P={prec:.4f}, R={rec:.4f}")
        
        print(f"\n🟢 Top 5 Labels:")
        for label, f1 in sorted_labels[-5:]:
            prec = metrics['per_label_precision'][label]
            rec = metrics['per_label_recall'][label]
            print(f"   {label:15s}: F1={f1:.4f}, P={prec:.4f}, R={rec:.4f}")
        print(f"{'='*80}\n")
    
    def train_stage(self, stage_num, train_df, epochs, learning_rate):
        """Train a single stage"""
        stage_name = f"Stage {stage_num}"
        print(f"\n{'='*100}")
        print(f"{stage_name.upper()} TRAINING")
        print(f"{'='*100}")
        print(f"Samples: {len(train_df):,}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"{'='*100}\n")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_df, self.val_df)
        
        # Compute pos_weights for this stage
        pos_weights = self.compute_pos_weights(train_df)
        
        # Initialize model
        model = MultiLabelEmotionModel(
            self.config.MODEL_NAME,
            self.num_labels,
            self.config,
            pos_weights=pos_weights if self.config.USE_CLASS_WEIGHTS and self.config.LOSS_TYPE == "bce" else None
        ).to(device)
        
        # Load from previous stage if Stage 2
        if stage_num == 2 and self.best_model_state is not None:
            print("🔄 Loading weights from Stage 1...")
            model.load_state_dict(self.best_model_state)
        
        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Scheduler
        total_steps = len(train_loader) * epochs // self.config.GRADIENT_ACCUMULATION_STEPS
        warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.FP16 else None
        
        # Training loop
        best_stage_metric = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n{'─'*100}")
            print(f"{stage_name} - Epoch {epoch+1}/{epochs}")
            print(f"{'─'*100}")
            
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, scheduler, scaler, epoch
            )
            
            print(f"\n📈 Training Metrics:")
            print(f"   Loss:              {train_metrics['loss']:.4f}")
            print(f"   Classification:    {train_metrics['cls_loss']:.4f}")
            if train_metrics['contrastive_loss'] > 0:
                print(f"   Contrastive:       {train_metrics['contrastive_loss']:.4f}")
            if train_metrics['rdrop_loss'] > 0:
                print(f"   R-Drop:            {train_metrics['rdrop_loss']:.4f}")
            
            # Validate
            optimize_thresh = (epoch == epochs - 1)  # Optimize on last epoch
            val_metrics, val_probs, val_preds, val_labels = self.validate(
                model, val_loader, optimize_thresholds=optimize_thresh
            )
            
            self.print_metrics(val_metrics, f"{stage_name} - Epoch {epoch+1}")
            
            # Check for improvement
            current_metric = val_metrics[self.config.METRIC_FOR_BEST_MODEL]
            
            if current_metric > best_stage_metric + self.config.EARLY_STOPPING_DELTA:
                best_stage_metric = current_metric
                patience_counter = 0
                
                # Save best model
                self.best_model_state = deepcopy(model.state_dict())
                
                # Save checkpoint
                checkpoint_path = self.config.OUTPUT_DIR / "checkpoints" / f"{stage_name.lower().replace(' ', '_')}_best.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': val_metrics,
                    'config': self.config.__dict__
                }, checkpoint_path)
                
                print(f"✅ New best {self.config.METRIC_FOR_BEST_MODEL}: {current_metric:.4f}")
            else:
                patience_counter += 1
                print(f"⏸️  No improvement ({patience_counter}/{self.config.EARLY_STOPPING_PATIENCE})")
            
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
                break
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
        
        # Load best model for this stage
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        
        print(f"\n✅ {stage_name} complete - Best {self.config.METRIC_FOR_BEST_MODEL}: {best_stage_metric:.4f}")
        
        return model
    
    def train(self):
        """Main training pipeline with two-stage approach"""
        print("\n" + "="*100)
        print("🚀 STARTING RESEARCH-GRADE MULTI-LABEL TRAINING")
        print("="*100)
        print(f"\n⚙️  Configuration:")
        print(f"   Model: {self.config.MODEL_NAME}")
        print(f"   Loss: {self.config.LOSS_TYPE}")
        print(f"   Two-Stage: {self.config.USE_TWO_STAGE}")
        print(f"   R-Drop: {self.config.USE_R_DROP}")
        print(f"   Contrastive: {self.config.USE_CONTRASTIVE}")
        print(f"   Threshold Opt: {self.config.OPTIMIZE_THRESHOLDS}")
        print("="*100)
        
        if self.config.USE_TWO_STAGE:
            # Stage 1: Balanced pre-training
            balanced_train_df = self.create_balanced_subset(
                self.train_df, 
                self.config.STAGE1_MAX_SAMPLES_PER_LABEL
            )
            
            model = self.train_stage(
                stage_num=1,
                train_df=balanced_train_df,
                epochs=self.config.STAGE1_EPOCHS,
                learning_rate=self.config.LEARNING_RATE
            )
            
            # Stage 2: Full data fine-tuning
            model = self.train_stage(
                stage_num=2,
                train_df=self.train_df,
                epochs=self.config.STAGE2_EPOCHS,
                learning_rate=self.config.STAGE2_LEARNING_RATE
            )
        else:
            # Single-stage training
            model = self.train_stage(
                stage_num=1,
                train_df=self.train_df,
                epochs=self.config.STAGE1_EPOCHS + self.config.STAGE2_EPOCHS,
                learning_rate=self.config.LEARNING_RATE
            )
        
        # Final evaluation on test set
        print("\n" + "="*100)
        print("🎯 FINAL EVALUATION ON TEST SET")
        print("="*100)
        
        test_dataset = MultiLabelEmotionDataset(
            self.test_df, self.tokenizer, self.mlb, self.config.MAX_LENGTH
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        test_metrics, test_probs, test_preds, test_labels = self.validate(
            model, test_loader, optimize_thresholds=False
        )
        
        self.print_metrics(test_metrics, "FINAL TEST SET")
        
        # Save final results
        self.save_final_results(model, test_metrics, test_probs, test_preds, test_labels)
        
        print("\n" + "="*100)
        print("✅ TRAINING COMPLETE!")
        print("="*100)
        
        return model, test_metrics
    
    def save_final_results(self, model, metrics, probs, preds, labels):
        """Save final model and results"""
        print("\n💾 Saving final results...")
        
        # Save final model
        final_model_path = self.config.OUTPUT_DIR / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'label_names': self.label_names,
            'mlb': self.mlb,
            'thresholds': self.threshold_optimizer.optimal_thresholds if hasattr(self, 'threshold_optimizer') else None
        }, final_model_path)
        
        # Save metrics
        metrics_path = self.config.OUTPUT_DIR / "metrics" / "final_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types for JSON
            metrics_json = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    metrics_json[k] = {str(kk): float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                                      for kk, vv in v.items()}
                elif isinstance(v, (np.floating, np.integer)):
                    metrics_json[k] = float(v)
                else:
                    metrics_json[k] = v
            json.dump(metrics_json, f, indent=2)
        
        # Plot per-label F1 scores
        self.plot_per_label_f1(metrics['per_label_f1'])
        
        # Plot confusion matrices for top labels
        self.plot_confusion_matrices(labels, preds)
        
        print("✅ Results saved successfully")
    
    def plot_per_label_f1(self, per_label_f1):
        """Plot per-label F1 scores"""
        labels = list(per_label_f1.keys())
        f1_scores = list(per_label_f1.values())
        
        # Sort by F1
        sorted_data = sorted(zip(labels, f1_scores), key=lambda x: x[1])
        labels, f1_scores = zip(*sorted_data)
        
        plt.figure(figsize=(14, 8))
        colors = ['red' if x < 0.5 else 'orange' if x < 0.7 else 'green' for x in f1_scores]
        
        plt.barh(labels, f1_scores, color=colors, alpha=0.7)
        plt.xlabel('F1 Score', fontsize=12)
        plt.title('Per-Label F1 Scores (Test Set)', fontsize=14, fontweight='bold')
        plt.axvline(np.mean(f1_scores), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(f1_scores):.4f}')
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(
            self.config.OUTPUT_DIR / "plots" / "per_label_f1_test.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def plot_confusion_matrices(self, y_true, y_pred):
        """Plot confusion matrices for each label"""
        # Compute per-label confusion matrices
        cm_multi = multilabel_confusion_matrix(y_true, y_pred)
        
        # Plot for top 9 labels by frequency
        label_frequencies = y_true.sum(axis=0)
        top_indices = np.argsort(label_frequencies)[-9:][::-1]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, label_idx in enumerate(top_indices):
            cm = cm_multi[label_idx]
            ax = axes[idx]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_title(f'{self.label_names[label_idx]}', fontsize=12, fontweight='bold')
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig(
            self.config.OUTPUT_DIR / "plots" / "confusion_matrices.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()


# ============================================================================
# Inference
# ============================================================================

class MultiLabelInference:
    """Inference with trained multi-label model"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        print(f"📥 Loading model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config
        config_dict = checkpoint['config']
        self.config = Config()
        for k, v in config_dict.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        # Extract label info
        self.label_names = checkpoint['label_names']
        self.num_labels = len(self.label_names)
        self.mlb = checkpoint['mlb']
        self.thresholds = checkpoint.get('thresholds', [0.5] * self.num_labels)
        
        # Initialize model
        self.model = MultiLabelEmotionModel(
            self.config.MODEL_NAME,
            self.num_labels,
            self.config
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.config.MODEL_NAME)
        
        print(f"✅ Model loaded: {self.num_labels} labels")
    
    def predict(self, text, top_k=5):
        """Predict emotions for text"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt'
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                encoding['input_ids'],
                encoding['attention_mask']
            )
            probs = torch.sigmoid(outputs['logits'])[0].cpu().numpy()
        
        # Apply thresholds
        predictions = []
        for i, (prob, threshold) in enumerate(zip(probs, self.thresholds)):
            if prob >= threshold:
                predictions.append({
                    'emotion': self.label_names[i],
                    'confidence': float(prob),
                    'threshold': float(threshold)
                })
        
        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Get top-k by probability
        top_emotions = sorted(
            [(self.label_names[i], float(probs[i])) for i in range(len(probs))],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return {
            'predicted_emotions': predictions,
            'top_k_emotions': top_emotions,
            'all_probabilities': {
                self.label_names[i]: float(probs[i]) 
                for i in range(len(probs))
            }
        }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "="*100)
    print("🎓 RESEARCH-GRADE MULTI-LABEL EMOTION CLASSIFICATION")
    print("Target: >80% Macro F1 for Publication")
    print("="*100)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    
    # Initialize config
    config = Config()
    
    # Train
    trainer = MultiLabelTrainer(config)
    model, test_metrics = trainer.train()
    
    # Demo inference
    print("\n" + "="*100)
    print("🎭 INFERENCE DEMO")
    print("="*100)
    
    inference = MultiLabelInference(config.OUTPUT_DIR / "final_model.pt")
    
    test_texts = [
        "I'm so happy and excited about my new job! Feeling grateful and proud.",
        "Today was terrible. I feel sad, angry, and disappointed with everything.",
        "I'm worried and anxious about the exam tomorrow. Can't sleep.",
        "Just had an amazing workout! Feeling confident and energized.",
        "I miss my family. Feeling lonely and a bit sad tonight."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. \"{text}\"")
        result = inference.predict(text, top_k=3)
        print(f"   Predicted: {', '.join([e['emotion'] for e in result['predicted_emotions']])}")
        print(f"   Top 3: {', '.join([f'{e}({p:.2f})' for e, p in result['top_k_emotions']])}")
    
    print("\n" + "="*100)
    print("✅ COMPLETE!")
    print("="*100)


if __name__ == "__main__":
    main()