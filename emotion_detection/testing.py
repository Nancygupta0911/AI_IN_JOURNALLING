import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import warnings
from collections import Counter, defaultdict
import argparse
import sys
import time
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings('ignore')

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from sklearn.metrics import (
    classification_report, f1_score, precision_recall_fscore_support,
    confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_KAGGLE = os.path.exists('/kaggle/working')


class TestConfig:
    """Testing configuration"""
    
    # Paths
    if IS_KAGGLE:
        DATA_DIR = Path("processed_emotion_data_v4")
        MODEL_DIR = Path("fold_5.pt")
        OUTPUT_DIR = Path("test_results")
    else:
        DATA_DIR = Path("./processed_emotion_data_v4")
        MODEL_DIR = Path("kfold_deberta_v4/fold_models")
        OUTPUT_DIR = Path("./test_results")
    
    # Model settings
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    
    # Testing options
    ENSEMBLE_METHOD = "average"  # "voting" or "average"
    TOP_K_PREDICTIONS = 5
    CONFIDENCE_THRESHOLD = 0.5
    
    # Output formats
    SAVE_PREDICTIONS = True
    GENERATE_PLOTS = True
    EXPORT_CSV = True
    EXPORT_JSON = True


# ============================================================================
# Enhanced Model Definition (matching training script)
# ============================================================================

class EnhancedDeBERTaModel(torch.nn.Module):
    """Enhanced DeBERTa model (matching training architecture)"""
    
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels
        
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        logits = self.dropout(outputs.logits)
        return {'logits': logits}


# ============================================================================
# Dataset for Testing
# ============================================================================

class TestDataset(Dataset):
    """Simple dataset for testing"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        label = self.labels[idx] if self.labels is not None else -1
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }


# ============================================================================
# Model Loader
# ============================================================================

class ModelLoader:
    """Load trained models (single or ensemble)"""
    
    def __init__(self, model_dir, config):
        self.model_dir = Path(model_dir)
        self.config = config
        self.models = []
        self.metadata = None
        self.label_names = None
        self.tokenizer = None
        
    def load_metadata(self):
        """Load dataset metadata"""
        metadata_path = self.config.DATA_DIR / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.num_labels = self.metadata['dataset_info']['num_labels']
        self.label_names = self.metadata['dataset_info']['label_names']
        
        print(f"Loaded metadata: {self.num_labels} emotion classes")
    
    def load_tokenizer(self):
        """Load tokenizer"""
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.config.MODEL_NAME)
        print(f"Loaded tokenizer: {self.config.MODEL_NAME}")
    
    def load_ensemble_models(self):
        """Load all fold models for ensemble"""
        fold_dir = self.model_dir 
        
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold models directory not found: {fold_dir}")
        
        fold_files = sorted(fold_dir.glob("fold_*.pt"))
        
        if not fold_files:
            raise FileNotFoundError(f"No fold models found in {fold_dir}")
        
        print(f"\nLoading {len(fold_files)} fold models...")
        
        for fold_file in fold_files:
            model = EnhancedDeBERTaModel(
                self.config.MODEL_NAME,
                self.num_labels,
                dropout_rate=0.3
            ).to(device)
            
            try:
                checkpoint = torch.load(fold_file, map_location=device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                self.models.append(model)
                
                print(f"  ✓ Loaded {fold_file.name}")
                
            except Exception as e:
                print(f"  ✗ Failed to load {fold_file.name}: {e}")
        
        if not self.models:
            raise RuntimeError("No models were successfully loaded")
        
        print(f"\nSuccessfully loaded {len(self.models)} models")
    
    def load_single_model(self, model_path):
        """Load a single model checkpoint"""
        model = EnhancedDeBERTaModel(
            self.config.MODEL_NAME,
            self.num_labels,
            dropout_rate=0.3
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        self.models.append(model)
        
        print(f"Loaded single model from {model_path}")


# ============================================================================
# Comprehensive Tester
# ============================================================================

class ComprehensiveTester:
    """Main testing class with multiple testing modes"""
    
    def __init__(self, config):
        self.config = config
        self.setup_output_dir()
        self.setup_logging()
        
        # Load models
        self.loader = ModelLoader(config.MODEL_DIR, config)
        self.loader.load_metadata()
        self.loader.load_tokenizer()
        self.loader.load_ensemble_models()
        
        self.models = self.loader.models
        self.tokenizer = self.loader.tokenizer
        self.label_names = self.loader.label_names
        self.num_labels = self.loader.num_labels
        
        self.test_data = None
        self.results = {}
    
    def setup_output_dir(self):
        """Create output directory structure"""
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.config.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
        (self.config.OUTPUT_DIR / "predictions").mkdir(exist_ok=True)
        (self.config.OUTPUT_DIR / "analysis").mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup logging"""
        log_file = self.config.OUTPUT_DIR / f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_test_data(self):
        """Load test dataset"""
        csv_dir = self.config.DATA_DIR / "csv_splits"
        test_file = csv_dir / "test.csv"
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        test_df = pd.read_csv(test_file, usecols=['text', 'emotions'])
        
        texts = test_df['text'].tolist()
        
        def parse_label(label_str):
            if isinstance(label_str, str):
                label_str = label_str.strip('"\'[]')
                return label_str.split(',')[0].strip().strip('"\'')
            return str(label_str)
        
        labels_str = [parse_label(l) for l in test_df['emotions'].tolist()]
        
        label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        labels = np.array([label_to_idx.get(l, 0) for l in labels_str])
        
        self.test_data = {
            'texts': texts,
            'labels': labels,
            'labels_str': labels_str
        }
        
        print(f"\nLoaded {len(texts)} test samples")
        print(f"Label distribution (top 10):")
        label_dist = Counter(labels)
        for label_idx, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {self.label_names[label_idx]:15s}: {count:5,} samples")
    
    def predict_batch(self, texts: List[str], method: str = 'average') -> Dict:
        """Predict emotions for a batch of texts"""
        dataset = TestDataset(texts, None, self.tokenizer, self.config.MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        all_fold_probs = []
        all_fold_preds = []
        
        # Get predictions from each model
        for model_idx, model in enumerate(self.models):
            fold_probs = []
            fold_preds = []
            
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    logits = outputs['logits']
                    probs = F.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    
                    fold_probs.extend(probs.cpu().numpy())
                    fold_preds.extend(preds.cpu().numpy())
            
            all_fold_probs.append(np.array(fold_probs))
            all_fold_preds.append(np.array(fold_preds))
        
        all_fold_probs = np.array(all_fold_probs)
        all_fold_preds = np.array(all_fold_preds)
        
        # Ensemble predictions
        if method == 'voting':
            final_preds = []
            confidences = []
            
            for i in range(len(texts)):
                votes = Counter(all_fold_preds[:, i])
                pred = votes.most_common(1)[0][0]
                confidence = votes[pred] / len(self.models)
                
                final_preds.append(pred)
                confidences.append(confidence)
            
            final_preds = np.array(final_preds)
            confidences = np.array(confidences)
            avg_probs = np.mean(all_fold_probs, axis=0)
        
        else:  # average
            avg_probs = np.mean(all_fold_probs, axis=0)
            final_preds = np.argmax(avg_probs, axis=1)
            confidences = np.max(avg_probs, axis=1)
        
        return {
            'predictions': final_preds,
            'confidences': confidences,
            'probabilities': avg_probs,
            'all_fold_predictions': all_fold_preds,
            'method': method
        }
    
    def predict_single(self, text: str, top_k: int = 5) -> Dict:
        """Predict emotion for a single text with detailed output"""
        result = self.predict_batch([text], method=self.config.ENSEMBLE_METHOD)
        
        pred_idx = result['predictions'][0]
        confidence = result['confidences'][0]
        probs = result['probabilities'][0]
        
        # Get top-k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_predictions = [
            {
                'emotion': self.label_names[idx],
                'probability': float(probs[idx]),
                'rank': rank + 1
            }
            for rank, idx in enumerate(top_indices)
        ]
        
        # Get fold agreement
        fold_preds = result['all_fold_predictions'][:, 0]
        fold_agreement = {
            self.label_names[pred]: int(count)
            for pred, count in Counter(fold_preds).items()
        }
        
        return {
            'text': text,
            'predicted_emotion': self.label_names[pred_idx],
            'confidence': float(confidence),
            'top_predictions': top_predictions,
            'fold_agreement': fold_agreement,
            'num_models': len(self.models),
            'method': result['method']
        }
    
    def automated_test(self):
        """Run automated testing on test dataset"""
        print("\n" + "="*80)
        print("AUTOMATED TESTING ON TEST DATASET")
        print("="*80)
        
        if self.test_data is None:
            self.load_test_data()
        
        texts = self.test_data['texts']
        true_labels = self.test_data['labels']
        
        print(f"\nRunning inference on {len(texts)} samples...")
        start_time = time.time()
        
        result = self.predict_batch(texts, method=self.config.ENSEMBLE_METHOD)
        
        inference_time = time.time() - start_time
        
        predictions = result['predictions']
        confidences = result['confidences']
        probabilities = result['probabilities']
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predictions, probabilities)
        
        # Store results
        self.results['automated'] = {
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'probabilities': probabilities,
            'metrics': metrics,
            'inference_time': inference_time,
            'samples_per_second': len(texts) / inference_time
        }
        
        # Print results
        print(f"\n{'='*80}")
        print("TEST RESULTS")
        print(f"{'='*80}")
        print(f"Inference Time: {inference_time:.2f}s ({len(texts)/inference_time:.1f} samples/sec)")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  F1 Macro:        {metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted:     {metrics['f1_weighted']:.4f}")
        print(f"  Precision Macro: {metrics['precision_macro']:.4f}")
        print(f"  Recall Macro:    {metrics['recall_macro']:.4f}")
        
        # Generate detailed analysis
        self.analyze_errors(true_labels, predictions, texts)
        self.analyze_confidence(true_labels, predictions, confidences)
        self.analyze_per_class_performance(true_labels, predictions)
        
        # Save outputs
        if self.config.SAVE_PREDICTIONS:
            self.save_predictions(texts, true_labels, predictions, confidences, probabilities)
        
        if self.config.GENERATE_PLOTS:
            self.generate_plots(true_labels, predictions, confidences)
        
        return metrics
    
    def calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1_macro': f1_score(true_labels, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, predictions, average='weighted', zero_division=0),
        }
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        
        metrics['precision_macro'] = precision
        metrics['recall_macro'] = recall
        
        return metrics
    
    def analyze_errors(self, true_labels, predictions, texts):
        """Detailed error analysis"""
        print(f"\n{'='*80}")
        print("ERROR ANALYSIS")
        print(f"{'='*80}")
        
        errors = []
        correct = []
        
        for idx, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
            if true_label != pred_label:
                errors.append({
                    'index': idx,
                    'text': texts[idx],
                    'true_label': self.label_names[true_label],
                    'predicted_label': self.label_names[pred_label],
                    'confidence': self.results['automated']['confidences'][idx]
                })
            else:
                correct.append(idx)
        
        print(f"\nTotal Errors: {len(errors)} / {len(texts)} ({len(errors)/len(texts)*100:.2f}%)")
        print(f"Total Correct: {len(correct)} / {len(texts)} ({len(correct)/len(texts)*100:.2f}%)")
        
        # Most common misclassifications
        misclassifications = Counter([
            (err['true_label'], err['predicted_label']) for err in errors
        ])
        
        print(f"\nTop 10 Most Common Misclassifications:")
        for (true_lbl, pred_lbl), count in misclassifications.most_common(10):
            print(f"  {true_lbl:15s} → {pred_lbl:15s}: {count:4d} times")
        
        # High-confidence errors
        high_conf_errors = [err for err in errors if err['confidence'] > 0.7]
        
        print(f"\nHigh-Confidence Errors (confidence > 0.7): {len(high_conf_errors)}")
        
        if high_conf_errors:
            print("\nExamples of high-confidence errors:")
            for err in high_conf_errors[:5]:
                print(f"\n  Text: \"{err['text'][:80]}...\"")
                print(f"  True: {err['true_label']:15s} | Predicted: {err['predicted_label']:15s} | Conf: {err['confidence']:.3f}")
        
        # Save error analysis
        error_df = pd.DataFrame(errors)
        if not error_df.empty:
            error_df.to_csv(
                self.config.OUTPUT_DIR / "analysis" / "error_analysis.csv",
                index=False
            )
    
    def analyze_confidence(self, true_labels, predictions, confidences):
        """Analyze confidence distribution"""
        print(f"\n{'='*80}")
        print("CONFIDENCE ANALYSIS")
        print(f"{'='*80}")
        
        correct_mask = true_labels == predictions
        
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        print(f"\nConfidence Statistics:")
        print(f"  All predictions:")
        print(f"    Mean:   {np.mean(confidences):.4f}")
        print(f"    Median: {np.median(confidences):.4f}")
        print(f"    Std:    {np.std(confidences):.4f}")
        
        print(f"\n  Correct predictions:")
        print(f"    Mean:   {np.mean(correct_confidences):.4f}")
        print(f"    Median: {np.median(correct_confidences):.4f}")
        
        print(f"\n  Incorrect predictions:")
        print(f"    Mean:   {np.mean(incorrect_confidences):.4f}")
        print(f"    Median: {np.median(incorrect_confidences):.4f}")
        
        # Confidence bins
        bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        bin_labels = ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
        
        print(f"\nAccuracy by Confidence Range:")
        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if i == len(bins)-2:  # Last bin includes 1.0
                mask = (confidences >= bins[i]) & (confidences <= bins[i+1])
            
            if mask.sum() > 0:
                acc = (true_labels[mask] == predictions[mask]).mean()
                count = mask.sum()
                print(f"  {bin_labels[i]:10s}: {acc:.4f} ({count:5d} samples)")
    
    def analyze_per_class_performance(self, true_labels, predictions):
        """Per-class performance analysis"""
        print(f"\n{'='*80}")
        print("PER-CLASS PERFORMANCE")
        print(f"{'='*80}")
        
        unique_labels = np.unique(true_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, labels=unique_labels, zero_division=0
        )
        
        class_metrics = []
        for idx, label in enumerate(unique_labels):
            class_metrics.append({
                'emotion': self.label_names[label],
                'precision': precision[idx],
                'recall': recall[idx],
                'f1': f1[idx],
                'support': int(support[idx])
            })
        
        # Sort by F1 score
        class_metrics.sort(key=lambda x: x['f1'])
        
        print("\nWorst Performing Classes (by F1):")
        for cm in class_metrics[:10]:
            print(f"  {cm['emotion']:15s}: F1={cm['f1']:.4f}, P={cm['precision']:.4f}, R={cm['recall']:.4f}, Support={cm['support']:5d}")
        
        print("\nBest Performing Classes (by F1):")
        for cm in class_metrics[-10:]:
            print(f"  {cm['emotion']:15s}: F1={cm['f1']:.4f}, P={cm['precision']:.4f}, R={cm['recall']:.4f}, Support={cm['support']:5d}")
        
        # Save per-class metrics
        df = pd.DataFrame(class_metrics)
        df.to_csv(
            self.config.OUTPUT_DIR / "analysis" / "per_class_performance.csv",
            index=False
        )
    
    def save_predictions(self, texts, true_labels, predictions, confidences, probabilities):
        """Save predictions to file"""
        print(f"\nSaving predictions...")
        
        results_list = []
        for i, text in enumerate(texts):
            pred_idx = predictions[i]
            true_idx = true_labels[i]
            
            # Get top-3 predictions
            top_3_indices = np.argsort(probabilities[i])[-3:][::-1]
            top_3 = [
                f"{self.label_names[idx]}({probabilities[i][idx]:.3f})"
                for idx in top_3_indices
            ]
            
            results_list.append({
                'text': text,
                'true_emotion': self.label_names[true_idx],
                'predicted_emotion': self.label_names[pred_idx],
                'confidence': confidences[i],
                'correct': true_idx == pred_idx,
                'top_3_predictions': ' | '.join(top_3)
            })
        
        df = pd.DataFrame(results_list)
        
        # Save as CSV
        csv_path = self.config.OUTPUT_DIR / "predictions" / "predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved predictions to {csv_path}")
        
        # Save as JSON
        if self.config.EXPORT_JSON:
            json_path = self.config.OUTPUT_DIR / "predictions" / "predictions.json"
            with open(json_path, 'w') as f:
                json.dump(results_list, f, indent=2)
            print(f"  Saved predictions to {json_path}")
    
    def generate_plots(self, true_labels, predictions, confidences):
        """Generate visualization plots"""
        print(f"\nGenerating plots...")
        
        # 1. Confusion Matrix
        self.plot_confusion_matrix(true_labels, predictions)
        
        # 2. Confidence Distribution
        self.plot_confidence_distribution(true_labels, predictions, confidences)
        
        # 3. Per-class F1 Scores
        self.plot_per_class_f1(true_labels, predictions)
        
        print(f"  Plots saved to {self.config.OUTPUT_DIR / 'plots'}")
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            self.config.OUTPUT_DIR / "plots" / "confusion_matrix.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def plot_confidence_distribution(self, true_labels, predictions, confidences):
        """Plot confidence distribution"""
        correct_mask = true_labels == predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(confidences[correct_mask], bins=50, alpha=0.7, 
                    label='Correct', color='green', edgecolor='black')
        axes[0].hist(confidences[~correct_mask], bins=50, alpha=0.7, 
                    label='Incorrect', color='red', edgecolor='black')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Confidence Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy by confidence
        bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                acc = (true_labels[mask] == predictions[mask]).mean()
                bin_accuracies.append(acc)
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        axes[1].plot(bin_centers, bin_accuracies, marker='o', linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Calibration Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            self.config.OUTPUT_DIR / "plots" / "confidence_analysis.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def plot_per_class_f1(self, true_labels, predictions):
        """Plot per-class F1 scores"""
        unique_labels = np.unique(true_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, labels=unique_labels, zero_division=0
        )
        
        df = pd.DataFrame({
            'Emotion': [self.label_names[i] for i in unique_labels],
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'Support': support
        })
        
        df = df.sort_values('F1', ascending=True)
        
        # Plot top and bottom classes
        n_show = min(20, len(df))
        df_show = pd.concat([df.head(n_show//2), df.tail(n_show//2)])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['red' if x < 0.3 else 'orange' if x < 0.5 else 'green' 
                  for x in df_show['F1']]
        
        ax.barh(df_show['Emotion'], df_show['F1'], color=colors, alpha=0.7)
        ax.axvline(np.mean(f1), color='blue', linestyle='--', 
                  label=f'Mean F1: {np.mean(f1):.4f}')
        ax.set_xlabel('F1 Score')
        ax.set_title('Per-Class F1 Scores (Best and Worst)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(
            self.config.OUTPUT_DIR / "plots" / "per_class_f1.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def manual_testing_mode(self):
        """Interactive manual testing mode"""
        print("\n" + "="*80)
        print("MANUAL TESTING MODE")
        print("="*80)
        print("\nCommands:")
        print("  - Enter text to classify")
        print("  - 'batch' to test multiple texts from file")
        print("  - 'examples' to see example predictions")
        print("  - 'stats' to see model statistics")
        print("  - 'quit' or 'exit' to exit")
        print("="*80)
        
        while True:
            print("\n" + "-"*80)
            user_input = input("\nEnter text (or command): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting manual testing mode...")
                break
            
            elif user_input.lower() == 'examples':
                self.show_example_predictions()
            
            elif user_input.lower() == 'stats':
                self.show_model_stats()
            
            elif user_input.lower() == 'batch':
                self.batch_file_testing()
            
            else:
                # Predict emotion
                self.predict_and_display(user_input)
    
    def predict_and_display(self, text: str):
        """Predict and display results for a single text"""
        print(f"\nAnalyzing text...")
        start_time = time.time()
        
        result = self.predict_single(text, top_k=self.config.TOP_K_PREDICTIONS)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        print(f"\n{'='*80}")
        print("PREDICTION RESULTS")
        print(f"{'='*80}")
        print(f"\nText: \"{text}\"")
        print(f"\nPredicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference Time: {inference_time:.2f}ms")
        
        print(f"\nTop {len(result['top_predictions'])} Predictions:")
        for pred in result['top_predictions']:
            bar_length = int(pred['probability'] * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"  {pred['rank']}. {pred['emotion']:15s} {bar} {pred['probability']:.4f}")
        
        print(f"\nFold Agreement (out of {result['num_models']} models):")
        for emotion, count in sorted(result['fold_agreement'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {emotion:15s}: {count}/{result['num_models']} models")
        
        print(f"\nEnsemble Method: {result['method']}")
    
    def show_example_predictions(self):
        """Show example predictions"""
        example_texts = [
            "I'm so happy today! Everything is going great!",
            "This is incredibly frustrating. Nothing works.",
            "I feel so alone and sad. Nobody understands me.",
            "I'm terrified of what might happen tomorrow.",
            "Wow! I can't believe I achieved this goal!",
            "I'm grateful for all the support from my friends.",
            "This makes me so angry! How could they do this?",
            "I'm worried about the upcoming exam.",
            "Life feels meaningless right now.",
            "I love spending time with my family."
        ]
        
        print(f"\n{'='*80}")
        print("EXAMPLE PREDICTIONS")
        print(f"{'='*80}")
        
        for i, text in enumerate(example_texts, 1):
            result = self.predict_single(text, top_k=3)
            
            print(f"\n{i}. \"{text}\"")
            print(f"   → {result['predicted_emotion']} ({result['confidence']:.3f})")
            
            top_3 = [f"{p['emotion']}({p['probability']:.2f})" 
                     for p in result['top_predictions'][:3]]
            print(f"   Top-3: {' | '.join(top_3)}")
    
    def show_model_stats(self):
        """Show model statistics"""
        print(f"\n{'='*80}")
        print("MODEL STATISTICS")
        print(f"{'='*80}")
        print(f"\nModel Architecture: {self.config.MODEL_NAME}")
        print(f"Number of Emotion Classes: {self.num_labels}")
        print(f"Number of Ensemble Models: {len(self.models)}")
        print(f"Ensemble Method: {self.config.ENSEMBLE_METHOD}")
        print(f"Max Sequence Length: {self.config.MAX_LENGTH}")
        
        if 'automated' in self.results:
            metrics = self.results['automated']['metrics']
            print(f"\nTest Set Performance:")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
            print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
            print(f"  Inference Speed: {self.results['automated']['samples_per_second']:.1f} samples/sec")
        
        print(f"\nTop 10 Emotion Classes:")
        for i, label in enumerate(self.label_names[:10]):
            print(f"  {i+1:2d}. {label}")
        
        if len(self.label_names) > 10:
            print(f"  ... and {len(self.label_names) - 10} more")
    
    def batch_file_testing(self):
        """Test multiple texts from a file"""
        print("\n" + "="*80)
        print("BATCH FILE TESTING")
        print("="*80)
        
        file_path = input("\nEnter file path (one text per line): ").strip()
        
        if not file_path:
            print("No file path provided.")
            return
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if not texts:
                print("No texts found in file.")
                return
            
            print(f"\nLoaded {len(texts)} texts from file.")
            print(f"Running predictions...")
            
            results = []
            for text in tqdm(texts, desc="Processing"):
                result = self.predict_single(text, top_k=3)
                results.append(result)
            
            # Display results
            print(f"\n{'='*80}")
            print("BATCH RESULTS")
            print(f"{'='*80}")
            
            for i, (text, result) in enumerate(zip(texts, results), 1):
                print(f"\n{i}. \"{text[:60]}...\"")
                print(f"   → {result['predicted_emotion']} ({result['confidence']:.3f})")
            
            # Save results
            save_option = input("\nSave results to file? (y/n): ").strip().lower()
            
            if save_option == 'y':
                output_path = self.config.OUTPUT_DIR / "predictions" / "batch_results.csv"
                
                results_df = pd.DataFrame([
                    {
                        'text': text,
                        'predicted_emotion': result['predicted_emotion'],
                        'confidence': result['confidence'],
                        'top_3': ' | '.join([f"{p['emotion']}({p['probability']:.3f})" 
                                            for p in result['top_predictions'][:3]])
                    }
                    for text, result in zip(texts, results)
                ])
                
                results_df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")
        
        except Exception as e:
            print(f"Error processing file: {e}")
    
    def benchmark_performance(self):
        """Benchmark model performance"""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARKING")
        print("="*80)
        
        test_texts = [
            "This is a test sentence for benchmarking.",
            "Another example to measure inference speed.",
            "Testing the model performance on various inputs."
        ] * 100  # 300 samples
        
        batch_sizes = [1, 4, 8, 16, 32]
        
        print("\nBenchmarking different batch sizes...")
        
        for batch_size in batch_sizes:
            original_batch_size = self.config.BATCH_SIZE
            self.config.BATCH_SIZE = batch_size
            
            start_time = time.time()
            _ = self.predict_batch(test_texts, method='average')
            elapsed_time = time.time() - start_time
            
            throughput = len(test_texts) / elapsed_time
            latency = elapsed_time / len(test_texts) * 1000  # ms per sample
            
            print(f"\nBatch Size: {batch_size}")
            print(f"  Total Time: {elapsed_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            print(f"  Latency: {latency:.2f}ms/sample")
            
            self.config.BATCH_SIZE = original_batch_size
    
    def compare_ensemble_methods(self):
        """Compare different ensemble methods"""
        if self.test_data is None:
            self.load_test_data()
        
        print("\n" + "="*80)
        print("ENSEMBLE METHOD COMPARISON")
        print("="*80)
        
        texts = self.test_data['texts']
        true_labels = self.test_data['labels']
        
        methods = ['voting', 'average']
        results = {}
        
        for method in methods:
            print(f"\nTesting {method} method...")
            result = self.predict_batch(texts, method=method)
            
            metrics = self.calculate_metrics(
                true_labels, 
                result['predictions'], 
                result['probabilities']
            )
            
            results[method] = metrics
        
        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}")
        
        for method in methods:
            print(f"\n{method.upper()} Method:")
            print(f"  Accuracy:    {results[method]['accuracy']:.4f}")
            print(f"  F1 Macro:    {results[method]['f1_macro']:.4f}")
            print(f"  F1 Weighted: {results[method]['f1_weighted']:.4f}")
        
        best_method = max(results.items(), key=lambda x: x[1]['f1_macro'])
        print(f"\nBest Method: {best_method[0].upper()} (F1 Macro: {best_method[1]['f1_macro']:.4f})")


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Testing Suite for DeBERTa Emotion Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run automated testing
  python test_model.py --mode auto
  
  # Manual testing mode
  python test_model.py --mode manual
  
  # Both automated and manual
  python test_model.py --mode both
  
  # Benchmark performance
  python test_model.py --mode benchmark
  
  # Compare ensemble methods
  python test_model.py --mode compare
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='both',
        choices=['auto', 'manual', 'both', 'benchmark', 'compare'],
        help='Testing mode (default: both)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Path to model directory (default: from config)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to data directory (default: from config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Path to output directory (default: from config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
    )
    
    parser.add_argument(
        '--ensemble-method',
        type=str,
        default='average',
        choices=['voting', 'average'],
        help='Ensemble method (default: average)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable saving predictions'
    )
    
    return parser.parse_args()


def main():
    """Main execution"""
    args = parse_args()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL TESTING SUITE")
    print("="*80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    # Setup configuration
    config = TestConfig()
    
    if args.model_dir:
        config.MODEL_DIR = Path(args.model_dir)
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
    
    config.BATCH_SIZE = args.batch_size
    config.ENSEMBLE_METHOD = args.ensemble_method
    config.GENERATE_PLOTS = not args.no_plots
    config.SAVE_PREDICTIONS = not args.no_save
    
    try:
        # Initialize tester
        tester = ComprehensiveTester(config)
        
        # Run based on mode
        if args.mode in ['auto', 'both']:
            tester.automated_test()
        
        if args.mode == 'benchmark':
            tester.benchmark_performance()
        
        if args.mode == 'compare':
            tester.compare_ensemble_methods()
        
        if args.mode in ['manual', 'both']:
            tester.manual_testing_mode()
        
        print("\n" + "="*80)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {config.OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()