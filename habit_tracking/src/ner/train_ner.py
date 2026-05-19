"""
Train NER Model for Habit Recognition using HuggingFace Transformers
Supports: BERT, RoBERTa, DeBERTa for token classification
Includes: Class weighting, early stopping, comprehensive evaluation
PyTorch-only implementation (no TensorFlow dependencies)
"""

import json
import argparse
import warnings
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Fix Windows Unicode issues
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Disable TensorFlow
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

# PyTorch imports
import torch
from torch.utils.data import Dataset

# HuggingFace imports - import only what we need to avoid TF
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback
    )
    from datasets import Dataset as HFDataset
    from evaluate import load as load_metric
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("\nPlease install required packages:")
    print("pip install torch transformers datasets evaluate seqeval")
    sys.exit(1)

warnings.filterwarnings('ignore')


class NERDataset:
    """Custom dataset class for NER training"""
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 128
    ):
        """
        Initialize NER dataset
        
        Args:
            jsonl_path: Path to JSONL file with BIO data
            tokenizer: HuggingFace tokenizer
            label2id: Mapping from label strings to IDs
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
        # Load data
        print(f"Loading data from: {jsonl_path}")
        self.examples = self._load_jsonl(jsonl_path)
        print(f"[OK] Loaded {len(self.examples)} examples")
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file"""
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        return examples
    
    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        """
        Tokenize and align labels for batch of examples
        
        This is crucial: spaCy tokens may not match subword tokens,
        so we need to align BIO tags to subword tokens
        """
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding=False  # Will pad in DataCollator
        )
        
        labels = []
        for i, label_list in enumerate(examples['tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens get -100 (ignored in loss)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word gets the label
                    label_ids.append(self.label2id[label_list[word_idx]])
                else:
                    # Subsequent subwords of same word
                    # Option 1: Use same label (uncomment below)
                    # label_ids.append(self.label2id[label_list[word_idx]])
                    # Option 2: Ignore in loss (standard practice)
                    label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def to_hf_dataset(self) -> HFDataset:
        """Convert to HuggingFace Dataset format"""
        
        # Prepare data dict
        data_dict = {
            'id': [ex['id'] for ex in self.examples],
            'tokens': [ex['tokens'] for ex in self.examples],
            'tags': [ex['tags'] for ex in self.examples]
        }
        
        # Create HuggingFace dataset
        dataset = HFDataset.from_dict(data_dict)
        
        # Tokenize and align labels
        tokenized_dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset


class NERTrainer:
    """Trainer for habit NER model"""
    
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        label_mappings_path: str = None,
        output_dir: str = "models/ner/hf_ner",
        use_class_weights: bool = True,
        device: str = None
    ):
        """
        Initialize NER trainer
        
        Args:
            model_name: Pretrained model name (bert-base-cased, roberta-base, etc.)
            label_mappings_path: Path to label_mappings.json
            output_dir: Directory to save model
            use_class_weights: Whether to use class weights for imbalanced data
            device: Device (cuda/cpu), auto-detected if None
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_class_weights = use_class_weights
        
        # Detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load label mappings
        print(f"Loading label mappings from: {label_mappings_path}")
        with open(label_mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        self.label2id = mappings['tag2id']
        self.id2label = {int(k): v for k, v in mappings['id2tag'].items()}
        self.num_labels = mappings['num_tags']
        
        print(f"[OK] Loaded {self.num_labels} labels")
        
        # Initialize tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        print(f"Loading model: {model_name}")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        
        self.model.to(self.device)
        print(f"[OK] Model initialized with {self.num_labels} labels")
        
        # Metrics
        try:
            self.seqeval = load_metric("seqeval")
        except Exception as e:
            print(f"Warning: Could not load seqeval metric: {e}")
            self.seqeval = None
    
    def compute_class_weights(self, train_dataset: HFDataset) -> torch.Tensor:
        """Compute class weights for imbalanced datasets"""
        
        print("\nComputing class weights...")
        
        # Collect all labels (excluding -100)
        all_labels = []
        for example in train_dataset:
            labels = [l for l in example['labels'] if l != -100]
            all_labels.extend(labels)
        
        # Count label frequencies
        label_counts = Counter(all_labels)
        print(f"Label distribution:")
        for label_id in sorted(label_counts.keys()):
            label_name = self.id2label[label_id]
            count = label_counts[label_id]
            print(f"  {label_name:20s}: {count:6d}")
        
        # Compute inverse frequency weights
        total_samples = len(all_labels)
        weights = []
        
        for label_id in range(self.num_labels):
            count = label_counts.get(label_id, 1)  # Avoid division by zero
            weight = total_samples / (self.num_labels * count)
            weights.append(weight)
        
        # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum() * len(weights)
        
        print(f"\nClass weights:")
        for label_id, weight in enumerate(weights):
            label_name = self.id2label[label_id]
            print(f"  {label_name:20s}: {weight:.4f}")
        
        return weights.to(self.device)
    
    def compute_metrics(self, eval_pred) -> Dict:
        """Compute evaluation metrics"""
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens) and convert to labels
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            true_pred = []
            true_label = []
            
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_pred.append(self.id2label[pred_id])
                    true_label.append(self.id2label[label_id])
            
            true_predictions.append(true_pred)
            true_labels.append(true_label)
        
        # Compute metrics using seqeval if available
        if self.seqeval is not None:
            try:
                results = self.seqeval.compute(
                    predictions=true_predictions,
                    references=true_labels,
                    mode='strict',
                    scheme='IOB2'
                )
                
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"]
                }
            except Exception as e:
                print(f"Warning: seqeval failed, using simple accuracy: {e}")
        
        # Fallback to simple accuracy
        correct = sum(
            1 for pred_seq, label_seq in zip(true_predictions, true_labels)
            for pred, label in zip(pred_seq, label_seq)
            if pred == label
        )
        total = sum(len(label_seq) for label_seq in true_labels)
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    def train(
        self,
        train_dataset: HFDataset,
        eval_dataset: Optional[HFDataset] = None,
        training_args: TrainingArguments = None
    ):
        """Train the NER model"""
        
        print(f"\n{'='*80}")
        print("TRAINING NER MODEL")
        print(f"{'='*80}")
        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Validation samples: {len(eval_dataset)}")
        
        # Compute class weights if requested
        class_weights = None
        if self.use_class_weights:
            class_weights = self.compute_class_weights(train_dataset)
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Custom Trainer with class weights
        if class_weights is not None:
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Compute weighted cross-entropy loss
                    loss_fct = torch.nn.CrossEntropyLoss(
                        weight=class_weights,
                        ignore_index=-100
                    )
                    loss = loss_fct(
                        logits.view(-1, self.model.config.num_labels),
                        labels.view(-1)
                    )
                    
                    return (loss, outputs) if return_outputs else loss
            
            trainer_class = WeightedTrainer
        else:
            trainer_class = Trainer
        
        # Initialize trainer
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print(f"\nStarting training...")
        train_result = trainer.train()
        
        # Save model
        print(f"\nSaving model to: {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        print(f"[OK] Training completed")
        print(f"\nTraining metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluate on validation set
        if eval_dataset:
            print(f"\n{'='*80}")
            print("EVALUATING ON VALIDATION SET")
            print(f"{'='*80}")
            
            eval_results = trainer.evaluate()
            
            print(f"\nValidation metrics:")
            for key, value in eval_results.items():
                print(f"  {key}: {value:.4f}")
            
            trainer.log_metrics("eval", eval_results)
            trainer.save_metrics("eval", eval_results)
            
            return train_result, eval_results
        
        return train_result, None
    
    def predict(
        self,
        test_dataset: HFDataset,
        output_path: Optional[str] = None
    ) -> Dict:
        """Run predictions on test set"""
        
        print(f"\n{'='*80}")
        print("RUNNING PREDICTIONS")
        print(f"{'='*80}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Create trainer for inference
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Predict
        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)
        
        # Convert to readable format
        results = []
        for i, (prediction, label) in enumerate(zip(predictions, labels)):
            pred_labels = []
            true_labels = []
            
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    pred_labels.append(self.id2label[pred_id])
                    true_labels.append(self.id2label[label_id])
            
            results.append({
                'predictions': pred_labels,
                'true_labels': true_labels
            })
        
        print(f"\nTest metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save predictions if path provided
        if output_path:
            print(f"\nSaving predictions to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': results,
                    'metrics': metrics
                }, f, indent=2)
        
        return metrics


def load_and_split_data(
    train_path: str,
    val_path: Optional[str],
    test_path: Optional[str],
    val_split: float,
    tokenizer,
    label2id: Dict,
    max_length: int
) -> Tuple[HFDataset, Optional[HFDataset], Optional[HFDataset]]:
    """Load data and create train/val/test splits"""
    
    print(f"\n{'='*80}")
    print("LOADING AND PREPARING DATA")
    print(f"{'='*80}")
    
    # Load training data
    train_dataset_obj = NERDataset(
        train_path,
        tokenizer,
        label2id,
        max_length
    )
    train_dataset = train_dataset_obj.to_hf_dataset()
    
    # Load or split validation data
    if val_path:
        print(f"\nLoading validation data from: {val_path}")
        val_dataset_obj = NERDataset(
            val_path,
            tokenizer,
            label2id,
            max_length
        )
        val_dataset = val_dataset_obj.to_hf_dataset()
    elif val_split > 0:
        print(f"\nSplitting training data (val_split={val_split})")
        split = train_dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split['train']
        val_dataset = split['test']
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
    else:
        val_dataset = None
    
    # Load test data
    if test_path:
        print(f"\nLoading test data from: {test_path}")
        test_dataset_obj = NERDataset(
            test_path,
            tokenizer,
            label2id,
            max_length
        )
        test_dataset = test_dataset_obj.to_hf_dataset()
    else:
        test_dataset = None
    
    return train_dataset, val_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Train NER model for habit recognition")
    
    # Data paths
    parser.add_argument('--train', type=str, required=True,
                       help="Path to training JSONL file")
    parser.add_argument('--val', type=str, default=None,
                       help="Path to validation JSONL file")
    parser.add_argument('--test', type=str, default=None,
                       help="Path to test JSONL file")
    parser.add_argument('--label-mappings', type=str, default=None,
                       help="Path to label_mappings.json (auto-detected if not provided)")
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='bert-base-cased',
                       help="Pretrained model name (bert-base-cased, roberta-base, microsoft/deberta-base)")
    parser.add_argument('--output-dir', type=str, default='models/ner/hf_ner',
                       help="Output directory for trained model")
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=16,
                       help="Training batch size")
    parser.add_argument('--num-epochs', type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument('--max-length', type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help="Weight decay")
    
    # Training options
    parser.add_argument('--val-split', type=float, default=0.15,
                       help="Validation split ratio (if --val not provided)")
    parser.add_argument('--no-class-weights', action='store_true',
                       help="Disable class weighting")
    parser.add_argument('--fp16', action='store_true',
                       help="Use mixed precision training (requires GPU)")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help="Gradient accumulation steps")
    
    # Evaluation options
    parser.add_argument('--eval-steps', type=int, default=100,
                       help="Evaluation frequency")
    parser.add_argument('--save-steps', type=int, default=500,
                       help="Model save frequency")
    parser.add_argument('--logging-steps', type=int, default=50,
                       help="Logging frequency")
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    parser.add_argument('--device', type=str, default=None,
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Auto-detect label mappings path
    if args.label_mappings is None:
        train_dir = Path(args.train).parent
        args.label_mappings = str(train_dir / 'label_mappings.json')
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"{'='*80}")
    print("NER MODEL TRAINING")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Training data: {args.train}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device or 'auto'}")
    
    # Initialize trainer
    trainer = NERTrainer(
        model_name=args.model_name,
        label_mappings_path=args.label_mappings,
        output_dir=args.output_dir,
        use_class_weights=not args.no_class_weights,
        device=args.device
    )
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_split_data(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        val_split=args.val_split if not args.val else 0,
        tokenizer=trainer.tokenizer,
        label2id=trainer.label2id,
        max_length=args.max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="f1" if val_dataset else None,
        greater_is_better=True,
        save_total_limit=3,
        seed=args.seed,
        report_to="none"  # Disable wandb/tensorboard for now
    )
    
    # Train model
    train_result, eval_result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args
    )
    
    # Test if test set provided
    if test_dataset:
        test_metrics = trainer.predict(
            test_dataset=test_dataset,
            output_path=str(Path(args.output_dir) / 'test_predictions.json')
        )
    
    print(f"\n{'='*80}")
    print("[OK] NER TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nModel saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review metrics in: {args.output_dir}")
    print(f"  2. Use model for inference")
    print(f"  3. Integrate into canonicalization pipeline")


if __name__ == "__main__":
    main()