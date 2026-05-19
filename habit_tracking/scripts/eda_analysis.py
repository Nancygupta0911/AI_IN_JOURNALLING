"""
Comprehensive EDA for Journal Emotion Datasets
Analyzes text data for NLP tasks including emotion detection and habit tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Text processing
import re
from typing import Dict, List, Tuple
import spacy

# For text statistics
from scipy import stats

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")  # Using a valid seaborn palette
RANDOM_STATE = 42

class JournalDatasetEDA:
    """Comprehensive EDA for journal emotion datasets"""
    
    def __init__(self, output_dir: str = "data/processed/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {}
        
        # Try to load spacy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def load_datasets(self, data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
        """Load all available datasets"""
        data_path = Path(data_dir)
        datasets = {}
        
        print("📂 Loading datasets...\n")
        
        # Load each dataset
        files = {
            'emotions_dataset': 'emotions_dataset.parquet',
            'emotion_dataset_2': 'emotion_dataset_2.csv',
            'goemotions': 'goemotions.csv',
            'test_journals': 'test_journals.parquet',
            'Daylio_journals': 'Daylio_Abid.csv'
        }
        
        for name, file in files.items():
            filepath = data_path / file
            if filepath.exists():
                try:
                    if file.endswith('.parquet'):
                        df = pd.read_parquet(filepath)
                    else:
                        df = pd.read_csv(filepath)
                    datasets[name] = df
                    print(f"✓ Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    print(f"✗ Error loading {name}: {e}")
            else:
                print(f"✗ File not found: {filepath}")
        
        print(f"\n📊 Total datasets loaded: {len(datasets)}\n")
        return datasets
    
    def analyze_dataset_structure(self, datasets: Dict[str, pd.DataFrame]):
        """Analyze structure and schema of each dataset"""
        print("="*80)
        print("DATASET STRUCTURE ANALYSIS")
        print("="*80 + "\n")
        
        for name, df in datasets.items():
            print(f"\n📋 Dataset: {name}")
            print("-" * 60)
            print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"\nColumns and Types:")
            print(df.dtypes.to_string())
            print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Missing values
            missing = df.isnull().sum()
            if missing.any():
                print(f"\n⚠️ Missing Values:")
                print(missing[missing > 0].to_string())
            else:
                print("\n✓ No missing values")
            
            # Sample data
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string())
            print("\n" + "="*60)
        
        # Save structure info
        structure_info = {}
        for name, df in datasets.items():
            structure_info[name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
            }
        
        with open(self.output_dir / 'dataset_structure.json', 'w') as f:
            json.dump(structure_info, f, indent=2)
    
    def identify_text_emotion_columns(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Automatically identify text and emotion/label columns"""
        print("\n" + "="*80)
        print("IDENTIFYING TEXT AND LABEL COLUMNS")
        print("="*80 + "\n")
        
        column_mapping = {}
        
        for name, df in datasets.items():
            text_col = None
            emotion_col = None
            
            # Common text column names
            text_candidates = ['text', 'content', 'journal', 'entry', 'message', 'comment']
            for col in df.columns:
                if col.lower() in text_candidates or 'text' in col.lower():
                    text_col = col
                    break
            
            # Common emotion/label column names
            emotion_candidates = ['emotion', 'label', 'emotions', 'sentiment', 'feeling']
            for col in df.columns:
                if col.lower() in emotion_candidates or 'label' in col.lower() or 'emotion' in col.lower():
                    emotion_col = col
                    break
            
            column_mapping[name] = {
                'text_column': text_col,
                'emotion_column': emotion_col,
                'all_columns': df.columns.tolist()
            }
            
            print(f"{name}:")
            print(f"  Text column: {text_col}")
            print(f"  Emotion column: {emotion_col}")
            print()
        
        return column_mapping
    
    def analyze_text_statistics(self, datasets: Dict[str, pd.DataFrame], 
                                column_mapping: Dict[str, Dict]):
        """Comprehensive text statistics analysis"""
        print("="*80)
        print("TEXT STATISTICS ANALYSIS")
        print("="*80 + "\n")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Text Length Distributions Across Datasets', fontsize=16, fontweight='bold')
        
        all_stats = {}
        
        for idx, (name, df) in enumerate(datasets.items()):
            text_col = column_mapping[name]['text_column']
            
            if text_col is None:
                print(f"⚠️ No text column found in {name}, skipping...")
                continue
            
            print(f"\n📝 Analyzing: {name}")
            print("-" * 60)
            
            # Ensure text is string
            df[text_col] = df[text_col].astype(str)
            
            # Calculate statistics
            df['char_count'] = df[text_col].str.len()
            df['word_count'] = df[text_col].str.split().str.len()
            df['sentence_count'] = df[text_col].str.count(r'[.!?]+')
            df['avg_word_length'] = df[text_col].apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            )
            
            stats_dict = {
                'total_entries': len(df),
                'char_count': {
                    'mean': float(df['char_count'].mean()),
                    'median': float(df['char_count'].median()),
                    'std': float(df['char_count'].std()),
                    'min': int(df['char_count'].min()),
                    'max': int(df['char_count'].max())
                },
                'word_count': {
                    'mean': float(df['word_count'].mean()),
                    'median': float(df['word_count'].median()),
                    'std': float(df['word_count'].std()),
                    'min': int(df['word_count'].min()),
                    'max': int(df['word_count'].max())
                },
                'avg_word_length': {
                    'mean': float(df['avg_word_length'].mean())
                }
            }
            
            all_stats[name] = stats_dict
            
            # Print statistics
            print(f"Total entries: {len(df):,}")
            print(f"\nCharacter count: mean={stats_dict['char_count']['mean']:.1f}, "
                  f"median={stats_dict['char_count']['median']:.1f}, "
                  f"range=[{stats_dict['char_count']['min']}, {stats_dict['char_count']['max']}]")
            print(f"Word count: mean={stats_dict['word_count']['mean']:.1f}, "
                  f"median={stats_dict['word_count']['median']:.1f}, "
                  f"range=[{stats_dict['word_count']['min']}, {stats_dict['word_count']['max']}]")
            print(f"Avg word length: {stats_dict['avg_word_length']['mean']:.2f} chars")
            
            # Plot distributions (if we have enough datasets)
            if idx < 4:
                ax_idx = idx
                ax = axes[ax_idx // 2, ax_idx % 2]
                
                # Plot word count distribution
                ax.hist(df['word_count'], bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(df['word_count'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {df["word_count"].mean():.1f}')
                ax.axvline(df['word_count'].median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {df["word_count"].median():.1f}')
                ax.set_xlabel('Word Count', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{name}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(datasets), 4):
            fig.delaxes(axes[idx // 2, idx % 2])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'text_length_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: text_length_distributions.png")
        
        # Save statistics
        with open(self.output_dir / 'text_statistics.json', 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        return all_stats
    
    def analyze_emotion_distribution(self, datasets: Dict[str, pd.DataFrame], 
                                    column_mapping: Dict[str, Dict]):
        """Analyze emotion label distribution and class imbalance"""
        print("\n" + "="*80)
        print("EMOTION LABEL DISTRIBUTION & IMBALANCE ANALYSIS")
        print("="*80 + "\n")
        
        emotion_stats = {}
        
        for name, df in datasets.items():
            emotion_col = column_mapping[name]['emotion_column']
            
            if emotion_col is None:
                print(f"⚠️ No emotion column found in {name}, skipping...")
                continue
            
            print(f"\n🎭 Analyzing: {name}")
            print("-" * 60)
            
            # Initialize flags
            is_multilabel = False
            
            # Handle multi-label (comma-separated) or single-label
            if df[emotion_col].dtype == 'object':
                # Check if multi-label
                sample = str(df[emotion_col].iloc[0])
                is_multilabel = ',' in sample or '[' in sample
                
                if is_multilabel:
                    # Flatten multi-label
                    all_emotions = []
                    for labels in df[emotion_col]:
                        if pd.notna(labels):
                            labels_str = str(labels).replace('[', '').replace(']', '').replace("'", "")
                            all_emotions.extend([l.strip() for l in labels_str.split(',')])
                    emotion_counts = Counter(all_emotions)
                else:
                    emotion_counts = df[emotion_col].value_counts()
            else:
                emotion_counts = df[emotion_col].value_counts()
            
            # Convert to dictionary for consistent handling
            if isinstance(emotion_counts, pd.Series):
                emotion_counts_dict = emotion_counts.to_dict()
            else:
                emotion_counts_dict = dict(emotion_counts)
            
            # Calculate imbalance metrics
            total = sum(emotion_counts_dict.values())
            n_classes = len(emotion_counts_dict)
            
            max_count = max(emotion_counts_dict.values())
            min_count = min(emotion_counts_dict.values())
            
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            emotion_stats[name] = {
                'n_classes': n_classes,
                'total_labels': total,
                'distribution': emotion_counts_dict,
                'imbalance_ratio': float(imbalance_ratio),
                'is_multilabel': is_multilabel
            }
            
            print(f"Number of emotion classes: {n_classes}")
            print(f"Total labels: {total:,}")
            print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
            print(f"\nTop 10 emotions:")
            sorted_emotions = sorted(emotion_counts_dict.items(), key=lambda x: x[1], reverse=True)
            for emotion, count in sorted_emotions[:10]:
                percentage = (count / total) * 100
                print(f"  {emotion}: {count:,} ({percentage:.2f}%)")
            
            # Calculate Imbalance Level
            if imbalance_ratio < 2:
                imbalance_level = "Balanced ✓"
            elif imbalance_ratio < 5:
                imbalance_level = "Slightly Imbalanced ⚠️"
            elif imbalance_ratio < 10:
                imbalance_level = "Moderately Imbalanced ⚠️⚠️"
            else:
                imbalance_level = "Highly Imbalanced ❗❗"
            
            print(f"\n📊 Imbalance Assessment: {imbalance_level}")
            
            # Visualize distribution
            plt.figure(figsize=(14, 6))
            
            sorted_emotions = sorted(emotion_counts_dict.items(), key=lambda x: x[1], reverse=True)
            emotions = [e[0] for e in sorted_emotions[:20]]  # Top 20
            counts = [e[1] for e in sorted_emotions[:20]]
            
            bars = plt.bar(range(len(emotions)), counts, alpha=0.8, edgecolor='black')
            
            # Color bars by frequency (gradient)
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.xlabel('Emotion Labels', fontsize=12, fontweight='bold')
            plt.ylabel('Frequency', fontsize=12, fontweight='bold')
            plt.title(f'Emotion Distribution: {name}\n{imbalance_level}', 
                     fontsize=14, fontweight='bold')
            plt.xticks(range(len(emotions)), emotions, rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            safe_name = name.replace(' ', '_').replace('/', '_')
            plt.savefig(self.output_dir / f'emotion_distribution_{safe_name}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: emotion_distribution_{safe_name}.png")
            plt.close()
        
        # Save emotion statistics
        # Convert numpy types to native Python types for JSON serialization
        for name in emotion_stats:
            if 'distribution' in emotion_stats[name]:
                emotion_stats[name]['distribution'] = {
                    k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                    for k, v in emotion_stats[name]['distribution'].items()
                }
        
        with open(self.output_dir / 'emotion_statistics.json', 'w') as f:
            json.dump(emotion_stats, f, indent=2)
        
        return emotion_stats
    
    def analyze_linguistic_features(self, datasets: Dict[str, pd.DataFrame], 
                                   column_mapping: Dict[str, Dict]):
        """Analyze linguistic features relevant for habit tracking"""
        print("\n" + "="*80)
        print("LINGUISTIC FEATURES ANALYSIS")
        print("="*80 + "\n")
        
        if self.nlp is None:
            print("⚠️ Skipping linguistic analysis (spaCy not available)")
            return {}
        
        linguistic_stats = {}
        
        # Habit-related patterns
        habit_patterns = {
            'time_expressions': r'\b(?:morning|afternoon|evening|night|today|yesterday|daily|weekly)\b',
            'duration_expressions': r'\b\d+\s*(?:hour|hr|minute|min|day|week|month)s?\b',
            'action_verbs': r'\b(?:did|went|ate|drank|exercised|studied|worked|slept|watched|played)\b',
            'frequency_adverbs': r'\b(?:always|often|sometimes|rarely|never|usually)\b',
            'negation': r'\b(?:not|no|never|didn\'t|don\'t|hasn\'t|haven\'t)\b'
        }
        
        for name, df in datasets.items():
            text_col = column_mapping[name]['text_column']
            
            if text_col is None:
                continue
            
            print(f"\n🔍 Analyzing: {name}")
            print("-" * 60)
            
            # Sample for efficiency (if dataset is large)
            sample_size = min(1000, len(df))
            df_sample = df.sample(n=sample_size, random_state=RANDOM_STATE) if len(df) > sample_size else df
            
            # Initialize counters
            feature_counts = {pattern: 0 for pattern in habit_patterns.keys()}
            pos_counts = Counter()
            entity_counts = Counter()
            
            for text in df_sample[text_col]:
                text_str = str(text).lower()
                
                # Count pattern matches
                for pattern_name, pattern in habit_patterns.items():
                    matches = len(re.findall(pattern, text_str, re.IGNORECASE))
                    feature_counts[pattern_name] += matches
                
                # POS tagging and NER (sample further for speed)
                if len(text_str) < 500:  # Only process shorter texts
                    try:
                        doc = self.nlp(text_str[:500])
                        pos_counts.update([token.pos_ for token in doc])
                        entity_counts.update([ent.label_ for ent in doc.ents])
                    except:
                        pass
            
            linguistic_stats[name] = {
                'sample_size': sample_size,
                'habit_patterns': feature_counts,
                'top_pos_tags': dict(pos_counts.most_common(10)),
                'top_entities': dict(entity_counts.most_common(10))
            }
            
            print(f"Sample size: {sample_size}")
            print(f"\n📌 Habit-related patterns (total occurrences):")
            for pattern, count in feature_counts.items():
                print(f"  {pattern}: {count}")
            
            print(f"\n📌 Top POS tags:")
            for pos, count in pos_counts.most_common(5):
                print(f"  {pos}: {count}")
        
        # Save linguistic statistics
        with open(self.output_dir / 'linguistic_features.json', 'w') as f:
            json.dump(linguistic_stats, f, indent=2)
        
        return linguistic_stats
    
    def analyze_vocabulary(self, datasets: Dict[str, pd.DataFrame], 
                          column_mapping: Dict[str, Dict]):
        """Analyze vocabulary richness and common terms"""
        print("\n" + "="*80)
        print("VOCABULARY ANALYSIS")
        print("="*80 + "\n")
        
        vocab_stats = {}
        
        for name, df in datasets.items():
            text_col = column_mapping[name]['text_column']
            
            if text_col is None:
                continue
            
            print(f"\n📚 Analyzing: {name}")
            print("-" * 60)
            
            # Combine all text
            all_text = ' '.join(df[text_col].astype(str).tolist()).lower()
            
            # Tokenize
            words = re.findall(r'\b[a-z]+\b', all_text)
            
            # Calculate statistics
            unique_words = set(words)
            word_freq = Counter(words)
            
            # Type-Token Ratio (vocabulary richness)
            ttr = len(unique_words) / len(words) if words else 0
            
            vocab_stats[name] = {
                'total_words': len(words),
                'unique_words': len(unique_words),
                'type_token_ratio': float(ttr),
                'top_30_words': dict(word_freq.most_common(30))
            }
            
            print(f"Total words: {len(words):,}")
            print(f"Unique words: {len(unique_words):,}")
            print(f"Type-Token Ratio (TTR): {ttr:.4f}")
            print(f"  (Higher TTR = more diverse vocabulary)")
            
            print(f"\n🔤 Top 20 most common words:")
            for word, count in word_freq.most_common(20):
                print(f"  {word}: {count:,}")
        
        # Save vocabulary statistics
        with open(self.output_dir / 'vocabulary_statistics.json', 'w') as f:
            json.dump(vocab_stats, f, indent=2)
        
        return vocab_stats
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80 + "\n")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("JOURNAL DATASET EDA SUMMARY REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        
        # Load all saved statistics
        stats_files = {
            'structure': 'dataset_structure.json',
            'text': 'text_statistics.json',
            'emotions': 'emotion_statistics.json',
            'linguistic': 'linguistic_features.json',
            'vocabulary': 'vocabulary_statistics.json'
        }
        
        all_stats = {}
        for stat_type, filename in stats_files.items():
            filepath = self.output_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    all_stats[stat_type] = json.load(f)
        
        # Dataset Overview
        if 'structure' in all_stats:
            report_lines.append("\n\n📊 DATASET OVERVIEW")
            report_lines.append("-" * 60)
            for name, info in all_stats['structure'].items():
                report_lines.append(f"\n{name}:")
                report_lines.append(f"  Rows: {info['shape'][0]:,}")
                report_lines.append(f"  Columns: {info['shape'][1]}")
                report_lines.append(f"  Memory: {info['memory_mb']:.2f} MB")
        
        # Text Statistics
        if 'text' in all_stats:
            report_lines.append("\n\n📝 TEXT STATISTICS SUMMARY")
            report_lines.append("-" * 60)
            for name, stats in all_stats['text'].items():
                report_lines.append(f"\n{name}:")
                report_lines.append(f"  Avg words per entry: {stats['word_count']['mean']:.1f}")
                report_lines.append(f"  Avg chars per entry: {stats['char_count']['mean']:.1f}")
        
        # Emotion Distribution
        if 'emotions' in all_stats:
            report_lines.append("\n\n🎭 EMOTION DISTRIBUTION SUMMARY")
            report_lines.append("-" * 60)
            for name, stats in all_stats['emotions'].items():
                report_lines.append(f"\n{name}:")
                report_lines.append(f"  Number of classes: {stats['n_classes']}")
                report_lines.append(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}")
                
                if stats['imbalance_ratio'] < 2:
                    imbalance = "✓ Balanced"
                elif stats['imbalance_ratio'] < 10:
                    imbalance = "⚠️ Moderately Imbalanced"
                else:
                    imbalance = "❗ Highly Imbalanced - Consider resampling/weighting"
                
                report_lines.append(f"  Status: {imbalance}")
        
        # Recommendations
        report_lines.append("\n\n💡 RECOMMENDATIONS FOR HABIT TRACKING PIPELINE")
        report_lines.append("-" * 60)
        
        if 'emotions' in all_stats:
            for name, stats in all_stats['emotions'].items():
                if stats['imbalance_ratio'] > 10:
                    report_lines.append(f"\n⚠️ {name}:")
                    report_lines.append("  - HIGH class imbalance detected")
                    report_lines.append("  - Recommend: Class weighting in loss function")
                    report_lines.append("  - Consider: SMOTE, focal loss, or stratified sampling")
                
                if stats['n_classes'] > 20:
                    report_lines.append(f"\n💭 {name}:")
                    report_lines.append(f"  - Large number of classes ({stats['n_classes']})")
                    report_lines.append("  - Consider: Hierarchical classification or emotion grouping")
        
        if 'text' in all_stats:
            for name, stats in all_stats['text'].items():
                if stats['word_count']['max'] > 512:
                    report_lines.append(f"\n⚠️ {name}:")
                    report_lines.append(f"  - Max word count: {stats['word_count']['max']}")
                    report_lines.append("  - Recommend: Truncation or sliding window for transformer models")
        
        report_lines.append("\n\n✓ All visualizations saved to: " + str(self.output_dir))
        report_lines.append("="*80)
        
        # Print and save report
        report_text = '\n'.join(report_lines)
        print(report_text)
        
        with open(self.output_dir / 'eda_summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Summary report saved to: eda_summary_report.txt")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("JOURNAL DATASET EDA")
    print("Comprehensive Analysis for Emotion Detection & Habit Tracking")
    print("="*80 + "\n")
    
    # Initialize EDA
    eda = JournalDatasetEDA(output_dir="data/processed/visualizations")
    
    # Load datasets
    datasets = eda.load_datasets(data_dir="data/raw")
    
    if not datasets:
        print("❌ No datasets found! Please check your data/raw directory.")
        return
    
    # Run analyses
    print("\n🚀 Starting comprehensive EDA...\n")
    
    # 1. Structure analysis
    eda.analyze_dataset_structure(datasets)
    
    # 2. Identify columns
    column_mapping = eda.identify_text_emotion_columns(datasets)
    
    # 3. Text statistics
    eda.analyze_text_statistics(datasets, column_mapping)
    
    # 4. Emotion distribution and imbalance
    eda.analyze_emotion_distribution(datasets, column_mapping)
    
    # 5. Linguistic features
    eda.analyze_linguistic_features(datasets, column_mapping)
    
    # 6. Vocabulary analysis
    eda.analyze_vocabulary(datasets, column_mapping)
    
    # 7. Generate summary report
    eda.generate_summary_report()
    
    print("\n" + "="*80)
    print("✅ EDA COMPLETE!")
    print("="*80)
    print(f"\n📁 All outputs saved to: data/processed/visualizations/")
    print("\nGenerated files:")
    print("  - dataset_structure.json")
    print("  - text_statistics.json")
    print("  - emotion_statistics.json")
    print("  - linguistic_features.json")
    print("  - vocabulary_statistics.json")
    print("  - text_length_distributions.png")
    print("  - emotion_distribution_*.png (per dataset)")
    print("  - eda_summary_report.txt")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()