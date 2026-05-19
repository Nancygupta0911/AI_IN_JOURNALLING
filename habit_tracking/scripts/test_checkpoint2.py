"""
Checkpoint 2 Validation Script
Test weak supervision pipeline
"""

import subprocess
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def print_step(step_num, text):
    """Print step header"""
    print(f"\n{'='*70}")
    print(f"Step {step_num}: {text}")
    print('='*70)


def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"\n▶ {description}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False


def check_file(filepath, description):
    """Check if file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} not found: {filepath}")
        return False


def validate_weak_labels(filepath):
    """Validate weak supervision output"""
    print("\n--- Validating Weak Labels ---")
    
    try:
        df = pd.read_parquet(filepath)
        print(f"✓ Loaded {len(df)} labeled spans")
        
        # Check required columns
        required_cols = ['span', 'weak_label', 'weak_label_name', 'max_prob', 'num_lfs_voted']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"✗ Missing columns: {missing}")
            return False
        print(f"✓ All required columns present")
        
        # Check label distribution
        print(f"\nLabel distribution:")
        label_counts = df['weak_label_name'].value_counts()
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Check coverage
        non_abstain = df[df['weak_label_name'] != 'ABSTAIN']
        coverage = len(non_abstain) / len(df) * 100
        print(f"\nCoverage: {len(non_abstain)} / {len(df)} ({coverage:.1f}%)")
        
        # Check confidence distribution
        if len(non_abstain) > 0:
            print(f"\nConfidence distribution (non-ABSTAIN):")
            print(f"  Mean: {non_abstain['max_prob'].mean():.3f}")
            print(f"  Median: {non_abstain['max_prob'].median():.3f}")
            print(f"  Min: {non_abstain['max_prob'].min():.3f}")
            print(f"  Max: {non_abstain['max_prob'].max():.3f}")
            
            # Confidence bins
            high_conf = (non_abstain['max_prob'] > 0.7).sum()
            med_conf = ((non_abstain['max_prob'] > 0.5) & (non_abstain['max_prob'] <= 0.7)).sum()
            low_conf = (non_abstain['max_prob'] <= 0.5).sum()
            
            print(f"\nConfidence bins:")
            print(f"  High (>0.7): {high_conf} ({high_conf/len(non_abstain)*100:.1f}%)")
            print(f"  Medium (0.5-0.7): {med_conf} ({med_conf/len(non_abstain)*100:.1f}%)")
            print(f"  Low (<=0.5): {low_conf} ({low_conf/len(non_abstain)*100:.1f}%)")
        
        # Check LF voting
        print(f"\nLabeling Function voting:")
        print(f"  Avg LFs per span: {df['num_lfs_voted'].mean():.2f}")
        print(f"  Max LFs: {df['num_lfs_voted'].max()}")
        print(f"  Spans with 0 LFs: {(df['num_lfs_voted'] == 0).sum()}")
        
        # Sample predictions
        print(f"\nSample predictions:")
        samples = df[df['weak_label_name'] != 'ABSTAIN'].nlargest(5, 'max_prob')
        for idx, row in samples.iterrows():
            print(f"  • [{row['weak_label_name']}] \"{row['span'][:50]}...\" (p={row['max_prob']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating weak labels: {e}")
        return False


def main():
    """Main validation pipeline"""
    
    print_header("CHECKPOINT 2: WEAK SUPERVISION VALIDATION")
    
    errors = []
    
    # Step 0: Check prerequisites
    print_step(0, "Check Prerequisites")
    
    # Check if Snorkel is installed
    try:
        import snorkel
        print(f"✓ Snorkel {snorkel.__version__} installed")
    except ImportError:
        print("✗ Snorkel not installed")
        print("\n  Install with: pip install snorkel")
        errors.append("Snorkel not installed")
    
    # Check for span extraction output
    if not check_file("results/spans/test_spans.parquet", "Span extraction output"):
        errors.append("Span extraction output missing - run checkpoint 1 first")
        print("\n⚠ Run checkpoint 1 first:")
        print("  python scripts/test_checkpoint1.py")
    
    # Check seed ontology
    if not check_file("seeds/seed_ontology.json", "Seed ontology"):
        errors.append("Seed ontology missing")
    
    # Stop if prerequisites missing
    if errors:
        print("\n" + "="*70)
        print("❌ PREREQUISITES NOT MET")
        print("="*70)
        print("\nErrors encountered:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\nPlease fix the errors above and run again.")
        return 1
    
    # Step 1: Run weak supervision (without semantic similarity)
    print_step(1, "Run Weak Supervision")
    
    print("⚠ Running without semantic similarity for faster testing")
    print("  (You can enable with --device cpu if sentence-transformers is installed)")
    
    if not run_command(
        [
            sys.executable, "src/supervision/weak_supervision.py",
            "--input", "results/spans/test_spans.parquet",
            "--output", "results/labels/weak_labels.parquet",
            "--seed-ontology", "seeds/seed_ontology.json",
            "--no-semantic",
            "--n-epochs", "100",  # Faster for testing
            "--device", "cpu"
        ],
        "Weak supervision"
    ):
        errors.append("Weak supervision failed")
    
    # Step 2: Validate outputs
    print_step(2, "Validate Outputs")
    
    if Path("results/labels/weak_labels.parquet").exists():
        if not validate_weak_labels("results/labels/weak_labels.parquet"):
            errors.append("Weak labels validation failed")
    else:
        errors.append("Weak labels output file not found")
    
    # Check label mappings
    print("\n--- Checking Label Mappings ---")
    if Path("results/labels/label_mappings.json").exists():
        with open("results/labels/label_mappings.json", 'r') as f:
            mappings = json.load(f)
        print(f"✓ Label mappings: {mappings['num_classes']} categories")
        print(f"  Categories: {list(mappings['label_to_category'].values())}")
    else:
        print("⚠ Label mappings not found (optional)")
    
    # Final report
    print("\n" + "="*70)
    if errors:
        print("❌ CHECKPOINT 2 VALIDATION FAILED")
        print("="*70)
        print("\nErrors encountered:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\nPlease fix the errors above and run again.")
        return 1
    else:
        print("✅ CHECKPOINT 2 COMPLETE")
        print("="*70)
        print("\n🎉 All tests passed!")
        print("\nGenerated files:")
        print("  - results/labels/weak_labels.parquet")
        print("  - results/labels/weak_labels.csv")
        print("  - results/labels/label_mappings.json")
        print("\nNext steps:")
        print("  1. Inspect the weak labels")
        print("  2. Analyze labeling function performance")
        print("  3. Move to Checkpoint 3: Model Training")
        print("     - Train end model on weak labels")
        print("     - Evaluate performance")
        return 0


if __name__ == "__main__":
    sys.exit(main())