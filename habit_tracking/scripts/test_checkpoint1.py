"""
Checkpoint 1 Validation Script (Python version)
Cross-platform test for extraction pipeline
"""

import subprocess
import sys
import json
from pathlib import Path
import pandas as pd


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


def validate_seed_ontology(filepath):
    """Validate seed ontology structure"""
    print("\n--- Validating Seed Ontology ---")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            habits = json.load(f)
        
        # Check if it's a list
        if not isinstance(habits, list):
            print(f"✗ Seed ontology must be a list, got {type(habits)}")
            return False
        
        if len(habits) == 0:
            print(f"✗ Seed ontology is empty")
            return False
        
        print(f"✓ Loaded {len(habits)} habits")
        
        # Check required fields
        required_fields = ['id', 'name', 'category']
        missing_fields = []
        
        for i, habit in enumerate(habits[:5]):  # Check first 5
            if not isinstance(habit, dict):
                print(f"✗ Habit {i} is not a dictionary")
                return False
            
            for field in required_fields:
                if field not in habit:
                    missing_fields.append(f"Habit {i} missing '{field}'")
        
        if missing_fields:
            print(f"✗ Validation errors:")
            for error in missing_fields:
                print(f"  - {error}")
            return False
        
        print(f"✓ All required fields present")
        
        # Count aliases
        total_aliases = sum(len(h.get('aliases', [])) for h in habits)
        print(f"  - Total habits: {len(habits)}")
        print(f"  - Total aliases: {total_aliases}")
        print(f"  - Categories: {len(set(h.get('category', 'unknown') for h in habits))}")
        
        return True
        
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ Error validating seed ontology: {e}")
        return False


def validate_spans(filepath):
    """Validate span extraction output"""
    print("\n--- Validating Spans ---")
    
    try:
        df = pd.read_parquet(filepath)
        print(f"✓ Loaded {len(df)} spans")
        
        # Check columns
        required_cols = ['doc_id', 'span', 'method', 'confidence']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"✗ Missing columns: {missing}")
            return False
        print(f"✓ All required columns present")
        
        # Check confidence scores
        if df['confidence'].min() < 0 or df['confidence'].max() > 1:
            print(f"✗ Invalid confidence range: [{df['confidence'].min()}, {df['confidence'].max()}]")
            return False
        print(f"✓ Confidence scores valid: [{df['confidence'].min():.2f}, {df['confidence'].max():.2f}]")
        
        # Statistics
        print(f"  - Unique spans: {df['span'].nunique()}")
        print(f"  - Extraction methods: {df['method'].nunique()}")
        print(f"  - Avg confidence: {df['confidence'].mean():.2f}")
        
        print("\nTop 5 methods by count:")
        print(df['method'].value_counts().head().to_string())
        
        print("\nSample spans:")
        sample = df[['span', 'method', 'confidence']].head(10)
        print(sample.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating spans: {e}")
        return False


def validate_keywords(filepath):
    """Validate keyword mining output"""
    print("\n--- Validating Keywords ---")
    
    try:
        df = pd.read_parquet(filepath)
        print(f"✓ Loaded {len(df)} keywords")
        
        # Check columns
        required_cols = ['phrase', 'composite_score', 'frequency']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"✗ Missing columns: {missing}")
            return False
        print(f"✓ All required columns present")
        
        # Statistics
        print(f"  - Unique phrases: {df['phrase'].nunique()}")
        print(f"  - Avg frequency: {df['frequency'].mean():.1f}")
        print(f"  - Avg composite score: {df['composite_score'].mean():.2f}")
        
        print("\nTop 10 mined keywords:")
        top = df[['phrase', 'composite_score', 'frequency', 'method_count']].head(10)
        print(top.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating keywords: {e}")
        return False


def main():
    """Main validation pipeline"""
    
    print_header("CHECKPOINT 1: EXTRACTION PIPELINE VALIDATION")
    
    errors = []
    
    # Step 0: Generate test data
    print_step(0, "Generate Test Dataset")
    if not run_command(
        [sys.executable, "scripts/generate_test_data.py"],
        "Test data generation"
    ):
        errors.append("Test data generation failed")
        print("\n⚠ Make sure scripts/generate_test_data.py exists")
    
    # Step 1: Check seed ontology
    print_step(1, "Check Seed Ontology")
    if check_file("seeds/seed_ontology.json", "Seed ontology"):
        if not validate_seed_ontology("seeds/seed_ontology.json"):
            errors.append("Seed ontology validation failed")
    else:
        errors.append("Seed ontology missing")
    
    # For testing, always skip KeyBERT (avoids dependency issues)
    print("⚠ Running without KeyBERT for faster testing")
    print("  (KeyBERT has complex dependencies - core pipeline works without it)")
    
    # Step 2: Run span extraction
    print_step(2, "Run Span Extraction")
    if not run_command(
        [
            sys.executable, "src/extraction/extract_regex.py",
            "--input", "data/raw/test_journals.csv",
            "--output", "results/spans/test_spans.parquet",
            "--seed-ontology", "seeds/seed_ontology.json",
            "--min-confidence", "0.3",
            "--no-keybert"
        ],
        "Span extraction"
    ):
        errors.append("Span extraction failed")
    
    # Step 3: Run keyword mining
    print_step(3, "Run Keyword Mining")
    if not run_command(
        [
            sys.executable, "src/extraction/keyword_mine.py",
            "--input", "data/raw/test_journals.csv",
            "--output", "results/spans/test_keywords.parquet",
            "--seed-ontology", "seeds/seed_ontology.json",
            "--min-freq", "2",
            "--top-n", "100",
            "--no-keybert"
        ],
        "Keyword mining"
    ):
        errors.append("Keyword mining failed")
    
    # Step 4: Validate outputs
    print_step(4, "Validate Outputs")
    
    # Validate spans
    if Path("results/spans/test_spans.parquet").exists():
        if not validate_spans("results/spans/test_spans.parquet"):
            errors.append("Span validation failed")
    else:
        errors.append("Span output file not found")
    
    # Validate keywords
    if Path("results/spans/test_keywords.parquet").exists():
        if not validate_keywords("results/spans/test_keywords.parquet"):
            errors.append("Keyword validation failed")
    else:
        errors.append("Keyword output file not found")
    
    # Check summaries (optional)
    print("\n--- Checking Summaries (optional) ---")
    if Path("results/spans/extraction_summary.json").exists():
        with open("results/spans/extraction_summary.json", 'r') as f:
            summary = json.load(f)
        print(f"✓ Extraction summary: {summary.get('total_spans')} spans from {summary.get('total_documents')} docs")
    else:
        print("⚠ Extraction summary not found (optional)")
    
    if Path("results/spans/keyword_mining_summary.json").exists():
        with open("results/spans/keyword_mining_summary.json", 'r') as f:
            summary = json.load(f)
        print(f"✓ Keyword mining summary: {summary.get('total_candidates')} candidates")
    else:
        print("⚠ Keyword mining summary not found (optional)")
    
    # Final report
    print("\n" + "="*70)
    if errors:
        print("❌ CHECKPOINT 1 VALIDATION FAILED")
        print("="*70)
        print("\nErrors encountered:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\nPlease fix the errors above and run again.")
        return 1
    else:
        print("✅ CHECKPOINT 1 COMPLETE")
        print("="*70)
        print("\n🎉 All tests passed!")
        print("\nGenerated files:")
        print("  - data/raw/test_journals.csv")
        print("  - results/spans/test_spans.parquet")
        print("  - results/spans/test_spans.csv")
        print("  - results/spans/test_keywords.parquet")
        print("  - results/spans/test_keywords.csv")
        print("\nNext steps:")
        print("  1. Inspect the extracted spans and keywords")
        print("  2. Verify the extraction quality")
        print("  3. Move to Checkpoint 2: Weak Supervision")
        print("     - Implement src/supervision/weak_supervision.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())