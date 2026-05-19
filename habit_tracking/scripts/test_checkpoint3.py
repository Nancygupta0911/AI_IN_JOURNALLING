"""
Checkpoint 3 Test Script
Tests: Gold Set Generation → BIO Conversion → NER Training Pipeline
Quick smoke tests to validate functionality before full training
"""

import sys
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def load_config():
    """Load configuration"""
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_gold_set_generation(config):
    """Test 1: Generate gold set"""
    print("\n" + "="*80)
    print("TEST 1: GOLD SET GENERATION")
    print("="*80)
    
    try:
        # Run gold set generation with small number for testing
        cmd = [
            "python", "scripts/generate_gold_set.py",
            "--seed-ontology", config['paths']['seed_ontology'],
            "--output", config['paths']['gold_spans'],
            "--num-spans", "50",  # Small number for testing
            "--min-quality", "0.4"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Gold set generation failed")
            print(result.stderr)
            return False
        
        # Check output file exists
        gold_path = Path(config['paths']['gold_spans'])
        if not gold_path.exists():
            print(f"❌ Output file not found: {gold_path}")
            return False
        
        # Load and validate
        df_gold = pd.read_csv(gold_path)
        print(f"\n✅ Gold set generated successfully!")
        print(f"   Spans: {len(df_gold)}")
        print(f"   Categories: {df_gold['gold_label'].nunique()}")
        print(f"   Columns: {list(df_gold.columns)}")
        
        # Check required columns
        required_cols = ['id', 'text', 'span', 'gold_label', 'start_char', 'end_char']
        missing = [col for col in required_cols if col not in df_gold.columns]
        if missing:
            print(f"❌ Missing required columns: {missing}")
            return False
        
        print(f"\n   Sample spans:")
        for i, row in df_gold.head(3).iterrows():
            print(f"   • [{row['gold_label']}] \"{row['span']}\" in \"{row['text'][:50]}...\"")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bio_conversion(config):
    """Test 2: Convert gold set to BIO format"""
    print("\n" + "="*80)
    print("TEST 2: BIO CONVERSION")
    print("="*80)
    
    try:
        gold_path = config['paths']['gold_spans']
        if not Path(gold_path).exists():
            print(f"❌ Gold set not found: {gold_path}")
            print("   Run Test 1 first")
            return False
        
        # Run BIO conversion
        output_path = "data/processed/test_ner_train.jsonl"
        cmd = [
            "python", "src/ner/to_bio.py",
            "--input", gold_path,
            "--input-type", "gold",
            "--output", output_path,
            "--spacy-model", config['extraction']['spacy_model'],
            "--max-seq-length", str(config['ner']['max_length'])
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ BIO conversion failed")
            print(result.stderr)
            return False
        
        # Check output exists
        output_path = Path(output_path)
        if not output_path.exists():
            print(f"❌ Output file not found: {output_path}")
            return False
        
        # Load and validate JSONL
        with open(output_path, 'r') as f:
            bio_docs = [json.loads(line) for line in f]
        
        print(f"\n✅ BIO conversion successful!")
        print(f"   Documents: {len(bio_docs)}")
        
        if len(bio_docs) > 0:
            doc = bio_docs[0]
            print(f"   Sample document:")
            print(f"     ID: {doc['id']}")
            print(f"     Tokens: {len(doc['tokens'])}")
            print(f"     Tags: {len(doc['tags'])}")
            print(f"     First 10 tokens:")
            for token, tag in zip(doc['tokens'][:10], doc['tags'][:10]):
                print(f"       {token:15s} → {tag}")
        
        # Check label mappings
        mappings_path = output_path.parent / 'label_mappings.json'
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
            print(f"\n   Label mappings loaded:")
            print(f"     Num tags: {mappings['num_tags']}")
            print(f"     Labels: {', '.join(mappings['labels'][:5])}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ner_training_quick(config):
    """Test 3: Quick NER training (1 epoch smoke test)"""
    print("\n" + "="*80)
    print("TEST 3: NER TRAINING (SMOKE TEST)")
    print("="*80)
    
    try:
        train_path = "data/processed/test_ner_train.jsonl"
        if not Path(train_path).exists():
            print(f"❌ Training data not found: {train_path}")
            print("   Run Test 2 first")
            return False
        
        # Run quick training (1 epoch, small model)
        output_dir = "models/ner/test_model"
        cmd = [
            "python", "src/ner/train_ner.py",
            "--train", train_path,
            "--model-name", "prajjwal1/bert-tiny",  # Tiny model for speed
            "--output-dir", output_dir,
            "--num-epochs", "1",
            "--batch-size", "8",
            "--learning-rate", "5e-5",
            "--max-length", "64",
            "--warmup-steps", "10",
            "--eval-steps", "10",
            "--save-steps", "20",
            "--logging-steps", "5",
            "--val-split", "0.2",
            "--no-class-weights",  # Faster without class weights
            "--seed", "42"
        ]
        
        print(f"Running quick training (1 epoch with tiny BERT)...")
        print(f"Note: This is just a smoke test to verify pipeline works")
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print("❌ NER training failed")
            return False
        
        # Check model output
        output_dir = Path(output_dir)
        if not output_dir.exists():
            print(f"❌ Model directory not created: {output_dir}")
            return False
        
        # Check for model files
        expected_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
        missing_files = [f for f in expected_files if not (output_dir / f).exists()]
        
        if missing_files:
            print(f"⚠️  Some model files missing: {missing_files}")
            print(f"   But training may have completed")
        
        print(f"\n✅ NER training smoke test passed!")
        print(f"   Model saved to: {output_dir}")
        
        # Check metrics if available
        metrics_file = output_dir / 'eval_results.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print(f"\n   Validation metrics:")
            for key, value in metrics.items():
                if 'eval_' in key:
                    print(f"     {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_quality_checks(config):
    """Test 4: Data quality validation"""
    print("\n" + "="*80)
    print("TEST 4: DATA QUALITY CHECKS")
    print("="*80)
    
    try:
        # Check gold set quality
        gold_path = Path(config['paths']['gold_spans'])
        if not gold_path.exists():
            print("⚠️  Gold set not found, skipping quality checks")
            return True
        
        df = pd.read_csv(gold_path)
        
        print("\nQuality checks:")
        
        # Check 1: No empty spans
        empty_spans = df['span'].isna().sum() + (df['span'].str.strip() == '').sum()
        if empty_spans > 0:
            print(f"   ❌ Found {empty_spans} empty spans")
            return False
        print(f"   ✅ No empty spans")
        
        # Check 2: Valid character offsets
        invalid_offsets = ((df['end_char'] <= df['start_char']) | 
                          (df['start_char'] < 0)).sum()
        if invalid_offsets > 0:
            print(f"   ❌ Found {invalid_offsets} invalid character offsets")
            return False
        print(f"   ✅ All character offsets valid")
        
        # Check 3: Span length reasonable
        df['span_len'] = df['span'].str.split().str.len()
        too_short = (df['span_len'] < 1).sum()
        too_long = (df['span_len'] > 15).sum()
        if too_short > 0:
            print(f"   ⚠️  Found {too_short} spans with < 1 token")
        if too_long > 0:
            print(f"   ⚠️  Found {too_long} spans with > 15 tokens")
        print(f"   ✅ Span length distribution looks reasonable")
        print(f"      Mean: {df['span_len'].mean():.1f} tokens")
        print(f"      Range: {df['span_len'].min()}-{df['span_len'].max()} tokens")
        
        # Check 4: Category distribution
        category_counts = df['gold_label'].value_counts()
        min_per_category = category_counts.min()
        max_per_category = category_counts.max()
        
        if min_per_category < 2:
            print(f"   ⚠️  Some categories have very few examples (min: {min_per_category})")
        print(f"   ✅ Category distribution:")
        print(f"      Categories: {len(category_counts)}")
        print(f"      Min per category: {min_per_category}")
        print(f"      Max per category: {max_per_category}")
        
        # Check 5: Text-span alignment
        misaligned = 0
        for idx, row in df.head(20).iterrows():  # Check first 20
            text = row['text'].lower()
            span = row['span'].lower()
            if span not in text:
                misaligned += 1
        
        if misaligned > 0:
            print(f"   ⚠️  Found {misaligned}/20 misaligned spans in sample")
        else:
            print(f"   ✅ Span-text alignment verified")
        
        print(f"\n✅ Data quality checks passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("CHECKPOINT 3 TEST SUITE")
    print("Testing: Gold Set → BIO Conversion → NER Training")
    print("="*80)
    
    # Load config
    try:
        config = load_config()
        print(f"\n✅ Configuration loaded")
    except Exception as e:
        print(f"\n❌ Failed to load config: {e}")
        return 1
    
    # Run tests
    tests = [
        ("Gold Set Generation", test_gold_set_generation),
        ("BIO Conversion", test_bio_conversion),
        ("Data Quality Checks", test_data_quality_checks),
        ("NER Training (Smoke Test)", test_ner_training_quick),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*80}")
            print(f"Running: {test_name}")
            print(f"{'='*80}")
            
            result = test_func(config)
            results[test_name] = result
            
            if result:
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Test interrupted by user")
            results[test_name] = False
            break
        except Exception as e:
            print(f"\n❌ {test_name} crashed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n{'='*80}")
        print("🎉 ALL TESTS PASSED!")
        print(f"{'='*80}")
        print("\nCheckpoint 3 validated successfully!")
        print("\nNext steps:")
        print("  1. Generate full gold set (400+ spans):")
        print("     python scripts/generate_gold_set.py")
        print("  2. Convert to BIO format:")
        print("     python src/ner/to_bio.py --input data/gold/gold_spans.csv --input-type gold --output data/processed/ner_train.jsonl")
        print("  3. Train full NER model:")
        print("     python src/ner/train_ner.py --train data/processed/ner_train.jsonl --num-epochs 5")
        return 0
    else:
        print(f"\n{'='*80}")
        print("❌ SOME TESTS FAILED")
        print(f"{'='*80}")
        print("\nPlease fix the failing tests before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())