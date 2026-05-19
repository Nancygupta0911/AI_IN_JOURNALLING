"""
Data Unification Script for Habit Extraction Pipeline
Unifies multiple journal/text datasets into a single canonical format
Focus: Habit extraction (no temporal analysis)
"""

import pandas as pd
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import hashlib
import re


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"unification_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def generate_text_hash(text: str) -> str:
    """Generate unique hash for text deduplication"""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    return text.strip()


def load_parquet_dataset(file_path: Path, logger) -> pd.DataFrame:
    """Load parquet file and extract text"""
    logger.info(f"Loading parquet: {file_path}")
    df = pd.read_parquet(file_path)
    
    # Try to identify text column
    text_cols = ['text', 'content', 'entry', 'journal', 'note', 'description']
    text_col = None
    
    for col in text_cols:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        # Try to find any column with string data
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col = col
                logger.warning(f"Using column '{col}' as text column")
                break
    
    if text_col is None:
        logger.error(f"No text column found in {file_path}")
        return pd.DataFrame()
    
    result = pd.DataFrame({
        'text': df[text_col],
        'source': file_path.stem
    })
    
    logger.info(f"Loaded {len(result)} entries from {file_path.name}")
    return result


def load_csv_dataset(file_path: Path, logger) -> pd.DataFrame:
    """Load CSV file and extract text"""
    logger.info(f"Loading CSV: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')
    
    # Identify text column
    text_cols = ['text', 'content', 'entry', 'journal', 'note', 'description', 
                 'activities', 'notes', 'full_date', 'title']
    text_col = None
    
    for col in text_cols:
        if col in df.columns:
            text_col = col
            break
    
    # Special handling for emotion datasets
    if 'goemotions' in file_path.stem.lower():
        text_col = 'text' if 'text' in df.columns else df.columns[0]
    elif 'daylio' in file_path.stem.lower():
        # Daylio exports often have activities column
        if 'activities' in df.columns:
            text_col = 'activities'
        elif 'note' in df.columns:
            text_col = 'note'
    
    if text_col is None:
        # Use first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col = col
                logger.warning(f"Using column '{col}' as text column")
                break
    
    if text_col is None:
        logger.error(f"No text column found in {file_path}")
        return pd.DataFrame()
    
    result = pd.DataFrame({
        'text': df[text_col],
        'source': file_path.stem
    })
    
    logger.info(f"Loaded {len(result)} entries from {file_path.name}")
    return result


def unify_datasets(input_dir: Path, output_path: Path, logger):
    """Unify all datasets in input directory"""
    
    logger.info("="*60)
    logger.info("Starting data unification for habit extraction")
    logger.info("="*60)
    
    all_data = []
    stats = {
        'files_processed': 0,
        'total_entries': 0,
        'empty_entries': 0,
        'duplicates_removed': 0,
        'final_entries': 0,
        'sources': {}
    }
    
    # Process all parquet files
    parquet_files = list(input_dir.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    for file_path in parquet_files:
        try:
            df = load_parquet_dataset(file_path, logger)
            if not df.empty:
                all_data.append(df)
                stats['files_processed'] += 1
                stats['sources'][file_path.stem] = len(df)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    # Process all CSV files
    csv_files = list(input_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")
    
    for file_path in csv_files:
        try:
            df = load_csv_dataset(file_path, logger)
            if not df.empty:
                all_data.append(df)
                stats['files_processed'] += 1
                stats['sources'][file_path.stem] = len(df)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not all_data:
        logger.error("No data loaded from any file!")
        return
    
    # Combine all datasets
    logger.info("Combining datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)
    stats['total_entries'] = len(combined_df)
    
    logger.info(f"Total entries before cleaning: {len(combined_df)}")
    
    # Clean text
    logger.info("Cleaning text...")
    combined_df['text'] = combined_df['text'].apply(clean_text)
    
    # Remove empty entries
    combined_df = combined_df[combined_df['text'].str.len() > 0]
    stats['empty_entries'] = stats['total_entries'] - len(combined_df)
    
    logger.info(f"Entries after removing empty: {len(combined_df)}")
    
    # Deduplicate based on text hash
    logger.info("Deduplicating entries...")
    combined_df['text_hash'] = combined_df['text'].apply(generate_text_hash)
    
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset='text_hash', keep='first')
    stats['duplicates_removed'] = before_dedup - len(combined_df)
    
    logger.info(f"Duplicates removed: {stats['duplicates_removed']}")
    logger.info(f"Final entries: {len(combined_df)}")
    
    # Add journal ID
    combined_df['journal_id'] = [f"j_{i:06d}" for i in range(len(combined_df))]
    
    # Reorder columns
    final_df = combined_df[['journal_id', 'text', 'source']].copy()
    stats['final_entries'] = len(final_df)
    
    # Add text statistics
    final_df['text_length'] = final_df['text'].str.len()
    final_df['word_count'] = final_df['text'].str.split().str.len()
    
    # Save unified dataset
    logger.info(f"Saving unified dataset to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    
    # Save statistics
    stats_path = output_path.parent / 'unification_stats.json'
    
    # Add summary statistics
    stats['text_length_stats'] = {
        'mean': float(final_df['text_length'].mean()),
        'median': float(final_df['text_length'].median()),
        'min': int(final_df['text_length'].min()),
        'max': int(final_df['text_length'].max())
    }
    
    stats['word_count_stats'] = {
        'mean': float(final_df['word_count'].mean()),
        'median': float(final_df['word_count'].median()),
        'min': int(final_df['word_count'].min()),
        'max': int(final_df['word_count'].max())
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_path}")
    
    # Print summary
    logger.info("="*60)
    logger.info("UNIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total entries collected: {stats['total_entries']}")
    logger.info(f"Empty entries removed: {stats['empty_entries']}")
    logger.info(f"Duplicates removed: {stats['duplicates_removed']}")
    logger.info(f"Final unified entries: {stats['final_entries']}")
    logger.info(f"\nText length (chars): {stats['text_length_stats']['mean']:.1f} avg, "
                f"{stats['text_length_stats']['median']:.0f} median")
    logger.info(f"Word count: {stats['word_count_stats']['mean']:.1f} avg, "
                f"{stats['word_count_stats']['median']:.0f} median")
    logger.info(f"\nSources breakdown:")
    for source, count in stats['sources'].items():
        logger.info(f"  {source}: {count} entries")
    logger.info("="*60)
    
    return final_df, stats


def main():
    parser = argparse.ArgumentParser(
        description='Unify multiple datasets for habit extraction pipeline'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Input directory containing raw datasets (default: data/raw)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/journals.parquet',
        help='Output path for unified dataset (default: data/processed/journals.parquet)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    logger = setup_logging(output_path.parent)
    
    try:
        final_df, stats = unify_datasets(input_dir, output_path, logger)
        logger.info("✓ Data unification completed successfully!")
        
        # Quick validation
        logger.info("\nQuick validation:")
        logger.info(f"Output file size: {output_path.stat().st_size / 1024:.2f} KB")
        logger.info(f"Sample entries (first 3):")
        for idx, row in final_df.head(3).iterrows():
            text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            logger.info(f"  [{row['journal_id']}] {text_preview}")
        
    except Exception as e:
        logger.error(f"Error during unification: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()