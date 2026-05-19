"""
Convert Weak Labels / Gold Labels to BIO Format for NER Training
Uses HuggingFace tokenizers (DeBERTa) for consistency with downstream training
Handles span-to-token alignment, confidence filtering, and JSONL export
"""

import json
import argparse
import warnings
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# Fix Windows Unicode issues
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class BIOConverter:
    """Convert span annotations to BIO-tagged token sequences using HuggingFace tokenizers"""
    
    def __init__(
        self,
        tokenizer_name: str = "microsoft/deberta-v3-small",
        bio_scheme: str = "BIO",
        max_seq_length: int = 128,
        min_confidence: float = 0.7
    ):
        """
        Initialize BIO converter with HuggingFace tokenizer
        
        Args:
            tokenizer_name: HuggingFace tokenizer (should match NER training model)
            bio_scheme: Tagging scheme ("BIO" or "BIOES")
            max_seq_length: Maximum sequence length (tokens)
            min_confidence: Minimum confidence for weak labels
        """
        self.bio_scheme = bio_scheme
        self.max_seq_length = max_seq_length
        self.min_confidence = min_confidence
        
        # Load HuggingFace tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Store special tokens
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        
        print(f"[OK] Initialized with {bio_scheme} scheme")
        print(f"Special tokens: CLS={self.cls_token}, SEP={self.sep_token}, PAD={self.pad_token}")
    
    def align_spans_to_tokens(
        self,
        text: str,
        spans: List[Dict],
        encoding
    ) -> List[str]:
        """
        Align labeled spans to tokenized sequence and generate BIO tags
        
        Args:
            text: Original text
            spans: List of span dicts with 'start_char', 'end_char', 'label'
            encoding: HuggingFace tokenizer encoding
        
        Returns:
            List of BIO tags (same length as tokens)
        """
        # Get number of tokens (excluding special tokens)
        num_tokens = len(encoding.tokens())
        
        # Initialize all tags as 'O' (Outside)
        tags = ['O'] * num_tokens
        
        # Handle special tokens (CLS, SEP) - always tag as 'O'
        special_token_mask = encoding.special_tokens_mask
        
        # Sort spans by start position to handle overlaps
        spans_sorted = sorted(spans, key=lambda s: s['start_char'])
        
        for span_info in spans_sorted:
            span_start = span_info['start_char']
            span_end = span_info['end_char']
            label = span_info['label']
            
            # Find tokens that overlap with this span
            # char_to_token returns None for special tokens and characters not in tokens
            matching_token_indices = []
            
            for char_idx in range(span_start, span_end):
                token_idx = encoding.char_to_token(char_idx)
                
                if token_idx is not None and token_idx not in matching_token_indices:
                    # Skip special tokens
                    if not special_token_mask[token_idx]:
                        matching_token_indices.append(token_idx)
            
            # Assign BIO tags
            if matching_token_indices:
                # Sort indices (should already be sorted, but ensure)
                matching_token_indices = sorted(matching_token_indices)
                
                # First token gets B- (Begin)
                if tags[matching_token_indices[0]] == 'O':  # Only if not already tagged
                    tags[matching_token_indices[0]] = f'B-{label}'
                    
                    # Remaining tokens get I- (Inside)
                    for idx in matching_token_indices[1:]:
                        if tags[idx] == 'O':  # Only if not already tagged
                            tags[idx] = f'I-{label}'
        
        return tags
    
    def tokenize_and_align(
        self,
        text: str,
        spans: List[Dict]
    ) -> Optional[Dict]:
        """
        Tokenize text and align spans to tokens
        
        Args:
            text: Original text
            spans: List of span dicts with 'start_char', 'end_char', 'label'
        
        Returns:
            Dict with 'tokens', 'tags', 'input_ids', 'attention_mask' or None if error
        """
        try:
            # Tokenize with HuggingFace tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_attention_mask=True
            )
            
            # Extract tokens
            tokens = encoding.tokens()
            
            # Align spans to tokens
            tags = self.align_spans_to_tokens(text, spans, encoding)
            
            # Validate lengths match
            if len(tokens) != len(tags):
                print(f"Warning: Token/tag length mismatch: {len(tokens)} vs {len(tags)}")
                return None
            
            return {
                'tokens': tokens,
                'tags': tags,
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'original_text': text
            }
            
        except Exception as e:
            print(f"Error in tokenize_and_align: {e}")
            return None
    
    def validate_bio_sequence(self, tags: List[str]) -> bool:
        """
        Validate BIO tag sequence consistency
        
        Rules:
        - I-X must follow B-X or I-X with same label X
        - No orphaned I- tags
        """
        if not tags:
            return False
        
        prev_tag = 'O'
        for tag in tags:
            if tag.startswith('I-'):
                label = tag[2:]
                
                # I- must follow B- or I- with same label
                if prev_tag == 'O':
                    return False
                
                if prev_tag.startswith('B-') or prev_tag.startswith('I-'):
                    prev_label = prev_tag[2:]
                    if prev_label != label:
                        return False
            
            prev_tag = tag
        
        return True
    
    def process_journal_entry(
        self,
        journal_id: str,
        text: str,
        spans: List[Dict]
    ) -> Optional[Dict]:
        """
        Process a single journal entry with multiple spans
        
        Args:
            journal_id: Journal entry identifier
            text: Original journal text
            spans: List of labeled spans in this entry
        
        Returns:
            Dict ready for JSONL export, or None if invalid
        """
        # Tokenize and align
        result = self.tokenize_and_align(text, spans)
        
        if result is None:
            return None
        
        # Validate BIO sequence
        if not self.validate_bio_sequence(result['tags']):
            print(f"Warning: Invalid BIO sequence for journal {journal_id}, skipping")
            return None
        
        # Create output format (HuggingFace token classification format)
        output = {
            'id': journal_id,
            'tokens': result['tokens'],
            'ner_tags': result['tags'],  # Primary field for HuggingFace
            'tags': result['tags'],  # Alias for compatibility
            'input_ids': result['input_ids'],
            'attention_mask': result['attention_mask']
        }
        
        return output
    
    def merge_journals_and_spans(
        self,
        journals_df: pd.DataFrame,
        spans_df: pd.DataFrame,
        text_column: str = 'text',
        journal_id_column: str = 'journal_id'
    ) -> pd.DataFrame:
        """
        Merge journals and spans DataFrames for processing
        
        Args:
            journals_df: DataFrame with journal entries
            spans_df: DataFrame with span annotations
            text_column: Column name for journal text
            journal_id_column: Column name for journal ID
        
        Returns:
            Merged DataFrame
        """
        print(f"\nMerging journals and spans...")
        print(f"Journals: {len(journals_df)} entries")
        print(f"Spans: {len(spans_df)} annotations")
        
        # Ensure journal_id column exists in both
        if journal_id_column not in journals_df.columns:
            print(f"Error: '{journal_id_column}' not found in journals DataFrame")
            return pd.DataFrame()
        
        if journal_id_column not in spans_df.columns:
            print(f"Error: '{journal_id_column}' not found in spans DataFrame")
            return pd.DataFrame()
        
        # Merge on journal_id
        merged = spans_df.merge(
            journals_df[[journal_id_column, text_column]],
            on=journal_id_column,
            how='left'
        )
        
        print(f"[OK] Merged: {len(merged)} span-journal pairs")
        
        # Check for missing text
        missing_text = merged[text_column].isna().sum()
        if missing_text > 0:
            print(f"Warning: {missing_text} spans have no matching journal text")
            merged = merged.dropna(subset=[text_column])
        
        return merged
    
    def convert_from_weak_labels(
        self,
        journals_df: pd.DataFrame,
        spans_df: pd.DataFrame,
        text_column: str = 'text',
        label_column: str = 'weak_label_name',
        journal_id_column: str = 'journal_id'
    ) -> List[Dict]:
        """
        Convert weak labels to BIO format
        
        Args:
            journals_df: DataFrame with journal entries (journal_id, date, text)
            spans_df: DataFrame with weak labels (journal_id, span, start_char, end_char, weak_label_name, max_prob)
            text_column: Column name for text
            label_column: Column name for label
            journal_id_column: Column name for journal ID
        
        Returns:
            List of BIO-formatted documents
        """
        print(f"\n{'='*80}")
        print("CONVERTING WEAK LABELS TO BIO FORMAT")
        print(f"{'='*80}")
        
        # Filter by confidence
        if 'max_prob' in spans_df.columns:
            print(f"Filtering spans with confidence >= {self.min_confidence}")
            initial_count = len(spans_df)
            spans_df = spans_df[spans_df['max_prob'] >= self.min_confidence].copy()
            filtered_count = len(spans_df)
            print(f"Kept {filtered_count}/{initial_count} spans ({filtered_count/initial_count*100:.1f}%)")
        
        # Remove ABSTAIN labels
        if label_column in spans_df.columns:
            initial_count = len(spans_df)
            spans_df = spans_df[spans_df[label_column] != 'ABSTAIN'].copy()
            filtered_count = len(spans_df)
            print(f"After removing ABSTAIN: {filtered_count}/{initial_count} spans")
        
        # Check for offset columns and warn
        print(f"\nChecking for offset columns...")
        has_offsets = False
        offset_info = []
        
        if 'start_char' in spans_df.columns and 'end_char' in spans_df.columns:
            has_offsets = True
            offset_info.append("✓ Found: start_char, end_char")
        elif 'start' in spans_df.columns and 'end' in spans_df.columns:
            has_offsets = True
            offset_info.append("✓ Found: start, end")
        elif 'start_idx' in spans_df.columns and 'end_idx' in spans_df.columns:
            has_offsets = True
            offset_info.append("✓ Found: start_idx, end_idx")
        
        if has_offsets:
            for info in offset_info:
                print(info)
        else:
            print("⚠ WARNING: No standard offset columns found!")
            print("  Will attempt to locate spans by text search (slower and less reliable)")
            print("  Available columns:", list(spans_df.columns))
            if 'span' not in spans_df.columns:
                print("  ERROR: 'span' column also missing - cannot proceed!")
                return []
        
        # Merge with journals to get full text
        merged_df = self.merge_journals_and_spans(
            journals_df,
            spans_df,
            text_column=text_column,
            journal_id_column=journal_id_column
        )
        
        if len(merged_df) == 0:
            print("[ERROR] No data after merging journals and spans")
            return []
        
        # Group spans by journal_id
        journal_groups = merged_df.groupby(journal_id_column)
        print(f"\nProcessing {len(journal_groups)} unique journal entries...")
        
        bio_documents = []
        skipped = 0
        error_types = Counter()
        
        for journal_id, group in tqdm(journal_groups, desc="Converting entries"):
            # Get journal text (should be same for all rows in group)
            text = group.iloc[0][text_column]
            
            # Prepare spans for this journal
            spans = []
            for _, row in group.iterrows():
                # Try different offset column names
                start_char = None
                end_char = None
                
                # Check multiple possible column names
                if 'start_char' in row and 'end_char' in row:
                    start_char = int(row['start_char'])
                    end_char = int(row['end_char'])
                elif 'start' in row and 'end' in row:
                    start_char = int(row['start'])
                    end_char = int(row['end'])
                elif 'start_idx' in row and 'end_idx' in row:
                    start_char = int(row['start_idx'])
                    end_char = int(row['end_idx'])
                elif 'span' in row:
                    # Fallback: search for span in text
                    span_text = row['span']
                    try:
                        start_char = text.index(span_text)
                        end_char = start_char + len(span_text)
                    except ValueError:
                        error_types['span_not_found_in_text'] += 1
                        continue
                else:
                    error_types['missing_offsets'] += 1
                    continue
                
                if start_char is None or end_char is None:
                    error_types['invalid_offsets'] += 1
                    continue
                
                # Validate offsets are within text bounds
                if start_char < 0 or end_char > len(text) or start_char >= end_char:
                    error_types['offsets_out_of_bounds'] += 1
                    continue
                
                spans.append({
                    'start_char': start_char,
                    'end_char': end_char,
                    'label': row[label_column],
                    'span_text': row.get('span', text[start_char:end_char])
                })
            
            if not spans:
                skipped += 1
                error_types['no_valid_spans'] += 1
                continue
            
            # Process journal entry
            bio_doc = self.process_journal_entry(journal_id, text, spans)
            
            if bio_doc is not None:
                bio_documents.append(bio_doc)
            else:
                skipped += 1
                error_types['processing_failed'] += 1
        
        print(f"\n[OK] Converted {len(bio_documents)} documents")
        print(f"[WARN] Skipped {skipped} documents")
        
        if error_types:
            print("\nError breakdown:")
            for error_type, count in error_types.most_common():
                print(f"  {error_type}: {count}")
        
        return bio_documents
    
    def convert_from_gold_labels(
        self,
        gold_df: pd.DataFrame,
        text_column: str = 'text',
        label_column: str = 'gold_label'
    ) -> List[Dict]:
        """
        Convert gold labels to BIO format
        
        Args:
            gold_df: DataFrame with gold annotations (id, text, span, start_char, end_char, gold_label)
            text_column: Column name for text
            label_column: Column name for label
        
        Returns:
            List of BIO-formatted documents
        """
        print(f"\n{'='*80}")
        print("CONVERTING GOLD LABELS TO BIO FORMAT")
        print(f"{'='*80}")
        print(f"Input: {len(gold_df)} gold spans")
        
        bio_documents = []
        skipped = 0
        
        for idx, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Converting gold spans"):
            doc_id = row.get('id', row.get('span_id', f'gold_{idx}'))
            text = row[text_column]
            
            # Single span per row in gold set
            spans = [{
                'start_char': int(row['start_char']),
                'end_char': int(row['end_char']),
                'label': row[label_column],
                'span_text': row['span']
            }]
            
            # Process document
            bio_doc = self.process_journal_entry(doc_id, text, spans)
            
            if bio_doc is not None:
                bio_documents.append(bio_doc)
            else:
                skipped += 1
        
        print(f"\n[OK] Converted {len(bio_documents)} documents")
        print(f"[WARN] Skipped {skipped} documents")
        
        return bio_documents
    
    def generate_label_mappings(
        self,
        documents: List[Dict]
    ) -> Dict:
        """
        Generate label-to-id mappings for model training
        
        Returns:
            Dict with tag2id, id2tag, labels, num_tags
        """
        # Collect all unique tags
        all_tags = set()
        for doc in documents:
            all_tags.update(doc['ner_tags'])
        
        # Sort tags: O first, then B- tags, then I- tags
        tags_sorted = sorted(all_tags, key=lambda t: (
            0 if t == 'O' else (1 if t.startswith('B-') else 2),
            t
        ))
        
        # Create mappings
        tag2id = {tag: idx for idx, tag in enumerate(tags_sorted)}
        id2tag = {idx: tag for tag, idx in tag2id.items()}
        
        # Extract unique entity labels (categories)
        labels = sorted(set(
            tag.split('-')[1] for tag in all_tags if tag != 'O'
        ))
        
        mappings = {
            'tag2id': tag2id,
            'id2tag': id2tag,
            'labels': labels,
            'num_tags': len(tag2id)
        }
        
        return mappings
    
    def save_to_jsonl(
        self,
        documents: List[Dict],
        output_path: str
    ):
        """Save BIO documents to JSONL format (one JSON object per line)"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving to JSONL: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"[OK] Saved {len(documents)} documents")
    
    def print_statistics(
        self,
        documents: List[Dict],
        mappings: Dict
    ):
        """Print detailed conversion statistics"""
        
        print(f"\n{'='*80}")
        print("BIO CONVERSION STATISTICS")
        print(f"{'='*80}")
        
        print(f"\nTotal documents: {len(documents)}")
        
        # Token statistics
        all_tokens = [token for doc in documents for token in doc['tokens']]
        all_tags = [tag for doc in documents for tag in doc['ner_tags']]
        
        print(f"Total tokens: {len(all_tokens)}")
        print(f"Average tokens per document: {len(all_tokens) / len(documents):.1f}")
        
        # Sequence length distribution
        seq_lengths = [len(doc['tokens']) for doc in documents]
        print(f"Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
              f"median={np.median(seq_lengths):.1f}")
        
        # Tag distribution
        tag_counts = Counter(all_tags)
        print(f"\nTag distribution:")
        print(f"  {'O (Outside)':25s}: {tag_counts['O']:6d} ({tag_counts['O']/len(all_tags)*100:5.1f}%)")
        
        for tag in sorted(tag_counts.keys()):
            if tag != 'O':
                count = tag_counts[tag]
                print(f"  {tag:25s}: {count:6d} ({count/len(all_tags)*100:5.1f}%)")
        
        # Entity statistics
        entity_counts = defaultdict(int)
        for doc in documents:
            tags = doc['ner_tags']
            for tag in tags:
                if tag.startswith('B-'):
                    label = tag[2:]
                    entity_counts[label] += 1
        
        print(f"\nEntity counts (by category):")
        total_entities = sum(entity_counts.values())
        for label in sorted(entity_counts.keys()):
            count = entity_counts[label]
            print(f"  {label:25s}: {count:6d} ({count/total_entities*100:5.1f}%)")
        
        print(f"\nTotal entities: {total_entities}")
        
        # Label mappings summary
        print(f"\nLabel mappings:")
        print(f"  Unique entity labels: {len(mappings['labels'])}")
        print(f"  Unique BIO tags: {mappings['num_tags']}")
        
        # Sample tag2id
        print(f"\nTag2ID mapping (first 15):")
        for tag, idx in sorted(mappings['tag2id'].items(), key=lambda x: x[1])[:15]:
            print(f"  {tag:20s} -> {idx:3d}")
        if len(mappings['tag2id']) > 15:
            print(f"  ... ({len(mappings['tag2id']) - 15} more)")
        
        # Sample documents
        print(f"\n{'='*80}")
        print("SAMPLE BIO SEQUENCES")
        print(f"{'='*80}")
        
        for i, doc in enumerate(documents[:2]):  # Show first 2
            print(f"\nDocument {i+1} (ID: {doc['id']}):")
            tokens = doc['tokens'][:25]  # Show first 25 tokens
            tags = doc['ner_tags'][:25]
            
            print(f"{'Token':20s} -> {'Tag':15s}")
            print("-" * 38)
            for token, tag in zip(tokens, tags):
                # Highlight entities
                if tag != 'O':
                    print(f"{token:20s} -> {tag:15s} <<<")
                else:
                    print(f"{token:20s} -> {tag:15s}")
            
            if len(doc['tokens']) > 25:
                print(f"... ({len(doc['tokens']) - 25} more tokens)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert span-level labels to BIO format for NER training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert weak labels
  python src/ner/to_bio.py \\
    --journals data/processed/journals.parquet \\
    --spans results/labels/weak_labels.parquet \\
    --out data/processed/ner_train.jsonl \\
    --input-type weak \\
    --min-conf 0.7
  
  # Convert gold labels
  python src/ner/to_bio.py \\
    --input data/gold/gold_spans.csv \\
    --out data/processed/ner_dev.jsonl \\
    --input-type gold
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--journals', type=str, default=None,
                            help="Journals parquet file (required for weak labels)")
    input_group.add_argument('--spans', type=str, default=None,
                            help="Spans parquet file (required for weak labels)")
    input_group.add_argument('--input', type=str, default=None,
                            help="Input file (for gold labels)")
    input_group.add_argument('--input-type', type=str, 
                            choices=['weak', 'gold'], required=True,
                            help="Type of input data")
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--out', '--output', type=str, required=True,
                             dest='output',
                             help="Output JSONL file path")
    output_group.add_argument('--mappings-out', type=str, default=None,
                             help="Output path for label mappings JSON (auto-generated if not specified)")
    
    # Column specifications
    column_group = parser.add_argument_group('Column Specifications')
    column_group.add_argument('--text-column', type=str, default='text',
                             help="Column name for text (default: text)")
    column_group.add_argument('--label-column', type=str, default=None,
                             help="Column name for labels (auto: weak_label_name or gold_label)")
    column_group.add_argument('--journal-id-column', type=str, default='journal_id',
                             help="Column name for journal ID (default: journal_id)")
    
    # Filtering and model options
    model_group = parser.add_argument_group('Model and Filtering Options')
    model_group.add_argument('--tokenizer', type=str, 
                            default='microsoft/deberta-v3-small',
                            help="HuggingFace tokenizer (default: microsoft/deberta-v3-small)")
    model_group.add_argument('--min-conf', '--confidence-threshold', 
                            type=float, default=0.7, dest='min_conf',
                            help="Minimum confidence for weak labels (default: 0.7)")
    model_group.add_argument('--max-seq-length', type=int, default=128,
                            help="Maximum sequence length (default: 128)")
    model_group.add_argument('--bio-scheme', type=str, default='BIO',
                            choices=['BIO', 'BIOES'],
                            help="BIO tagging scheme (default: BIO)")
    
    args = parser.parse_args()
    
    # Validate input arguments
    if args.input_type == 'weak':
        if args.journals is None or args.spans is None:
            parser.error("--journals and --spans are required for weak labels")
    elif args.input_type == 'gold':
        if args.input is None:
            parser.error("--input is required for gold labels")
    
    # Auto-detect label column if not specified
    if args.label_column is None:
        args.label_column = 'weak_label_name' if args.input_type == 'weak' else 'gold_label'
    
    print(f"{'='*80}")
    print("HABIT TRACKING PIPELINE - BIO CONVERSION")
    print(f"{'='*80}")
    print(f"Input type: {args.input_type}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Min confidence: {args.min_conf}")
    print(f"Max sequence length: {args.max_seq_length}")
    
    # Initialize converter
    print(f"\n{'='*80}")
    print("INITIALIZING BIO CONVERTER")
    print(f"{'='*80}")
    
    converter = BIOConverter(
        tokenizer_name=args.tokenizer,
        bio_scheme=args.bio_scheme,
        max_seq_length=args.max_seq_length,
        min_confidence=args.min_conf
    )
    
    # Load and convert data
    if args.input_type == 'weak':
        # Load journals
        print(f"\nLoading journals: {args.journals}")
        if args.journals.endswith('.parquet'):
            journals_df = pd.read_parquet(args.journals)
        else:
            journals_df = pd.read_csv(args.journals)
        print(f"[OK] Loaded {len(journals_df)} journal entries")
        
        # Load spans
        print(f"\nLoading spans: {args.spans}")
        if args.spans.endswith('.parquet'):
            spans_df = pd.read_parquet(args.spans)
        else:
            spans_df = pd.read_csv(args.spans)
        print(f"[OK] Loaded {len(spans_df)} span annotations")
        
        # Convert
        documents = converter.convert_from_weak_labels(
            journals_df=journals_df,
            spans_df=spans_df,
            text_column=args.text_column,
            label_column=args.label_column,
            journal_id_column=args.journal_id_column
        )
    
    else:  # gold
        print(f"\nLoading gold labels: {args.input}")
        if args.input.endswith('.parquet'):
            gold_df = pd.read_parquet(args.input)
        else:
            gold_df = pd.read_csv(args.input)
        print(f"[OK] Loaded {len(gold_df)} gold annotations")
        
        # Convert
        documents = converter.convert_from_gold_labels(
            gold_df=gold_df,
            text_column=args.text_column,
            label_column=args.label_column
        )
    
    # Check results
    if len(documents) == 0:
        print("\n[ERROR] No valid documents generated!")
        print("Please check:")
        print("  - Input data format and columns")
        print("  - Confidence threshold (try lowering --min-conf)")
        print("  - Character offset alignment")
        sys.exit(1)
    
    # Generate label mappings
    print(f"\n{'='*80}")
    print("GENERATING LABEL MAPPINGS")
    print(f"{'='*80}")
    
    mappings = converter.generate_label_mappings(documents)
    
    # Print statistics
    converter.print_statistics(documents, mappings)
    
    # Save outputs
    print(f"\n{'='*80}")
    print("SAVING OUTPUTS")
    print(f"{'='*80}")
    
    # Save BIO documents
    converter.save_to_jsonl(documents, args.output)
    
    # Save label mappings
    if args.mappings_out is None:
        mappings_path = Path(args.output).parent / 'label_mappings.json'
    else:
        mappings_path = Path(args.mappings_out)
    
    mappings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mappings_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved label mappings to: {mappings_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("✓ BIO CONVERSION COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  BIO data:  {args.output}")
    print(f"  Mappings:  {mappings_path}")
    print(f"\nStatistics:")
    print(f"  Documents: {len(documents)}")
    print(f"  Tags:      {mappings['num_tags']}")
    print(f"  Labels:    {len(mappings['labels'])}")
    print(f"\nNext step:")
    print(f"  python src/ner/train_ner.py --train {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()