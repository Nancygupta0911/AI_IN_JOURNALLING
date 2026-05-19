"""
Enhanced Habit Span Extraction Pipeline
Extracts habit mentions from journal entries with improved patterns and filtering
Focus: High-precision habit extraction for personal journaling data
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

import spacy
from spacy.matcher import Matcher, PhraseMatcher
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class HabitSpanExtractor:
    """Multi-strategy habit span extraction optimized for journal entries"""
    
    def __init__(
        self, 
        seed_ontology_path: str,
        spacy_model: str = "en_core_web_sm",
        min_span_length: int = 2,
        max_span_length: int = 8,
        context_window: int = 80,
        min_confidence: float = 0.35
    ):
        """
        Initialize span extractor
        
        Args:
            seed_ontology_path: Path to seed_ontology.json
            spacy_model: spaCy model name
            min_span_length: Minimum tokens in span
            max_span_length: Maximum tokens in span  
            context_window: Context characters around span
            min_confidence: Minimum confidence threshold
        """
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.context_window = context_window
        self.min_confidence = min_confidence
        
        # Load spaCy
        print(f"Loading spaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading {spacy_model}...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        # Increase max length for longer journal entries
        self.nlp.max_length = 3000000
        
        # Load seed ontology
        print(f"Loading seed ontology: {seed_ontology_path}")
        self.seed_habits = self._load_seed_ontology(seed_ontology_path)
        
        # Build extraction components
        self._build_regex_patterns()
        self._build_spacy_matchers()
        self._build_stopword_filter()
        
        print(f"✓ Extractor initialized with {len(self.seed_habits)} seed habits")
    
    def _load_seed_ontology(self, path: str) -> List[Dict]:
        """Load and validate seed ontology"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                habits = json.load(f)
            
            if not isinstance(habits, list) or len(habits) == 0:
                raise ValueError("Seed ontology must be non-empty list")
            
            # Validate structure
            for i, habit in enumerate(habits):
                required = ['id', 'name', 'category']
                for field in required:
                    if field not in habit:
                        raise ValueError(f"Habit {i} missing field: {field}")
            
            return habits
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Seed ontology not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in seed ontology: {e}")
    
    def _build_stopword_filter(self):
        """Build comprehensive stopword list for filtering"""
        self.stop_phrases = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
            'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your',
            'it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'shall',
            'not', 'no', 'nor', 'as', 'if', 'or', 'because', 'while', 'during'
        }
    
    def _build_regex_patterns(self):
        """Build targeted regex patterns for habit extraction"""
        
        # 1. Collect aliases from seed ontology
        alias_patterns = []
        for habit in self.seed_habits:
            # Add exact name
            alias_patterns.append(re.escape(habit['name'].lower()))
            # Add all aliases
            for alias in habit.get('aliases', []):
                alias_patterns.append(re.escape(alias.lower()))
        
        # 2. Core habit verbs (activities people track)
        habit_verbs = [
            # Physical activities
            'exercised', 'worked out', 'ran', 'walked', 'cycled', 'swam', 'stretched',
            'meditated', 'did yoga', 'played sports',
            
            # Work/Study
            'studied', 'worked', 'coded', 'practiced', 'learned', 'read', 'wrote',
            'attended', 'completed', 'finished', 'started',
            
            # Social
            'met', 'called', 'texted', 'hung out', 'talked', 'visited', 'socialized',
            
            # Consumption
            'ate', 'drank', 'had coffee', 'smoked', 'vaped', 'consumed',
            'ordered food', 'cooked', 'meal prepped',
            
            # Digital
            'scrolled', 'browsed', 'watched', 'played games', 'binged', 
            'used phone', 'checked social media',
            
            # Self-care
            'slept', 'napped', 'showered', 'groomed', 'relaxed', 'rested',
            
            # Avoidance/Procrastination  
            'procrastinated', 'avoided', 'skipped', 'missed', 'postponed', 'delayed',
            
            # Mental/Emotional
            'felt anxious', 'felt depressed', 'felt happy', 'felt stressed',
            'worried', 'panicked', 'overthought'
        ]
        
        # 3. Build patterns with confidence weights
        self.regex_patterns = []
        
        # HIGHEST CONFIDENCE: Direct seed aliases
        if alias_patterns:
            self.regex_patterns.append({
                'name': 'seed_alias',
                'pattern': re.compile(
                    r'\b(' + '|'.join(alias_patterns) + r')\b',
                    re.IGNORECASE
                ),
                'confidence': 0.95
            })
        
        # HIGH CONFIDENCE: Verb + duration
        duration_expr = r'\d+\s*(?:hours?|hrs?|h|minutes?|mins?|m)'
        self.regex_patterns.append({
            'name': 'verb_numeric_duration',
            'pattern': re.compile(
                rf'\b\w+(?:ed|ing)\s+for\s+{duration_expr}\b',
                re.IGNORECASE
            ),
            'confidence': 0.85
        })
        
        # Duration + activity
        self.regex_patterns.append({
            'name': 'duration_activity',
            'pattern': re.compile(
                rf'\b{duration_expr}\s+(?:of\s+)?\w+(?:ing|ed)?\b',
                re.IGNORECASE
            ),
            'confidence': 0.80
        })
        
        # Time expressions (early/late)
        time_expr = r'(?:early|late|all day|all night|until late|till late|past midnight)'
        self.regex_patterns.append({
            'name': 'verb_time',
            'pattern': re.compile(
                rf'\b(?:went|stayed|was|woke)\s+(?:\w+\s+){{0,2}}{time_expr}\b',
                re.IGNORECASE
            ),
            'confidence': 0.75
        })
        
        # "Too much X" pattern
        self.regex_patterns.append({
            'name': 'too_much',
            'pattern': re.compile(
                r'\btoo much\s+\w+(?:ing)?\b',
                re.IGNORECASE
            ),
            'confidence': 0.80
        })
        
        # Frequency indicators
        freq_expr = r'(?:again|still|always|constantly|repeatedly|multiple times)'
        self.regex_patterns.append({
            'name': 'frequency',
            'pattern': re.compile(
                rf'\b\w+(?:ed|ing)\s+{freq_expr}\b',
                re.IGNORECASE
            ),
            'confidence': 0.70
        })
        
        # Negations (important for habit tracking)
        self.regex_patterns.append({
            'name': 'negation',
            'pattern': re.compile(
                r'\b(?:didn\'?t|did not|couldn\'?t|haven\'?t|skipped|missed|avoided)\s+\w+(?:ing)?\b',
                re.IGNORECASE
            ),
            'confidence': 0.75
        })
        
        # "I + habit verb" structure
        verb_pattern = '|'.join([re.escape(v) for v in habit_verbs])
        self.regex_patterns.append({
            'name': 'i_verb_habit',
            'pattern': re.compile(
                rf'\bI\s+(?:{verb_pattern})\b',
                re.IGNORECASE
            ),
            'confidence': 0.85
        })
        
        # Goal/intention patterns
        self.regex_patterns.append({
            'name': 'goal_pattern',
            'pattern': re.compile(
                r'\b(?:tried to|attempted to|managed to|failed to|planning to)\s+\w+\b',
                re.IGNORECASE
            ),
            'confidence': 0.70
        })
        
        print(f"✓ Built {len(self.regex_patterns)} regex patterns")
    
    def _build_spacy_matchers(self):
        """Build spaCy pattern matchers for linguistic structures"""
        
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # 1. Phrase matcher for seed aliases
        alias_docs = []
        for habit in self.seed_habits:
            alias_docs.append(self.nlp.make_doc(habit['name'].lower()))
            for alias in habit.get('aliases', []):
                alias_docs.append(self.nlp.make_doc(alias.lower()))
        
        if alias_docs:
            self.phrase_matcher.add("HABIT_ALIAS", alias_docs)
        
        # 2. Linguistic patterns
        
        # VERB + NOUN (basic action-object)
        self.matcher.add("VERB_NOUN", [[
            {"POS": "VERB"},
            {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}
        ]])
        
        # VERB + ADV + NOUN (e.g., "ate too much")
        self.matcher.add("VERB_ADV_NOUN", [[
            {"POS": "VERB"},
            {"POS": "ADV"},
            {"POS": "NOUN"}
        ]])
        
        # ADJ + NOUN (e.g., "late night", "early morning")
        self.matcher.add("ADJ_NOUN", [[
            {"POS": "ADJ"},
            {"POS": "NOUN"}
        ]])
        
        # Feel/Get + ADJ (emotional states)
        self.matcher.add("FEEL_STATE", [[
            {"LEMMA": {"IN": ["feel", "get", "become", "be"]}},
            {"POS": "ADJ"}
        ]])
        
        # Negation + VERB
        self.matcher.add("NEG_VERB", [[
            {"LOWER": {"IN": ["didn't", "didnt", "couldn't", "couldnt", "not", "no"]}},
            {"POS": "VERB"}
        ]])
        
        # VERB + VERB (progressive: "kept scrolling", "started working")
        self.matcher.add("VERB_VERB_PROG", [[
            {"LEMMA": {"IN": ["keep", "start", "stop", "continue", "try", "attempt"]}},
            {"POS": "VERB", "TAG": {"IN": ["VBG", "VB"]}}
        ]])
        
        # VERB + for + duration
        self.matcher.add("VERB_FOR_DURATION", [[
            {"POS": "VERB"},
            {"LOWER": "for"},
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["hour", "hours", "minute", "minutes", "hr", "hrs", "min", "mins"]}}
        ]])
        
        print(f"✓ Built spaCy matchers with {len(self.matcher)} patterns")
    
    def extract_regex_spans(self, text: str, doc_id: str) -> List[Dict]:
        """Extract spans using regex patterns"""
        spans = []
        
        for pattern_info in self.regex_patterns:
            matches = pattern_info['pattern'].finditer(text)
            
            for match in matches:
                span_text = match.group(0).strip()
                
                # Skip if too short or just punctuation
                if len(span_text) < 3 or not any(c.isalnum() for c in span_text):
                    continue
                
                token_count = len(span_text.split())
                
                # Filter by length
                if token_count < self.min_span_length or token_count > self.max_span_length:
                    continue
                
                spans.append({
                    'journal_id': doc_id,
                    'span': span_text,
                    'start_idx': match.start(),
                    'end_idx': match.end(),
                    'method': f"regex_{pattern_info['name']}",
                    'confidence': pattern_info['confidence'],
                    'context': self._get_context(text, match.start(), match.end())
                })
        
        return spans
    
    def extract_spacy_spans(self, text: str, doc_id: str) -> List[Dict]:
        """Extract spans using spaCy matchers"""
        spans = []
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Phrase matches (highest confidence)
        phrase_matches = self.phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            spans.append({
                'journal_id': doc_id,
                'span': span.text,
                'start_idx': span.start_char,
                'end_idx': span.end_char,
                'method': 'spacy_phrase_match',
                'confidence': 0.95,
                'context': self._get_context(text, span.start_char, span.end_char)
            })
        
        # Pattern matches
        pattern_matches = self.matcher(doc)
        for match_id, start, end in pattern_matches:
            span = doc[start:end]
            
            # Length filter
            if len(span) < self.min_span_length or len(span) > self.max_span_length:
                continue
            
            pattern_name = self.nlp.vocab.strings[match_id]
            confidence = self._get_pattern_confidence(pattern_name)
            
            spans.append({
                'journal_id': doc_id,
                'span': span.text,
                'start_idx': span.start_char,
                'end_idx': span.end_char,
                'method': f'spacy_{pattern_name.lower()}',
                'confidence': confidence,
                'context': self._get_context(text, span.start_char, span.end_char)
            })
        
        return spans
    
    def _get_pattern_confidence(self, pattern_name: str) -> float:
        """Get confidence score for spaCy pattern"""
        confidences = {
            'HABIT_ALIAS': 0.95,
            'FEEL_STATE': 0.90,
            'VERB_NOUN': 0.75,
            'VERB_ADV_NOUN': 0.80,
            'ADJ_NOUN': 0.65,
            'NEG_VERB': 0.80,
            'VERB_VERB_PROG': 0.75,
            'VERB_FOR_DURATION': 0.85
        }
        return confidences.get(pattern_name, 0.65)
    
    def _get_context(self, text: str, start: int, end: int) -> str:
        """Get context window around span"""
        context_start = max(0, start - self.context_window)
        context_end = min(len(text), end + self.context_window)
        context = text[context_start:context_end]
        
        # Clean context
        context = ' '.join(context.split())
        return context
    
    def deduplicate_spans(self, spans: List[Dict]) -> List[Dict]:
        """Remove overlapping duplicates, keeping highest confidence"""
        if not spans:
            return []
        
        # Sort by start position
        sorted_spans = sorted(spans, key=lambda s: (s['start_idx'], -s['confidence']))
        
        deduped = []
        
        for span in sorted_spans:
            # Check for overlap with already selected spans
            overlaps = False
            for selected in deduped:
                # Check if spans overlap
                if (span['start_idx'] < selected['end_idx'] and 
                    span['end_idx'] > selected['start_idx']):
                    overlaps = True
                    # Keep higher confidence span
                    if span['confidence'] > selected['confidence']:
                        deduped.remove(selected)
                        deduped.append(span)
                    break
            
            if not overlaps:
                deduped.append(span)
        
        return deduped
    
    def filter_quality(self, spans: List[Dict]) -> List[Dict]:
        """Apply quality filters to spans"""
        filtered = []
        
        for span in spans:
            span_text = span['span'].strip().lower()
            
            # Skip if below confidence threshold
            if span['confidence'] < self.min_confidence:
                continue
            
            # Skip very short spans
            if len(span_text) < 3:
                continue
            
            # Skip if only stopwords
            words = span_text.split()
            if all(w in self.stop_phrases for w in words):
                continue
            
            # Skip if no alphabetic characters
            if not any(c.isalpha() for c in span_text):
                continue
            
            # Skip pure punctuation or numbers
            if re.match(r'^[\d\s\W]+$', span_text):
                continue
            
            # Skip generic pronouns
            if span_text in {'i', 'me', 'my', 'you', 'it', 'they', 'them'}:
                continue
            
            filtered.append(span)
        
        return filtered
    
    def extract_from_dataframe(
        self, 
        df: pd.DataFrame,
        text_column: str = 'text',
        id_column: str = 'journal_id'
    ) -> pd.DataFrame:
        """Main extraction pipeline"""
        
        print(f"\n{'='*60}")
        print("HABIT SPAN EXTRACTION")
        print(f"{'='*60}")
        print(f"Processing {len(df)} journal entries...")
        
        all_spans = []
        entries_with_spans = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            doc_id = str(row[id_column])
            text = str(row[text_column])
            
            # Skip very short entries
            if not text or len(text) < 15:
                continue
            
            # Extract using both methods
            regex_spans = self.extract_regex_spans(text, doc_id)
            spacy_spans = self.extract_spacy_spans(text, doc_id)
            
            # Combine and process
            doc_spans = regex_spans + spacy_spans
            
            if doc_spans:
                entries_with_spans += 1
            
            all_spans.extend(doc_spans)
        
        print(f"\n✓ Extracted {len(all_spans)} raw spans from {entries_with_spans} entries")
        
        # Deduplicate overlaps
        all_spans = self.deduplicate_spans(all_spans)
        print(f"✓ After deduplication: {len(all_spans)} spans")
        
        # Quality filtering
        all_spans = self.filter_quality(all_spans)
        print(f"✓ After quality filtering: {len(all_spans)} spans")
        
        # Convert to DataFrame
        if not all_spans:
            print("\n⚠ No spans extracted! Check your seed ontology and input data.")
            return pd.DataFrame()
        
        df_spans = pd.DataFrame(all_spans)
        
        # Add span ID
        df_spans['span_id'] = [f"s_{i:06d}" for i in range(len(df_spans))]
        
        # Reorder columns
        cols = ['span_id', 'journal_id', 'span', 'start_idx', 'end_idx', 
                'method', 'confidence', 'context']
        df_spans = df_spans[cols]
        
        # Print statistics
        print(f"\n{'='*60}")
        print("EXTRACTION STATISTICS")
        print(f"{'='*60}")
        print(f"Total journal entries: {len(df)}")
        print(f"Entries with spans: {entries_with_spans} ({entries_with_spans/len(df)*100:.1f}%)")
        print(f"Total spans extracted: {len(df_spans)}")
        print(f"Avg spans per entry: {len(df_spans) / len(df):.2f}")
        print(f"Avg confidence: {df_spans['confidence'].mean():.3f}")
        
        print(f"\n{'Method Distribution':^60}")
        print("-" * 60)
        for method, count in df_spans['method'].value_counts().head(10).items():
            pct = count / len(df_spans) * 100
            print(f"{method:.<50} {count:>5} ({pct:>5.1f}%)")
        
        print(f"\n{'Confidence Distribution':^60}")
        print("-" * 60)
        print(df_spans['confidence'].describe().to_string())
        
        return df_spans


def main():
    parser = argparse.ArgumentParser(
        description="Extract habit spans from journal entries"
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help="Input parquet/CSV with journal entries"
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help="Output path for extracted spans"
    )
    parser.add_argument(
        '--seeds', type=str, default='seeds/seed_ontology.json',
        help="Path to seed ontology JSON"
    )
    parser.add_argument(
        '--text-column', type=str, default='text',
        help="Column name containing text"
    )
    parser.add_argument(
        '--id-column', type=str, default='journal_id',
        help="Column name containing document ID"
    )
    parser.add_argument(
        '--spacy-model', type=str, default='en_core_web_sm',
        help="spaCy model name"
    )
    parser.add_argument(
        '--min-span-len', type=int, default=2,
        help="Minimum span length (tokens)"
    )
    parser.add_argument(
        '--max-span-len', type=int, default=8,
        help="Maximum span length (tokens)"
    )
    parser.add_argument(
        '--min-confidence', type=float, default=0.35,
        help="Minimum confidence threshold"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    print(f"Loading data from: {args.input}")
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    print(f"✓ Loaded {len(df)} journal entries")
    
    # Check columns
    if args.text_column not in df.columns:
        print(f"Error: Text column '{args.text_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    if args.id_column not in df.columns:
        print(f"Warning: ID column '{args.id_column}' not found, using index")
        df[args.id_column] = [f"j_{i:06d}" for i in range(len(df))]
    
    # Initialize extractor
    try:
        extractor = HabitSpanExtractor(
            seed_ontology_path=args.seeds,
            spacy_model=args.spacy_model,
            min_span_length=args.min_span_len,
            max_span_length=args.max_span_len,
            min_confidence=args.min_confidence
        )
    except Exception as e:
        print(f"Error initializing extractor: {e}")
        return
    
    # Extract spans
    df_spans = extractor.extract_from_dataframe(
        df,
        text_column=args.text_column,
        id_column=args.id_column
    )
    
    if df_spans.empty:
        print("\n❌ No spans extracted. Exiting.")
        return
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save parquet
    df_spans.to_parquet(args.out, index=False)
    print(f"✓ Saved parquet: {args.out}")
    
    # Save CSV for easy inspection
    csv_path = output_path.with_suffix('.csv')
    df_spans.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Save summary statistics
    summary = {
        'input_file': str(args.input),
        'total_entries': len(df),
        'entries_with_spans': int(df_spans['journal_id'].nunique()),
        'total_spans': len(df_spans),
        'avg_spans_per_entry': float(len(df_spans) / len(df)),
        'avg_confidence': float(df_spans['confidence'].mean()),
        'method_distribution': df_spans['method'].value_counts().to_dict(),
        'confidence_stats': {
            'min': float(df_spans['confidence'].min()),
            'max': float(df_spans['confidence'].max()),
            'mean': float(df_spans['confidence'].mean()),
            'median': float(df_spans['confidence'].median())
        },
        'sample_spans': df_spans.nlargest(10, 'confidence')[['span', 'method', 'confidence']].to_dict('records')
    }
    
    summary_path = output_path.parent / 'extraction_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved summary: {summary_path}")
    
    # Show top spans
    print(f"\n{'='*60}")
    print("TOP 10 EXTRACTED SPANS (by confidence)")
    print(f"{'='*60}")
    top_spans = df_spans.nlargest(10, 'confidence')
    for _, row in top_spans.iterrows():
        print(f"{row['confidence']:.2f} | {row['span']:<40} | {row['method']}")
    
    print(f"\n{'='*60}")
    print("✅ EXTRACTION COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()