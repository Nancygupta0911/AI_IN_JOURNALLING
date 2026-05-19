"""
Enhanced Habit Keyword Mining Pipeline
Discovers habit-related phrases using linguistic analysis, TF-IDF, and collocation mining
Optimized for personal journal data and habit tracking
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter, defaultdict
import math

import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class HabitKeywordMiner:
    """Mine habit-related keywords and phrases from journal corpus"""
    
    def __init__(
        self,
        seed_ontology_path: str,
        spacy_model: str = "en_core_web_sm",
        min_freq: int = 3,
        max_phrase_length: int = 4,
        top_n: int = 300
    ):
        """
        Initialize keyword miner
        
        Args:
            seed_ontology_path: Path to seed ontology JSON
            spacy_model: spaCy model name
            min_freq: Minimum frequency threshold
            max_phrase_length: Max tokens in phrase
            top_n: Number of top keywords to return
        """
        self.min_freq = min_freq
        self.max_phrase_length = max_phrase_length
        self.top_n = top_n
        
        # Load spaCy
        print(f"Loading spaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading {spacy_model}...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        self.nlp.max_length = 3000000
        
        # Load seed ontology
        print(f"Loading seed ontology: {seed_ontology_path}")
        with open(seed_ontology_path, 'r', encoding='utf-8') as f:
            self.seed_habits = json.load(f)
        
        # Build reference vocabulary
        self.seed_keywords = self._extract_seed_keywords()
        self.habit_vocab = self._build_habit_vocabulary()
        
        print(f"✓ Initialized with {len(self.seed_habits)} seed habits")
        print(f"✓ Built vocabulary of {len(self.habit_vocab)} habit-related terms")
    
    def _extract_seed_keywords(self) -> Set[str]:
        """Extract all keywords from seed ontology"""
        keywords = set()
        
        for habit in self.seed_habits:
            # Name words
            keywords.update(habit['name'].lower().split())
            
            # Aliases
            for alias in habit.get('aliases', []):
                keywords.update(alias.lower().split())
            
            # Description lemmas
            if 'description' in habit:
                doc = self.nlp(habit['description'].lower())
                for token in doc:
                    if token.pos_ in ['VERB', 'NOUN', 'ADJ'] and not token.is_stop:
                        keywords.add(token.lemma_)
        
        return keywords
    
    def _build_habit_vocabulary(self) -> Set[str]:
        """Build comprehensive habit-related vocabulary"""
        
        # Core habit verbs (lemmatized)
        habit_verbs = {
            # Actions
            'do', 'go', 'have', 'take', 'make', 'get', 'work', 'study',
            'practice', 'exercise', 'train', 'play', 'watch', 'read', 'write',
            'eat', 'drink', 'sleep', 'wake', 'cook', 'clean', 'organize',
            
            # Digital
            'scroll', 'browse', 'surf', 'stream', 'binge', 'game', 'text', 'call',
            
            # Social
            'meet', 'hang', 'talk', 'chat', 'socialize', 'party', 'visit',
            
            # Mental
            'think', 'worry', 'stress', 'panic', 'overthink', 'ruminate',
            'meditate', 'reflect', 'journal',
            
            # Self-care
            'shower', 'groom', 'dress', 'relax', 'rest', 'nap',
            
            # Avoidance
            'procrastinate', 'avoid', 'skip', 'miss', 'postpone', 'delay', 'ignore'
        }
        
        # Habit nouns
        habit_nouns = {
            'work', 'study', 'exercise', 'workout', 'training', 'practice',
            'sleep', 'nap', 'rest', 'food', 'meal', 'snack', 'breakfast', 'lunch', 'dinner',
            'coffee', 'tea', 'alcohol', 'cigarette', 'vape',
            'gym', 'run', 'walk', 'jog', 'yoga', 'meditation',
            'game', 'phone', 'tv', 'netflix', 'youtube', 'social media',
            'book', 'article', 'video', 'movie', 'show', 'music', 'podcast',
            'friend', 'family', 'people', 'party', 'event',
            'task', 'assignment', 'project', 'deadline', 'meeting',
            'anxiety', 'stress', 'depression', 'mood', 'emotion', 'feeling'
        }
        
        # Combine all
        vocab = habit_verbs | habit_nouns | self.seed_keywords
        return vocab
    
    def extract_verb_object_pairs(self, texts: List[str]) -> Dict[str, int]:
        """Extract verb-object pairs from dependency parsing"""
        print("\n=== Extracting Verb-Object Pairs ===")
        
        pairs = []
        
        for text in tqdm(texts, desc="V-O extraction"):
            doc = self.nlp(text)
            
            for token in doc:
                # Look for main verbs
                if token.pos_ == "VERB":
                    verb_lemma = token.lemma_.lower()
                    
                    # Find direct objects
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            # Include modifiers
                            obj_words = []
                            for grandchild in child.children:
                                if grandchild.dep_ in ["amod", "compound"]:
                                    obj_words.append(grandchild.text.lower())
                            obj_words.append(child.text.lower())
                            
                            pair = f"{verb_lemma} {' '.join(obj_words)}"
                            
                            # Length filter
                            if len(pair.split()) <= self.max_phrase_length:
                                pairs.append(pair)
        
        pair_counts = Counter(pairs)
        filtered = {p: c for p, c in pair_counts.items() if c >= self.min_freq}
        
        print(f"✓ Found {len(filtered)} V-O pairs (freq >= {self.min_freq})")
        return filtered
    
    def extract_noun_phrases(self, texts: List[str]) -> Dict[str, int]:
        """Extract meaningful noun phrases"""
        print("\n=== Extracting Noun Phrases ===")
        
        phrases = []
        
        for text in tqdm(texts, desc="NP extraction"):
            doc = self.nlp(text)
            
            for chunk in doc.noun_chunks:
                # Filter by length
                if 2 <= len(chunk) <= self.max_phrase_length:
                    # Lemmatize
                    phrase = ' '.join([t.lemma_.lower() for t in chunk])
                    phrases.append(phrase)
        
        phrase_counts = Counter(phrases)
        filtered = {p: c for p, c in phrase_counts.items() if c >= self.min_freq}
        
        print(f"✓ Found {len(filtered)} noun phrases")
        return filtered
    
    def extract_bigrams_trigrams(self, texts: List[str]) -> Dict[str, int]:
        """Extract frequent bigrams and trigrams"""
        print("\n=== Extracting N-grams ===")
        
        ngrams = []
        
        for text in tqdm(texts, desc="N-grams"):
            doc = self.nlp(text)
            
            # Filter tokens
            tokens = [
                t.text.lower() for t in doc 
                if not t.is_stop and not t.is_punct and t.is_alpha
            ]
            
            # Bigrams
            for i in range(len(tokens) - 1):
                ngrams.append(f"{tokens[i]} {tokens[i+1]}")
            
            # Trigrams
            for i in range(len(tokens) - 2):
                ngrams.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        
        ngram_counts = Counter(ngrams)
        filtered = {n: c for n, c in ngram_counts.items() if c >= self.min_freq}
        
        print(f"✓ Found {len(filtered)} n-grams")
        return filtered
    
    def compute_tfidf_keywords(self, texts: List[str]) -> Dict[str, float]:
        """Extract keywords using TF-IDF"""
        print("\n=== Computing TF-IDF Scores ===")
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=self.top_n * 3,
            min_df=self.min_freq,
            max_df=0.7,  # Filter too common words
            token_pattern=r'\b[a-zA-Z]{3,}\b',
            lowercase=True,
            stop_words='english',
            sublinear_tf=True
        )
        
        try:
            X = vectorizer.fit_transform(texts)
            features = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF score for each term
            tfidf_scores = {}
            for idx, term in enumerate(features):
                score = float(X[:, idx].mean())
                tfidf_scores[term] = score
            
            print(f"✓ Computed TF-IDF for {len(tfidf_scores)} terms")
            return tfidf_scores
        
        except Exception as e:
            print(f"⚠ TF-IDF failed: {e}")
            return {}
    
    def compute_pmi_scores(
        self, 
        bigrams: Dict[str, int],
        texts: List[str]
    ) -> Dict[str, float]:
        """Compute Pointwise Mutual Information for bigrams"""
        print("\n=== Computing PMI Scores ===")
        
        # Count unigrams
        unigram_counts = Counter()
        total_tokens = 0
        
        for text in texts:
            tokens = text.lower().split()
            unigram_counts.update(tokens)
            total_tokens += len(tokens)
        
        pmi_scores = {}
        
        for bigram, bigram_freq in tqdm(bigrams.items(), desc="PMI"):
            words = bigram.split()
            if len(words) != 2:
                continue
            
            w1, w2 = words
            w1_freq = unigram_counts.get(w1, 1)
            w2_freq = unigram_counts.get(w2, 1)
            
            # PMI = log2(P(w1,w2) / (P(w1) * P(w2)))
            p_bigram = bigram_freq / (total_tokens - len(texts))
            p_w1 = w1_freq / total_tokens
            p_w2 = w2_freq / total_tokens
            
            if p_w1 > 0 and p_w2 > 0 and p_bigram > 0:
                pmi = math.log2(p_bigram / (p_w1 * p_w2))
                pmi_scores[bigram] = max(0, pmi)  # Keep only positive PMI
        
        print(f"✓ Computed PMI for {len(pmi_scores)} bigrams")
        return pmi_scores
    
    def filter_habit_relevance(self, candidates: Dict[str, float]) -> Dict[str, float]:
        """Filter candidates for habit relevance"""
        print("\n=== Filtering for Habit Relevance ===")
        
        filtered = {}
        
        for phrase, score in candidates.items():
            phrase_lower = phrase.lower()
            phrase_tokens = set(phrase_lower.split())
            
            # Check 1: Overlaps with habit vocabulary
            if phrase_tokens & self.habit_vocab:
                filtered[phrase] = score
                continue
            
            # Check 2: Overlaps with seed keywords
            if phrase_tokens & self.seed_keywords:
                filtered[phrase] = score
                continue
            
            # Check 3: Linguistic analysis
            doc = self.nlp(phrase_lower)
            
            # Has habit-related POS pattern (VERB + NOUN, ADJ + NOUN)
            pos_tags = [t.pos_ for t in doc]
            if ('VERB' in pos_tags and 'NOUN' in pos_tags) or \
               ('ADJ' in pos_tags and 'NOUN' in pos_tags):
                # Check if verb/noun is habit-related
                for token in doc:
                    if token.lemma_ in self.habit_vocab:
                        filtered[phrase] = score
                        break
        
        print(f"✓ Filtered {len(candidates)} → {len(filtered)} habit-relevant phrases")
        return filtered
    
    def merge_and_rank(
        self,
        vo_pairs: Dict[str, int],
        noun_phrases: Dict[str, int],
        ngrams: Dict[str, int],
        tfidf_scores: Dict[str, float],
        pmi_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """Merge all sources and compute composite ranking"""
        print("\n=== Merging and Ranking ===")
        
        # Collect all phrases
        all_phrases = set()
        all_phrases.update(vo_pairs.keys())
        all_phrases.update(noun_phrases.keys())
        all_phrases.update(ngrams.keys())
        all_phrases.update(tfidf_scores.keys())
        all_phrases.update(pmi_scores.keys())
        
        candidates = []
        
        for phrase in all_phrases:
            # Aggregate frequency from all sources
            freq = (vo_pairs.get(phrase, 0) + 
                   noun_phrases.get(phrase, 0) + 
                   ngrams.get(phrase, 0))
            
            # Get scores
            tfidf = tfidf_scores.get(phrase, 0.0)
            pmi = pmi_scores.get(phrase, 0.0)
            
            # Count how many extraction methods found this phrase
            method_count = sum([
                phrase in vo_pairs,
                phrase in noun_phrases,
                phrase in ngrams,
                tfidf > 0,
                pmi > 0
            ])
            
            # Check seed overlap
            overlaps_seed = bool(set(phrase.split()) & self.seed_keywords)
            
            candidates.append({
                'phrase': phrase,
                'frequency': freq,
                'tfidf_score': tfidf,
                'pmi_score': pmi,
                'method_count': method_count,
                'overlaps_seed': overlaps_seed,
                'word_count': len(phrase.split())
            })
        
        df = pd.DataFrame(candidates)
        
        # Normalize scores (0-1 range)
        for col in ['frequency', 'tfidf_score', 'pmi_score']:
            max_val = df[col].max()
            if max_val > 0:
                df[f'{col}_norm'] = df[col] / max_val
            else:
                df[f'{col}_norm'] = 0.0
        
        # Composite score (weighted combination)
        df['composite_score'] = (
            0.30 * df['frequency_norm'] +       # Frequency is important
            0.25 * df['tfidf_score_norm'] +     # TF-IDF uniqueness
            0.20 * df['pmi_score_norm'] +       # Collocation strength
            0.15 * (df['method_count'] / 5.0) + # Multiple sources
            0.10 * df['overlaps_seed'].astype(float)  # Seed relevance
        )
        
        # Sort by composite score
        df = df.sort_values('composite_score', ascending=False)
        
        print(f"✓ Ranked {len(df)} candidates")
        return df
    
    def mine_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """Main mining pipeline"""
        
        print(f"\n{'='*60}")
        print("HABIT KEYWORD MINING")
        print(f"{'='*60}")
        print(f"Processing {len(df)} journal entries...")
        
        texts = df[text_column].astype(str).tolist()
        
        # Extract from multiple sources
        vo_pairs = self.extract_verb_object_pairs(texts)
        noun_phrases = self.extract_noun_phrases(texts)
        ngrams = self.extract_bigrams_trigrams(texts)
        tfidf_scores = self.compute_tfidf_keywords(texts)
        
        # Compute PMI for bigrams
        bigrams_only = {k: v for k, v in ngrams.items() if len(k.split()) == 2}
        pmi_scores = self.compute_pmi_scores(bigrams_only, texts)
        
        # Merge and rank
        df_candidates = self.merge_and_rank(
            vo_pairs, noun_phrases, ngrams, tfidf_scores, pmi_scores
        )
        
        # Filter for habit relevance
        habit_relevant = self.filter_habit_relevance(
            dict(zip(df_candidates['phrase'], df_candidates['composite_score']))
        )
        
        # Keep only habit-relevant, top N
        df_final = df_candidates[
            df_candidates['phrase'].isin(habit_relevant.keys())
        ].head(self.top_n)
        
        print(f"\n{'='*60}")
        print(f"✅ Mining Complete: {len(df_final)} keywords extracted")
        print(f"{'='*60}\n")
        
        return df_final


def main():
    parser = argparse.ArgumentParser(
        description="Mine habit keywords from journal corpus"
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help="Input parquet/CSV file"
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help="Output path for mined keywords"
    )
    parser.add_argument(
        '--seed-ontology', type=str, default='seeds/seed_ontology.json',
        help="Path to seed ontology"
    )
    parser.add_argument(
        '--text-column', type=str, default='text',
        help="Column name for text"
    )
    parser.add_argument(
        '--spacy-model', type=str, default='en_core_web_sm',
        help="spaCy model"
    )
    parser.add_argument(
        '--min-freq', type=int, default=3,
        help="Minimum phrase frequency"
    )
    parser.add_argument(
        '--max-phrase-length', type=int, default=4,
        help="Maximum phrase length"
    )
    parser.add_argument(
        '--top-n', type=int, default=300,
        help="Number of top keywords"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.input}")
    input_file = Path(args.input)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    if input_file.suffix == '.parquet':
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    print(f"✓ Loaded {len(df)} entries")
    
    # Check text column
    if args.text_column not in df.columns:
        print(f"Error: Column '{args.text_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Initialize miner
    try:
        miner = HabitKeywordMiner(
            seed_ontology_path=args.seed_ontology,
            spacy_model=args.spacy_model,
            min_freq=args.min_freq,
            max_phrase_length=args.max_phrase_length,
            top_n=args.top_n
        )
    except Exception as e:
        print(f"Error initializing miner: {e}")
        return
    
    # Mine keywords
    df_keywords = miner.mine_from_dataframe(df, text_column=args.text_column)
    
    if df_keywords.empty:
        print("\n⚠ No keywords extracted!")
        return
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save main output
    if output_path.suffix == '.parquet':
        df_keywords.to_parquet(args.output, index=False)
    else:
        df_keywords.to_csv(args.output, index=False)
    print(f"✓ Saved: {args.output}")
    
    # Always save CSV for inspection
    csv_path = output_path.with_suffix('.csv')
    df_keywords.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Save summary
    summary = {
        'total_entries': len(df),
        'total_keywords': len(df_keywords),
        'avg_frequency': float(df_keywords['frequency'].mean()),
        'avg_composite_score': float(df_keywords['composite_score'].mean()),
        'top_20': df_keywords.head(20)[['phrase', 'composite_score', 'frequency']].to_dict('records')
    }
    
    summary_path = output_path.parent / 'keyword_mining_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved summary: {summary_path}")
    
    # Display top keywords
    print(f"\n{'='*60}")
    print("TOP 20 MINED KEYWORDS")
    print(f"{'='*60}")
    top_df = df_keywords.head(20)[['phrase', 'composite_score', 'frequency', 'method_count', 'overlaps_seed']]
    print(top_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("✅ KEYWORD MINING COMPLETED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()