"""
Enhanced Weak Supervision Pipeline for Habit Labeling
Uses Snorkel to combine multiple noisy labeling functions
Focused on habit category classification without temporal features
"""

# Prevent TensorFlow import issues
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'

import re
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

# Snorkel imports
from snorkel.labeling import labeling_function, LabelingFunction, PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

warnings.filterwarnings('ignore')


class HabitWeakSupervision:
    """Multi-class weak supervision for habit span labeling"""
    
    ABSTAIN = -1
    
    def __init__(
        self,
        seed_ontology_path: str,
        spacy_model: str = "en_core_web_sm",
        use_semantic: bool = False,
        similarity_threshold: float = 0.70,
        device: str = "cpu"
    ):
        """
        Initialize weak supervision system
        
        Args:
            seed_ontology_path: Path to seed_ontology.json
            spacy_model: spaCy model name
            use_semantic: Enable semantic similarity LFs
            similarity_threshold: Cosine similarity threshold (0.70 default)
            device: Device for embeddings (cpu/cuda)
        """
        self.similarity_threshold = similarity_threshold
        self.use_semantic = use_semantic
        self.device = device
        
        print(f"Loading seed ontology: {seed_ontology_path}")
        self.seed_habits = self._load_seed_ontology(seed_ontology_path)
        
        # Build mappings
        self._build_category_mappings()
        self._build_category_resources()
        
        # Load spaCy
        print(f"Loading spaCy: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading {spacy_model}...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        # Load embeddings (optional)
        self.semantic_model = None
        self.util = None
        if self.use_semantic:
            self._load_semantic_model()
        
        # Build labeling functions
        self.labeling_functions = self._build_labeling_functions()
        
        print(f"\n✅ Initialized with {self.num_classes} categories")
        print(f"✅ Created {len(self.labeling_functions)} labeling functions")
    
    def _load_seed_ontology(self, path: str) -> List[Dict]:
        """Load and validate seed ontology"""
        with open(path, 'r', encoding='utf-8') as f:
            habits = json.load(f)
        
        if not isinstance(habits, list) or len(habits) == 0:
            raise ValueError("Seed ontology must be non-empty list")
        
        required = ['id', 'name', 'category']
        for i, habit in enumerate(habits):
            for field in required:
                if field not in habit:
                    raise ValueError(f"Habit {i} missing required field: {field}")
        
        print(f"✓ Loaded {len(habits)} seed habits")
        return habits
    
    def _build_category_mappings(self):
        """Build category ↔ label mappings"""
        # Extract unique categories
        categories = sorted(set(h['category'].upper() for h in self.seed_habits))
        
        self.category_to_label = {cat: idx for idx, cat in enumerate(categories)}
        self.label_to_category = {idx: cat for cat, idx in self.category_to_label.items()}
        self.num_classes = len(categories)
        
        # Group habits by category
        self.category_habits = {cat: [] for cat in categories}
        for habit in self.seed_habits:
            category = habit['category'].upper()
            self.category_habits[category].append(habit)
        
        print(f"Categories: {list(self.category_to_label.keys())}")
    
    def _build_category_resources(self):
        """Build aliases, keywords, and patterns per category"""
        self.category_aliases = {}
        self.category_keywords = {}
        self.category_patterns = {}
        
        for category, habits in self.category_habits.items():
            # Collect aliases
            aliases = set()
            for habit in habits:
                aliases.add(habit['name'].lower())
                for alias in habit.get('aliases', []):
                    aliases.add(alias.lower())
            self.category_aliases[category] = aliases
            
            # Extract keywords from descriptions
            keywords = set()
            for habit in habits:
                desc = habit.get('description', '').lower()
                # Extract meaningful words (4+ chars)
                words = re.findall(r'\b[a-z]{4,}\b', desc)
                keywords.update(words)
            
            # Remove generic stopwords
            stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 
                        'will', 'when', 'what', 'which', 'their', 'there', 'where'}
            keywords = keywords - stopwords
            self.category_keywords[category] = keywords
            
            # Build regex patterns
            if aliases:
                pattern = re.compile(
                    r'\b(' + '|'.join(re.escape(a) for a in aliases) + r')\b',
                    re.IGNORECASE
                )
                self.category_patterns[category] = pattern
            else:
                self.category_patterns[category] = None
    
    def _load_semantic_model(self):
        """Load sentence transformer for semantic matching"""
        try:
            from sentence_transformers import SentenceTransformer, util as st_util
            
            print(f"Loading sentence transformer on {self.device}...")
            self.semantic_model = SentenceTransformer(
                'all-mpnet-base-v2', 
                device=self.device
            )
            self.util = st_util
            
            # Precompute embeddings
            self._precompute_embeddings()
            
        except Exception as e:
            print(f"⚠ Failed to load semantic model: {e}")
            print("⚠ Disabling semantic LFs")
            self.use_semantic = False
            self.semantic_model = None
    
    def _precompute_embeddings(self):
        """Precompute embeddings for all aliases"""
        if not self.semantic_model:
            return
        
        print("Precomputing alias embeddings...")
        self.category_embeddings = {}
        
        try:
            import torch
            
            for category, aliases in self.category_aliases.items():
                if aliases:
                    alias_list = list(aliases)
                    with torch.no_grad():
                        embeddings = self.semantic_model.encode(
                            alias_list,
                            convert_to_tensor=True,
                            show_progress_bar=False,
                            batch_size=32
                        )
                    self.category_embeddings[category] = {
                        'aliases': alias_list,
                        'embeddings': embeddings
                    }
            
            print(f"✓ Precomputed embeddings for {len(self.category_embeddings)} categories")
        
        except Exception as e:
            print(f"⚠ Embedding precomputation failed: {e}")
            self.use_semantic = False
    
    def _build_labeling_functions(self) -> List[LabelingFunction]:
        """Build all labeling functions"""
        lfs = []
        
        # 1. Exact alias matching (highest confidence)
        for category in self.category_to_label.keys():
            lf = self._make_alias_lf(category)
            lfs.append(lf)
        
        # 2. Partial alias matching (substring)
        for category in self.category_to_label.keys():
            lf = self._make_partial_alias_lf(category)
            lfs.append(lf)
        
        # 3. Keyword density (multiple keywords)
        for category in self.category_to_label.keys():
            lf = self._make_keyword_density_lf(category)
            lfs.append(lf)
        
        # 4. Semantic similarity (if enabled)
        if self.use_semantic and self.semantic_model:
            for category in self.category_to_label.keys():
                lf = self._make_semantic_lf(category)
                lfs.append(lf)
        
        # 5. POS + Verb pattern LFs
        lfs.append(self._make_verb_pattern_lf())
        lfs.append(self._make_negation_pattern_lf())
        
        # 6. Domain-specific heuristics
        lfs.append(self._make_duration_pattern_lf())
        lfs.append(self._make_frequency_pattern_lf())
        lfs.append(self._make_intensity_pattern_lf())
        lfs.append(self._make_location_pattern_lf())
        
        # 7. Contextual LFs (using context field)
        lfs.append(self._make_context_keyword_lf())
        
        return lfs
    
    # ============= ALIAS MATCHING LFs =============
    
    def _make_alias_lf(self, category: str) -> LabelingFunction:
        """Exact alias match LF"""
        label = self.category_to_label[category]
        pattern = self.category_patterns[category]
        
        @labeling_function(name=f"exact_{category.lower()}")
        def lf(x):
            if pattern and pattern.search(x.span.lower()):
                return label
            return self.ABSTAIN
        
        return lf
    
    def _make_partial_alias_lf(self, category: str) -> LabelingFunction:
        """Partial alias match (substring)"""
        label = self.category_to_label[category]
        aliases = self.category_aliases[category]
        
        @labeling_function(name=f"partial_{category.lower()}")
        def lf(x):
            span_lower = x.span.lower()
            # Check if any alias is substring
            matches = sum(1 for alias in aliases if alias in span_lower)
            # Require substantial match
            if matches >= 1 and len(span_lower) >= 5:
                return label
            return self.ABSTAIN
        
        return lf
    
    # ============= KEYWORD LFs =============
    
    def _make_keyword_density_lf(self, category: str) -> LabelingFunction:
        """Multiple keyword presence LF"""
        label = self.category_to_label[category]
        keywords = self.category_keywords[category]
        
        @labeling_function(name=f"keywords_{category.lower()}")
        def lf(x):
            span_lower = x.span.lower()
            context_lower = str(getattr(x, 'context', '')).lower()
            combined = span_lower + ' ' + context_lower
            
            # Count keyword hits
            hits = sum(1 for kw in keywords if kw in combined)
            
            # Require multiple hits
            if hits >= 2:
                return label
            return self.ABSTAIN
        
        return lf
    
    # ============= SEMANTIC SIMILARITY LFs =============
    
    def _make_semantic_lf(self, category: str) -> LabelingFunction:
        """Semantic similarity using embeddings"""
        label = self.category_to_label[category]
        
        @labeling_function(name=f"semantic_{category.lower()}")
        def lf(x):
            if not self.use_semantic or not self.semantic_model:
                return self.ABSTAIN
            
            if category not in self.category_embeddings:
                return self.ABSTAIN
            
            try:
                import torch
                
                # Encode span
                with torch.no_grad():
                    span_emb = self.semantic_model.encode(
                        x.span,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                
                # Compute similarity
                emb_data = self.category_embeddings[category]
                similarities = self.util.cos_sim(span_emb, emb_data['embeddings'])
                max_sim = similarities.max().item()
                
                if max_sim >= self.similarity_threshold:
                    return label
            
            except Exception:
                pass
            
            return self.ABSTAIN
        
        return lf
    
    # ============= LINGUISTIC PATTERN LFs =============
    
    def _make_verb_pattern_lf(self) -> LabelingFunction:
        """Verb lemma → category mapping"""
        
        # Define verb → category mappings
        verb_mappings = {
            'SLEEP': ['sleep', 'nap', 'rest', 'doze', 'snooze', 'wake', 'oversleep'],
            'FITNESS': ['exercise', 'run', 'jog', 'walk', 'workout', 'train', 'swim', 'cycle', 'lift', 'stretch', 'gym'],
            'ACADEMICS': ['study', 'learn', 'read', 'revise', 'prepare', 'research', 'review', 'memorize', 'practice'],
            'DIGITAL': ['scroll', 'browse', 'surf', 'stream', 'binge', 'game', 'play', 'watch', 'post', 'check', 'text'],
            'NUTRITION': ['eat', 'drink', 'cook', 'order', 'snack', 'feast', 'binge', 'consume', 'meal'],
            'SOCIAL': ['meet', 'hang', 'talk', 'chat', 'call', 'visit', 'socialize', 'party', 'date'],
            'WORK': ['work', 'code', 'design', 'develop', 'build', 'create', 'manage', 'lead'],
            'WELLNESS': ['meditate', 'relax', 'breathe', 'journal', 'reflect', 'practice', 'yoga'],
            'SUBSTANCE': ['smoke', 'vape', 'drink', 'consume', 'use'],
            'LEISURE': ['watch', 'play', 'read', 'listen', 'enjoy', 'relax', 'chill'],
            'HYGIENE': ['shower', 'brush', 'wash', 'clean', 'groom', 'bathe'],
            'PROCRASTINATION': ['procrastinate', 'delay', 'postpone', 'avoid', 'skip', 'miss', 'waste']
        }
        
        @labeling_function(name="verb_pattern")
        def lf(x):
            doc = self.nlp(x.span)
            verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]
            
            if not verbs:
                return self.ABSTAIN
            
            # Check each category
            for category, verb_list in verb_mappings.items():
                if category in self.category_to_label:
                    if any(v in verb_list for v in verbs):
                        return self.category_to_label[category]
            
            return self.ABSTAIN
        
        return lf
    
    def _make_negation_pattern_lf(self) -> LabelingFunction:
        """Negation + activity pattern"""
        
        negation_words = ['didn\'t', 'didnt', 'couldn\'t', 'couldnt', 'haven\'t',
                         'havent', 'not', 'never', 'no', 'avoided', 'skipped', 'missed']
        
        # Activity keywords per category
        neg_mappings = {
            'SLEEP': ['sleep', 'bed', 'rest', 'nap'],
            'FITNESS': ['exercise', 'gym', 'workout', 'run', 'walk'],
            'ACADEMICS': ['study', 'homework', 'assignment', 'work', 'revision'],
            'NUTRITION': ['eat', 'meal', 'breakfast', 'lunch', 'dinner', 'food'],
            'SOCIAL': ['meet', 'friends', 'people', 'party', 'hangout'],
            'WORK': ['work', 'project', 'task', 'meeting', 'deadline']
        }
        
        @labeling_function(name="negation_pattern")
        def lf(x):
            span_lower = x.span.lower()
            
            # Check for negation
            has_neg = any(neg in span_lower for neg in negation_words)
            
            if has_neg:
                # Check which category is negated
                for category, keywords in neg_mappings.items():
                    if category in self.category_to_label:
                        if any(kw in span_lower for kw in keywords):
                            return self.category_to_label[category]
            
            return self.ABSTAIN
        
        return lf
    
    # ============= DOMAIN HEURISTIC LFs =============
    
    def _make_duration_pattern_lf(self) -> LabelingFunction:
        """Duration mention + activity"""
        
        duration_re = re.compile(r'\b\d+\s*(?:hour|hr|minute|min)s?\b', re.IGNORECASE)
        
        duration_mappings = {
            'ACADEMICS': ['study', 'studied', 'homework', 'work', 'revision', 'learning'],
            'FITNESS': ['exercise', 'workout', 'run', 'gym', 'training'],
            'SLEEP': ['sleep', 'slept', 'nap', 'rest'],
            'DIGITAL': ['scroll', 'watch', 'youtube', 'netflix', 'gaming', 'browsing', 'instagram', 'tiktok'],
            'WORK': ['work', 'worked', 'coding', 'meeting', 'project']
        }
        
        @labeling_function(name="duration_mention")
        def lf(x):
            span_lower = x.span.lower()
            
            if duration_re.search(span_lower):
                # Check activity
                for category, keywords in duration_mappings.items():
                    if category in self.category_to_label:
                        if any(kw in span_lower for kw in keywords):
                            return self.category_to_label[category]
            
            return self.ABSTAIN
        
        return lf
    
    def _make_frequency_pattern_lf(self) -> LabelingFunction:
        """Frequency expressions (again, still, always)"""
        
        freq_words = ['again', 'still', 'always', 'constantly', 'repeatedly', 
                     'multiple times', 'all day', 'whole day', 'entire day']
        
        @labeling_function(name="frequency_pattern")
        def lf(x):
            span_lower = x.span.lower()
            context_lower = str(getattr(x, 'context', '')).lower()
            combined = span_lower + ' ' + context_lower
            
            has_freq = any(fw in combined for fw in freq_words)
            
            if has_freq:
                # Scroll/browse → DIGITAL
                if any(w in combined for w in ['scroll', 'browse', 'watch', 'youtube', 'social media']):
                    return self.category_to_label.get('DIGITAL', self.ABSTAIN)
                
                # Work/study → ACADEMICS or WORK
                if any(w in combined for w in ['study', 'work', 'coding', 'homework']):
                    if 'homework' in combined or 'study' in combined:
                        return self.category_to_label.get('ACADEMICS', self.ABSTAIN)
                    return self.category_to_label.get('WORK', self.ABSTAIN)
                
                # Procrastination
                if any(w in combined for w in ['procrastinate', 'waste', 'avoid']):
                    return self.category_to_label.get('PROCRASTINATION', self.ABSTAIN)
            
            return self.ABSTAIN
        
        return lf
    
    def _make_intensity_pattern_lf(self) -> LabelingFunction:
        """Intensity modifiers (too much, excessive)"""
        
        intensity_patterns = ['too much', 'excessive', 'excessively', 'way too',
                            'overly', 'really', 'very', 'extremely', 'heavily']
        
        @labeling_function(name="intensity_pattern")
        def lf(x):
            span_lower = x.span.lower()
            
            has_intensity = any(ip in span_lower for ip in intensity_patterns)
            
            if has_intensity:
                # Digital usage
                if any(w in span_lower for w in ['scroll', 'phone', 'screen', 'social', 'youtube', 'netflix']):
                    return self.category_to_label.get('DIGITAL', self.ABSTAIN)
                
                # Food/drink
                if any(w in span_lower for w in ['eat', 'ate', 'food', 'snack', 'drink', 'coffee']):
                    return self.category_to_label.get('NUTRITION', self.ABSTAIN)
                
                # Substances
                if any(w in span_lower for w in ['smoke', 'vape', 'alcohol', 'drink']):
                    return self.category_to_label.get('SUBSTANCE', self.ABSTAIN)
                
                # Work/study
                if any(w in span_lower for w in ['work', 'study', 'stress']):
                    return self.category_to_label.get('WORK', self.ABSTAIN)
            
            return self.ABSTAIN
        
        return lf
    
    def _make_location_pattern_lf(self) -> LabelingFunction:
        """Location mentions"""
        
        location_mappings = {
            'FITNESS': ['gym', 'park', 'track', 'pool', 'court', 'field', 'studio'],
            'ACADEMICS': ['library', 'class', 'classroom', 'lecture', 'school', 'university'],
            'NUTRITION': ['restaurant', 'cafe', 'kitchen', 'dining', 'cafeteria', 'food court'],
            'SOCIAL': ['party', 'bar', 'club', 'friend', 'gathering', 'event'],
            'WORK': ['office', 'desk', 'workplace', 'meeting room', 'conference']
        }
        
        @labeling_function(name="location_mention")
        def lf(x):
            span_lower = x.span.lower()
            context_lower = str(getattr(x, 'context', '')).lower()
            combined = span_lower + ' ' + context_lower
            
            for category, locations in location_mappings.items():
                if category in self.category_to_label:
                    if any(loc in combined for loc in locations):
                        return self.category_to_label[category]
            
            return self.ABSTAIN
        
        return lf
    
    def _make_context_keyword_lf(self) -> LabelingFunction:
        """Use context field for additional signals"""
        
        @labeling_function(name="context_keywords")
        def lf(x):
            context = str(getattr(x, 'context', '')).lower()
            
            if not context or len(context) < 10:
                return self.ABSTAIN
            
            # Strong category indicators in context
            context_signals = {
                'SLEEP': ['bed', 'sleep', 'tired', 'exhausted', 'wake', 'alarm', 'dream'],
                'FITNESS': ['gym', 'exercise', 'workout', 'cardio', 'weights', 'running', 'training'],
                'DIGITAL': ['phone', 'screen', 'youtube', 'instagram', 'tiktok', 'facebook', 'twitter', 'reddit'],
                'ACADEMICS': ['exam', 'test', 'assignment', 'homework', 'chapter', 'notes', 'lecture'],
                'NUTRITION': ['hungry', 'food', 'meal', 'breakfast', 'lunch', 'dinner', 'restaurant'],
                'SOCIAL': ['friends', 'people', 'party', 'hangout', 'together', 'conversation'],
                'WELLNESS': ['meditate', 'mindfulness', 'breathing', 'calm', 'peaceful', 'relaxed']
            }
            
            for category, signals in context_signals.items():
                if category in self.category_to_label:
                    hits = sum(1 for sig in signals if sig in context)
                    # Require multiple signals
                    if hits >= 2:
                        return self.category_to_label[category]
            
            return self.ABSTAIN
        
        return lf
    
    # ============= PIPELINE METHODS =============
    
    def apply_lfs(self, df: pd.DataFrame) -> np.ndarray:
        """Apply all labeling functions to dataframe"""
        print(f"\n{'='*60}")
        print(f"Applying {len(self.labeling_functions)} labeling functions...")
        print(f"{'='*60}")
        
        applier = PandasLFApplier(lfs=self.labeling_functions)
        L_train = applier.apply(df=df)
        
        print(f"✓ Label matrix shape: {L_train.shape}")
        print(f"  ({L_train.shape[0]} spans × {L_train.shape[1]} LFs)")
        
        return L_train
    
    def analyze_lfs(
        self, 
        L_train: np.ndarray, 
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Analyze LF performance and coverage"""
        print(f"\n{'='*60}")
        print("LABELING FUNCTION ANALYSIS")
        print(f"{'='*60}")
        
        lf_summary = LFAnalysis(L=L_train, lfs=self.labeling_functions).lf_summary()
        
        # Display summary
        print("\nLF Statistics:")
        print(lf_summary[['j', 'Polarity', 'Coverage', 'Overlaps', 'Conflicts']].to_string())
        
        # Coverage analysis
        covered = (L_train != self.ABSTAIN).any(axis=1).sum()
        total = len(L_train)
        coverage_pct = covered / total * 100
        
        avg_lfs = (L_train != self.ABSTAIN).sum(axis=1).mean()
        max_lfs = (L_train != self.ABSTAIN).sum(axis=1).max()
        
        print(f"\n{'='*60}")
        print("COVERAGE STATISTICS")
        print(f"{'='*60}")
        print(f"Total spans: {total}")
        print(f"Covered (≥1 LF): {covered} ({coverage_pct:.1f}%)")
        print(f"Uncovered: {total - covered} ({100-coverage_pct:.1f}%)")
        print(f"Avg LFs per span: {avg_lfs:.2f}")
        print(f"Max LFs per span: {max_lfs}")
        
        # Conflict analysis
        def count_conflicts(row):
            labels = row[row != self.ABSTAIN]
            return len(set(labels)) > 1
        
        conflicts = np.apply_along_axis(count_conflicts, 1, L_train).sum()
        conflict_pct = conflicts / total * 100
        
        print(f"\nConflicting spans: {conflicts} ({conflict_pct:.1f}%)")
        
        # Per-category coverage
        print(f"\n{'='*60}")
        print("PER-CATEGORY LF COVERAGE")
        print(f"{'='*60}")
        
        for category, label in self.category_to_label.items():
            cat_votes = (L_train == label).sum()
            cat_pct = cat_votes / total * 100 if total > 0 else 0
            print(f"{category:.<25} {cat_votes:>6} votes ({cat_pct:>5.1f}%)")
        
        # Save analysis
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            lf_summary.to_csv(save_path, index=False)
            print(f"\n✓ Saved LF analysis: {save_path}")
        
        return lf_summary
    
    def train_label_model(
        self,
        L_train: np.ndarray,
        n_epochs: int = 200,
        lr: float = 0.01,
        seed: int = 42
    ) -> LabelModel:
        """Train Snorkel label model"""
        print(f"\n{'='*60}")
        print("TRAINING LABEL MODEL")
        print(f"{'='*60}")
        print(f"Config: epochs={n_epochs}, lr={lr}, seed={seed}")
        print(f"Classes: {self.num_classes}")
        
        label_model = LabelModel(
            cardinality=self.num_classes,
            verbose=True
        )
        
        label_model.fit(
            L_train=L_train,
            n_epochs=n_epochs,
            lr=lr,
            seed=seed,
            log_freq=max(50, n_epochs // 4)
        )
        
        print("\n✅ Label model training complete")
        
        return label_model
    
    def generate_labels(
        self,
        label_model: LabelModel,
        L_train: np.ndarray,
        tie_break_policy: str = "abstain"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate probabilistic labels"""
        print(f"\n{'='*60}")
        print("GENERATING PROBABILISTIC LABELS")
        print(f"{'='*60}")
        
        # Get probabilities
        probs = label_model.predict_proba(L=L_train)
        
        # Get hard predictions
        preds = label_model.predict(L=L_train, tie_break_policy=tie_break_policy)
        
        # Statistics
        labeled = (preds != self.ABSTAIN).sum()
        total = len(preds)
        
        print(f"\nLabeled: {labeled} / {total} ({labeled/total*100:.1f}%)")
        print(f"\nLabel distribution:")
        
        unique, counts = np.unique(preds[preds != self.ABSTAIN], return_counts=True)
        for label, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
            category = self.label_to_category.get(label, 'UNKNOWN')
            pct = count / labeled * 100
            print(f"  {category:.<20} {count:>6} ({pct:>5.1f}%)")
        
        abstain_count = (preds == self.ABSTAIN).sum()
        print(f"  {'ABSTAIN':.<20} {abstain_count:>6} ({abstain_count/total*100:>5.1f}%)")
        
        # Confidence distribution
        labeled_mask = preds != self.ABSTAIN
        if labeled_mask.sum() > 0:
            max_probs = probs.max(axis=1)
            high_conf = (max_probs[labeled_mask] > 0.7).sum()
            med_conf = ((max_probs[labeled_mask] > 0.5) & (max_probs[labeled_mask] <= 0.7)).sum()
            low_conf = (max_probs[labeled_mask] <= 0.5).sum()
            
            print(f"\nConfidence distribution:")
            print(f"  High (>0.7):  {high_conf:>6} ({high_conf/labeled*100:>5.1f}%)")
            print(f"  Medium (0.5-0.7): {med_conf:>6} ({med_conf/labeled*100:>5.1f}%)")
            print(f"  Low (≤0.5):   {low_conf:>6} ({low_conf/labeled*100:>5.1f}%)")
        
        return preds, probs
    
    def save_results(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        L_train: np.ndarray,
        output_path: str
    ) -> pd.DataFrame:
        """Save weak supervision results"""
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        
        # Create output dataframe
        df_out = df.copy()
        
        # Add predictions
        df_out['weak_label'] = predictions
        df_out['weak_label_name'] = [
            self.label_to_category.get(p, 'ABSTAIN') if p != self.ABSTAIN else 'ABSTAIN'
            for p in predictions
        ]
        
        # Add confidence scores
        df_out['max_prob'] = probabilities.max(axis=1)
        
        # Add full probability vector (as JSON)
        df_out['prob_vec'] = [
            json.dumps([float(p) for p in probs])
            for probs in probabilities
        ]
        
        # Add LF voting statistics
        df_out['num_lfs_voted'] = (L_train != self.ABSTAIN).sum(axis=1)
        
        # Add second-best probability (for confidence calibration)
        sorted_probs = np.sort(probabilities, axis=1)
        df_out['second_prob'] = sorted_probs[:, -2]
        df_out['prob_margin'] = df_out['max_prob'] - df_out['second_prob']
        
        # Save parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_out.to_parquet(output_path, index=False)
        print(f"✓ Saved parquet: {output_path}")
        
        # Save CSV
        csv_path = output_path.with_suffix('.csv')
        df_out.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")
        
        # Save label mappings
        mappings = {
            'category_to_label': self.category_to_label,
            'label_to_category': self.label_to_category,
            'num_classes': self.num_classes,
            'abstain_value': self.ABSTAIN
        }
        
        mappings_path = output_path.parent / 'label_mappings.json'
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"✓ Saved mappings: {mappings_path}")
        
        return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Weak supervision for habit span labeling"
    )
    
    # I/O paths
    parser.add_argument(
        '--input', type=str, required=True,
        help="Input parquet/CSV with extracted spans"
    )
    parser.add_argument(
        '--output', type=str, default='results/labels/weak_labels.parquet',
        help="Output parquet file path"
    )
    parser.add_argument(
        '--seed-ontology', type=str, default='seeds/seed_ontology.json',
        help="Path to seed ontology JSON"
    )
    
    # Model configuration
    parser.add_argument(
        '--spacy-model', type=str, default='en_core_web_sm',
        help="spaCy model name"
    )
    parser.add_argument(
        '--use-semantic', action='store_true',
        help="Enable semantic similarity LFs (requires sentence-transformers)"
    )
    parser.add_argument(
        '--similarity-threshold', type=float, default=0.70,
        help="Semantic similarity threshold (default: 0.70)"
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help="Device for embeddings (cpu/cuda)"
    )
    
    # Training parameters
    parser.add_argument(
        '--n-epochs', type=int, default=200,
        help="Label model training epochs"
    )
    parser.add_argument(
        '--lr', type=float, default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        '--tie-break-policy', type=str, default='abstain',
        choices=['abstain', 'random'],
        help="Tie-breaking policy"
    )
    
    # Analysis options
    parser.add_argument(
        '--analyze-only', action='store_true',
        help="Only analyze LFs without training"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Load data
    print(f"{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    print(f"✓ Loaded {len(df)} spans")
    
    # Validate required columns
    if 'span' not in df.columns:
        print("Error: Input must have 'span' column")
        return
    
    if 'context' not in df.columns:
        print("⚠ Warning: 'context' column missing, using span as context")
        df['context'] = df['span']
    
    # Show sample
    print(f"\nSample spans:")
    for i, row in df.head(3).iterrows():
        print(f"  • {row['span']}")
    
    # Initialize weak supervision
    print(f"\n{'='*60}")
    print("INITIALIZING WEAK SUPERVISION")
    print(f"{'='*60}")
    
    try:
        ws = HabitWeakSupervision(
            seed_ontology_path=args.seed_ontology,
            spacy_model=args.spacy_model,
            use_semantic=args.use_semantic,
            similarity_threshold=args.similarity_threshold,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing weak supervision: {e}")
        return
    
    # Apply labeling functions
    L_train = ws.apply_lfs(df)
    
    # Analyze LFs
    lf_analysis_path = Path(args.output).parent / 'lf_analysis.csv'
    lf_summary = ws.analyze_lfs(L_train, save_path=str(lf_analysis_path))
    
    if args.analyze_only:
        print(f"\n{'='*60}")
        print("✅ ANALYSIS COMPLETE (--analyze-only mode)")
        print(f"{'='*60}")
        print(f"\nOutput: {lf_analysis_path}")
        return
    
    # Train label model
    label_model = ws.train_label_model(
        L_train=L_train,
        n_epochs=args.n_epochs,
        lr=args.lr,
        seed=args.seed
    )
    
    # Generate probabilistic labels
    predictions, probabilities = ws.generate_labels(
        label_model=label_model,
        L_train=L_train,
        tie_break_policy=args.tie_break_policy
    )
    
    # Save results
    df_output = ws.save_results(
        df=df,
        predictions=predictions,
        probabilities=probabilities,
        L_train=L_train,
        output_path=args.output
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    total = len(df)
    labeled = (predictions != ws.ABSTAIN).sum()
    coverage = labeled / total * 100
    
    print(f"\nInput spans: {total}")
    print(f"Labeled spans: {labeled} ({coverage:.1f}%)")
    print(f"Abstained: {total - labeled} ({100-coverage:.1f}%)")
    
    # Show high-confidence samples per category
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS (High Confidence)")
    print(f"{'='*60}")
    
    labeled_df = df_output[df_output['weak_label'] != ws.ABSTAIN]
    
    for category in sorted(ws.category_to_label.keys()):
        cat_df = labeled_df[labeled_df['weak_label_name'] == category]
        
        if len(cat_df) > 0:
            samples = cat_df.nlargest(3, 'max_prob')
            print(f"\n{category}:")
            for _, row in samples.iterrows():
                span_preview = row['span'][:50] + '...' if len(row['span']) > 50 else row['span']
                print(f"  • \"{span_preview}\"")
                print(f"    Prob: {row['max_prob']:.3f}, LFs: {row['num_lfs_voted']}")
    
    # Quality metrics
    high_conf = (df_output['max_prob'] > 0.7).sum()
    med_conf = ((df_output['max_prob'] > 0.5) & (df_output['max_prob'] <= 0.7)).sum()
    low_conf = (df_output['max_prob'] <= 0.5).sum()
    
    print(f"\n{'='*60}")
    print("QUALITY METRICS")
    print(f"{'='*60}")
    print(f"High confidence (>0.7): {high_conf} spans")
    print(f"Medium confidence (0.5-0.7): {med_conf} spans")
    print(f"Low confidence (≤0.5): {low_conf} spans")
    
    avg_margin = df_output[df_output['weak_label'] != ws.ABSTAIN]['prob_margin'].mean()
    print(f"Avg probability margin: {avg_margin:.3f}")
    
    # Files generated
    print(f"\n{'='*60}")
    print("✅ WEAK SUPERVISION COMPLETED")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  • Weak labels: {args.output}")
    print(f"  • CSV export: {Path(args.output).with_suffix('.csv')}")
    print(f"  • Label mappings: {Path(args.output).parent / 'label_mappings.json'}")
    print(f"  • LF analysis: {lf_analysis_path}")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Review LF analysis: {lf_analysis_path}")
    print(f"  2. Filter high-confidence labels (>0.7) for NER training")
    print(f"  3. Proceed to Step 6: Convert to BIO format (to_bio.py)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()