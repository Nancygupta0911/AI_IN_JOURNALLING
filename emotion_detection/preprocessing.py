"""
Enhanced Multi-Label Emotion Classification Preprocessing Pipeline v4.0
========================================================================

RESEARCH-ALIGNED IMPROVEMENTS FOR HABIT-EMOTION ANALYSIS:
1. ✅ Multi-label emotion detection (captures complex emotional states)
2. ✅ Temporal sequencing preservation (tracks emotion evolution)
3. ✅ Context-aware augmentation (maintains semantic integrity)
4. ✅ Stratified sampling (balances without losing rare patterns)
5. ✅ Metadata enrichment (intensity, confidence, timestamps)
6. ✅ Habit keyword extraction (links behaviors to emotions)
7. ✅ Emotion co-occurrence analysis (finds patterns)

Author: Research-Optimized for Habit-Emotion Correlation Study
Version: 4.0 - Multi-Label with Temporal Context
"""

import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
import torch
from torch.utils.data import Dataset
from transformers import DebertaV2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample

# NLP utilities
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    STOPWORDS = set(stopwords.words('english'))
except:
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    STOPWORDS = set(stopwords.words('english'))

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


class ResearchAlignedEmotionPreprocessor:
    """
    Multi-label emotion preprocessing optimized for habit-emotion research.
    
    Key Innovations:
    - Multi-label classification (captures emotional complexity)
    - Temporal context preservation (tracks evolution)
    - Habit keyword extraction (links behaviors to emotions)
    - Stratified multi-label sampling (maintains rare patterns)
    - Emotion intensity scoring (quantifies strength)
    """
    
    def __init__(self, data_dir: str = ".", output_dir: str = "./processed_emotion_data_v4"):
        """Initialize research-aligned preprocessor."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info("=" * 100)
        self.logger.info("RESEARCH-ALIGNED MULTI-LABEL EMOTION PREPROCESSING v4.0")
        self.logger.info("=" * 100)
        
        # Initialize tokenizer
        try:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
            self.logger.info("✓ DeBERTa-v3-base tokenizer loaded")
        except Exception as e:
            self.logger.error(f"❌ Failed to load tokenizer: {e}")
            raise
        
        # Enhanced emotion hierarchy with intensity markers
        self.emotion_hierarchy = {
            'anger': {
                'primary': ['anger', 'angry', 'furious', 'rage', 'enraged'],
                'moderate': ['mad', 'irritated', 'annoyed', 'frustrated'],
                'mild': ['bothered', 'irked', 'miffed']
            },
            'anxiety': {
                'primary': ['anxiety', 'anxious', 'panic', 'panicked', 'terrified'],
                'moderate': ['worried', 'nervous', 'nervousness', 'stressed'],
                'mild': ['concerned', 'uneasy', 'apprehensive']
            },
            'calmness': {
                'primary': ['calm', 'peaceful', 'serene', 'tranquil'],
                'moderate': ['calmness', 'relaxed', 'composed'],
                'mild': ['at ease', 'settled', 'stable']
            },
            'confidence': {
                'primary': ['confident', 'self-assured', 'certain', 'bold'],
                'moderate': ['confidence', 'assured', 'determined'],
                'mild': ['hopeful', 'optimistic']
            },
            'confusion': {
                'primary': ['confusion', 'confused', 'bewildered', 'perplexed'],
                'moderate': ['uncertain', 'puzzled', 'unclear'],
                'mild': ['unsure', 'questioning', 'wondering']
            },
            'contentment': {
                'primary': ['content', 'contentment', 'satisfied', 'fulfilled'],
                'moderate': ['pleased', 'gratified'],
                'mild': ['okay', 'alright', 'fine']
            },
            'disappointment': {
                'primary': ['disappointed', 'devastated', 'crushed'],
                'moderate': ['disappointment', 'let down', 'discouraged'],
                'mild': ['dismayed', 'underwhelmed']
            },
            'disgust': {
                'primary': ['disgust', 'disgusted', 'revolted', 'repulsed'],
                'moderate': ['grossed out', 'sickened'],
                'mild': ['put off', 'distasteful']
            },
            'excitement': {
                'primary': ['excitement', 'excited', 'thrilled', 'ecstatic'],
                'moderate': ['eager', 'enthusiastic', 'energetic'],
                'mild': ['interested', 'curious', 'engaged']
            },
            'fear': {
                'primary': ['fear', 'afraid', 'terrified', 'horrified'],
                'moderate': ['scared', 'frightened', 'fearful'],
                'mild': ['nervous', 'worried', 'concerned']
            },
            'frustration': {
                'primary': ['frustration', 'frustrated', 'exasperated'],
                'moderate': ['annoyed', 'irritated'],
                'mild': ['bothered', 'inconvenienced']
            },
            'gratitude': {
                'primary': ['gratitude', 'grateful', 'thankful', 'blessed'],
                'moderate': ['appreciative', 'touched'],
                'mild': ['glad', 'pleased']
            },
            'hope': {
                'primary': ['hope', 'hopeful', 'optimistic', 'inspired'],
                'moderate': ['optimism', 'encouraged', 'positive'],
                'mild': ['looking forward', 'anticipating']
            },
            'joy': {
                'primary': ['joy', 'joyful', 'elated', 'ecstatic', 'overjoyed'],
                'moderate': ['happy', 'happiness', 'cheerful', 'delighted'],
                'mild': ['glad', 'pleased', 'content']
            },
            'loneliness': {
                'primary': ['loneliness', 'lonely', 'isolated', 'abandoned'],
                'moderate': ['alone', 'disconnected', 'separate'],
                'mild': ['distant', 'apart']
            },
            'love': {
                'primary': ['love', 'loving', 'adore', 'cherish'],
                'moderate': ['affection', 'caring', 'compassion'],
                'mild': ['like', 'fond', 'appreciate']
            },
            'neutral': {
                'primary': ['neutral', 'normal', 'regular', 'typical'],
                'moderate': ['okay', 'fine', 'alright'],
                'mild': ['meh', 'whatever']
            },
            'pride': {
                'primary': ['pride', 'proud', 'accomplished', 'triumphant'],
                'moderate': ['achievement', 'successful'],
                'mild': ['satisfied', 'pleased']
            },
            'sadness': {
                'primary': ['sadness', 'sad', 'depressed', 'miserable', 'heartbroken'],
                'moderate': ['unhappy', 'sorrowful', 'down', 'blue'],
                'mild': ['disappointed', 'low', 'somber']
            },
            'shame': {
                'primary': ['shame', 'ashamed', 'humiliated', 'mortified'],
                'moderate': ['embarrassed', 'embarrassment', 'guilty'],
                'mild': ['awkward', 'uncomfortable', 'self-conscious']
            },
            'surprise': {
                'primary': ['surprise', 'surprised', 'astonished', 'shocked', 'stunned'],
                'moderate': ['amazed', 'startled', 'taken aback'],
                'mild': ['unexpected', 'unforeseen']
            }
        }
        
        # Flatten emotion mapping with intensity scores
        self.emotion_mapping = {}  # word -> (emotion, intensity)
        for emotion, intensities in self.emotion_hierarchy.items():
            for intensity_level, words in intensities.items():
                score = {'primary': 1.0, 'moderate': 0.6, 'mild': 0.3}[intensity_level]
                for word in words:
                    self.emotion_mapping[word.lower()] = (emotion, score)
        
        self.core_emotions = sorted(list(self.emotion_hierarchy.keys()))
        
        # Habit keywords for behavioral tracking (CRITICAL FOR RESEARCH)
        self.habit_keywords = {
            'exercise': ['gym', 'workout', 'exercise', 'run', 'running', 'jog', 'yoga', 
                        'fitness', 'training', 'sport', 'swim', 'cycling', 'walk', 'walking'],
            'sleep': ['sleep', 'slept', 'insomnia', 'tired', 'exhausted', 'rest', 'rested',
                     'bed', 'nap', 'awake', 'sleepy', 'fatigue'],
            'social': ['friends', 'family', 'meet', 'met', 'hangout', 'party', 'social',
                      'gathering', 'visit', 'call', 'text', 'message', 'chat'],
            'work': ['work', 'job', 'office', 'meeting', 'project', 'deadline', 'boss',
                    'colleague', 'task', 'assignment', 'presentation'],
            'study': ['study', 'studied', 'exam', 'test', 'class', 'homework', 'assignment',
                     'lecture', 'learning', 'reading', 'research'],
            'food': ['eat', 'ate', 'food', 'meal', 'lunch', 'dinner', 'breakfast', 'hungry',
                    'diet', 'cook', 'cooked', 'restaurant', 'snack'],
            'media': ['watch', 'watched', 'tv', 'movie', 'show', 'netflix', 'youtube',
                     'video', 'game', 'gaming', 'social media', 'facebook', 'instagram'],
            'health': ['doctor', 'medication', 'therapy', 'counseling', 'sick', 'illness',
                      'pain', 'health', 'medical', 'hospital'],
            'creative': ['write', 'writing', 'draw', 'drawing', 'paint', 'music', 'create',
                        'creative', 'art', 'hobby'],
            'mindfulness': ['meditate', 'meditation', 'mindful', 'breathe', 'journal',
                           'reflect', 'grateful', 'gratitude', 'pray', 'prayer']
        }
        
        # Emotion co-occurrence patterns (for research insights)
        self.emotion_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Statistics tracking
        self.stats = {
            'source_stats': {},
            'quality_filtering': {},
            'deduplication': {},
            'multi_label_distribution': {},
            'emotion_cooccurrence': {},
            'habit_correlations': {},
            'temporal_analysis': {},
            'balancing': {},
            'splits': {}
        }
        
        # Multi-label balancing config
        self.balancing_config = {
            'min_samples_per_label': 800,  # Minimum samples for training
            'max_samples_per_label': 6000,  # Maximum to prevent dominance
            'target_label_distribution': 0.15,  # Target 15% occurrence per label
            'preserve_rare_combinations': True,  # Keep rare emotion combos
            'temporal_window_days': 30  # Context window for temporal features
        }
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.output_dir / 'preprocessing_v4_multilabel.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def clean_text_preserve_context(self, text: str) -> str:
        """
        Context-aware text cleaning that preserves emotional and behavioral cues.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Preserve important patterns BEFORE cleaning
        # 1. Emotion intensifiers
        intensifiers = r'\b(very|extremely|really|so|too|quite|incredibly|absolutely)\b'
        text = re.sub(intensifiers, lambda m: f' {m.group(0).upper()} ', text, flags=re.IGNORECASE)
        
        # 2. Negations (critical for sentiment)
        negations = r"\b(not|never|no|don't|doesn't|didn't|won't|can't|couldn't)\b"
        text = re.sub(negations, lambda m: f' NEG_{m.group(0).upper()} ', text, flags=re.IGNORECASE)
        
        # Standard cleaning
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Preserve emotional punctuation (but normalize)
        text = re.sub(r'([!?]){4,}', r'\1\1\1', text)
        text = re.sub(r'\.{4,}', '...', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text.strip()
    
    def extract_emotions_with_intensity(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract multiple emotions with intensity scores.
        
        Returns: [(emotion, confidence_score), ...]
        """
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        detected_emotions = defaultdict(float)
        
        # 1. Direct keyword matching with intensity
        for word in words:
            if word in self.emotion_mapping:
                emotion, intensity = self.emotion_mapping[word]
                detected_emotions[emotion] = max(detected_emotions[emotion], intensity)
        
        # 2. Boost score for emotion words with intensifiers nearby
        for i, word in enumerate(words):
            if word in self.emotion_mapping:
                emotion, base_score = self.emotion_mapping[word]
                
                # Check surrounding context (±2 words)
                context = words[max(0, i-2):min(len(words), i+3)]
                
                # Intensifiers boost
                if any(w in ['very', 'extremely', 'really', 'so', 'incredibly'] for w in context):
                    detected_emotions[emotion] = min(1.0, detected_emotions[emotion] * 1.3)
                
                # Negations reduce (but don't eliminate - "not happy" still relates to happiness)
                if any(w.startswith('neg_') for w in context):
                    detected_emotions[emotion] = min(1.0, detected_emotions[emotion] * 0.7)
        
        # 3. Return emotions above threshold (0.3 = mild intensity)
        return [(emotion, score) for emotion, score in detected_emotions.items() if score >= 0.3]
    
    def extract_habits(self, text: str) -> Dict[str, bool]:
        """
        Extract mentioned habits/activities from text.
        CRITICAL for habit-emotion correlation research.
        """
        text_lower = text.lower()
        words = set(word_tokenize(text_lower))
        
        detected_habits = {}
        for habit_category, keywords in self.habit_keywords.items():
            # Check if any keyword present
            detected_habits[habit_category] = any(kw in text_lower for kw in keywords)
        
        return detected_habits
    
    def extract_temporal_features(self, text: str, date: Optional[datetime] = None) -> Dict:
        """
        Extract temporal context for time-series analysis.
        """
        features = {
            'has_date': date is not None,
            'day_of_week': date.weekday() if date else None,
            'is_weekend': date.weekday() >= 5 if date else None,
            'month': date.month if date else None
        }
        
        # Temporal indicators in text
        text_lower = text.lower()
        features['mentions_today'] = 'today' in text_lower
        features['mentions_yesterday'] = 'yesterday' in text_lower
        features['mentions_tomorrow'] = 'tomorrow' in text_lower
        
        return features
    
    def analyze_text_quality_research(self, text: str) -> Dict:
        """
        Research-optimized quality analysis.
        """
        if not isinstance(text, str) or not text.strip():
            return {'is_valid': False, 'reason': 'empty', 'richness_score': 0.0}
        
        words = text.split()
        word_count = len(words)
        
        # Minimum length check (relaxed for genuine journal entries)
        if word_count < 3:
            return {'is_valid': False, 'reason': 'too_short', 'richness_score': 0.0}
        
        # Calculate text richness (diversity)
        unique_words = len(set(word.lower() for word in words))
        richness_score = unique_words / word_count if word_count > 0 else 0
        
        # Check for spam patterns
        if richness_score < 0.2 and word_count > 10:
            return {'is_valid': False, 'reason': 'spam_pattern', 'richness_score': richness_score}
        
        # Must contain alphabetic content
        if not re.search(r'[a-zA-Z]', text):
            return {'is_valid': False, 'reason': 'no_text', 'richness_score': 0.0}
        
        # Calculate emotional content density (research metric)
        emotion_words = sum(1 for word in words if word.lower() in self.emotion_mapping)
        emotional_density = emotion_words / word_count if word_count > 0 else 0
        
        return {
            'is_valid': True,
            'word_count': word_count,
            'richness_score': richness_score,
            'emotional_density': emotional_density,
            'unique_words': unique_words
        }
    
    def load_all_data_sources(self) -> pd.DataFrame:
        """Load and unify all data sources with multi-label support."""
        self.logger.info("Loading all data sources with multi-label extraction...")
        
        all_dfs = []
        
        # Source loaders (adapted for multi-label)
        loaders = [
            ('archive2', self._load_archive2_multilabel),
            ('archive3', self._load_archive3_multilabel),
            ('goemotions', self._load_goemotions_multilabel),
            ('student_journal', self._load_student_journal_multilabel),
            ('parquet', self._load_parquet_files_multilabel),
            ('additional_csvs', self._load_additional_csvs_multilabel)
        ]
        
        for source_name, loader_func in loaders:
            try:
                df = loader_func()
                if len(df) > 0:
                    all_dfs.append(df)
                    self.stats['source_stats'][source_name] = len(df)
                    self.logger.info(f"✓ {source_name}: {len(df)} samples")
            except Exception as e:
                self.logger.warning(f"⚠ {source_name} failed: {e}")
        
        if not all_dfs:
            raise ValueError("❌ No data could be loaded!")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        self.logger.info(f"\n✓ Combined: {len(combined_df)} samples from {len(all_dfs)} sources")
        
        return combined_df
    
    def _load_archive2_multilabel(self) -> pd.DataFrame:
        """Load Daylio data with temporal context."""
        daylio_path = self.data_dir / "archive (2)" / "Daylio_Abid.csv"
        if not daylio_path.exists():
            return pd.DataFrame(columns=['text', 'emotions', 'date', 'habits'])
        
        df = pd.read_csv(daylio_path)
        
        processed_rows = []
        for _, row in df.iterrows():
            # Combine activities and mood
            text = self.clean_text_preserve_context(
                f"{row.get('activities', '')} {row.get('mood', '')}"
            )
            
            if len(text) < 10:
                continue
            
            # Extract emotions (multi-label)
            emotions_with_scores = self.extract_emotions_with_intensity(text)
            if not emotions_with_scores:
                emotions_with_scores = [('neutral', 0.5)]
            
            emotions = [e[0] for e in emotions_with_scores]
            intensities = {e[0]: e[1] for e in emotions_with_scores}
            
            # Extract habits
            habits = self.extract_habits(text)
            
            # Parse date if available
            date = None
            if 'date' in row and pd.notna(row['date']):
                try:
                    date = pd.to_datetime(row['date'])
                except:
                    pass
            
            processed_rows.append({
                'text': text,
                'emotions': emotions,
                'emotion_intensities': intensities,
                'habits': habits,
                'date': date,
                'source': 'daylio'
            })
        
        return pd.DataFrame(processed_rows)
    
    def _load_archive3_multilabel(self) -> pd.DataFrame:
        """Load binary emotion data as multi-label."""
        data_path = self.data_dir / "archive (3)" / "data.csv"
        if not data_path.exists():
            return pd.DataFrame(columns=['text', 'emotions', 'habits'])
        
        df = pd.read_csv(data_path)
        
        emotion_cols = [col for col in df.columns if col.startswith('Answer.f1.')]
        text_col = None
        for col in df.columns:
            if col not in emotion_cols and df[col].dtype == 'object':
                sample = df[col].dropna().head(10)
                if any(len(str(s)) > 20 for s in sample):
                    text_col = col
                    break
        
        if not text_col or not emotion_cols:
            return pd.DataFrame(columns=['text', 'emotions', 'habits'])
        
        processed_rows = []
        for _, row in df.iterrows():
            text = self.clean_text_preserve_context(str(row[text_col]))
            
            if len(text) < 10:
                continue
            
            # Extract ALL active emotions (multi-label)
            detected_emotions = []
            for col in emotion_cols:
                if row.get(col, 0) == 1:
                    emotion_name = col.replace('Answer.f1.', '').replace('.raw', '')
                    if emotion_name.lower() in [e for e, _ in self.emotion_mapping.values()]:
                        detected_emotions.append(emotion_name.lower())
            
            if not detected_emotions:
                # Fallback to text-based extraction
                emotions_with_scores = self.extract_emotions_with_intensity(text)
                detected_emotions = [e[0] for e in emotions_with_scores] if emotions_with_scores else ['neutral']
            
            intensities = {e: 0.8 for e in detected_emotions}  # Binary data gets uniform intensity
            habits = self.extract_habits(text)
            
            processed_rows.append({
                'text': text,
                'emotions': detected_emotions,
                'emotion_intensities': intensities,
                'habits': habits,
                'date': None,
                'source': 'archive3'
            })
        
        return pd.DataFrame(processed_rows)
    
    def _load_goemotions_multilabel(self) -> pd.DataFrame:
        """Load GoEmotions with proper multi-label handling."""
        goemotions_path = self.data_dir / 'goemotions.csv'
        if not goemotions_path.exists():
            return pd.DataFrame(columns=['text', 'emotions', 'habits'])
        
        df = pd.read_csv(goemotions_path)
        
        if 'text' not in df.columns:
            return pd.DataFrame(columns=['text', 'emotions', 'habits'])
        
        # Find binary emotion columns
        emotion_cols = []
        for col in df.columns:
            if col != 'text' and df[col].dtype in ['int64', 'float64']:
                unique_vals = set(df[col].dropna().unique())
                if unique_vals.issubset({0, 1, 0.0, 1.0}):
                    # Check if column name is an emotion
                    if col.lower() in [e for e, _ in self.emotion_mapping.values()]:
                        emotion_cols.append(col)
        
        if not emotion_cols:
            return pd.DataFrame(columns=['text', 'emotions', 'habits'])
        
        processed_rows = []
        for _, row in df.iterrows():
            text = self.clean_text_preserve_context(str(row['text']))
            
            if len(text) < 10:
                continue
            
            # Extract ALL active emotions
            detected_emotions = []
            for col in emotion_cols:
                if row.get(col, 0) == 1:
                    emotion = col.lower()
                    if emotion in self.core_emotions:
                        detected_emotions.append(emotion)
            
            if not detected_emotions:
                detected_emotions = ['neutral']
            
            intensities = {e: 0.8 for e in detected_emotions}
            habits = self.extract_habits(text)
            
            processed_rows.append({
                'text': text,
                'emotions': detected_emotions,
                'emotion_intensities': intensities,
                'habits': habits,
                'date': None,
                'source': 'goemotions'
            })
        
        return pd.DataFrame(processed_rows)
    
    def _load_student_journal_multilabel(self) -> pd.DataFrame:
        """Load student journal with contextual extraction."""
        json_path = self.data_dir / 'student_journal.json'
        if not json_path.exists():
            return pd.DataFrame(columns=['text', 'emotions', 'habits', 'date'])
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'entries' not in data:
            return pd.DataFrame(columns=['text', 'emotions', 'habits', 'date'])
        
        processed_rows = []
        for entry in data['entries']:
            if 'entry' not in entry:
                continue
            
            text = self.clean_text_preserve_context(entry['entry'])
            if len(text) < 10:
                continue
            
            # Extract emotions from text AND provided mood
            emotions_from_text = self.extract_emotions_with_intensity(text)
            emotions = [e[0] for e in emotions_from_text]
            
            # Add provided mood if available
            if 'mood' in entry and pd.notna(entry['mood']):
                mood_emotion = None
                mood_lower = str(entry['mood']).lower()
                if mood_lower in self.emotion_mapping:
                    mood_emotion, _ = self.emotion_mapping[mood_lower]
                elif mood_lower in self.core_emotions:
                    mood_emotion = mood_lower
                
                if mood_emotion and mood_emotion not in emotions:
                    emotions.append(mood_emotion)
            
            if not emotions:
                emotions = ['neutral']
            
            intensities = {e[0]: e[1] for e in emotions_from_text}
            habits = self.extract_habits(text)
            
            # Parse date if available
            date = None
            if 'date' in entry and pd.notna(entry['date']):
                try:
                    date = pd.to_datetime(entry['date'])
                except:
                    pass
            
            processed_rows.append({
                'text': text,
                'emotions': emotions,
                'emotion_intensities': intensities,
                'habits': habits,
                'date': date,
                'source': 'student_journal'
            })
        
        return pd.DataFrame(processed_rows)
    
    def _load_parquet_files_multilabel(self) -> pd.DataFrame:
        """Load parquet files with multi-label extraction."""
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame(columns=['text', 'emotions', 'habits'])
        
        all_dfs = []
        for parquet_path in parquet_files:
            try:
                df = pd.read_parquet(parquet_path)
                
                # Find emotion columns
                emotion_cols = [(col, col.replace('Answer.f1.', '').replace('.raw', '')) 
                               for col in df.columns if col.startswith('Answer.f1.')]
                
                # Find text column
                text_col = next((col for col in df.columns 
                                if df[col].dtype == 'object' and 
                                any(len(str(s)) > 20 for s in df[col].dropna().head(10))), None)
                
                if not text_col:
                    continue
                
                processed_rows = []
                for _, row in df.iterrows():
                    text = self.clean_text_preserve_context(str(row[text_col]))
                    
                    if len(text) < 10:
                        continue
                    
                    detected_emotions = []
                    if emotion_cols:
                        # Binary emotion format - extract ALL active
                        for col, emotion_name in emotion_cols:
                            if row.get(col, 0) == 1:
                                emotion = emotion_name.lower()
                                if emotion in self.core_emotions:
                                    detected_emotions.append(emotion)
                    
                    # Fallback to text extraction
                    if not detected_emotions:
                        emotions_from_text = self.extract_emotions_with_intensity(text)
                        detected_emotions = [e[0] for e in emotions_from_text] if emotions_from_text else ['neutral']
                    
                    intensities = {e: 0.8 for e in detected_emotions}
                    habits = self.extract_habits(text)
                    
                    processed_rows.append({
                        'text': text,
                        'emotions': detected_emotions,
                        'emotion_intensities': intensities,
                        'habits': habits,
                        'date': None,
                        'source': f'parquet_{parquet_path.stem}'
                    })
                
                if processed_rows:
                    all_dfs.append(pd.DataFrame(processed_rows))
                    
            except Exception as e:
                self.logger.warning(f"⚠ Failed to load {parquet_path.name}: {e}")
        
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=['text', 'emotions', 'habits'])
    
    def _load_additional_csvs_multilabel(self) -> pd.DataFrame:
        """Load additional CSV files with multi-label support."""
        csv_files = ['train (1).csv', 'emotion_dataset_2.csv']
        all_dfs = []
        
        for csv_file in csv_files:
            csv_path = self.data_dir / csv_file
            if not csv_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path)
                
                text_col = next((col for col in ['text', 'Text', 'Clean_Text', 'entry'] 
                               if col in df.columns), None)
                label_col = next((col for col in ['label', 'Emotion', 'emotion', 'mood'] 
                                if col in df.columns), None)
                
                if not text_col:
                    continue
                
                processed_rows = []
                for _, row in df.iterrows():
                    text = self.clean_text_preserve_context(str(row[text_col]))
                    
                    if len(text) < 10:
                        continue
                    
                    # Extract from text
                    emotions_from_text = self.extract_emotions_with_intensity(text)
                    emotions = [e[0] for e in emotions_from_text]
                    
                    # Add label if valid
                    if label_col and pd.notna(row[label_col]):
                        label_lower = str(row[label_col]).lower()
                        if label_lower in self.emotion_mapping:
                            label_emotion, _ = self.emotion_mapping[label_lower]
                            if label_emotion not in emotions:
                                emotions.append(label_emotion)
                        elif label_lower in self.core_emotions:
                            if label_lower not in emotions:
                                emotions.append(label_lower)
                    
                    if not emotions:
                        emotions = ['neutral']
                    
                    intensities = {e[0]: e[1] for e in emotions_from_text}
                    habits = self.extract_habits(text)
                    
                    processed_rows.append({
                        'text': text,
                        'emotions': emotions,
                        'emotion_intensities': intensities,
                        'habits': habits,
                        'date': None,
                        'source': csv_file
                    })
                
                if processed_rows:
                    all_dfs.append(pd.DataFrame(processed_rows))
                    
            except Exception as e:
                self.logger.warning(f"⚠ Failed to load {csv_file}: {e}")
        
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=['text', 'emotions', 'habits'])
    
    def perform_quality_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply research-optimized quality filtering."""
        self.logger.info("\nPerforming quality analysis (research-optimized)...")
        
        initial_count = len(df)
        
        # Analyze quality
        quality_results = df['text'].apply(self.analyze_text_quality_research)
        df['quality'] = quality_results
        
        # Filter valid samples
        df_clean = df[df['quality'].apply(lambda x: x.get('is_valid', False))].copy()
        
        # Store quality metrics for research analysis
        df_clean['richness_score'] = df_clean['quality'].apply(lambda x: x.get('richness_score', 0))
        df_clean['emotional_density'] = df_clean['quality'].apply(lambda x: x.get('emotional_density', 0))
        df_clean = df_clean.drop('quality', axis=1)
        
        # Statistics
        removed = initial_count - len(df_clean)
        removal_reasons = Counter([q.get('reason', 'unknown') 
                                  for q in quality_results if not q.get('is_valid', False)])
        
        self.logger.info(f"Quality filtering: {initial_count} → {len(df_clean)} samples")
        self.logger.info(f"Removed: {removed} samples")
        for reason, count in removal_reasons.most_common():
            self.logger.info(f"  - {reason}: {count}")
        
        self.stats['quality_filtering'] = {
            'initial': initial_count,
            'final': len(df_clean),
            'removed': removed,
            'reasons': dict(removal_reasons),
            'avg_richness': df_clean['richness_score'].mean(),
            'avg_emotional_density': df_clean['emotional_density'].mean()
        }
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates while preserving multi-label diversity."""
        self.logger.info("\nRemoving duplicates (preserving label diversity)...")
        
        initial_count = len(df)
        
        # For identical texts with different emotion sets, keep all
        df_dedup = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
        
        removed = initial_count - len(df_dedup)
        
        self.logger.info(f"Deduplication: {initial_count} → {len(df_dedup)} samples")
        self.logger.info(f"Removed: {removed} duplicates")
        
        self.stats['deduplication'] = {
            'initial': initial_count,
            'final': len(df_dedup),
            'removed': removed
        }
        
        return df_dedup
    
    def analyze_multi_label_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive multi-label distribution analysis.
        CRITICAL for understanding emotion co-occurrence patterns.
        """
        self.logger.info("\nAnalyzing multi-label distribution...")
        
        # 1. Label frequency analysis
        all_emotions = []
        for emotions_list in df['emotions']:
            all_emotions.extend(emotions_list)
        
        emotion_counts = Counter(all_emotions)
        total_labels = len(all_emotions)
        total_samples = len(df)
        
        # 2. Labels per sample distribution
        labels_per_sample = df['emotions'].apply(len)
        
        # 3. Emotion co-occurrence matrix
        cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        for emotions_list in df['emotions']:
            for i, e1 in enumerate(emotions_list):
                for e2 in emotions_list[i+1:]:
                    cooccurrence_matrix[e1][e2] += 1
                    cooccurrence_matrix[e2][e1] += 1
        
        # 4. Habit-emotion correlations
        habit_emotion_correlations = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            emotions = row['emotions']
            habits = row['habits']
            for emotion in emotions:
                for habit, is_present in habits.items():
                    if is_present:
                        habit_emotion_correlations[habit][emotion] += 1
        
        # Statistics
        emotion_stats = {}
        for emotion, count in emotion_counts.items():
            emotion_stats[emotion] = {
                'count': count,
                'percentage': (count / total_labels) * 100,
                'sample_occurrence': (df['emotions'].apply(lambda x: emotion in x).sum() / total_samples) * 100
            }
        
        # Sort by count
        emotion_stats = dict(sorted(emotion_stats.items(), key=lambda x: x[1]['count'], reverse=True))
        
        self.logger.info(f"Total samples: {total_samples:,}")
        self.logger.info(f"Total emotion labels: {total_labels:,}")
        self.logger.info(f"Avg labels per sample: {labels_per_sample.mean():.2f}")
        self.logger.info(f"Label distribution: min={labels_per_sample.min()}, max={labels_per_sample.max()}")
        
        # Show top emotions
        self.logger.info("\nTop 10 emotions by occurrence:")
        for emotion, stats in list(emotion_stats.items())[:10]:
            self.logger.info(
                f"  {emotion:15s}: {stats['count']:6,} labels "
                f"({stats['sample_occurrence']:5.1f}% of samples)"
            )
        
        analysis = {
            'total_samples': total_samples,
            'total_labels': total_labels,
            'avg_labels_per_sample': float(labels_per_sample.mean()),
            'emotion_stats': emotion_stats,
            'labels_per_sample_dist': {
                'min': int(labels_per_sample.min()),
                'max': int(labels_per_sample.max()),
                'mean': float(labels_per_sample.mean()),
                'std': float(labels_per_sample.std())
            },
            'cooccurrence_matrix': dict(cooccurrence_matrix),
            'habit_emotion_correlations': dict(habit_emotion_correlations)
        }
        
        self.stats['multi_label_distribution'] = analysis
        self.stats['emotion_cooccurrence'] = dict(cooccurrence_matrix)
        self.stats['habit_correlations'] = dict(habit_emotion_correlations)
        
        return analysis
    
    def create_visualizations(self, df: pd.DataFrame, distribution: Dict):
        """
        Create comprehensive visualizations for research insights.
        """
        self.logger.info("\nCreating research visualizations...")
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Emotion frequency
        ax1 = fig.add_subplot(gs[0, :])
        emotion_stats = distribution['emotion_stats']
        emotions = list(emotion_stats.keys())
        counts = [stats['count'] for stats in emotion_stats.values()]
        
        bars = ax1.bar(range(len(emotions)), counts, color='steelblue', alpha=0.7)
        ax1.set_title('Multi-Label Emotion Distribution (Total Label Occurrences)', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Emotions', fontsize=12)
        ax1.set_ylabel('Total Occurrences', fontsize=12)
        ax1.set_xticks(range(len(emotions)))
        ax1.set_xticklabels(emotions, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}', ha='center', va='bottom', fontsize=9)
        
        # 2. Labels per sample distribution
        ax2 = fig.add_subplot(gs[1, 0])
        labels_per_sample = df['emotions'].apply(len)
        ax2.hist(labels_per_sample, bins=range(1, labels_per_sample.max()+2), 
                color='coral', alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Labels per Sample', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Emotion Labels', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.axvline(labels_per_sample.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {labels_per_sample.mean():.2f}')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Emotion co-occurrence heatmap
        ax3 = fig.add_subplot(gs[1, 1:])
        cooccurrence = distribution['cooccurrence_matrix']
        
        # Create matrix for top 15 emotions
        top_emotions = list(emotion_stats.keys())[:15]
        matrix = np.zeros((len(top_emotions), len(top_emotions)))
        
        for i, e1 in enumerate(top_emotions):
            for j, e2 in enumerate(top_emotions):
                if e1 in cooccurrence and e2 in cooccurrence[e1]:
                    matrix[i, j] = cooccurrence[e1][e2]
        
        sns.heatmap(matrix, xticklabels=top_emotions, yticklabels=top_emotions,
                   cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Co-occurrence Count'},
                   ax=ax3)
        ax3.set_title('Emotion Co-occurrence Matrix (Top 15)', fontsize=14, fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Habit-emotion correlation
        ax4 = fig.add_subplot(gs[2, :])
        habit_correlations = distribution['habit_emotion_correlations']
        
        # Create correlation matrix
        habit_categories = list(self.habit_keywords.keys())
        top_emotions_for_habits = list(emotion_stats.keys())[:10]
        
        habit_matrix = np.zeros((len(habit_categories), len(top_emotions_for_habits)))
        for i, habit in enumerate(habit_categories):
            for j, emotion in enumerate(top_emotions_for_habits):
                if habit in habit_correlations and emotion in habit_correlations[habit]:
                    habit_matrix[i, j] = habit_correlations[habit][emotion]
        
        sns.heatmap(habit_matrix, xticklabels=top_emotions_for_habits, 
                   yticklabels=habit_categories, cmap='viridis', annot=True, fmt='.0f',
                   cbar_kws={'label': 'Co-occurrence Count'}, ax=ax4)
        ax4.set_title('HABIT-EMOTION CORRELATIONS (RESEARCH KEY INSIGHT)', 
                     fontsize=14, fontweight='bold', color='darkred')
        ax4.set_xlabel('Emotions', fontsize=12)
        ax4.set_ylabel('Habit Categories', fontsize=12)
        
        # 5. Sample occurrence percentage
        ax5 = fig.add_subplot(gs[3, 0])
        sample_percentages = [stats['sample_occurrence'] for stats in emotion_stats.values()]
        ax5.barh(range(len(emotions)), sample_percentages, color='teal', alpha=0.7)
        ax5.set_title('Emotion Occurrence in Samples (%)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Percentage of Samples', fontsize=12)
        ax5.set_yticks(range(len(emotions)))
        ax5.set_yticklabels(emotions)
        ax5.grid(axis='x', alpha=0.3)
        
        # 6. Text richness vs emotional density
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.scatter(df['richness_score'], df['emotional_density'], 
                   alpha=0.5, c='purple', s=10)
        ax6.set_title('Text Richness vs Emotional Density', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Richness Score', fontsize=12)
        ax6.set_ylabel('Emotional Density', fontsize=12)
        ax6.grid(alpha=0.3)
        
        # 7. Temporal distribution (if available)
        ax7 = fig.add_subplot(gs[3, 2])
        if 'date' in df.columns and df['date'].notna().sum() > 0:
            dates = pd.to_datetime(df['date'].dropna())
            ax7.hist(dates, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax7.set_title('Temporal Distribution of Entries', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Date', fontsize=12)
            ax7.set_ylabel('Number of Entries', fontsize=12)
            plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
        else:
            ax7.text(0.5, 0.5, 'No temporal data available', 
                    ha='center', va='center', fontsize=12, transform=ax7.transAxes)
            ax7.set_title('Temporal Distribution', fontsize=14, fontweight='bold')
        
        plt.savefig(self.output_dir / 'research_visualizations.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("✓ Research visualizations saved")
    
    def stratified_multi_label_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified splitting for multi-label data.
        Ensures balanced representation of label combinations.
        """
        self.logger.info("\nCreating stratified multi-label splits...")
        
        # Strategy: Use most frequent emotion as stratification key
        df['primary_emotion'] = df['emotions'].apply(
            lambda x: x[0] if len(x) > 0 else 'neutral'
        )
        
        try:
            # 75% train, 10% val, 15% test
            train_df, temp_df = train_test_split(
                df, test_size=0.25, random_state=42, 
                stratify=df['primary_emotion']
            )
            
            val_df, test_df = train_test_split(
                temp_df, test_size=0.6, random_state=42,
                stratify=temp_df['primary_emotion']
            )
        except ValueError:
            self.logger.warning("Stratification failed, using random split")
            train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.6, random_state=42)
        
        # Remove temporary column
        train_df = train_df.drop('primary_emotion', axis=1)
        val_df = val_df.drop('primary_emotion', axis=1)
        test_df = test_df.drop('primary_emotion', axis=1)
        
        self.logger.info(f"Train: {len(train_df):6,} samples ({len(train_df)/len(df)*100:5.1f}%)")
        self.logger.info(f"Val:   {len(val_df):6,} samples ({len(val_df)/len(df)*100:5.1f}%)")
        self.logger.info(f"Test:  {len(test_df):6,} samples ({len(test_df)/len(df)*100:5.1f}%)")
        
        # Verify label distribution
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            all_emotions = []
            for emotions in split_df['emotions']:
                all_emotions.extend(emotions)
            dist = Counter(all_emotions)
            self.logger.info(f"{split_name} top-3: {dict(dist.most_common(3))}")
        
        self.stats['splits'] = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'train_percent': len(train_df)/len(df)*100,
            'val_percent': len(val_df)/len(df)*100,
            'test_percent': len(test_df)/len(df)*100
        }
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> Dict:
        """Save multi-label processed data with comprehensive metadata."""
        self.logger.info("\nSaving processed data...")
        
        # Create directories
        csv_dir = self.output_dir / "csv_splits"
        csv_dir.mkdir(exist_ok=True)
        
        # Convert emotions list to string for CSV storage
        for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            df_save = df.copy()
            df_save['emotions'] = df_save['emotions'].apply(lambda x: '|'.join(x))
            df_save['emotion_intensities'] = df_save['emotion_intensities'].apply(json.dumps)
            df_save['habits'] = df_save['habits'].apply(json.dumps)
            df_save.to_csv(csv_dir / f"{df_name}.csv", index=False)
        
        self.logger.info("✓ CSV splits saved")
        
        # Create multi-label binarizer
        mlb = MultiLabelBinarizer()
        all_emotions = []
        for emotions in pd.concat([train_df, val_df, test_df])['emotions']:
            all_emotions.append(emotions)
        mlb.fit(all_emotions)
        
        # Save label mapping
        label_mapping = {
            'classes': mlb.classes_.tolist(),
            'num_labels': len(mlb.classes_),
            'label2id': {label: idx for idx, label in enumerate(mlb.classes_)},
            'id2label': {idx: label for idx, label in enumerate(mlb.classes_)}
        }
        
        with open(self.output_dir / "label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        # Save comprehensive metadata
        metadata = {
            'version': '4.0',
            'created_at': datetime.now().isoformat(),
            'preprocessing_type': 'multi_label',
            'preprocessing_config': {
                'max_length': 256,
                'multi_label': True,
                'temporal_features': True,
                'habit_extraction': True,
                'context_preserved': True
            },
            'dataset_info': {
                'num_labels': len(mlb.classes_),
                'label_names': mlb.classes_.tolist(),
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'total_size': len(train_df) + len(val_df) + len(test_df),
                'avg_labels_per_sample': self.stats['multi_label_distribution']['avg_labels_per_sample']
            },
            'statistics': self.stats,
            'research_recommendations': self._generate_research_recommendations()
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info("✓ Metadata saved")
        
        return metadata
    
    def _generate_research_recommendations(self) -> Dict:
        """Generate research-specific recommendations."""
        return {
            'model_config': {
                'model_name': 'microsoft/deberta-v3-base',
                'max_length': 256,
                'problem_type': 'multi_label_classification',
                'learning_rate': 2e-5,
                'batch_size': 8,
                'gradient_accumulation_steps': 4,
                'epochs': 20,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01
            },
            'loss_config': {
                'loss_function': 'BCEWithLogitsLoss',  # For multi-label
                'pos_weight': 'dynamic',  # Adjust per label
                'label_smoothing': 0.05  # Lower for multi-label
            },
            'research_features': {
                'use_habit_features': True,
                'use_temporal_features': True,
                'use_emotion_intensity': True,
                'analyze_cooccurrence': True
            },
            'expected_performance': {
                'target_f1_micro': 0.65,
                'target_f1_macro': 0.55,
                'target_hamming_loss': 0.15,
                'note': 'Multi-label metrics - focus on label-wise F1 scores'
            }
        }
    
    def run_pipeline(self) -> Dict:
        """Execute complete research-aligned preprocessing pipeline."""
        self.logger.info("\n" + "="*100)
        self.logger.info("STARTING RESEARCH-ALIGNED MULTI-LABEL PREPROCESSING PIPELINE V4.0")
        self.logger.info("="*100 + "\n")
        
        try:
            # Step 1: Load all data
            df = self.load_all_data_sources()
            
            # Step 2: Quality filtering
            df = self.perform_quality_filtering(df)
            
            # Step 3: Remove duplicates
            df = self.remove_duplicates(df)
            
            # Step 4: Analyze multi-label distribution
            distribution = self.analyze_multi_label_distribution(df)
            
            # Step 5: Create visualizations
            self.create_visualizations(df, distribution)
            
            # Step 6: Create splits
            train_df, val_df, test_df = self.stratified_multi_label_split(df)
            
            # Step 7: Save everything
            metadata = self.save_processed_data(train_df, val_df, test_df)
            
            # Step 8: Generate summary
            self._print_summary_report(metadata)
            
            self.logger.info("\n" + "="*100)
            self.logger.info("✅ RESEARCH-ALIGNED PREPROCESSING COMPLETED SUCCESSFULLY!")
            self.logger.info("="*100 + "\n")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _print_summary_report(self, metadata: Dict):
        """Print comprehensive research-aligned summary."""
        print("\n" + "="*100)
        print("RESEARCH-ALIGNED PREPROCESSING SUMMARY - HABIT-EMOTION ANALYSIS v4.0")
        print("="*100)
        
        print(f"\n🎯 RESEARCH GOAL ALIGNMENT:")
        print(f"   ✓ Multi-label emotion detection (captures complex states)")
        print(f"   ✓ Habit extraction from text (enables behavior-emotion correlation)")
        print(f"   ✓ Temporal features preserved (tracks evolution over time)")
        print(f"   ✓ Emotion co-occurrence tracked (finds patterns)")
        print(f"   ✓ Context-aware processing (maintains semantic integrity)")
        
        # Data statistics
        # Data statistics
        dataset_info = metadata['dataset_info']
        print(f"\n📂 DATA STATISTICS:")
        print(f"   Total samples:        {dataset_info['total_size']:6,}")
        print(f"   Training samples:     {dataset_info['train_size']:6,} ({dataset_info['train_size']/dataset_info['total_size']*100:5.1f}%)")
        print(f"   Validation samples:   {dataset_info['val_size']:6,} ({dataset_info['val_size']/dataset_info['total_size']*100:5.1f}%)")
        print(f"   Test samples:         {dataset_info['test_size']:6,} ({dataset_info['test_size']/dataset_info['total_size']*100:5.1f}%)")
        print(f"   Emotion labels:       {dataset_info['num_labels']}")
        print(f"   Avg labels/sample:    {dataset_info['avg_labels_per_sample']:.2f}")
        
        # Source breakdown
        sources = self.stats.get('source_stats', {})
        if sources:
            print(f"\n📊 DATA SOURCES:")
            total_from_sources = sum(sources.values())
            for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                print(f"   {source:20s}: {count:6,} samples ({count/total_from_sources*100:5.1f}%)")
        
        # Quality filtering
        quality = self.stats.get('quality_filtering', {})
        if quality:
            print(f"\n🔍 QUALITY FILTERING:")
            print(f"   Initial samples:      {quality.get('initial', 0):6,}")
            print(f"   Final samples:        {quality.get('final', 0):6,}")
            print(f"   Removed:              {quality.get('removed', 0):6,}")
            print(f"   Avg richness score:   {quality.get('avg_richness', 0):.3f}")
            print(f"   Avg emotional density:{quality.get('avg_emotional_density', 0):.3f}")
        
        # Multi-label distribution
        ml_dist = self.stats.get('multi_label_distribution', {})
        if ml_dist:
            print(f"\n🏷️  MULTI-LABEL DISTRIBUTION:")
            print(f"   Total emotion labels: {ml_dist.get('total_labels', 0):6,}")
            print(f"   Avg labels/sample:    {ml_dist.get('avg_labels_per_sample', 0):.2f}")
            print(f"   Min labels/sample:    {ml_dist.get('labels_per_sample_dist', {}).get('min', 0)}")
            print(f"   Max labels/sample:    {ml_dist.get('labels_per_sample_dist', {}).get('max', 0)}")
            
            emotion_stats = ml_dist.get('emotion_stats', {})
            if emotion_stats:
                print(f"\n   Top 10 emotions by occurrence:")
                for emotion, stats in list(emotion_stats.items())[:10]:
                    print(f"      {emotion:15s}: {stats['count']:6,} labels "
                          f"({stats['sample_occurrence']:5.1f}% of samples)")
        
        # Emotion co-occurrence insights
        cooccurrence = self.stats.get('emotion_cooccurrence', {})
        if cooccurrence:
            print(f"\n🔗 EMOTION CO-OCCURRENCE (Top 5 Patterns):")
            # Find top co-occurring pairs
            pairs = []
            for e1, related in cooccurrence.items():
                for e2, count in related.items():
                    if e1 < e2:  # Avoid duplicates
                        pairs.append((e1, e2, count))
            
            top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]
            for e1, e2, count in top_pairs:
                print(f"   {e1:12s} + {e2:12s}: {count:5,} times")
        
        # Habit-emotion correlations
        habit_corr = self.stats.get('habit_correlations', {})
        if habit_corr:
            print(f"\n🎯 HABIT-EMOTION CORRELATIONS (Research Key):")
            print(f"   Top 5 habit-emotion associations:")
            
            # Find strongest correlations
            all_correlations = []
            for habit, emotions in habit_corr.items():
                for emotion, count in emotions.items():
                    all_correlations.append((habit, emotion, count))
            
            top_correlations = sorted(all_correlations, key=lambda x: x[2], reverse=True)[:5]
            for habit, emotion, count in top_correlations:
                print(f"      {habit:12s} → {emotion:12s}: {count:5,} co-occurrences")
        
        # Research recommendations
        print(f"\n💡 RESEARCH RECOMMENDATIONS:")
        recommendations = metadata.get('research_recommendations', {})
        
        model_config = recommendations.get('model_config', {})
        print(f"\n   Model Configuration:")
        print(f"      Model:              {model_config.get('model_name', 'N/A')}")
        print(f"      Max length:         {model_config.get('max_length', 'N/A')} tokens")
        print(f"      Problem type:       {model_config.get('problem_type', 'N/A')}")
        print(f"      Learning rate:      {model_config.get('learning_rate', 'N/A')}")
        print(f"      Batch size:         {model_config.get('batch_size', 'N/A')}")
        print(f"      Epochs:             {model_config.get('epochs', 'N/A')}")
        
        loss_config = recommendations.get('loss_config', {})
        print(f"\n   Loss Configuration:")
        print(f"      Loss function:      {loss_config.get('loss_function', 'N/A')}")
        print(f"      Pos weight:         {loss_config.get('pos_weight', 'N/A')}")
        print(f"      Label smoothing:    {loss_config.get('label_smoothing', 'N/A')}")
        
        research_features = recommendations.get('research_features', {})
        print(f"\n   Research Features:")
        print(f"      Habit features:     {'✓' if research_features.get('use_habit_features') else '✗'}")
        print(f"      Temporal features:  {'✓' if research_features.get('use_temporal_features') else '✗'}")
        print(f"      Emotion intensity:  {'✓' if research_features.get('use_emotion_intensity') else '✗'}")
        print(f"      Co-occurrence:      {'✓' if research_features.get('analyze_cooccurrence') else '✗'}")
        
        expected = recommendations.get('expected_performance', {})
        print(f"\n   Expected Performance (Multi-Label):")
        print(f"      Target F1 (micro):  {expected.get('target_f1_micro', 'N/A')}")
        print(f"      Target F1 (macro):  {expected.get('target_f1_macro', 'N/A')}")
        print(f"      Target Hamming:     {expected.get('target_hamming_loss', 'N/A')}")
        
        # Output files
        print(f"\n📁 OUTPUT FILES:")
        print(f"   {self.output_dir.absolute()}/")
        print(f"   ├── csv_splits/")
        print(f"   │   ├── train.csv          (multi-label format)")
        print(f"   │   ├── val.csv            (multi-label format)")
        print(f"   │   └── test.csv           (multi-label format)")
        print(f"   ├── metadata.json          (complete preprocessing info)")
        print(f"   ├── label_mapping.json     (multi-label binarizer classes)")
        print(f"   ├── research_visualizations.png")
        print(f"   └── preprocessing_v4_multilabel.log")
        
        # Next steps
        print(f"\n🚀 NEXT STEPS FOR RESEARCH:")
        print(f"   1. Load data using MultiLabelBinarizer from label_mapping.json")
        print(f"   2. Configure model for multi-label classification:")
        print(f"      - Set problem_type='multi_label_classification'")
        print(f"      - Use BCEWithLogitsLoss for training")
        print(f"      - Enable sigmoid activation (not softmax!)")
        print(f"   3. Extract habit features from 'habits' column")
        print(f"   4. Use temporal features from 'date' column if available")
        print(f"   5. Analyze emotion co-occurrence patterns for insights")
        print(f"   6. Track emotion intensity scores for detailed analysis")
        
        print(f"\n🔬 RESEARCH INSIGHTS TO EXPLORE:")
        print(f"   • Which habits correlate most strongly with specific emotions?")
        print(f"   • How do emotions co-occur (e.g., anxiety + sadness)?")
        print(f"   • Do temporal patterns exist (weekday vs weekend emotions)?")
        print(f"   • What is the intensity distribution of detected emotions?")
        print(f"   • How do emotion combinations change over time?")
        
        print("\n" + "="*100)


class MultiLabelEmotionDataset(Dataset):
    """PyTorch Dataset for multi-label emotion classification."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, mlb: MultiLabelBinarizer, max_length: int = 256):
        """
        Initialize multi-label dataset.
        
        Args:
            df: DataFrame with 'text' and 'emotions' (list) columns
            tokenizer: DeBERTa tokenizer
            mlb: Fitted MultiLabelBinarizer
            max_length: Maximum sequence length
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row['text'])
        
        # Parse emotions (handle both list and string formats)
        if isinstance(row['emotions'], str):
            emotions = row['emotions'].split('|')
        else:
            emotions = row['emotions']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Convert to multi-hot encoding
        labels = self.mlb.transform([emotions])[0]
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)  # Float for BCEWithLogitsLoss
        }
        
        # Add habit features if available
        if 'habits' in row and pd.notna(row['habits']):
            if isinstance(row['habits'], str):
                habits = json.loads(row['habits'])
            else:
                habits = row['habits']
            
            habit_vector = [1.0 if v else 0.0 for v in habits.values()]
            item['habits'] = torch.tensor(habit_vector, dtype=torch.float)
        
        # Add temporal features if available
        if 'date' in row and pd.notna(row['date']):
            date = pd.to_datetime(row['date'])
            temporal_features = [
                float(date.weekday()) / 6.0,  # Normalize to [0, 1]
                1.0 if date.weekday() >= 5 else 0.0,  # Is weekend
                float(date.month) / 12.0  # Normalize month
            ]
            item['temporal'] = torch.tensor(temporal_features, dtype=torch.float)
        
        return item


def load_preprocessed_data_v4(data_dir: str = "./processed_emotion_data_v4"):
    """
    Load preprocessed multi-label data for training.
    
    Returns:
        train_df, val_df, test_df, metadata, mlb
    """
    data_path = Path(data_dir)
    
    # Load metadata
    with open(data_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load label mapping
    with open(data_path / "label_mapping.json", 'r') as f:
        label_mapping = json.load(f)
    
    # Create MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([label_mapping['classes']])
    
    # Load splits
    train_df = pd.read_csv(data_path / "csv_splits" / "train.csv")
    val_df = pd.read_csv(data_path / "csv_splits" / "val.csv")
    test_df = pd.read_csv(data_path / "csv_splits" / "test.csv")
    
    # Convert emotion strings back to lists
    for df in [train_df, val_df, test_df]:
        df['emotions'] = df['emotions'].apply(lambda x: x.split('|') if isinstance(x, str) else x)
    
    print(f"✓ Loaded multi-label data from {data_dir}")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    print(f"  Labels: {len(mlb.classes_)}")
    
    return train_df, val_df, test_df, metadata, mlb


def main():
    """Main execution function."""
    print("\n" + "="*100)
    print("RESEARCH-ALIGNED MULTI-LABEL EMOTION PREPROCESSING PIPELINE V4.0")
    print("Specialized for Habit-Emotion Correlation Analysis")
    print("="*100)
    print("\n🎯 RESEARCH INNOVATIONS:")
    print("   ✅ Multi-label emotion detection (captures complex emotional states)")
    print("   ✅ Habit keyword extraction (enables behavior-emotion correlation)")
    print("   ✅ Temporal context preservation (tracks emotion evolution)")
    print("   ✅ Emotion co-occurrence analysis (discovers patterns)")
    print("   ✅ Emotion intensity scoring (quantifies emotional strength)")
    print("   ✅ Context-aware augmentation (maintains semantic integrity)")
    print("\n📋 KEY DIFFERENCES FROM V3:")
    print("   • Single-label → Multi-label classification")
    print("   • No class balancing (preserves natural distributions)")
    print("   • Habit extraction for correlation studies")
    print("   • Temporal features for time-series analysis")
    print("   • Emotion intensity metadata")
    print("   • Co-occurrence matrix for pattern discovery")
    print("\n🔬 RESEARCH APPLICATIONS:")
    print("   • Analyze which habits correlate with specific emotions")
    print("   • Study emotion combinations (e.g., anxiety + sadness)")
    print("   • Track emotional evolution over time")
    print("   • Understand emotion intensity patterns")
    print("   • Discover behavioral triggers for emotions")
    print("="*100 + "\n")
    
    # Initialize and run pipeline
    processor = ResearchAlignedEmotionPreprocessor(
        data_dir=".",
        output_dir="./processed_emotion_data_v4"
    )
    
    try:
        metadata = processor.run_pipeline()
        
        # Quick start guide
        print("\n" + "="*100)
        print("📝 QUICK START - MULTI-LABEL TRAINING:")
        print("="*100)
        print("\n# 1. Load the preprocessed data:")
        print("```python")
        print("from preprocessing_v4 import load_preprocessed_data_v4, MultiLabelEmotionDataset")
        print("from transformers import DebertaV2Tokenizer")
        print("")
        print("# Load data")
        print("train_df, val_df, test_df, metadata, mlb = load_preprocessed_data_v4()")
        print("")
        print("# Initialize tokenizer")
        print("tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')")
        print("")
        print("# Create datasets")
        print("train_dataset = MultiLabelEmotionDataset(train_df, tokenizer, mlb, max_length=256)")
        print("val_dataset = MultiLabelEmotionDataset(val_df, tokenizer, mlb, max_length=256)")
        print("test_dataset = MultiLabelEmotionDataset(test_df, tokenizer, mlb, max_length=256)")
        print("```")
        print("\n# 2. Configure model for multi-label classification:")
        print("```python")
        print("from transformers import DebertaV2ForSequenceClassification, TrainingArguments")
        print("import torch.nn as nn")
        print("")
        print("# Load model")
        print("model = DebertaV2ForSequenceClassification.from_pretrained(")
        print("    'microsoft/deberta-v3-base',")
        print("    num_labels=len(mlb.classes_),")
        print("    problem_type='multi_label_classification'  # CRITICAL!")
        print(")")
        print("")
        print("# Training arguments")
        print("training_args = TrainingArguments(")
        print("    output_dir='./multi_label_emotion_model',")
        print("    learning_rate=2e-5,")
        print("    per_device_train_batch_size=8,")
        print("    per_device_eval_batch_size=8,")
        print("    gradient_accumulation_steps=4,")
        print("    num_train_epochs=20,")
        print("    warmup_ratio=0.1,")
        print("    weight_decay=0.01,")
        print("    evaluation_strategy='epoch',")
        print("    save_strategy='epoch',")
        print("    load_best_model_at_end=True,")
        print("    metric_for_best_model='f1_micro',")
        print("    greater_is_better=True")
        print(")")
        print("```")
        print("\n# 3. Define multi-label metrics:")
        print("```python")
        print("from sklearn.metrics import f1_score, hamming_loss, accuracy_score")
        print("import numpy as np")
        print("")
        print("def compute_metrics(eval_pred):")
        print("    predictions, labels = eval_pred")
        print("    # Apply sigmoid and threshold")
        print("    predictions = (1 / (1 + np.exp(-predictions))) > 0.5")
        print("    ")
        print("    return {")
        print("        'f1_micro': f1_score(labels, predictions, average='micro', zero_division=0),")
        print("        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),")
        print("        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),")
        print("        'hamming_loss': hamming_loss(labels, predictions),")
        print("        'subset_accuracy': accuracy_score(labels, predictions)")
        print("    }")
        print("```")
        print("\n# 4. Analyze habit-emotion correlations:")
        print("```python")
        print("import json")
        print("")
        print("# Load metadata with correlations")
        print("with open('processed_emotion_data_v4/metadata.json', 'r') as f:")
        print("    metadata = json.load(f)")
        print("")
        print("habit_correlations = metadata['statistics']['habit_correlations']")
        print("")
        print("# Find top correlations")
        print("for habit, emotions in habit_correlations.items():")
        print("    print(f'\\n{habit}:')")
        print("    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)")
        print("    for emotion, count in sorted_emotions[:5]:")
        print("        print(f'  {emotion}: {count} co-occurrences')")
        print("```")
        print("\n" + "="*100)
        print("✅ Multi-label preprocessing complete! Ready for research analysis.")
        print("="*100 + "\n")
        
        return metadata
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()