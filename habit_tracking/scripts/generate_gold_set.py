"""
Generate Gold Standard Dataset for Habit NER Evaluation
Creates 300-500 manually-labeled spans with realistic journal language
Balanced across categories with edge cases and negation patterns
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)


class GoldSetGenerator:
    """Generate realistic gold-labeled habit spans for NER evaluation"""
    
    def __init__(self, seed_ontology_path: str, num_spans: int = 400):
        """
        Initialize generator
        
        Args:
            seed_ontology_path: Path to seed_ontology.json
            num_spans: Target number of gold spans (300-500)
        """
        self.num_spans = num_spans
        
        # Load seed ontology
        print(f"Loading seed ontology: {seed_ontology_path}")
        with open(seed_ontology_path, 'r', encoding='utf-8') as f:
            self.seed_habits = json.load(f)
        
        # Build category mappings
        self.categories = sorted(set(h['category'].upper() for h in self.seed_habits))
        self.category_habits = defaultdict(list)
        for habit in self.seed_habits:
            self.category_habits[habit['category'].upper()].append(habit)
        
        print(f"✓ Loaded {len(self.seed_habits)} habits, {len(self.categories)} categories")
        
        # Build templates
        self._build_templates()
    
    def _build_templates(self):
        """Build templates for natural journal language"""
        
        # Positive habit templates
        self.positive_templates = [
            "{habit} today. Felt really good!",
            "Finally {habit} after putting it off.",
            "{habit} in the morning. Great start.",
            "Managed to {habit} even though I was tired.",
            "{habit} and it felt amazing.",
            "Actually {habit} like I planned.",
        ]
        
        # Negative/problematic templates
        self.negative_templates = [
            "{habit} again. Need to stop this.",
            "Can't believe I {habit} for the third time this week.",
            "{habit} when I should have been working.",
            "Wasted time {habit}. So unproductive.",
            "{habit} all night. Regret it now.",
            "Another day of {habit}. Getting worried.",
        ]
        
        # Neutral templates
        self.neutral_templates = [
            "{habit} today.",
            "Spent some time {habit}.",
            "{habit} like usual.",
            "Did my regular {habit}.",
            "{habit} for about an hour.",
        ]
        
        # Duration templates
        self.duration_templates = [
            "{habit} for {duration}.",
            "Spent {duration} {habit}.",
            "{duration} of {habit} today.",
        ]
        
        # Time templates
        self.time_templates = [
            "{habit} at {time}.",
            "Woke up and {habit} around {time}.",
            "Late night {habit} at {time}.",
        ]
        
        # Negation templates
        self.negation_templates = [
            "Didn't {habit} today. Feel bad about it.",
            "Skipped {habit} again.",
            "Wanted to {habit} but couldn't find time.",
            "Failed to {habit} like I planned.",
            "Avoided {habit} all day.",
            "Couldn't bring myself to {habit}.",
            "No {habit} today. Too exhausted.",
            "Missed my usual {habit}.",
        ]
        
        # Resources
        self.durations = ["30 minutes", "an hour", "2 hours", "3 hours", "45 minutes"]
        self.times = ["2am", "3am", "6am", "8am", "11pm", "midnight"]
    
    def generate_span_from_habit(
        self, 
        habit: Dict, 
        template_type: str = "neutral"
    ) -> Tuple[str, str, str, str]:
        """
        Generate journal text with span
        
        Returns:
            (journal_text, span_text, category, habit_id)
        """
        # Choose alias
        alias = random.choice(habit.get('aliases', [habit['name'].lower()]))
        category = habit['category'].upper()
        habit_id = habit['id']
        
        # Select template
        if template_type == "positive":
            template = random.choice(self.positive_templates)
        elif template_type == "negative":
            template = random.choice(self.negative_templates)
        elif template_type == "duration":
            template = random.choice(self.duration_templates)
            duration = random.choice(self.durations)
            text = template.format(habit=alias, duration=duration)
            return text, alias, category, habit_id
        elif template_type == "time":
            template = random.choice(self.time_templates)
            time = random.choice(self.times)
            text = template.format(habit=alias, time=time)
            return text, alias, category, habit_id
        elif template_type == "negation":
            template = random.choice(self.negation_templates)
        else:  # neutral
            template = random.choice(self.neutral_templates)
        
        text = template.format(habit=alias)
        return text, alias, category, habit_id
    
    def generate_edge_cases(self) -> List[Tuple[str, str, str, str]]:
        """Generate edge case spans (ambiguous/tricky)"""
        
        return [
            # Ambiguous spans
            ("feeling really anxious lately", "anxious", "MENTAL_STATE", "anxiety"),
            ("slept through my alarm", "slept through", "SLEEP", "sleep_late"),
            ("kept procrastinating on assignment", "procrastinating", "PRODUCTIVITY", "procrastinate"),
            ("ate junk food again", "ate junk food", "NUTRITION", "junk_food"),
            ("couldn't sleep at all", "sleep", "SLEEP", "insomnia"),
            
            # Multi-word habits
            ("went for morning run", "morning run", "FITNESS", "running"),
            ("did yoga practice", "yoga practice", "FITNESS", "yoga"),
            ("worked on side project", "side project", "PROFESSIONAL", "side_project"),
            ("called my parents", "called my parents", "SOCIAL", "family_time"),
            
            # Negations (important for NER)
            ("didn't study at all", "study", "ACADEMICS", "study"),
            ("skipped breakfast again", "skipped breakfast", "NUTRITION", "skip_meal"),
            ("avoided work all day", "avoided work", "PROFESSIONAL", "procrastinate"),
            
            # Context/modifiers
            ("late night gaming session", "gaming", "ENTERTAINMENT", "gaming"),
            ("excessive coffee drinking", "coffee", "NUTRITION", "caffeine"),
            ("mindless scrolling", "scrolling", "DIGITAL", "social_media"),
            
            # Partial mentions
            ("gym time", "gym", "FITNESS", "gym"),
            ("work stuff", "work", "PROFESSIONAL", "work"),
            ("sleep schedule", "sleep", "SLEEP", "sleep"),
            
            # Colloquial
            ("pulled all-nighter", "all-nighter", "SLEEP", "all_nighter"),
            ("doom scrolling", "scrolling", "DIGITAL", "social_media"),
        ]
    
    def generate_balanced_gold_set(self) -> pd.DataFrame:
        """Generate balanced gold set across all categories"""
        
        gold_spans = []
        span_id = 1
        
        # Calculate distribution
        base_per_category = max(8, (self.num_spans - 50) // len(self.categories))
        edge_case_count = 50
        
        print(f"\nGenerating gold set:")
        print(f"  Target: {self.num_spans} spans")
        print(f"  Base per category: {base_per_category}")
        print(f"  Edge cases: {edge_case_count}")
        
        # Generate base spans (balanced across categories)
        for category in self.categories:
            habits = self.category_habits[category]
            
            if not habits:
                continue
            
            # Distribution across template types
            templates = ["neutral", "positive", "negative", "duration", "time", "negation"]
            spans_per_type = base_per_category // len(templates)
            
            for template_type in templates:
                for _ in range(spans_per_type):
                    habit = random.choice(habits)
                    
                    try:
                        text, span, cat, habit_id = self.generate_span_from_habit(
                            habit, template_type
                        )
                        
                        # Find span position
                        text_lower = text.lower()
                        span_lower = span.lower()
                        start_char = text_lower.find(span_lower)
                        
                        if start_char == -1:
                            # Fallback
                            text = span
                            start_char = 0
                        
                        end_char = start_char + len(span)
                        
                        gold_spans.append({
                            'span_id': f'gold_{span_id:04d}',
                            'journal_id': f'journal_{span_id}',
                            'text': text,
                            'span': span,
                            'gold_label': category,
                            'start_char': start_char,
                            'end_char': end_char,
                            'span_type': template_type,
                            'habit_id': habit_id
                        })
                        
                        span_id += 1
                    
                    except Exception as e:
                        print(f"⚠ Failed for {habit['id']}: {e}")
                        continue
        
        # Add edge cases
        print(f"\nAdding {edge_case_count} edge cases...")
        edge_cases = self.generate_edge_cases()
        
        for text, span, category, habit_id in edge_cases * (edge_case_count // len(edge_cases) + 1):
            if len(gold_spans) >= self.num_spans:
                break
            
            text_lower = text.lower()
            span_lower = span.lower()
            start_char = text_lower.find(span_lower)
            
            if start_char == -1:
                text = span
                start_char = 0
            
            end_char = start_char + len(span)
            
            gold_spans.append({
                'span_id': f'gold_{span_id:04d}',
                'journal_id': f'journal_{span_id}',
                'text': text,
                'span': span,
                'gold_label': category,
                'start_char': start_char,
                'end_char': end_char,
                'span_type': 'edge_case',
                'habit_id': habit_id
            })
            
            span_id += 1
        
        # Convert to DataFrame
        df_gold = pd.DataFrame(gold_spans)
        
        # Shuffle
        df_gold = df_gold.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Limit to target
        df_gold = df_gold.head(self.num_spans)
        
        print(f"\n✓ Generated {len(df_gold)} gold spans")
        
        return df_gold
    
    def split_dev_test(
        self, 
        df: pd.DataFrame, 
        dev_size: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into dev (100) and test (rest)"""
        
        # Stratified split by category
        dev_spans = []
        test_spans = []
        
        dev_per_category = dev_size // len(self.categories)
        
        for category in self.categories:
            category_spans = df[df['gold_label'] == category]
            
            if len(category_spans) >= dev_per_category:
                dev_cat = category_spans.head(dev_per_category)
                test_cat = category_spans.iloc[dev_per_category:]
            else:
                dev_cat = category_spans
                test_cat = pd.DataFrame()
            
            dev_spans.append(dev_cat)
            if len(test_cat) > 0:
                test_spans.append(test_cat)
        
        df_dev = pd.concat(dev_spans, ignore_index=True)
        df_test = pd.concat(test_spans, ignore_index=True) if test_spans else pd.DataFrame()
        
        # Shuffle
        df_dev = df_dev.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✓ Split: dev={len(df_dev)}, test={len(df_test)}")
        
        return df_dev, df_test
    
    def print_statistics(self, df: pd.DataFrame, name: str = "Gold Set"):
        """Print statistics"""
        
        print(f"\n{'='*60}")
        print(f"{name.upper()} STATISTICS")
        print(f"{'='*60}")
        
        print(f"\nTotal spans: {len(df)}")
        
        print(f"\nCategory distribution:")
        category_dist = df['gold_label'].value_counts().sort_index()
        for category, count in category_dist.items():
            pct = count/len(df)*100
            print(f"  {category:20s}: {count:3d} ({pct:5.1f}%)")
        
        print(f"\nSpan types:")
        type_dist = df['span_type'].value_counts()
        for span_type, count in type_dist.items():
            pct = count/len(df)*100
            print(f"  {span_type:15s}: {count:3d} ({pct:5.1f}%)")
        
        # Sample spans
        print(f"\n{'='*60}")
        print(f"SAMPLE SPANS (3 per category)")
        print(f"{'='*60}")
        
        for category in sorted(df['gold_label'].unique()):
            samples = df[df['gold_label'] == category].head(3)
            if len(samples) > 0:
                print(f"\n{category}:")
                for _, row in samples.iterrows():
                    print(f"  • \"{row['text'][:60]}\" -> [{row['span']}]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate gold standard dataset for habit NER"
    )
    parser.add_argument(
        '--seed-ontology', type=str, default='seeds/seed_ontology.json',
        help="Path to seed ontology"
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/gold',
        help="Output directory for gold files"
    )
    parser.add_argument(
        '--num-spans', type=int, default=400,
        help="Target number of spans (300-500)"
    )
    parser.add_argument(
        '--dev-size', type=int, default=100,
        help="Number of spans for dev set"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print("GOLD SET GENERATION")
    print(f"{'='*60}")
    print(f"Seed ontology: {args.seed_ontology}")
    print(f"Target spans: {args.num_spans}")
    print(f"Dev size: {args.dev_size}")
    print(f"Output dir: {args.output_dir}")
    
    # Initialize generator
    generator = GoldSetGenerator(
        seed_ontology_path=args.seed_ontology,
        num_spans=args.num_spans
    )
    
    # Generate gold set
    df_gold = generator.generate_balanced_gold_set()
    
    # Print full statistics
    generator.print_statistics(df_gold, "Full Gold Set")
    
    # Split into dev/test
    df_dev, df_test = generator.split_dev_test(df_gold, dev_size=args.dev_size)
    
    generator.print_statistics(df_dev, "Dev Set")
    generator.print_statistics(df_test, "Test Set")
    
    # Save files
    print(f"\n{'='*60}")
    print("SAVING FILES")
    print(f"{'='*60}")
    
    # Save full gold set
    gold_path = output_dir / 'gold_spans.csv'
    df_gold.to_csv(gold_path, index=False, encoding='utf-8')
    print(f"✓ Saved full gold set: {gold_path}")
    
    # Save minimal version (for easy manual review)
    minimal_cols = ['span_id', 'journal_id', 'text', 'span', 'gold_label']
    minimal_path = output_dir / 'gold_spans_minimal.csv'
    df_gold[minimal_cols].to_csv(minimal_path, index=False, encoding='utf-8')
    print(f"✓ Saved minimal version: {minimal_path}")
    
    # Save dev set
    dev_path = output_dir / 'gold_dev.csv'
    df_dev.to_csv(dev_path, index=False, encoding='utf-8')
    print(f"✓ Saved dev set: {dev_path}")
    
    # Save test set
    test_path = output_dir / 'gold_test.csv'
    df_test.to_csv(test_path, index=False, encoding='utf-8')
    print(f"✓ Saved test set: {test_path}")
    
    # Save statistics
    stats = {
        'total_spans': len(df_gold),
        'dev_spans': len(df_dev),
        'test_spans': len(df_test),
        'num_categories': df_gold['gold_label'].nunique(),
        'categories': sorted(df_gold['gold_label'].unique().tolist()),
        'category_distribution': df_gold['gold_label'].value_counts().to_dict(),
        'span_type_distribution': df_gold['span_type'].value_counts().to_dict()
    }
    
    stats_path = output_dir / 'gold_set_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics: {stats_path}")
    
    print(f"\n{'='*60}")
    print("✅ GOLD SET GENERATION COMPLETED")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  - gold_spans.csv: {len(df_gold)} spans (full set)")
    print(f"  - gold_spans_minimal.csv: minimal columns")
    print(f"  - gold_dev.csv: {len(df_dev)} spans (for tuning)")
    print(f"  - gold_test.csv: {len(df_test)} spans (for evaluation)")
    print(f"  - gold_set_stats.json: statistics summary")
    
    print(f"\nNext steps:")
    print(f"  1. Review gold sets (especially edge cases)")
    print(f"  2. Optionally manually refine spans")
    print(f"  3. Run BIO conversion: python src/ner/to_bio.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()