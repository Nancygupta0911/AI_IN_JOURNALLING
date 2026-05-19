# 🔄 Habit Tracking Pipeline

A 5-stage NLP pipeline that reads free-text journal entries and **automatically extracts, labels, and classifies** habit mentions into 18 life-domain categories — with **zero manual annotation**.

## Pipeline Architecture

```
Raw Journal Text
      │
      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Stage 1  │ →  │ Stage 2  │ →  │ Stage 3  │ →  │ Stage 4  │ →  │   Stage 5    │
│ Span     │    │ Weak     │    │ Gold Set │    │ NER      │    │ Canonicali-  │
│ Extract  │    │ Supervis.│    │ Gen.     │    │ Training │    │ zation       │
│          │    │          │    │          │    │          │    │              │
│• Regex   │    │• 30+ LFs │    │• 400     │    │• BERT /  │    │• Semantic    │
│• spaCy   │    │• Snorkel │    │  spans   │    │  DeBERTa │    │  matching    │
│• TF-IDF  │    │• Probab. │    │• Edge    │    │• BIO     │    │• HDBSCAN     │
│• PMI     │    │  labels  │    │  cases   │    │  tags    │    │• Fuzzy match │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────────┘
      │               │               │               │               │
      ▼               ▼               ▼               ▼               ▼
  Spans +         Weak Labels     Gold Spans      Trained NER     Canonical
  Keywords        + Confidence    + Stats          Model           Mappings
```

## 18 Habit Categories

| Category | Examples |
|----------|----------|
| Sleep | slept late, insomnia, napped, pulled all-nighter |
| Fitness | gym, running, yoga, workout |
| Academics | studied, exam prep, homework, lecture |
| Digital/Screen | scrolled Instagram, doom scrolling, social media |
| Nutrition | ate junk food, skipped breakfast, cooked meal |
| Social | hung out with friends, called mom, party |
| Mental Health | felt anxious, meditation, therapy session |
| Productivity | procrastinated, finished project, time management |
| Hygiene | showered, brushed teeth, skincare routine |
| Creative | drew, wrote poetry, played guitar |
| Spiritual | prayed, gratitude journal, church |
| Substance Use | drank coffee, smoked, alcohol |
| Financial | budgeted, impulse purchase, savings |
| Entertainment | watched Netflix, read a book, gaming |
| Commute | walked to class, bus ride, drove |
| Environmental | cleaned room, organized desk, gardening |
| Household | did laundry, cooked dinner, grocery shopping |
| Other | miscellaneous activities |

## Key Technical Components

### Stage 1: Span Extraction (`src/extraction/`)

**`extract_regex.py`** — Multi-strategy extraction:
- 9 regex pattern types (seed alias, verb+duration, negation, frequency, etc.)
- 7 spaCy linguistic patterns (VERB_NOUN, FEEL_STATE, NEG_VERB, etc.)
- spaCy PhraseMatcher for direct ontology matching
- Post-processing: deduplication, quality filtering

**`keyword_mine.py`** — Unsupervised discovery:
- TF-IDF for corpus-specific terms
- PMI (Pointwise Mutual Information) for collocations
- Verb-Object pairs via dependency parsing
- Composite ranking: `0.30×freq + 0.25×TF-IDF + 0.20×PMI + 0.15×method_count + 0.10×seed_overlap`

### Stage 2: Weak Supervision (`src/supervision/`)

**`weak_supervision.py`** — Snorkel-based labeling:
- 30+ labeling functions across 7 types (alias match, keyword density, verb pattern, negation, duration, location, context)
- Snorkel `LabelModel` resolves conflicts between disagreeing LFs
- Outputs probabilistic labels with confidence scores

### Stages 3-4: NER Pipeline (`src/ner/`)

**`to_bio.py`** — BIO tag conversion:
- Character-level span offsets → subword token alignment
- Supports both gold and weak label inputs

**`train_ner.py`** — Transformer fine-tuning:
- BERT-base-cased or DeBERTa-v3-small
- Class-weighted CrossEntropyLoss for 18-class imbalance
- seqeval evaluation (entity-level F1)
- Early stopping on validation F1

### Stage 5: Canonicalization (`src/canonicalization/`)
- Semantic matching to canonical ontology IDs using sentence-transformers
- HDBSCAN clustering for novel habit discovery
- Fuzzy string matching as fallback

## Seed Ontology

The knowledge base (`seeds/seed_ontology.json`) contains:
- **80+ habits** with structured metadata
- **800+ aliases** (natural language variations)
- **18 categories** with descriptions
- Example: `running` → ["went for a run", "jogged", "morning 5k", "went jogging", "cardio"]

## Running the Pipeline

See the [main README](../README.md) for full installation and pipeline commands, or refer to the [DEMONSTRATION_GUIDE.md](DEMONSTRATION_GUIDE.md) for a comprehensive walkthrough.

```bash
cd habit_tracking/

# Quick start
python scripts/generate_test_data.py
python scripts/unify_data.py --input-dir data/raw --output data/processed/journals.parquet
python src/extraction/extract_regex.py --input data/raw/test_journals.csv --out results/spans/extracted_spans.parquet --seeds seeds/seed_ontology.json
python src/supervision/weak_supervision.py --input results/spans/extracted_spans.parquet --output results/labels/weak_labels.parquet --seed-ontology seeds/seed_ontology.json
```

## Connection to Emotion Detection

This pipeline consumes the **same journal entries** analyzed by the emotion detection model. The two systems together answer:
- **Emotion Detection:** _What does the user feel?_ (anxiety, joy, sadness, etc.)
- **Habit Tracking:** _What behaviors are mentioned?_ (slept late, scrolled social media, exercised)
- **Combined:** _How do habits correlate with emotional states?_

The emotion detection model's preprocessing already extracts habit keywords — this pipeline provides granular, span-level extraction with category classification.
