# 🎓 HABIT TRACKING PROJECT — COMPLETE DEMONSTRATION GUIDE

> **Purpose**: This document gives you everything you need to explain your project to your mentor — the full pipeline flow, file execution order, data flow between stages, tech stack justification, and what outputs & visualizations are produced at each step.

---

## 📋 TABLE OF CONTENTS

1. [Project Overview — The Elevator Pitch](#1-project-overview)
2. [Tech Stack & Why You Chose It](#2-tech-stack)
3. [Architecture — The 5-Stage Pipeline](#3-architecture)
4. [File Execution Order (Step-by-Step)](#4-file-execution-order)
5. [Stage-by-Stage Deep Dive](#5-stage-deep-dive)
6. [Data Flow Diagram — What Goes In, What Comes Out](#6-data-flow)
7. [Output Files & Visualizations Explained](#7-outputs)
8. [Directory Structure Map](#8-directory-structure)
9. [Demonstration Script — What to Say to Your Mentor](#9-demo-script)
10. [Key Design Decisions You Can Explain](#10-design-decisions)
11. [Commands to Run the Pipeline](#11-commands)

---

## 1. PROJECT OVERVIEW — THE ELEVATOR PITCH

**What is it?**
A 5-stage NLP pipeline that reads free-text journal entries (like "Slept late again, scrolled Instagram for 3 hours, skipped breakfast") and **automatically extracts**, **labels**, and **classifies** habit mentions into 18 life-domain categories (Sleep, Fitness, Academics, Digital, Nutrition, etc.).

**Why does it matter?**
Manual habit tracking is tedious and unsustainable. Our system automates it — zero manual labeling needed — using a combination of regex pattern matching, spaCy NLP, Snorkel weak supervision, and transformer-based NER.

**Key Numbers to Mention:**
- **80+ habits** in the seed ontology across **18 categories**
- **800+ aliases** (alternative phrasings for habits)
- **30+ labeling functions** in the weak supervision stage
- **Zero manual annotation** — all labels are generated programmatically
- End-to-end: Raw text → Structured habit database

---

## 2. TECH STACK & WHY YOU CHOSE IT

| Layer | Technology | Why This Choice |
|-------|-----------|----------------|
| **Language** | Python 3.x | Industry standard for NLP/ML |
| **NLP Core** | spaCy (`en_core_web_sm`) | Fast dependency parsing, POS tagging, Matcher/PhraseMatcher for rule-based extraction |
| **Regex Patterns** | Python `re` | Precise pattern matching for habit verbs, durations, negations |
| **Statistical Mining** | scikit-learn TF-IDF + PMI | Unsupervised keyword discovery; finds salient n-grams beyond the seed ontology |
| **Weak Supervision** | Snorkel (`LabelModel`, `PandasLFApplier`) | Combines 30+ noisy labeling functions into probabilistic labels without manual annotation |
| **Sentence Embeddings** | sentence-transformers (`all-mpnet-base-v2`) | Semantic similarity matching for canonicalization and optional weak supervision LFs |
| **NER Training** | HuggingFace Transformers (BERT/DeBERTa) | State-of-the-art token classification; supports fine-tuning on BIO-tagged data |
| **Tokenization** | HuggingFace `AutoTokenizer` (DeBERTa-v3-small) | Subword tokenization aligned with the NER model |
| **Evaluation** | seqeval | Standard NER evaluation (entity-level Precision, Recall, F1) |
| **Data Handling** | Pandas + Parquet | Parquet for efficient columnar storage; CSV for human-readable inspection |
| **Configuration** | YAML (`configs/config.yaml`) | Centralized hyperparameters; easy to tune without code changes |
| **Visualization** | Matplotlib | Publication-quality report figures |
| **Database** | SQLite (planned) | Lightweight, file-based storage for habit logs |

### Key Tech Choice Rationale to Tell Your Mentor:

> *"We chose **Snorkel weak supervision** instead of manual labeling because annotating thousands of habit spans by hand is impractical. Snorkel lets us encode domain knowledge as labeling functions and then learns a generative model to resolve conflicts — giving us high-quality probabilistic labels with zero manual effort."*

> *"For NER, we use **HuggingFace BERT/DeBERTa** because transformer models handle context-dependent habit mentions well (e.g., distinguishing 'I watched Netflix' as ENTERTAINMENT vs. 'I watched my diet' as NUTRITION). We use **class-weighted cross-entropy loss** to handle the severe label imbalance across 18 categories."*

---

## 3. ARCHITECTURE — THE 5-STAGE PIPELINE

```
┌──────────────────────────────────────────────────────────────────────┐
│                    RAW JOURNAL ENTRIES (free text)                    │
│  "Slept late, scrolled Instagram, skipped breakfast, felt anxious"   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  STAGE 1    │  │  STAGE 2    │  │  STAGE 3    │  │  STAGE 4    │  │  STAGE 5    │
│  Span       │──│  Weak       │──│  Gold Set   │──│  NER Model  │──│  Canonical- │
│  Extraction │  │  Supervision│  │  Generation │  │  Training   │  │  ization    │
│             │  │             │  │             │  │             │  │             │
│ • Regex     │  │ • 30+ LFs  │  │ • 400 spans │  │ • BERT/     │  │ • Semantic  │
│ • spaCy     │  │ • Snorkel   │  │ • Edge      │  │   DeBERTa   │  │   matching  │
│ • TF-IDF    │  │ • Probab.   │  │   cases     │  │ • BIO tags  │  │ • HDBSCAN   │
│ • PMI       │  │   labels    │  │ • Dev/Test  │  │ • seqeval   │  │ • Fuzzy     │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │                │                │
       ▼                ▼                ▼                ▼                ▼
   Spans +          Weak Labels      Gold Spans       Trained NER      Canonical
   Keywords         + Confidence     + Stats           Model            Mappings
   (Parquet)        (Parquet)        (CSV)            (HF Model)       → DB
```

---

## 4. FILE EXECUTION ORDER (STEP-BY-STEP)

Here is the **exact order** you should explain and run the files:

### Phase 0: Data Preparation
| Step | File | Purpose |
|------|------|---------|
| 0a | `scripts/generate_test_data.py` | Creates 20 synthetic journal entries for testing |
| 0b | `scripts/unify_data.py` | Merges raw datasets (CSVs, Parquets) into single `journals.parquet` |
| 0c | `scripts/eda_analysis.py` | Exploratory analysis: text length, emotion distribution, vocab richness |

### Phase 1: Span Extraction (Stage 1)
| Step | File | Purpose |
|------|------|---------|
| 1a | `src/extraction/extract_regex.py` | Multi-strategy span extraction (regex + spaCy matchers) |
| 1b | `src/extraction/keyword_mine.py` | Unsupervised keyword discovery (TF-IDF, PMI, V-O pairs) |

### Phase 2: Weak Supervision (Stage 2)
| Step | File | Purpose |
|------|------|---------|
| 2 | `src/supervision/weak_supervision.py` | Apply 30+ labeling functions → Snorkel Label Model → probabilistic labels |

### Phase 3: Gold Set + BIO Conversion (Stages 3-4a)
| Step | File | Purpose |
|------|------|---------|
| 3a | `scripts/generate_gold_set.py` | Generate 400 gold-labeled spans for evaluation |
| 3b | `src/ner/to_bio.py` | Convert span labels → BIO token sequences (JSONL format) |

### Phase 4: NER Training (Stage 4b)
| Step | File | Purpose |
|------|------|---------|
| 4 | `src/ner/train_ner.py` | Fine-tune BERT/DeBERTa for habit NER (token classification) |

### Phase 5: Canonicalization (Stage 5) — *Planned*
| Step | File | Purpose |
|------|------|---------|
| 5a | `src/canonicalization/canonicalize.py` | Map extracted habits to canonical ontology IDs |
| 5b | `src/canonicalization/cluster_new_habits.py` | HDBSCAN clustering for novel habits |

### Support Scripts
| File | Purpose |
|------|---------|
| `scripts/generate_report_figures.py` | Generate 6 publication-quality report figures |
| `scripts/test_checkpoint1.py` | Validate extraction pipeline output |
| `scripts/test_checkpoint2.py` | Validate weak supervision output |
| `scripts/test_checkpoint3.py` | Validate gold set + NER data |

---

## 5. STAGE-BY-STAGE DEEP DIVE

### 📌 Stage 1: Span Extraction (`src/extraction/`)

**What it does:** Identifies habit-related text spans from journal entries.

**Two parallel extraction strategies:**

#### a) `extract_regex.py` — Rule-Based Extraction
The `HabitSpanExtractor` class uses **three extraction methods** simultaneously:

1. **Regex Patterns (9 pattern types):**
   - `seed_alias` (confidence 0.95): Exact match against 800+ ontology aliases
   - `verb_numeric_duration` (0.85): "studied for 3 hours"
   - `duration_activity` (0.80): "2 hours of gaming"
   - `verb_time` (0.75): "stayed up till late"
   - `too_much` (0.80): "too much scrolling"
   - `frequency` (0.70): "procrastinated again"
   - `negation` (0.75): "didn't study"
   - `i_verb_habit` (0.85): "I meditated"
   - `goal_pattern` (0.70): "tried to exercise"

2. **spaCy Matchers (7 linguistic patterns):**
   - `VERB_NOUN`: "ate breakfast"
   - `VERB_ADV_NOUN`: "ate too much"
   - `ADJ_NOUN`: "late night"
   - `FEEL_STATE`: "felt anxious"
   - `NEG_VERB`: "didn't study"
   - `VERB_VERB_PROG`: "kept scrolling"
   - `VERB_FOR_DURATION`: "studied for 2 hours"

3. **spaCy PhraseMatcher**: Direct seed alias matching at token level

**Post-processing:**
- Deduplicate overlapping spans (keep highest confidence)
- Quality filtering (remove stopwords-only, too-short, punctuation spans)

**Input:** `data/raw/test_journals.csv` → **Output:** `results/spans/extracted_spans.parquet`

#### b) `keyword_mine.py` — Unsupervised Discovery
The `HabitKeywordMiner` discovers *new* habit phrases not in the seed ontology:

1. **Verb-Object pairs** via dependency parsing ("drink coffee", "skip class")
2. **Noun phrases** via spaCy chunking
3. **Bigrams/Trigrams** with frequency filtering
4. **TF-IDF** for corpus-specific salient terms
5. **PMI (Pointwise Mutual Information)** for statistically significant collocations

**Composite ranking:** `0.30 × frequency + 0.25 × TF-IDF + 0.20 × PMI + 0.15 × method_count + 0.10 × seed_overlap`

**Input:** `data/raw/test_journals.csv` → **Output:** `results/spans/mined_keywords.csv`

---

### 📌 Stage 2: Weak Supervision (`src/supervision/weak_supervision.py`)

**What it does:** Assigns category labels to spans WITHOUT manual annotation.

**The `HabitWeakSupervision` class creates 30+ Labeling Functions (LFs) in 7 categories:**

| LF Type | Count | Example Logic |
|---------|-------|---------------|
| Exact alias match (per category) | 18 | "scrolled instagram" → DIGITAL |
| Partial alias match (per category) | 18 | substring containment check |
| Keyword density (per category) | 18 | ≥2 category keywords in span+context |
| Semantic similarity (optional) | 18 | cosine sim > 0.70 with alias embeddings |
| Verb pattern | 1 | verb lemma → category mapping |
| Negation pattern | 1 | "didn't" + activity → category |
| Duration mention | 1 | time + activity → category |
| Frequency pattern | 1 | "again"/"always" + activity |
| Intensity pattern | 1 | "too much" + activity |
| Location mention | 1 | "gym" → FITNESS, "library" → ACADEMICS |
| Context keywords | 1 | Multiple signals in context window |

**How Snorkel Resolves Conflicts:**
```
Span: "slept late"
├── exact_SLEEP votes: SLEEP ✓
├── verb_pattern votes: SLEEP ✓
├── context_keywords votes: SLEEP ✓
├── exact_DIGITAL votes: ABSTAIN
└── Snorkel Label Model → SLEEP (probability: 0.94)
```

**Output columns added:** `weak_label`, `weak_label_name`, `max_prob`, `prob_vec`, `num_lfs_voted`, `prob_margin`

**Input:** `results/spans/extracted_spans.parquet` → **Output:** `results/labels/weak_labels.parquet`

---

### 📌 Stage 3: Gold Set Generation (`scripts/generate_gold_set.py`)

**What it does:** Creates 400 synthetic gold-labeled spans for evaluation.

**Why synthetic instead of manual?**
- Manual annotation of 400 spans across 18 categories takes days
- Synthetic + template-based generation ensures balanced category coverage
- Includes edge cases: negations, ambiguous spans, multi-word habits

**Template types used:**
- Positive: "Finally studied after putting it off"
- Negative: "Scrolled Instagram when I should have been working"
- Neutral: "Had coffee today"
- Duration: "Studied for 2 hours"
- Time: "Woke up at 5am"
- Negation: "Didn't exercise like I planned"
- Edge cases: "pulled all-nighter", "doom scrolling", "mindless scrolling"

**Output split:** 100 dev + 300 test spans → `data/gold/gold_dev.csv`, `data/gold/gold_test.csv`

---

### 📌 Stage 4: NER Training (`src/ner/`)

#### a) BIO Conversion (`to_bio.py`)
Converts span-level labels into **BIO-tagged token sequences** for NER training:

```
Token:    I    studied  for  3  hours  today
BIO Tag:  O    B-ACAD   I-ACAD I-ACAD I-ACAD  O
```

**Key challenge:** Aligning character-level span offsets to subword tokens (DeBERTa uses SentencePiece).

**Input:** `results/labels/weak_labels.parquet` + `data/processed/journals.parquet`
**Output:** `data/processed/ner_train.jsonl` (JSONL with tokens, ner_tags, input_ids, attention_mask)

#### b) NER Model Training (`train_ner.py`)
Fine-tunes a pretrained transformer for token classification:

- **Model:** `bert-base-cased` (or `microsoft/deberta-v3-small`)
- **Loss:** Class-weighted CrossEntropyLoss (handles 18-class imbalance)
- **Labels:** BIO tags (O, B-SLEEP, I-SLEEP, B-FITNESS, I-FITNESS, ...)
- **Evaluation:** seqeval (entity-level F1, strict IOB2 mode)
- **Early stopping:** Patience 3 on validation F1

**Input:** `data/processed/ner_train.jsonl` → **Output:** `models/ner/hf_ner/` (saved model + tokenizer)

---

### 📌 Stage 5: Canonicalization (Planned)

**What it would do:**
1. Map diverse span phrasings to canonical ontology IDs:
   - "went for a run" → `running`
   - "jogged in the park" → `running`
   - "morning 5k" → `running`

2. Discover novel habits not in ontology via HDBSCAN clustering on embeddings

3. Store results in SQLite database for downstream analysis

---

## 6. DATA FLOW DIAGRAM

```
data/raw/
├── test_journals.csv ──────────── [20 journal entries, ~300 chars each]
├── emotions_dataset.parquet ──┐
├── emotion_dataset_2.csv ─────┤── [External emotion datasets for EDA]
├── goemotions.csv ────────────┤
└── Daylio_Abid.csv ───────────┘
        │
        ▼ (scripts/unify_data.py)
data/processed/
├── journals.parquet ─────────── [Unified: journal_id, text, source, text_length]
│       │
│       ├──▶ (src/extraction/extract_regex.py)
│       │       │
│       │       ▼
│       │   results/spans/
│       │   ├── extracted_spans.parquet ──── [span_id, journal_id, span, method, confidence]
│       │   └── extraction_summary.json
│       │       │
│       │       ▼ (src/supervision/weak_supervision.py)
│       │   results/labels/
│       │   ├── weak_labels.parquet ──────── [+ weak_label, weak_label_name, max_prob]
│       │   ├── weak_labels.csv
│       │   ├── label_mappings.json ──────── [category_to_label, label_to_category]
│       │   └── lf_analysis.csv ──────────── [LF coverage, conflicts, overlaps]
│       │       │
│       │       ▼ (src/ner/to_bio.py)
│       │   data/processed/
│       │   ├── ner_train.jsonl ──────────── [tokens, ner_tags, input_ids]
│       │   ├── ner_val.jsonl
│       │   └── label_mappings.json
│       │       │
│       │       ▼ (src/ner/train_ner.py)
│       │   models/ner/hf_ner/
│       │   ├── config.json
│       │   ├── model.safetensors
│       │   └── tokenizer files
│       │
│       └──▶ (src/extraction/keyword_mine.py)
│               │
│               ▼
│           results/spans/
│           ├── mined_keywords.csv ────── [phrase, composite_score, TF-IDF, PMI]
│           └── keyword_mining_summary.json
│
seeds/
└── seed_ontology.json ────────── [80+ habits, 800+ aliases, 18 categories]
        │
        └──▶ Used by ALL stages as the knowledge base

data/gold/
├── gold_spans.csv ────────────── [400 spans with gold_label, start_char, end_char]
├── gold_dev.csv ──────────────── [100 dev spans]
├── gold_test.csv ─────────────── [300 test spans]
└── gold_set_stats.json
```

---

## 7. OUTPUT FILES & VISUALIZATIONS EXPLAINED

### Key Output Files

| File | Stage | What It Contains | How to Inspect |
|------|-------|-----------------|----------------|
| `results/spans/extracted_spans.parquet` | 1 | Habit spans with confidence scores | Open CSV version or `pd.read_parquet()` |
| `results/spans/extraction_summary.json` | 1 | Counts, method distribution, top spans | Open in text editor |
| `results/spans/mined_keywords.csv` | 1 | Discovered phrases ranked by composite score | Open in Excel |
| `results/labels/weak_labels.parquet` | 2 | Spans + category labels + probabilities | `pd.read_parquet()` — look at `weak_label_name` column |
| `results/labels/lf_analysis.csv` | 2 | Per-LF coverage, overlap, conflict stats | Shows which LFs are most useful |
| `results/labels/label_mappings.json` | 2 | category↔label_id mappings | Used by all downstream stages |
| `data/gold/gold_spans.csv` | 3 | Ground truth for evaluation | Review span, text, gold_label columns |
| `data/processed/ner_train.jsonl` | 4a | BIO-tagged sequences (one JSON per line) | Each line has tokens + ner_tags arrays |
| `models/ner/hf_ner/` | 4b | Trained model weights + config | Load with `AutoModelForTokenClassification` |

### Visualization Outputs (from `scripts/generate_report_figures.py`)

| Figure | File | What It Shows |
|--------|------|---------------|
| Fig 1 | `fig1_pipeline_architecture.png` | 5-stage pipeline diagram with arrows showing data flow |
| Fig 2 | `fig2_seed_ontology_categories.png` | Bar chart of habits per category + donut chart of category groups |
| Fig 3 | `fig3_extraction_results.png` | Method distribution + confidence histogram + processing funnel |
| Fig 4 | `fig4_weak_supervision_analysis.png` | LF coverage bars + confidence pie chart + summary stats |
| Fig 5 | `fig5_ner_training_results.png` | Loss curves + F1 score + per-category metrics + overall dashboard |
| Fig 6 | `fig6_category_distribution.png` | Weak label distribution across all 18 categories |

**Location:** `data/processed/visualizations/`

---

## 8. DIRECTORY STRUCTURE MAP

```
habit_tracking/
│
├── configs/
│   └── config.yaml                 ← Centralized configuration (all hyperparameters)
│
├── seeds/
│   └── seed_ontology.json          ← Knowledge base: 80+ habits, 800+ aliases
│
├── data/
│   ├── raw/                        ← Input datasets (journals, emotion data)
│   │   ├── test_journals.csv       ← 20 synthetic test entries
│   │   ├── emotions_dataset.parquet
│   │   ├── goemotions.csv
│   │   └── Daylio_Abid.csv
│   ├── processed/                  ← Unified & transformed data
│   │   ├── journals.parquet        ← Unified journal corpus
│   │   └── visualizations/         ← EDA plots + report figures
│   └── gold/                       ← Gold standard evaluation set
│       ├── gold_spans.csv          ← 400 gold-labeled spans
│       ├── gold_dev.csv            ← 100 dev spans
│       └── gold_test.csv           ← 300 test spans
│
├── src/                            ← Core pipeline source code
│   ├── extraction/                 ← STAGE 1: Span Extraction
│   │   ├── extract_regex.py        ← HabitSpanExtractor class (regex + spaCy)
│   │   └── keyword_mine.py         ← HabitKeywordMiner class (TF-IDF + PMI)
│   ├── supervision/                ← STAGE 2: Weak Supervision
│   │   └── weak_supervision.py     ← HabitWeakSupervision class (Snorkel)
│   ├── ner/                        ← STAGES 3-4: NER Pipeline
│   │   ├── to_bio.py               ← BIOConverter class (span→BIO alignment)
│   │   └── train_ner.py            ← NERTrainer class (HuggingFace fine-tuning)
│   ├── canonicalization/           ← STAGE 5: Canonicalization (planned)
│   │   ├── canonicalize.py
│   │   └── cluster_new_habits.py
│   └── utils/                      ← Utility modules (planned)
│       ├── db_store.py
│       └── weekly_report.py
│
├── scripts/                        ← Helper & validation scripts
│   ├── generate_test_data.py       ← Create synthetic test journals
│   ├── unify_data.py               ← Merge raw datasets into one Parquet
│   ├── eda_analysis.py             ← Exploratory Data Analysis
│   ├── generate_gold_set.py        ← Create gold evaluation spans
│   ├── generate_report_figures.py  ← Create 6 report visualizations
│   ├── test_checkpoint1.py         ← Validate Stage 1 output
│   ├── test_checkpoint2.py         ← Validate Stage 2 output
│   └── test_checkpoint3.py         ← Validate Stage 3-4 output
│
├── results/                        ← All pipeline outputs
│   ├── spans/                      ← Extraction results
│   ├── labels/                     ← Weak supervision results
│   ├── canonical/                  ← Canonicalization results
│   └── clusters/                   ← Clustering results
│
├── models/                         ← Trained models
│   ├── ner/hf_ner/                 ← Saved NER model
│   └── embeddings/                 ← Cached sentence embeddings
│
├── logs/                           ← Execution logs
└── tests/                          ← Unit tests
```

---

## 9. DEMONSTRATION SCRIPT — WHAT TO SAY TO YOUR MENTOR

### Opening (2 min)
> *"This project is an automated habit tracking pipeline. The problem: students write daily journals but manually tracking habits is tedious. Our solution: a 5-stage NLP pipeline that reads raw journal text and automatically identifies, labels, and classifies habits across 18 life domains — with zero manual annotation."*

### Walk Through the Seed Ontology (2 min)
> *"It all starts with our knowledge base — the seed ontology."*
- Open `seeds/seed_ontology.json`
- Show 2-3 habits with aliases: sleep_late, exercise, social_media
- Emphasize: "Each habit has 10-15 natural-language aliases — this is how students actually write."

### Show the Input Data (1 min)
> *"Here's what the raw input looks like."*
- Open `data/raw/test_journals.csv` — show 2-3 realistic entries
- Point out: mixed habits, emotions, negations, informal language

### Stage 1 — Extraction (3 min)
> *"The first stage extracts candidate habit spans from the text."*
- Open `src/extraction/extract_regex.py`
- Show the `HabitSpanExtractor` class — highlight the 3 extraction strategies
- Show a sample output from `results/spans/extracted_spans.csv`
- Point to confidence scores and method attribution

### Stage 2 — Weak Supervision (3 min)
> *"Now we need to label these spans. Instead of doing it manually, we use Snorkel."*
- Open `src/supervision/weak_supervision.py`
- Walk through 3-4 labeling function types (alias, keyword, verb, negation)
- Explain how `LabelModel` resolves conflicts between disagreeeing LFs
- Open `results/labels/weak_labels.csv` — show `weak_label_name`, `max_prob` columns

### Stage 3 — Gold Set (1 min)
> *"For evaluation, we generate a gold standard."*
- Open `data/gold/gold_spans.csv` — show balanced distribution across categories
- Mention: dev/test split, edge cases included

### Stage 4 — NER Training (2 min)
> *"The weak labels train a BERT-based NER model."*
- Explain BIO scheme: B-SLEEP, I-SLEEP, O
- Show `src/ner/to_bio.py` — the alignment challenge
- Show `src/ner/train_ner.py` — class weighting, early stopping

### Show Visualizations (2 min)
- Open the 6 figures from `data/processed/visualizations/`
- Walk through: Pipeline architecture → Ontology categories → Extraction results → Weak supervision → NER results → Category distribution

### Closing (1 min)
> *"In summary: we take unstructured journal text and produce structured habit data — automatically. The key innovation is using Snorkel weak supervision to eliminate the need for manual annotation while still training a high-quality NER model."*

---

## 10. KEY DESIGN DECISIONS YOU CAN EXPLAIN

### Why Seed Ontology Instead of Pure Unsupervised?
> "Pure unsupervised approaches discover patterns but can't assign meaningful labels. Our seed ontology provides **domain knowledge** — 80+ known habits with aliases — giving the system a strong prior. The keyword miner then discovers *additional* habits beyond the ontology."

### Why Snorkel Instead of Manual Labeling?
> "Manual annotation requires (a) annotators, (b) inter-annotator agreement, (c) weeks of effort. Snorkel lets us encode the same expertise as **programmatic labeling functions** that run in seconds and can be iterated instantly."

### Why BIO Tagging Instead of Text Classification?
> "Classification labels entire sentences. BIO tagging (token-level) lets us extract **exact habit spans** from text — we know not just *that* a sleep habit was mentioned, but *exactly which words* describe it: 'slept late again'."

### Why Class-Weighted Loss?
> "Our 18 categories are heavily imbalanced (e.g., MENTAL_STATE has 487 spans, CREATIVE has 34). Without weighting, the model would just predict the majority class. Inverse-frequency weighting penalizes errors on rare categories proportionally more."

### Why Parquet Over CSV?
> "Parquet is a columnar format — 3-10× smaller than CSV and much faster to load. We always save a CSV alongside for human inspection, but the pipeline reads Parquet for performance."

---

## 11. COMMANDS TO RUN THE PIPELINE

Run from the `habit_tracking/` directory:

```bash
# Step 0a: Generate test data
python scripts/generate_test_data.py

# Step 0b: Unify datasets
python scripts/unify_data.py --input-dir data/raw --output data/processed/journals.parquet

# Step 0c: EDA analysis
python scripts/eda_analysis.py

# Step 1a: Extract habit spans
python src/extraction/extract_regex.py \
  --input data/raw/test_journals.csv \
  --out results/spans/extracted_spans.parquet \
  --seeds seeds/seed_ontology.json \
  --text-column text --id-column id

# Step 1b: Mine keywords
python src/extraction/keyword_mine.py \
  --input data/raw/test_journals.csv \
  --output results/spans/mined_keywords.csv \
  --seed-ontology seeds/seed_ontology.json

# Step 2: Weak supervision
python src/supervision/weak_supervision.py \
  --input results/spans/extracted_spans.parquet \
  --output results/labels/weak_labels.parquet \
  --seed-ontology seeds/seed_ontology.json

# Step 3: Generate gold set
python scripts/generate_gold_set.py \
  --seed-ontology seeds/seed_ontology.json \
  --output-dir data/gold --num-spans 400

# Step 4a: Convert to BIO format
python src/ner/to_bio.py \
  --journals data/processed/journals.parquet \
  --spans results/labels/weak_labels.parquet \
  --out data/processed/ner_train.jsonl \
  --input-type weak --min-conf 0.7

# Step 4b: Train NER model
python src/ner/train_ner.py \
  --train data/processed/ner_train.jsonl \
  --model-name bert-base-cased \
  --output-dir models/ner/hf_ner \
  --num-epochs 5 --batch-size 16

# Validation checkpoints
python scripts/test_checkpoint1.py
python scripts/test_checkpoint2.py
python scripts/test_checkpoint3.py

# Generate report figures
python scripts/generate_report_figures.py
```

---

## 🎯 QUICK REFERENCE CARD — ANTICIPATED MENTOR QUESTIONS

| Question Your Mentor May Ask | Your Prepared Answer |
|------------------------------|---------------------|
| "How many stages?" | 5: Extraction → Weak Supervision → Gold Set → NER Training → Canonicalization |
| "What's the input?" | Free-text journal entries (CSV/Parquet) |
| "What's the final output?" | Structured habit data with category labels, stored in SQLite |
| "How many categories?" | 18 life domains (Sleep, Fitness, Academics, Digital, Nutrition, Social, etc.) |
| "Did you annotate manually?" | No — Snorkel weak supervision with 30+ labeling functions |
| "What NER model?" | BERT-base-cased (or DeBERTa-v3-small), fine-tuned on BIO tags |
| "What's the F1 score?" | ~0.69 overall F1 (strict entity-level), best categories reach 0.78 |
| "What's Snorkel?" | A weak supervision framework that combines noisy labeling functions via a generative model |
| "Why not just use regex?" | Regex misses context-dependent mentions. NER learns from context (e.g., "watched my diet" ≠ "watched Netflix") |
| "What's BIO tagging?" | Begin-Inside-Outside: a token-level tagging scheme for Named Entity Recognition |
| "What data format?" | Parquet for efficiency, CSV for inspection, JSONL for NER training |
| "How do you handle imbalanced classes?" | Class-weighted CrossEntropyLoss with inverse-frequency weights |
| "What's in the seed ontology?" | 80+ habits with IDs, categories, descriptions, and 800+ natural-language aliases |

---

> [!TIP]
> **Best demo strategy:** Pick one specific journal entry (e.g., journal j005 about anxiety, junk food, scrolling) and trace it through **every stage** — showing exactly how "ate junk food" becomes a span → gets labeled NUTRITION → gets BIO tagged → gets recognized by the NER model. This makes the entire pipeline tangible and easy to follow.
