<p align="center">
  <h1 align="center">🧠 AI in Journalling</h1>
  <p align="center">
    <strong>Emotion Detection & Habit Tracking from Free-Text Journal Entries</strong>
  </p>
  <p align="center">
    <a href="#emotion-detection-model"><img src="https://img.shields.io/badge/Module_1-Emotion_Detection-667eea?style=for-the-badge" alt="Emotion Detection"/></a>
    <a href="#habit-tracking-pipeline"><img src="https://img.shields.io/badge/Module_2-Habit_Tracking-764ba2?style=for-the-badge" alt="Habit Tracking"/></a>
  </p>
</p>

---

## 📖 Overview

This repository contains two interconnected AI systems designed for **automated behavioral analysis from journal entries**:

| Module | Description | Key Tech |
|--------|-------------|----------|
| **[`emotion_detection/`](#emotion-detection-model)** | Multi-label emotion classifier (21 emotions) using DeBERTa-v3-base with 5-fold ensemble | PyTorch, HuggingFace Transformers, Streamlit |
| **[`habit_tracking/`](#habit-tracking-pipeline)** | 5-stage NLP pipeline that extracts, labels, and classifies habits across 18 life domains — with zero manual annotation | spaCy, Snorkel, HuggingFace NER, HDBSCAN |

**Combined Workflow:**  
Raw journal text → **Emotion Detection** (identifies _what_ the user feels) → **Habit Tracking** (identifies _what behaviors_ are mentioned) → Structured behavioral insights.

---

## 🏗️ Repository Structure

```
AI_IN_JOURNALLING/
│
├── emotion_detection/              ← Module 1: Multi-label Emotion Classification
│   ├── preprocessing.py            ← Data preprocessing & multi-label encoding
│   ├── training.py                 ← Research-grade training (ASL, R-Drop, Contrastive)
│   ├── testing.py                  ← Comprehensive evaluation & ensemble testing
│   ├── streamlit_app/              ← Interactive demo application
│   │   ├── app.py                  ← Streamlit UI with visualizations
│   │   ├── config.py               ← App configuration
│   │   └── run_app.py              ← Startup validation & launcher
│   ├── metrics/                    ← Training & evaluation results
│   │   ├── classification_report.txt
│   │   ├── ensemble_results.json
│   │   ├── fold_summary.csv
│   │   └── per_class_metrics.csv
│   ├── plots/                      ← Evaluation visualizations
│   │   ├── confusion_matrix_ensemble_average.png
│   │   ├── fold_comparison.png
│   │   └── per_class_f1.png
│   ├── label_mapping.json          ← 21 emotion class definitions
│   └── metadata.json               ← Dataset statistics & config
│
├── habit_tracking/                 ← Module 2: Habit Tracking Pipeline
│   ├── configs/
│   │   └── config.yaml             ← Centralized pipeline configuration
│   ├── seeds/
│   │   └── seed_ontology.json      ← Knowledge base (80+ habits, 800+ aliases)
│   ├── src/                        ← Core pipeline source code
│   │   ├── extraction/             ← Stage 1: Span Extraction
│   │   │   ├── extract_regex.py    ← Regex + spaCy pattern extraction
│   │   │   └── keyword_mine.py     ← TF-IDF + PMI keyword discovery
│   │   ├── supervision/            ← Stage 2: Weak Supervision
│   │   │   └── weak_supervision.py ← Snorkel labeling functions
│   │   ├── ner/                    ← Stages 3-4: NER Pipeline
│   │   │   ├── to_bio.py           ← Span → BIO tag conversion
│   │   │   └── train_ner.py        ← Transformer NER training
│   │   ├── canonicalization/       ← Stage 5: Canonicalization
│   │   │   ├── canonicalize.py     ← Semantic matching to ontology
│   │   │   └── cluster_new_habits.py ← HDBSCAN novel habit discovery
│   │   └── utils/                  ← Utilities
│   │       ├── db_store.py         ← SQLite storage
│   │       └── weekly_report.py    ← Reporting module
│   ├── scripts/                    ← Helper & validation scripts
│   │   ├── generate_test_data.py   ← Create synthetic test journals
│   │   ├── unify_data.py           ← Merge raw datasets
│   │   ├── eda_analysis.py         ← Exploratory data analysis
│   │   ├── generate_gold_set.py    ← Gold evaluation set creation
│   │   ├── generate_report_figures.py ← Report visualization
│   │   ├── test_checkpoint1.py     ← Stage 1 validation
│   │   ├── test_checkpoint2.py     ← Stage 2 validation
│   │   └── test_checkpoint3.py     ← Stage 3-4 validation
│   ├── data/
│   │   ├── raw/test_journals.csv   ← Sample test entries
│   │   └── gold/                   ← Gold standard evaluation
│   ├── results/                    ← Pipeline outputs
│   │   ├── spans/                  ← Extraction results
│   │   └── labels/                 ← Weak supervision labels
│   └── DEMONSTRATION_GUIDE.md      ← Comprehensive walkthrough
│
├── .gitignore
└── README.md                       ← This file
```

---

## 🎭 Emotion Detection Model

### Problem Statement
Detecting nuanced emotions from free-text journal entries, capturing **multi-label** emotional states (e.g., a single entry can express both _joy_ and _anxiety_ simultaneously).

### Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | Microsoft DeBERTa-v3-base |
| **Training Strategy** | 5-Fold Cross-Validation → Ensemble |
| **Loss Function** | Asymmetric Loss (ICCV 2021) for class imbalance |
| **Regularization** | R-Drop (NeurIPS 2021) + Supervised Contrastive Learning |
| **Threshold Optimization** | Per-label threshold tuning on validation set |
| **Inference** | Ensemble averaging across 5 fold models |

### 21 Emotion Classes

```
anger • anxiety • calmness • confidence • confusion • contentment
disappointment • disgust • excitement • fear • frustration • gratitude
hope • joy • loneliness • love • neutral • pride • sadness • shame • surprise
```

### Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 58.4% |
| **F1 Macro** | 63.0% |
| **F1 Weighted** | 58.2% |
| **Best Class** (Loneliness) | 92.2% F1 |
| **Best Class** (Confidence) | 89.2% F1 |

> **Note:** These metrics are for a 21-class multi-label classification task — far more granular than typical binary sentiment analysis.

### Data Sources
The model was trained on a unified dataset combining:
- **GoEmotions** (Reddit comments with emotion labels)
- **Emotion Dataset v2** (multi-class emotion corpus)
- **Daylio Export** (mood tracking app data)
- **Student Journal Entries** (custom collected)
- **Parquet emotion datasets** (HuggingFace Hub)

### Preprocessing Pipeline Highlights
- Context-aware text cleaning (preserves intensifiers & negations)
- Multi-label emotion extraction with intensity scoring
- Habit keyword co-extraction (links behaviors to emotions)
- Emotion co-occurrence analysis
- Stratified multi-label train/val/test splits

### Streamlit Demo App
An interactive web application for real-time emotion analysis:

```bash
cd emotion_detection/streamlit_app
streamlit run app.py
```

**Features:**
- Real-time multi-label emotion detection
- Emotion confidence bar charts, radar charts, valence distribution
- Explicit vs Implicit emotion classification
- Fold agreement visualization
- Quick-test example buttons

---

## 🔄 Habit Tracking Pipeline

### Problem Statement
Automatically extract, label, and classify habit mentions from free-text journal entries into 18 life-domain categories — **without any manual annotation**.

### 5-Stage Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAW JOURNAL ENTRIES (free text)                          │
│  "Slept late, scrolled Instagram, skipped breakfast, felt anxious"         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │ Stage 1  │→│ Stage 2  │→│ Stage 3  │→│ Stage 4  │→│   Stage 5    │
    │ Span     │ │ Weak     │ │ Gold Set │ │ NER      │ │ Canonicali-  │
    │ Extract  │ │ Supervis.│ │ Gen.     │ │ Training │ │ zation       │
    │          │ │          │ │          │ │          │ │              │
    │• Regex   │ │• 30+ LFs │ │• 400     │ │• BERT/   │ │• Semantic    │
    │• spaCy   │ │• Snorkel │ │  spans   │ │  DeBERTa │ │  matching    │
    │• TF-IDF  │ │• Probab. │ │• Edge    │ │• BIO     │ │• HDBSCAN     │
    │• PMI     │ │  labels  │ │  cases   │ │  tags    │ │• Fuzzy match │
    └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

### 18 Habit Categories

```
Sleep • Fitness • Academics • Digital/Screen • Nutrition • Social
Mental Health • Productivity • Hygiene • Creative • Spiritual
Substance Use • Financial • Entertainment • Commute • Environmental
Household • Other
```

### Key Technical Highlights

| Stage | Technology | What It Does |
|-------|-----------|--------------|
| **Extraction** | spaCy + Regex + TF-IDF/PMI | 9 regex pattern types + 7 linguistic patterns + unsupervised keyword discovery |
| **Weak Supervision** | Snorkel LabelModel | 30+ labeling functions (alias match, keyword density, verb pattern, negation, semantic similarity) resolve via generative model |
| **Gold Set** | Template-based generation | 400 synthetic gold spans (dev/test split) with edge cases & negations |
| **NER Training** | HuggingFace Transformers | BERT/DeBERTa fine-tuned on BIO tags with class-weighted CrossEntropyLoss |
| **Canonicalization** | Sentence-BERT + HDBSCAN | Map diverse phrasings → canonical habit IDs; discover novel habits |

### Seed Ontology
- **80+ habits** across 18 categories
- **800+ aliases** (natural language phrasings)
- Example: `running` → ["went for a run", "jogged", "morning 5k", "went jogging"]

### Running the Pipeline

```bash
cd habit_tracking/

# 1. Generate test data
python scripts/generate_test_data.py

# 2. Unify datasets
python scripts/unify_data.py --input-dir data/raw --output data/processed/journals.parquet

# 3. Extract habit spans
python src/extraction/extract_regex.py \
  --input data/raw/test_journals.csv \
  --out results/spans/extracted_spans.parquet \
  --seeds seeds/seed_ontology.json

# 4. Apply weak supervision
python src/supervision/weak_supervision.py \
  --input results/spans/extracted_spans.parquet \
  --output results/labels/weak_labels.parquet \
  --seed-ontology seeds/seed_ontology.json

# 5. Generate gold set
python scripts/generate_gold_set.py \
  --seed-ontology seeds/seed_ontology.json \
  --output-dir data/gold --num-spans 400

# 6. Convert to BIO format
python src/ner/to_bio.py \
  --journals data/processed/journals.parquet \
  --spans results/labels/weak_labels.parquet \
  --out data/processed/ner_train.jsonl

# 7. Train NER model
python src/ner/train_ner.py \
  --train data/processed/ner_train.jsonl \
  --model-name bert-base-cased \
  --output-dir models/ner/hf_ner

# Validation
python scripts/test_checkpoint1.py
python scripts/test_checkpoint2.py
python scripts/test_checkpoint3.py
```

---

## 🔗 How the Two Modules Connect

The Emotion Detection model and Habit Tracking pipeline work together to provide comprehensive behavioral analysis:

```
Journal Entry: "I slept late, scrolled through Instagram for 3 hours,
               skipped breakfast, and felt really anxious about exams."
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  Emotion Detection       Habit Tracking
  ┌──────────────┐     ┌────────────────────┐
  │ anxiety: 0.82│     │ "slept late" → SLEEP│
  │ fear:    0.45│     │ "scrolled Instagram"│
  │ sadness: 0.38│     │          → DIGITAL  │
  └──────────────┘     │ "skipped breakfast" │
                       │          → NUTRITION│
                       └────────────────────┘
                    │
                    ▼
           Combined Insight:
    "High anxiety correlates with poor sleep,
     excessive screen time, and skipped meals"
```

**The emotion detection model's preprocessing pipeline** (`preprocessing.py`) already includes habit keyword extraction — linking detected emotions to behavioral patterns. The habit tracking pipeline provides fine-grained, span-level habit identification that complements the sentence-level emotion analysis.

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/PRASANNA-THE-PRASANN1/AI_IN_JOURNALLING.git
cd AI_IN_JOURNALLING

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers
pip install pandas numpy scikit-learn matplotlib seaborn
pip install spacy snorkel sentence-transformers
pip install streamlit plotly
pip install nltk hdbscan

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Model Weights

The trained DeBERTa-v3 fold models (~741MB each) are **not included** in this repository due to size constraints. The models can be reproduced by running the training pipeline with the provided preprocessing and training scripts.

---

## 📊 Results & Visualizations

### Emotion Detection
| File | Description |
|------|-------------|
| `emotion_detection/metrics/classification_report.txt` | Per-class precision, recall, F1 |
| `emotion_detection/metrics/ensemble_results.json` | Ensemble (voting, averaging, calibrated) metrics |
| `emotion_detection/plots/confusion_matrix_ensemble_average.png` | 21×21 confusion matrix |
| `emotion_detection/plots/per_class_f1.png` | Per-class F1 score comparison |

### Habit Tracking
| File | Description |
|------|-------------|
| `habit_tracking/results/spans/` | Extracted habit spans with confidence scores |
| `habit_tracking/results/labels/` | Weak supervision output with category labels |
| `habit_tracking/data/gold/` | Gold standard evaluation set |

---

## 📚 References

- **DeBERTa:** He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" (ICLR 2021)
- **Asymmetric Loss:** Ben-Baruch et al., "Asymmetric Loss for Multi-Label Classification" (ICCV 2021)
- **R-Drop:** Wu et al., "R-Drop: Regularized Dropout for Neural Networks" (NeurIPS 2021)
- **Supervised Contrastive Learning:** Khosla et al. (NeurIPS 2020)
- **Snorkel:** Ratner et al., "Data Programming: Creating Large Training Sets, Quickly" (NeurIPS 2016)
- **GoEmotions:** Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions" (ACL 2020)
- **seqeval:** Standard evaluation for sequence labeling

---

## 👤 Author

**Prasanna Saxena**

---

## 📝 License

This project is open source and available for academic and research purposes.
