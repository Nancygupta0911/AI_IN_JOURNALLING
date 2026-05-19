<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" alt="spaCy"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

<h1 align="center">🧠 AI in Journalling</h1>

<p align="center">
  <strong>Emotion Detection & Habit Tracking from Free-Text Journal Entries</strong>
</p>

<p align="center">
  <em>A dual-module AI system that analyzes journal entries to understand <b>what you feel</b> and <b>what you do</b> — combining deep learning emotion classification with zero-annotation habit extraction.</em>
</p>

<p align="center">
  <a href="#-emotion-detection-model"><img src="https://img.shields.io/badge/Module_1-Emotion_Detection-667eea?style=for-the-badge" alt="Emotion Detection"/></a>
  <a href="#-habit-tracking-pipeline"><img src="https://img.shields.io/badge/Module_2-Habit_Tracking-764ba2?style=for-the-badge" alt="Habit Tracking"/></a>
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Repository Structure](#️-repository-structure)
- [Emotion Detection Model](#-emotion-detection-model)
- [Habit Tracking Pipeline](#-habit-tracking-pipeline)
- [How the Two Modules Connect](#-how-the-two-modules-connect)
- [Installation & Setup](#️-installation--setup)
- [Results & Visualizations](#-results--visualizations)
- [Tech Stack](#-tech-stack)
- [References](#-references)
- [Author](#-author)
- [License](#-license)

---

## 📖 Overview

This repository contains two interconnected AI systems designed for **automated behavioral analysis from journal entries**:

| Module | Description | Key Tech |
|--------|-------------|----------|
| **[`emotion_detection/`](#-emotion-detection-model)** | Multi-label emotion classifier (21 emotions) using DeBERTa-v3-base with 5-fold ensemble | PyTorch, HuggingFace Transformers, Streamlit |
| **[`habit_tracking/`](#-habit-tracking-pipeline)** | 5-stage NLP pipeline that extracts, labels, and classifies habits across 18 life domains — with zero manual annotation | spaCy, Snorkel, HuggingFace NER, HDBSCAN |

### Combined Workflow

```
📝 Raw Journal Text
        │
        ├──► 🎭 Emotion Detection  →  Identifies WHAT the user feels
        │
        └──► 🔄 Habit Tracking     →  Identifies WHAT behaviors are mentioned
                    │
                    ▼
        📊 Structured Behavioral Insights
           (emotion–habit correlations)
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎭 **21 Emotion Classes** | Fine-grained multi-label emotion detection beyond simple sentiment (positive/negative) |
| 🏋️ **18 Habit Categories** | Comprehensive life-domain coverage from Sleep to Spiritual to Substance Use |
| 🤖 **Zero Manual Annotation** | Habit tracking pipeline uses weak supervision (Snorkel) — no hand-labeling needed |
| 🔬 **Research-Grade Training** | Asymmetric Loss, R-Drop, Supervised Contrastive Learning, per-label threshold tuning |
| 🧩 **Modular Architecture** | Each module works independently or together for combined insights |
| 🌐 **Interactive Demo** | Streamlit web app with real-time analysis, radar charts, and confidence visualizations |
| 📊 **Reproducible Results** | All metrics, plots, and evaluation artifacts are version-controlled |

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

Detecting nuanced emotions from free-text journal entries, capturing **multi-label** emotional states — because a single entry can express both _joy_ and _anxiety_ simultaneously.

### Model Architecture

```
                        ┌──────────────────────────┐
                        │  DeBERTa-v3-base Encoder  │
                        └────────────┬─────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ Multi-Label Head  │   │  Contrastive     │   │   R-Drop         │
   │ (21 classes)      │   │  Learning Head   │   │   Regularization │
   └──────────────────┘   └──────────────────┘   └──────────────────┘
```

| Component | Details |
|-----------|---------| 
| **Base Model** | Microsoft DeBERTa-v3-base |
| **Training Strategy** | 5-Fold Cross-Validation → Ensemble |
| **Loss Function** | Asymmetric Loss (ICCV 2021) for class imbalance |
| **Regularization** | R-Drop (NeurIPS 2021) + Supervised Contrastive Learning |
| **Threshold Optimization** | Per-label threshold tuning on validation set |
| **Inference** | Ensemble averaging across 5 fold models |

### 21 Emotion Classes

<table>
<tr>
<td>😠 anger</td><td>😰 anxiety</td><td>😌 calmness</td><td>💪 confidence</td><td>😕 confusion</td><td>☺️ contentment</td><td>😞 disappointment</td>
</tr>
<tr>
<td>🤢 disgust</td><td>🎉 excitement</td><td>😨 fear</td><td>😤 frustration</td><td>🙏 gratitude</td><td>🌟 hope</td><td>😊 joy</td>
</tr>
<tr>
<td>😔 loneliness</td><td>❤️ love</td><td>😐 neutral</td><td>🏆 pride</td><td>😢 sadness</td><td>😳 shame</td><td>😲 surprise</td>
</tr>
</table>

### Performance (5-Fold Ensemble)

| Metric | Score |
|--------|-------|
| **Accuracy** | 58.4% |
| **F1 Macro** | 63.0% |
| **F1 Weighted** | 58.2% |

**Top Performing Classes:**

| Class | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| 😔 Loneliness | **92.2%** | 88.7% | 95.9% |
| 💪 Confidence | **89.2%** | 89.5% | 89.0% |
| ☺️ Contentment | **77.1%** | 75.5% | 78.7% |
| 😳 Shame | **76.1%** | 71.3% | 81.6% |
| 🙏 Gratitude | **75.9%** | 73.5% | 78.5% |

> **Note:** These metrics are for a 21-class multi-label classification task — far more granular than typical binary sentiment analysis.

### Data Sources

The model was trained on a unified dataset combining:

| Source | Type |
|--------|------|
| **GoEmotions** | Reddit comments with emotion labels (Google Research) |
| **Emotion Dataset v2** | Multi-class emotion corpus |
| **Daylio Export** | Mood tracking app data with temporal context |
| **Student Journal Entries** | Custom collected journal data |
| **HuggingFace Parquet** | Additional emotion-labeled datasets |

### Preprocessing Highlights

- ✅ Context-aware text cleaning (preserves intensifiers & negations)
- ✅ Multi-label emotion extraction with intensity scoring
- ✅ Habit keyword co-extraction (links behaviors to emotions)
- ✅ Emotion co-occurrence analysis
- ✅ Stratified multi-label train/val/test splits

### 🖥️ Streamlit Demo App

An interactive web application for real-time emotion analysis:

```bash
cd emotion_detection/streamlit_app
streamlit run app.py
```

**Features:**
- 📊 Real-time multi-label emotion detection
- 📈 Emotion confidence bar charts, radar charts, valence distribution
- 🔍 Explicit vs Implicit emotion classification
- 🤝 Fold agreement visualization
- ⚡ Quick-test example buttons

---

## 🔄 Habit Tracking Pipeline

### Problem Statement

Automatically extract, label, and classify habit mentions from free-text journal entries into 18 life-domain categories — **without any manual annotation**.

### 5-Stage Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     📝 RAW JOURNAL ENTRIES (free text)                       │
│   "Slept late, scrolled Instagram, skipped breakfast, felt anxious"          │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
    ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐
    │  Stage 1  │→│  Stage 2  │→│  Stage 3  │→│  Stage 4  │→│    Stage 5    │
    │  Span     │ │  Weak     │ │  Gold Set │ │  NER      │ │  Canonicali-  │
    │  Extract  │ │  Supervis.│ │  Gen.     │ │  Training │ │  zation       │
    │           │ │           │ │           │ │           │ │               │
    │ • Regex   │ │ • 30+ LFs │ │ • 400     │ │ • BERT/   │ │ • Semantic    │
    │ • spaCy   │ │ • Snorkel │ │   spans   │ │   DeBERTa │ │   matching    │
    │ • TF-IDF  │ │ • Probab. │ │ • Edge    │ │ • BIO     │ │ • HDBSCAN     │
    │ • PMI     │ │   labels  │ │   cases   │ │   tags    │ │ • Fuzzy match │
    └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────────┘
```

### 18 Habit Categories

| Category | Example Habits |
|----------|----------------|
| 😴 **Sleep** | slept late, insomnia, napped, pulled all-nighter |
| 🏋️ **Fitness** | gym, running, yoga, workout |
| 📚 **Academics** | studied, exam prep, homework, lecture |
| 📱 **Digital/Screen** | scrolled Instagram, doom scrolling, social media |
| 🍎 **Nutrition** | ate junk food, skipped breakfast, cooked meal |
| 👥 **Social** | hung out with friends, called mom, party |
| 🧘 **Mental Health** | felt anxious, meditation, therapy session |
| ⏰ **Productivity** | procrastinated, finished project, time management |
| 🚿 **Hygiene** | showered, brushed teeth, skincare routine |
| 🎨 **Creative** | drew, wrote poetry, played guitar |
| 🙏 **Spiritual** | prayed, gratitude journal, church |
| ☕ **Substance Use** | drank coffee, smoked, alcohol |
| 💰 **Financial** | budgeted, impulse purchase, savings |
| 🎮 **Entertainment** | watched Netflix, read a book, gaming |
| 🚗 **Commute** | walked to class, bus ride, drove |
| 🌿 **Environmental** | cleaned room, organized desk, gardening |
| 🏠 **Household** | did laundry, cooked dinner, grocery shopping |
| 📦 **Other** | miscellaneous activities |

### Key Technical Highlights

| Stage | Technology | What It Does |
|-------|-----------|--------------| 
| **1. Extraction** | spaCy + Regex + TF-IDF/PMI | 9 regex pattern types + 7 linguistic patterns + unsupervised keyword discovery |
| **2. Weak Supervision** | Snorkel LabelModel | 30+ labeling functions resolve via generative model → probabilistic labels |
| **3. Gold Set** | Template-based generation | 400 synthetic gold spans (dev/test split) with edge cases & negations |
| **4. NER Training** | HuggingFace Transformers | BERT/DeBERTa fine-tuned on BIO tags with class-weighted loss |
| **5. Canonicalization** | Sentence-BERT + HDBSCAN | Map diverse phrasings → canonical habit IDs; discover novel habits |

### Seed Ontology

The knowledge base powering the pipeline:
- 🔹 **80+ habits** across 18 categories
- 🔹 **800+ aliases** (natural language phrasings)
- 🔹 Example: `running` → `["went for a run", "jogged", "morning 5k", "went jogging", "cardio"]`

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

# ✅ Validation checkpoints
python scripts/test_checkpoint1.py    # Stage 1
python scripts/test_checkpoint2.py    # Stage 2
python scripts/test_checkpoint3.py    # Stages 3-4
```

---

## 🔗 How the Two Modules Connect

The Emotion Detection model and Habit Tracking pipeline work together to provide **comprehensive behavioral analysis**:

```
Journal Entry: "I slept late, scrolled through Instagram for 3 hours,
               skipped breakfast, and felt really anxious about exams."
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  🎭 Emotion Detection     🔄 Habit Tracking
  ┌──────────────────┐     ┌─────────────────────┐
  │ anxiety:  0.82   │     │ "slept late"  → SLEEP│
  │ fear:     0.45   │     │ "scrolled IG"       │
  │ sadness:  0.38   │     │           → DIGITAL  │
  └──────────────────┘     │ "skipped breakfast"  │
                           │          → NUTRITION  │
                           └─────────────────────┘
                    │
                    ▼
          📊 Combined Insight:
    "High anxiety correlates with poor sleep,
     excessive screen time, and skipped meals"
```

The emotion detection model's preprocessing pipeline (`preprocessing.py`) already includes habit keyword extraction — linking detected emotions to behavioral patterns. The habit tracking pipeline provides fine-grained, span-level habit identification that complements the sentence-level emotion analysis.

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python** 3.8+
- **CUDA-capable GPU** (recommended for training)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Nancygupta0911/AI_IN_JOURNALLING.git
cd AI_IN_JOURNALLING

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers
pip install pandas numpy scikit-learn matplotlib seaborn

# Install NLP dependencies
pip install spacy snorkel sentence-transformers
pip install nltk hdbscan

# Install web app dependencies
pip install streamlit plotly

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Model Weights

> The trained DeBERTa-v3 fold models (~741MB each) are **not included** in this repository due to size constraints. The models can be reproduced by running the training pipeline with the provided preprocessing and training scripts.

---

## 📊 Results & Visualizations

### Emotion Detection

| File | Description |
|------|-------------|
| `emotion_detection/metrics/classification_report.txt` | Per-class precision, recall, F1 |
| `emotion_detection/metrics/ensemble_results.json` | Ensemble (voting, averaging, calibrated) metrics |
| `emotion_detection/plots/confusion_matrix_ensemble_average.png` | 21×21 confusion matrix |
| `emotion_detection/plots/per_class_f1.png` | Per-class F1 score comparison |
| `emotion_detection/plots/fold_comparison.png` | Fold-wise performance comparison |

### Habit Tracking

| File | Description |
|------|-------------|
| `habit_tracking/results/spans/` | Extracted habit spans with confidence scores |
| `habit_tracking/results/labels/` | Weak supervision output with category labels |
| `habit_tracking/data/gold/` | Gold standard evaluation set (400 spans) |

---

## 🛠️ Tech Stack

<table>
<tr><th>Category</th><th>Technologies</th></tr>
<tr><td><b>Deep Learning</b></td><td>PyTorch, HuggingFace Transformers, DeBERTa-v3</td></tr>
<tr><td><b>NLP</b></td><td>spaCy, NLTK, Sentence-Transformers, seqeval</td></tr>
<tr><td><b>Weak Supervision</b></td><td>Snorkel (Data Programming)</td></tr>
<tr><td><b>Clustering</b></td><td>HDBSCAN, scikit-learn</td></tr>
<tr><td><b>Data Processing</b></td><td>Pandas, NumPy, Parquet</td></tr>
<tr><td><b>Visualization</b></td><td>Matplotlib, Seaborn, Plotly</td></tr>
<tr><td><b>Web App</b></td><td>Streamlit</td></tr>
<tr><td><b>Storage</b></td><td>SQLite, JSON, CSV, Parquet</td></tr>
</table>

---

## 📚 References

| Paper | Venue |
|-------|-------|
| **DeBERTa:** He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" | ICLR 2021 |
| **Asymmetric Loss:** Ben-Baruch et al., "Asymmetric Loss for Multi-Label Classification" | ICCV 2021 |
| **R-Drop:** Wu et al., "R-Drop: Regularized Dropout for Neural Networks" | NeurIPS 2021 |
| **Supervised Contrastive Learning:** Khosla et al. | NeurIPS 2020 |
| **Snorkel:** Ratner et al., "Data Programming: Creating Large Training Sets, Quickly" | NeurIPS 2016 |
| **GoEmotions:** Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions" | ACL 2020 |
| **seqeval:** Standard evaluation framework for sequence labeling | — |

---

## 👤 Author

**NANCY GUPTA**  
🔗 [GitHub](https://github.com/Nancygupta0911)

---

## 📝 License

This project is open source and available for academic and research purposes.

---

<p align="center">
  <sub>⭐ Star this repo if you found it useful!</sub>
</p>
