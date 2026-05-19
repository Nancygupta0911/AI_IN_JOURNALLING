# 🎭 Emotion Detection Module

Multi-label emotion classification system using **DeBERTa-v3-base** with 5-fold cross-validation ensemble.

## Architecture

```
Input Text → DeBERTa-v3-base Encoder → Multi-Label Classification Head (21 classes)
                                     → Contrastive Learning Head (emotion relationships)
                                     → R-Drop Regularization (generalization)
```

### Training Strategy
- **Two-Stage Training:** Stage 1 trains on balanced subset, Stage 2 fine-tunes on full data
- **Loss Function:** Asymmetric Loss (ASL) — SOTA for imbalanced multi-label classification
- **Regularization:** R-Drop (passes same input twice with different dropout masks, minimizes KL divergence)
- **Contrastive Learning:** Supervised contrastive loss to learn emotion relationships (similar emotions cluster together)
- **Threshold Optimization:** Per-label threshold tuning on validation set

### 21 Emotion Classes
`anger` · `anxiety` · `calmness` · `confidence` · `confusion` · `contentment` · `disappointment` · `disgust` · `excitement` · `fear` · `frustration` · `gratitude` · `hope` · `joy` · `loneliness` · `love` · `neutral` · `pride` · `sadness` · `shame` · `surprise`

## Files

| File | Description |
|------|-------------|
| `preprocessing.py` | Data loading from 6 sources, multi-label encoding, quality filtering, text cleaning, emotion hierarchy, habit keyword extraction, stratified splits |
| `training.py` | Full training pipeline: AsymmetricLoss, R-Drop, SupConLoss, MultiLabelEmotionModel, ThresholdOptimizer, 5-fold CV |
| `testing.py` | Comprehensive evaluation: ensemble inference (voting/averaging), per-class analysis, confidence calibration, error analysis, visualization |
| `streamlit_app/app.py` | Interactive Streamlit web interface with plotly charts |
| `streamlit_app/config.py` | Configuration for model paths, UI settings |
| `streamlit_app/run_app.py` | Startup validation script |
| `label_mapping.json` | Class ↔ ID mappings for 21 emotions |
| `metadata.json` | Dataset statistics (split sizes, label distribution) |

## Performance (5-Fold Ensemble)

| Method | Accuracy | F1 Macro | F1 Weighted |
|--------|----------|----------|-------------|
| Voting | 58.2% | 62.7% | 57.9% |
| **Averaging** | **58.3%** | **62.8%** | **58.0%** |
| Calibrated | 58.4% | 63.0% | 58.2% |

### Top-Performing Classes (F1 Score)
| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Loneliness | 0.922 | 0.887 | 0.959 |
| Confidence | 0.892 | 0.895 | 0.890 |
| Contentment | 0.771 | 0.755 | 0.787 |
| Shame | 0.761 | 0.713 | 0.816 |
| Gratitude | 0.759 | 0.735 | 0.785 |

## Data Sources
1. **GoEmotions** — Reddit comments with emotion labels (Google Research)
2. **Emotion Dataset v2** — Multi-class emotion corpus
3. **Daylio Export** — Mood tracking app data with temporal context
4. **Student Journal Entries** — Custom collected journal data
5. **HuggingFace Parquet datasets** — Additional emotion-labeled data
6. **Additional CSV datasets** — Supplementary emotion corpora

## Running

### Training (on Kaggle/GPU)
```python
# Preprocessing
python preprocessing.py

# Training (requires GPU, ~4 hours per fold)
python training.py

# Testing
python testing.py
```

### Streamlit Demo
```bash
cd streamlit_app
streamlit run app.py
```

> **Note:** Trained model weights (~741MB × 5 folds) are not included. Reproduce by running the training pipeline.

## Connection to Habit Tracking
The `preprocessing.py` module extracts **habit keywords** alongside emotions, establishing the foundational link between emotional states and behavioral patterns. The habit tracking pipeline builds on this by providing fine-grained, span-level habit identification.
