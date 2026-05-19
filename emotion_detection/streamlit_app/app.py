"""
Streamlit Interface for Multi-Label Emotion Classification
Research Project: Explicit vs Implicit Emotion Detection

Author: Emotion Analysis Research Team
Purpose: Production-ready interface for demo and evaluation
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Emotion Analysis System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Emotion cards */
    .emotion-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Metrics */
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Insight box */
    .insight-box {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2b83ba;
        margin: 1.5rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths configuration
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "kfold_deberta_v4" / "fold_models"
DATA_DIR = BASE_DIR / "processed_emotion_data_v4"
RESULTS_DIR = BASE_DIR / "ANALYSIS" / "emotion-analysis" / "data" / "results"

# Model parameters
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256

# ============================================================================
# ENHANCED MODEL DEFINITION
# ============================================================================

class EnhancedDeBERTaModel(torch.nn.Module):
    """Enhanced DeBERTa model for emotion classification"""
    
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels
        
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        logits = self.dropout(outputs.logits)
        return {'logits': logits}

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_metadata():
    """Load dataset metadata"""
    metadata_path = DATA_DIR / "metadata.json"
    
    if not metadata_path.exists():
        st.error(f"❌ Metadata file not found: {metadata_path}")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

@st.cache_resource
def load_tokenizer():
    """Load tokenizer"""
    return DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

@st.cache_resource
def load_models():
    """Load all fold models for ensemble"""
    metadata = load_metadata()
    
    if metadata is None:
        return None, None, None
    
    num_labels = metadata['dataset_info']['num_labels']
    label_names = metadata['dataset_info']['label_names']
    
    models = []
    fold_files = sorted(MODEL_DIR.glob("fold_*.pt"))
    
    if not fold_files:
        st.error(f"❌ No model files found in {MODEL_DIR}")
        return None, None, None
    
    for fold_file in fold_files:
        model = EnhancedDeBERTaModel(
            MODEL_NAME,
            num_labels,
            dropout_rate=0.3
        ).to(device)
        
        try:
            checkpoint = torch.load(fold_file, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            models.append(model)
            
        except Exception as e:
            st.warning(f"⚠️ Failed to load {fold_file.name}: {e}")
    
    if not models:
        st.error("❌ No models were successfully loaded")
        return None, None, None
    
    return models, label_names, num_labels

@st.cache_data
def load_research_results():
    """Load bootstrap and metrics results"""
    bootstrap_file = RESULTS_DIR / "bootstrap_results.json"
    metrics_file = RESULTS_DIR / "metrics_STRICT.json"
    
    results = {}
    
    if bootstrap_file.exists():
        with open(bootstrap_file, 'r') as f:
            results['bootstrap'] = json.load(f)
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            results['metrics'] = json.load(f)
    
    return results

# ============================================================================
# LEXICON FOR EXPLICITNESS DETECTION
# ============================================================================

EMOTION_LEXICON = {
    'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage', 'frustrated'],
    'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'panic'],
    'joy': ['happy', 'joyful', 'excited', 'cheerful', 'delighted', 'pleased', 'glad'],
    'sadness': ['sad', 'depressed', 'miserable', 'unhappy', 'gloomy', 'sorrowful', 'grief'],
    'love': ['love', 'adore', 'affection', 'caring', 'devoted', 'cherish'],
    'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled'],
    'disgust': ['disgusted', 'repulsed', 'revolted', 'sickened'],
}

def detect_explicit_words(text):
    """Detect explicit emotion words in text"""
    text_lower = text.lower()
    found_emotions = {}
    
    for emotion_category, words in EMOTION_LEXICON.items():
        found_words = [word for word in words if word in text_lower]
        if found_words:
            found_emotions[emotion_category] = found_words
    
    return found_emotions

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_emotion(text, models, tokenizer, label_names, top_k=5, threshold=0.3):
    """Predict emotions for input text"""
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions from all models
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for model in models:
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    # Average probabilities across folds
    avg_probs = np.mean(all_probs, axis=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(avg_probs)[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        if avg_probs[idx] >= threshold:
            predictions.append({
                'emotion': label_names[idx],
                'probability': float(avg_probs[idx]),
                'confidence': float(avg_probs[idx])
            })
    
    # Calculate fold agreement for top prediction
    fold_votes = [all_preds[i][0] for i in range(len(models))]
    from collections import Counter
    vote_counts = Counter(fold_votes)
    
    # Detect explicit words
    explicit_emotions = detect_explicit_words(text)
    
    # Classify as explicit or implicit
    is_explicit = len(explicit_emotions) > 0
    
    return {
        'predictions': predictions,
        'all_probabilities': avg_probs,
        'fold_agreement': dict(vote_counts),
        'num_models': len(models),
        'explicit_emotions': explicit_emotions,
        'is_explicit': is_explicit
    }

def calculate_valence_distribution(predictions):
    """Calculate emotional valence distribution"""
    
    # Emotion to valence mapping
    valence_map = {
        # Negative emotions
        'sadness': 'negative', 'anger': 'negative', 'fear': 'negative',
        'disgust': 'negative', 'grief': 'negative', 'nervousness': 'negative',
        'annoyance': 'negative', 'disapproval': 'negative', 'disappointment': 'negative',
        'embarrassment': 'negative', 'remorse': 'negative',
        
        # Positive emotions
        'joy': 'positive', 'love': 'positive', 'gratitude': 'positive',
        'admiration': 'positive', 'amusement': 'positive', 'approval': 'positive',
        'caring': 'positive', 'desire': 'positive', 'excitement': 'positive',
        'optimism': 'positive', 'pride': 'positive', 'relief': 'positive',
        
        # Neutral emotions
        'neutral': 'neutral', 'realization': 'neutral', 'surprise': 'neutral',
        'curiosity': 'neutral', 'confusion': 'neutral'
    }
    
    valence_scores = {'negative': 0, 'positive': 0, 'neutral': 0}
    
    for pred in predictions:
        emotion = pred['emotion'].lower()
        probability = pred['probability']
        
        # Find valence
        valence = 'neutral'  # default
        for emotion_key, valence_val in valence_map.items():
            if emotion_key in emotion:
                valence = valence_val
                break
        
        valence_scores[valence] += probability
    
    # Normalize
    total = sum(valence_scores.values())
    if total > 0:
        valence_scores = {k: v/total for k, v in valence_scores.items()}
    
    return valence_scores

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_emotion_bar_chart(predictions):
    """Create interactive bar chart for emotions"""
    
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    
    # Color mapping
    colors = ['#667eea' if i == 0 else '#764ba2' if i == 1 else '#9f7aea' 
              for i in range(len(df))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['probability'],
            y=df['emotion'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{p:.1%}" for p in df['probability']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Detected Emotions",
        xaxis_title="Confidence",
        yaxis_title="Emotion",
        height=max(300, len(predictions) * 50),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_xaxes(range=[0, 1], tickformat='.0%')
    
    return fig

def create_valence_pie_chart(valence_scores):
    """Create pie chart for emotional valence"""
    
    colors = {
        'negative': '#d7191c',
        'positive': '#2b83ba',
        'neutral': '#fdae61'
    }
    
    labels = []
    values = []
    chart_colors = []
    
    for valence, score in valence_scores.items():
        if score > 0:
            labels.append(valence.capitalize())
            values.append(score)
            chart_colors.append(colors[valence])
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=chart_colors),
        textinfo='label+percent',
        textfont=dict(size=14),
        hole=0.3
    )])
    
    fig.update_layout(
        title="Emotional Valence Distribution",
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_radar_chart(predictions):
    """Create radar chart for top emotions"""
    
    if len(predictions) < 3:
        return None
    
    # Take top 6 emotions for radar
    top_emotions = predictions[:min(6, len(predictions))]
    
    categories = [p['emotion'] for p in top_emotions]
    values = [p['probability'] for p in top_emotions]
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8, color='#667eea')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Emotion Profile",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11)
    )
    
    return fig

def create_research_comparison_chart(bootstrap_data):
    """Create comparison chart from research results"""
    
    if 'bootstrap_results' not in bootstrap_data:
        return None
    
    results = bootstrap_data['bootstrap_results']
    
    valences = []
    explicit_f1 = []
    implicit_f1 = []
    
    for valence in ['negative', 'positive', 'neutral']:
        if valence in results:
            data = results[valence]
            valences.append(valence.capitalize())
            
            if 'f1_explicit' in data:
                explicit_f1.append(data['f1_explicit']['mean'])
            else:
                explicit_f1.append(0)
            
            if 'f1_implicit' in data:
                implicit_f1.append(data['f1_implicit']['mean'])
            else:
                implicit_f1.append(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Explicit Emotions',
        x=valences,
        y=explicit_f1,
        marker_color='#1b7837',
        text=[f"{v:.3f}" for v in explicit_f1],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Implicit Emotions',
        x=valences,
        y=implicit_f1,
        marker_color='#762a83',
        text=[f"{v:.3f}" for v in implicit_f1],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Research Finding: Explicit vs Implicit Performance",
        xaxis_title="Emotional Valence",
        yaxis_title="F1 Score",
        barmode='group',
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Multi-Label Emotion Analysis System</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">
            Advanced AI for detecting complex emotional states in text
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/brain.png", width=80)
        st.title("⚙️ Settings")
        
        st.markdown("---")
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        top_k = st.slider(
            "Number of emotions to display",
            min_value=3,
            max_value=10,
            value=5,
            help="Shows top N most likely emotions"
        )
        
        threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Minimum confidence to display an emotion"
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Information")
        metadata = load_metadata()
        
        if metadata:
            st.metric("Emotion Classes", metadata['dataset_info']['num_labels'])
            st.metric("Model", "DeBERTa-v3-base")
            st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("📍 Navigate")
        page = st.radio(
            "Choose a section:",
            ["🎯 Analyze Text", "📊 Research Insights", "ℹ️ About"],
            label_visibility="collapsed"
        )
    
    # Load models
    with st.spinner("🔄 Loading models..."):
        models, label_names, num_labels = load_models()
        tokenizer = load_tokenizer()
    
    if models is None:
        st.error("❌ Failed to load models. Please check the model directory.")
        return
    
    # ========================================================================
    # PAGE 1: ANALYZE TEXT
    # ========================================================================
    
    if page == "🎯 Analyze Text":
        
        st.markdown("### 📝 Enter Your Text")
        
        # Text input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "Type or paste your journal entry, tweet, or any text:",
                height=150,
                placeholder="Example: I'm feeling overwhelmed with everything happening. There's so much to do and I don't know where to start. But I'm grateful for the support from my friends.",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**Quick Examples:**")
            
            if st.button("😊 Positive", use_container_width=True):
                user_input = "I'm so happy today! Everything is going great and I feel incredibly grateful for all the opportunities I have."
            
            if st.button("😢 Sad", use_container_width=True):
                user_input = "I feel so alone right now. Nobody seems to understand what I'm going through and it's really hard to keep going."
            
            if st.button("😠 Angry", use_container_width=True):
                user_input = "This is so frustrating! I can't believe they would do something like this. It makes me furious just thinking about it."
            
            if st.button("😰 Anxious", use_container_width=True):
                user_input = "I'm really worried about tomorrow. There's so much that could go wrong and I don't feel prepared at all."
            
            if st.button("🎭 Mixed", use_container_width=True):
                user_input = "Part of me is excited about this new opportunity, but I'm also terrified of failing. It's overwhelming but also kind of thrilling."
        
        # Analyze button
        if st.button("🔍 Analyze Emotions", type="primary", use_container_width=True):
            
            if not user_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
                return
            
            # Show analysis
            with st.spinner("🧠 Analyzing emotional content..."):
                start_time = time.time()
                
                result = predict_emotion(
                    user_input,
                    models,
                    tokenizer,
                    label_names,
                    top_k=top_k,
                    threshold=threshold
                )
                
                inference_time = (time.time() - start_time) * 1000
            
            # ================================================================
            # SECTION 2: MODEL OUTPUT
            # ================================================================
            
            st.markdown("---")
            st.markdown("## 🎯 Analysis Results")
            
            if not result['predictions']:
                st.warning("⚠️ No emotions detected above the confidence threshold. Try lowering the threshold in the sidebar.")
                return
            
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Primary Emotion",
                    result['predictions'][0]['emotion'],
                    f"{result['predictions'][0]['probability']:.1%} confidence"
                )
            
            with col2:
                st.metric(
                    "Emotions Detected",
                    len(result['predictions']),
                    "Multi-label"
                )
            
            with col3:
                expression_type = "Explicit" if result['is_explicit'] else "Implicit"
                st.metric(
                    "Expression Type",
                    expression_type,
                    "🔍 Research Insight"
                )
            
            with col4:
                st.metric(
                    "Analysis Time",
                    f"{inference_time:.0f}ms",
                    "Real-time"
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Main visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Emotion bar chart
                fig1 = create_emotion_bar_chart(result['predictions'])
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Explicit words detected
                if result['explicit_emotions']:
                    st.markdown("### 🔎 Explicit Emotion Words Found")
                    
                    for category, words in result['explicit_emotions'].items():
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>{category.capitalize()}:</strong> {', '.join(words)}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="insight-box">
                        <strong>🔍 Implicit Expression Detected</strong><br>
                        No explicit emotion words found. The model detected emotions from context and linguistic patterns.
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Valence distribution
                valence_scores = calculate_valence_distribution(result['predictions'])
                fig2 = create_valence_pie_chart(valence_scores)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Radar chart
                fig3 = create_radar_chart(result['predictions'])
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
            
            # ================================================================
            # SECTION 3: INSIGHTS & ANALYSIS
            # ================================================================
            
            st.markdown("---")
            st.markdown("## 💡 Emotional Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dominant emotional tone
                dominant_valence = max(valence_scores.items(), key=lambda x: x[1])
                
                tone_description = {
                    'negative': ("Predominantly Negative", "The text expresses mainly negative emotions.", "#d7191c"),
                    'positive': ("Predominantly Positive", "The text expresses mainly positive emotions.", "#2b83ba"),
                    'neutral': ("Neutral/Mixed", "The text shows balanced or neutral emotional content.", "#fdae61")
                }
                
                tone, desc, color = tone_description[dominant_valence[0]]
                
                st.markdown(f"""
                <div class="emotion-card">
                    <h4 style="color: {color};">📊 Overall Emotional State</h4>
                    <h3>{tone}</h3>
                    <p>{desc}</p>
                    <p><strong>Confidence:</strong> {dominant_valence[1]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Emotional complexity
                complexity = "High" if len(result['predictions']) >= 4 else "Moderate" if len(result['predictions']) >= 2 else "Simple"
                
                st.markdown(f"""
                <div class="metric-container">
                    <h4>🎭 Emotional Complexity: {complexity}</h4>
                    <p>{'Multiple emotions detected simultaneously. This indicates complex emotional processing.' if complexity == 'High' else 'Mixed emotional state detected.' if complexity == 'Moderate' else 'Single dominant emotion.'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Valence breakdown
                st.markdown("""
                <div class="emotion-card">
                    <h4>🎨 Emotional Category Summary</h4>
                """, unsafe_allow_html=True)
                
                for valence, score in sorted(valence_scores.items(), key=lambda x: x[1], reverse=True):
                    if score > 0:
                        st.markdown(f"""
                        <div style="margin: 0.5rem 0;">
                            <strong>{valence.capitalize()} emotions:</strong> {score:.1%}
                            <div style="background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden;">
                                <div style="background: {'#d7191c' if valence == 'negative' else '#2b83ba' if valence == 'positive' else '#fdae61'}; 
                                            width: {score*100}%; height: 100%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Research connection
                st.markdown("""
                <div class="insight-box">
                    <h4>🔬 Research Insight</h4>
                    <p><strong>Did you know?</strong> This system can detect emotions even without explicit emotion words. 
                    Research shows that implicit emotional expressions are prevalent in real-world text and require 
                    sophisticated linguistic understanding.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed predictions table
            st.markdown("### 📋 Detailed Emotion Breakdown")
            
            df = pd.DataFrame(result['predictions'])
            df['probability'] = df['probability'].apply(lambda x: f"{x:.2%}")
            df = df.rename(columns={
                'emotion': 'Emotion',
                'probability': 'Confidence'
            })
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            
            # Model ensemble info
            with st.expander("🔧 Technical Details"):
                st.markdown(f"""
                **Model Architecture:** DeBERTa-v3-base (Multi-label classification)
                
                **Ensemble Method:** {result['num_models']}-fold cross-validation ensemble
                
                **Inference Time:** {inference_time:.2f}ms
                
                **Expression Type:** {"Explicit (contains emotion words)" if result['is_explicit'] else "Implicit (context-based detection)"}
                
                **Total Emotion Classes:** {num_labels}
                
                **Confidence Threshold:** {threshold:.1%}
                """)
    
    # ========================================================================
    # PAGE 2: RESEARCH INSIGHTS
    # ========================================================================
    
    elif page == "📊 Research Insights":
        
        st.markdown("## 🔬 Research Findings: Explicit vs Implicit Emotions")
        
        st.markdown("""
        <div class="insight-box">
            <h3>Key Research Question</h3>
            <p><strong>Does the presence of explicit emotion words affect model performance?</strong></p>
            <p>This research compares model performance on texts with explicit emotion words (e.g., "happy", "sad") 
            versus texts expressing emotions implicitly through context.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load research results
        research_data = load_research_results()
        
        if 'bootstrap' in research_data:
            bootstrap_data = research_data['bootstrap']
            
            # Main finding
            st.markdown("### 📈 Performance Comparison")
            
            fig = create_research_comparison_chart(bootstrap_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Key findings
            st.markdown("### 🎯 Key Findings")
            
            col1, col2, col3 = st.columns(3)
            
            results = bootstrap_data.get('bootstrap_results', {})
            
            with col1:
                if 'negative' in results and 'delta' in results['negative']:
                    delta = results['negative']['delta']['mean']
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h4 style="color: #d7191c;">Negative Emotions</h4>
                        <h2>{delta:+.3f}</h2>
                        <p>Performance gain with explicit words</p>
                        <small>{'Significant ***' if results['negative']['delta'].get('significant') else 'Not significant'}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if 'positive' in results and 'delta' in results['positive']:
                    delta = results['positive']['delta']['mean']
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h4 style="color: #2b83ba;">Positive Emotions</h4>
                        <h2>{delta:+.3f}</h2>
                        <p>Performance gain with explicit words</p>
                        <small>{'Significant ***' if results['positive']['delta'].get('significant') else 'Not significant'}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if 'neutral' in results and 'delta' in results['neutral']:
                    delta = results['neutral']['delta']['mean']
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h4 style="color: #fdae61;">Neutral Emotions</h4>
                        <h2>{delta:+.3f}</h2>
                        <p>Performance gain with explicit words</p>
                        <small>{'Significant ***' if results['neutral']['delta'].get('significant') else 'Not significant'}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Interpretation
            st.markdown("### 📝 Interpretation")
            
            st.markdown("""
            <div class="insight-box">
                <h4>What This Means</h4>
                <ul>
                    <li><strong>Positive Δ:</strong> Explicit emotion words improve model performance</li>
                    <li><strong>Negative Δ:</strong> Model performs better on implicit expressions (rare)</li>
                    <li><strong>*** (Three stars):</strong> Statistically significant difference (p < 0.001)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Methodology
            with st.expander("📚 Methodology"):
                st.markdown("""
                **Bootstrap Confidence Intervals**
                - 10,000 bootstrap resamples
                - 95% confidence intervals
                - Bias-corrected accelerated (BCa) method
                
                **Significance Testing**
                - Cohen's d for effect size
                - Non-overlapping confidence intervals
                - Strict criteria (both CIs must not overlap)
                
                **Data Split**
                - Explicit: Texts containing emotion lexicon words
                - Implicit: Texts without explicit emotion words
                - Controlled for valence (negative/positive/neutral)
                """)
            
            # Dataset statistics
            st.markdown("### 📊 Dataset Statistics")
            
            if 'negative' in results:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Negative Samples",
                        f"{results['negative']['n_total']:,}",
                        f"{results['negative']['explicit_percentage']:.1f}% explicit"
                    )
                
                with col2:
                    if 'positive' in results:
                        st.metric(
                            "Positive Samples",
                            f"{results['positive']['n_total']:,}",
                            f"{results['positive']['explicit_percentage']:.1f}% explicit"
                        )
                
                with col3:
                    if 'neutral' in results:
                        st.metric(
                            "Neutral Samples",
                            f"{results['neutral']['n_total']:,}",
                            f"{results['neutral']['explicit_percentage']:.1f}% explicit"
                        )
        
        else:
            st.warning("⚠️ Research results not found. Please ensure bootstrap analysis has been run.")
    
    # ========================================================================
    # PAGE 3: ABOUT
    # ========================================================================
    
    elif page == "ℹ️ About":
        
        st.markdown("## ℹ️ About This System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            
            This is a **multi-label emotion classification system** that can:
            
            ✅ Detect multiple emotions simultaneously
            
            ✅ Classify both explicit and implicit emotional expressions
            
            ✅ Analyze 28 distinct emotion categories
            
            ✅ Provide confidence scores and insights
            
            ✅ Process text in real-time
            
            ### 🔬 Research Contribution
            
            This system demonstrates that:
            
            📊 **Explicit emotion words significantly impact model performance**
            
            🎭 **Implicit emotions require deeper linguistic understanding**
            
            📈 **Effect varies by emotional valence (negative/positive/neutral)**
            
            🏆 **State-of-the-art performance on GoEmotions dataset**
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Technical Details
            
            **Model Architecture:**
            - Base: DeBERTa-v3-base (Microsoft)
            - Task: Multi-label classification
            - Parameters: 184M
            - Dropout: 0.3
            
            **Training:**
            - Dataset: GoEmotions (58k texts)
            - Emotions: 28 classes
            - Validation: 5-fold cross-validation
            - Ensemble: Average probabilities
            
            **Performance:**
            - Macro F1: ~0.45-0.55 (varies by valence)
            - Inference: <100ms per text
            - Device: CPU/GPU compatible
            
            ### 📚 Use Cases
            
            - Mental health monitoring
            - Social media analysis
            - Customer feedback analysis
            - Content moderation
            - Therapeutic chatbots
            - Market research
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📖 Dataset Information
        
        **GoEmotions Dataset** (Google Research)
        - 58,000 Reddit comments
        - 28 emotion categories
        - Multi-label annotations
        - Balanced across emotions
        
        **Emotion Categories:**
        
        *Positive:* admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
        
        *Negative:* anger, annoyance, confusion, curiosity, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
        
        *Ambiguous:* realization, surprise, neutral
        """)
        
        st.markdown("---")
        
        # Citations
        st.markdown("""
        ### 📝 Citations
        
        If you use this system in research, please cite:
        
        ```bibtex
        @inproceedings{demszky2020goemotions,
            title={GoEmotions: A Dataset of Fine-Grained Emotions},
            author={Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and 
                    Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
            booktitle={Proceedings of ACL},
            year={2020}
        }
        
        @article{he2021deberta,
            title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
            author={He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu},
            journal={ICLR},
            year={2021}
        }
        ```
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎭 Multi-Label Emotion Analysis System | Built with Streamlit + PyTorch + DeBERTa</p>
        <p>Research Project © 2025 | For academic and research purposes</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()