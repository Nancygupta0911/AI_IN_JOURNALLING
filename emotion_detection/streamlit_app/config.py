"""
Configuration file for Emotion Analysis Streamlit App
Modify these paths to match your directory structure
"""

from pathlib import Path

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================

# Option 1: Auto-detect (recommended)
# The app will look for directories in the current working directory
BASE_DIR = Path(".")

# Option 2: Manual paths (uncomment and modify if needed)
# BASE_DIR = Path("/path/to/your/project")
# MODEL_DIR = Path("/custom/path/to/models")
# DATA_DIR = Path("/custom/path/to/data")
# RESULTS_DIR = Path("/custom/path/to/results")

# ============================================================================
# DEFAULT PATHS (modify if your structure is different)
# ============================================================================

# Model directory (contains fold_*.pt files)
MODEL_DIR = BASE_DIR / "kfold_deberta_v4" / "fold_models"

# Data directory (contains metadata.json)
DATA_DIR = BASE_DIR / "processed_emotion_data_v4"

# Results directory (contains bootstrap_results.json, metrics_STRICT.json)
RESULTS_DIR = BASE_DIR / "ANALYSIS" / "emotion-analysis" / "data" / "results"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256

# Number of models to load (set to 1 for faster demo, 5 for best accuracy)
NUM_FOLDS_TO_LOAD = 5  # Options: 1, 3, 5

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Default analysis parameters
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.3

# Color scheme (modify for different branding)
COLORS = {
    'negative': '#d7191c',      # Red
    'positive': '#2b83ba',      # Blue
    'neutral': '#fdae61',       # Orange
    'explicit': '#1b7837',      # Dark green
    'implicit': '#762a83'       # Purple
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Force CPU (set to True if you want to disable GPU)
FORCE_CPU = False

# Enable caching (recommended for production)
ENABLE_CACHING = True

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Show research insights page
SHOW_RESEARCH_PAGE = True

# Show technical details in expanders
SHOW_TECHNICAL_DETAILS = True

# Enable batch analysis from file
ENABLE_BATCH_UPLOAD = False  # Set to True to enable file upload feature

# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Validate that required paths exist"""
    errors = []
    
    if not MODEL_DIR.exists():
        errors.append(f"❌ Model directory not found: {MODEL_DIR}")
    
    if not DATA_DIR.exists():
        errors.append(f"❌ Data directory not found: {DATA_DIR}")
    
    metadata_file = DATA_DIR / "metadata.json"
    if not metadata_file.exists():
        errors.append(f"❌ Metadata file not found: {metadata_file}")
    
    fold_files = list(MODEL_DIR.glob("fold_*.pt")) if MODEL_DIR.exists() else []
    if not fold_files:
        errors.append(f"❌ No model files found in: {MODEL_DIR}")
    
    if SHOW_RESEARCH_PAGE and not RESULTS_DIR.exists():
        errors.append(f"⚠️ Research results directory not found: {RESULTS_DIR}")
        errors.append("   (Research Insights page will not be available)")
    
    return errors

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_models():
    """Get list of available model files"""
    if not MODEL_DIR.exists():
        return []
    
    fold_files = sorted(MODEL_DIR.glob("fold_*.pt"))
    return fold_files[:NUM_FOLDS_TO_LOAD]

def get_device():
    """Get the computation device"""
    import torch
    
    if FORCE_CPU:
        return torch.device('cpu')
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')