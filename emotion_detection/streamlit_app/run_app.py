#!/usr/bin/env python
"""
Startup script to validate environment and launch Streamlit app
Run this instead of 'streamlit run app.py' for better error handling
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_packages():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'plotly',
        'pandas',
        'numpy',
        'sklearn'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"Package installed: {package}")
        except ImportError:
            missing.append(package)
            print_error(f"Package missing: {package}")
    
    if missing:
        print("\n" + "-" * 80)
        print("Missing packages detected. Install them with:")
        print(f"  pip install -r requirements.txt")
        print("-" * 80)
        return False
    
    return True

def check_paths():
    """Check if required files and directories exist"""
    from config import validate_paths
    
    errors = validate_paths()
    
    if not errors:
        print_success("All required paths exist")
        return True
    
    has_critical_error = False
    
    for error in errors:
        if "❌" in error:
            print(error)
            has_critical_error = True
        else:
            print(error)
    
    if has_critical_error:
        print("\n" + "-" * 80)
        print("CRITICAL ERRORS FOUND")
        print("-" * 80)
        print("\nPlease fix the errors above before running the app.")
        print("See README_SETUP.md for detailed setup instructions.")
        print("-" * 80)
        return False
    
    return True

def check_model_files():
    """Check if model files are accessible"""
    from config import get_available_models
    
    models = get_available_models()
    
    if not models:
        print_error("No model files found")
        return False
    
    print_success(f"Found {len(models)} model file(s)")
    for model in models:
        print(f"  📦 {model.name}")
    
    return True

def check_device():
    """Check computation device"""
    import torch
    from config import get_device
    
    device = get_device()
    
    if device.type == 'cuda':
        print_success(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print_warning("GPU not available, using CPU (slower but works)")
    
    return True

def check_metadata():
    """Check if metadata file exists and is valid"""
    from config import DATA_DIR
    import json
    
    metadata_file = DATA_DIR / "metadata.json"
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        num_labels = metadata['dataset_info']['num_labels']
        print_success(f"Metadata loaded: {num_labels} emotion classes")
        return True
    
    except FileNotFoundError:
        print_error(f"Metadata file not found: {metadata_file}")
        return False
    except Exception as e:
        print_error(f"Error loading metadata: {e}")
        return False

def run_app():
    """Launch Streamlit app"""
    print("\n" + "=" * 80)
    print("  🚀 Launching Streamlit App...")
    print("=" * 80 + "\n")
    
    try:
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("  👋 App stopped by user")
        print("=" * 80)
    except Exception as e:
        print_error(f"Failed to launch app: {e}")
        sys.exit(1)

def main():
    """Main validation and launch sequence"""
    
    print_header("🎭 EMOTION ANALYSIS SYSTEM - STARTUP VALIDATION")
    
    print("📋 Step 1: Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Step 2: Checking required packages...")
    if not check_packages():
        sys.exit(1)
    
    print("\n📁 Step 3: Checking file paths...")
    if not check_paths():
        sys.exit(1)
    
    print("\n🤖 Step 4: Checking model files...")
    if not check_model_files():
        sys.exit(1)
    
    print("\n💾 Step 5: Checking metadata...")
    if not check_metadata():
        sys.exit(1)
    
    print("\n🖥️  Step 6: Checking computation device...")
    check_device()
    
    print("\n" + "=" * 80)
    print("  ✅ ALL CHECKS PASSED!")
    print("=" * 80)
    
    input("\nPress ENTER to launch the app... ")
    
    run_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Startup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)