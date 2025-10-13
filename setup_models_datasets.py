#!/usr/bin/env python3
"""
Model and Dataset Setup for LLM Safety Research
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Missing packages: {e}")
    print("Run setup_environment.sh first")
    sys.exit(1)

MODELS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-4B": "Qwen/Qwen3-4B", 
    "Qwen3-8B": "Qwen/Qwen3-8B"
}

DATASETS = {
    "PKU-Alignment/PKU-SafeRLHF-prompt": "train",
    "tatsu-lab/alpaca": "train"
}

def setup_directories():
    """Create required directories"""
    Path("../llm_models").mkdir(exist_ok=True)
    Path("../dataset_folder").mkdir(exist_ok=True)
    print("✅ Directories created")

def download_model(model_name):
    """Download model"""
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    model_path = f"../llm_models/{model_name}"
    
    if Path(model_path).exists():
        print(f"✅ Model {model_name} already exists")
        return True
    
    print(f"📥 Downloading {model_name}...")
    
    try:
        hf_name = MODELS[model_name]
        
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(hf_name, trust_remote_code=True, torch_dtype=torch.float16)
        
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"✅ {model_name} downloaded")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def download_datasets():
    """Download datasets"""
    for dataset_name, split in DATASETS.items():
        dataset_path = f"../dataset_folder/{dataset_name}"
        
        if Path(dataset_path).exists():
            print(f"✅ Dataset {dataset_name} already exists")
            continue
            
        print(f"📥 Downloading {dataset_name}...")
        
        try:
            dataset = load_dataset(dataset_name, split=split)
            dataset.save_to_disk(dataset_path)
            print(f"✅ {dataset_name} downloaded ({len(dataset)} samples)")
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download models and datasets for LLM safety research")
    parser.add_argument("--include-8b", action="store_true", 
                       help="Also download the large 8B model (requires ~20GB VRAM)")
    args = parser.parse_args()
    
    print("🚀 Setting up models and datasets...")
    
    setup_directories()
    
    # Download required models (0.6B and 4B)
    required_models = ["Qwen3-0.6B", "Qwen3-4B"]
    optional_models = ["Qwen3-8B"] if args.include_8b else []
    all_models = required_models + optional_models
    
    print(f"📥 Downloading {len(all_models)} models...")
    for model in all_models:
        print(f"  - {model}")
    
    if args.include_8b:
        print("\n⚠️  Note: 8B model requires significant disk space (~15GB) and VRAM (~20GB)")
    
    # Download all specified models
    success = True
    for model_name in all_models:
        if not download_model(model_name):
            success = False
    
    # Download datasets
    if not download_datasets():
        success = False
    
    if success:
        print("\n🎉 Setup complete!")
        print(f"✅ Downloaded {len(all_models)} models and datasets")
    else:
        print("\n❌ Setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()