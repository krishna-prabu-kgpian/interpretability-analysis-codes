#!/usr/bin/env python3
"""
Refusal Direction Analysis - CLI Script
Automated pipeline for extracting refusal vectors from harmful vs harmless prompts
Based on identify_refusal_location.ipynb
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Refusal Direction Analysis Pipeline")
    parser.add_argument("--model-name", required=True, 
                       help="Model name (e.g., Qwen3-0.6B, Qwen3-4B, Qwen3-8B)")
    parser.add_argument("--target-size", type=int, default=20000,
                       help="Number of examples to process (default: 20000)")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Processing batch size (default: 512)")
    parser.add_argument("--num-last-tokens", type=int, default=3,
                       help="Number of last tokens to analyze (default: 3)")
    parser.add_argument("--device", default="cuda:1",
                       help="Device to use (default: cuda:1)")
    parser.add_argument("--output-dir", default="refusal_analysis_results",
                       help="Output directory for results (default: refusal_analysis_results)")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip activation extraction if files already exist")
    parser.add_argument("--force-recompute", action="store_true",
                       help="Force recomputation even if results exist")
    
    return parser.parse_args()

class RefusalAnalysisPipeline:
    def __init__(self, args):
        self.args = args
        self.model_path = self.model_path = f"../llm_models/{args.model_name}"
        self.output_base  = args.output_dir
        self.nb_model_dir = os.path.join(self.output_base, self.args.model_name)
        self.harmful_dir  = os.path.join(self.nb_model_dir, "harmful_prompt_activations")
        self.harmless_dir = os.path.join(self.nb_model_dir, "harmless_prompt_activations")

        # Create output directories
        os.makedirs(self.nb_model_dir,  exist_ok=True)
        os.makedirs(self.harmful_dir,   exist_ok=True)
        os.makedirs(self.harmless_dir,  exist_ok=True)

        print(f"🚀 Refusal Direction Analysis Pipeline")
        print(f"Model: {args.model_name}")
        print(f"Target size: {args.target_size:,}")
        print(f"Device: {args.device}")
        print(f"Output root: {self.output_base}")
        print(f"Model folder: {self.nb_model_dir}")
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"📥 Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.args.device)
        self.model.eval()
        
        print("✅ Model loaded successfully")
        
    def load_datasets(self):
        """Load harmful and harmless datasets"""
        print("📥 Loading datasets...")
        
        # Load harmful dataset
        harmful_dataset = load_from_disk("../dataset_folder/PKU-Alignment/PKU-SafeRLHF-prompt")
        if "train" in harmful_dataset:
            harmful_ds = harmful_dataset["train"]
        else:
            harmful_ds = harmful_dataset
        
        harmful_prompts = harmful_ds["prompt"]
        total_harmful = len(harmful_prompts)
        step_harmful = total_harmful // self.args.target_size
        self.harmful_prompts = harmful_prompts[::step_harmful][:self.args.target_size]
        
        # Load harmless dataset
        harmless_dataset = load_from_disk("../dataset_folder/tatsu-lab/alpaca")
        if "train" in harmless_dataset:
            harmless_ds = harmless_dataset["train"]
        else:
            harmless_ds = harmless_dataset
            
        harmless_prompts = harmless_ds["instruction"]
        total_harmless = len(harmless_prompts)
        step_harmless = total_harmless // self.args.target_size
        self.harmless_prompts = harmless_prompts[::step_harmless][:self.args.target_size]
        
        print(f"✅ Loaded {len(self.harmful_prompts):,} harmful prompts")
        print(f"✅ Loaded {len(self.harmless_prompts):,} harmless prompts")
        
    def setup_hooks(self):
        """Setup activation hooks for all layers"""
        self.residuals_by_layer = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, inputs, outputs):
                with torch.no_grad():
                    activation = outputs[0] if isinstance(outputs, tuple) else outputs
                    act_cpu = activation.detach().cpu()
                    seq_len = act_cpu.size(1)

                    if seq_len >= self.args.num_last_tokens:
                        self.residuals_by_layer[layer_idx] = act_cpu[0, -self.args.num_last_tokens:, :].clone()
                    else:
                        self.residuals_by_layer[layer_idx] = act_cpu[0, :, :].clone()
            return hook_fn

        self.hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            self.hooks.append(layer.register_forward_hook(make_hook(idx)))
            
    def remove_hooks(self):
        """Remove all hooks"""
        for h in self.hooks:
            h.remove()
            
    def extract_activations(self, prompts, output_dir, mode_name):
        """Extract activations for a set of prompts"""
        print(f"🔄 Extracting activations for {mode_name} prompts...")
        print(f"   Processing {len(prompts):,} prompts...")
        
        batch_data = []
        batch_counter = 0
        
        for idx, prompt in enumerate(prompts):
            if idx % 1000 == 0:
                print(f"   Progress: {idx:,}/{len(prompts):,} ({idx/len(prompts)*100:.1f}%)")
                
            self.residuals_by_layer.clear()
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            chat_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.args.device)

            with torch.no_grad():
                _ = self.model(**inputs)
                
            example_record = {
                "prompt": prompt,
                "residuals": {layer_idx: tensor for layer_idx, tensor in self.residuals_by_layer.items()}
            }
            batch_data.append(example_record)

            is_last_example = (idx + 1) == len(prompts)
            if (len(batch_data) >= self.args.batch_size) or is_last_example:
                batch_id = batch_counter
                save_path = os.path.join(output_dir, f"batch_{batch_id:04d}.pt")
                torch.save(batch_data, save_path)
                print(f"   Saved batch {batch_id} with {len(batch_data)} examples → {save_path}")
                batch_counter += 1
                batch_data = []

        print(f"✅ Finished extracting {mode_name} activations")
        
    def compute_means_for_directory(self, dir_path):
        """Compute mean activations for all files in directory"""
        means = {}        
        count = 0         
        
        filenames = sorted([f for f in os.listdir(dir_path) if f.endswith(".pt")])
        for fname in filenames:
            batch_path = os.path.join(dir_path, fname)
            batch_list = torch.load(batch_path) 

            for example in batch_list:
                count += 1
                residuals = example["residuals"] 

                if count == 1:
                    for layer_idx, tensor in residuals.items():
                        means[layer_idx] = tensor.float().clone()
                else:
                    for layer_idx, tensor in residuals.items():
                        x = tensor.float()  
                        old_mean = means[layer_idx]
                        means[layer_idx] = old_mean + (x - old_mean) / count

            del batch_list
            torch.cuda.empty_cache()

        return means
        
    def compute_statistics(self):
        """Compute mean activations and refusal vectors"""
        print("📊 Computing statistics...")
        
        # Compute means
        print("   Computing harmful means...")
        means_harmful = self.compute_means_for_directory(self.harmful_dir)
        harmful_mean_path   = os.path.join(self.nb_model_dir, "harmful_mean.pt")
        torch.save(means_harmful, harmful_mean_path)
        print(f"   Saved harmful means → {harmful_mean_path}")

        print("   Computing harmless means...")
        means_harmless = self.compute_means_for_directory(self.harmless_dir)
        harmless_mean_path  = os.path.join(self.nb_model_dir, "harmless_mean.pt")
        torch.save(means_harmless, harmless_mean_path)
        print(f"   Saved harmless means → {harmless_mean_path}")
        
        # Compute refusal vector
        print("   Computing refusal vector...")
        refusal_vector = {}
        for layer_idx in means_harmful:
            refusal_vector[layer_idx] = means_harmful[layer_idx] - means_harmless[layer_idx]

        refusal_vector_path = os.path.join(self.nb_model_dir, "refusal_vector.pt")
        torch.save(refusal_vector, refusal_vector_path)
        print(f"   Saved refusal vector → {refusal_vector_path}")
        
        return means_harmful, means_harmless, refusal_vector
        
    def create_visualizations(self, means_harmful, means_harmless):
        """Create cosine similarity visualization"""
        print("📈 Creating visualizations...")
        
        layer_indices = sorted(means_harmful.keys())
        cos_sims = {i: [] for i in range(self.args.num_last_tokens)}

        for layer_idx in layer_indices:
            v_h = means_harmful[layer_idx]
            v_safe = means_harmless[layer_idx]

            for token_pos in range(self.args.num_last_tokens):
                a = v_h[token_pos].unsqueeze(0)    
                b = v_safe[token_pos].unsqueeze(0) 
                cos_val = F.cosine_similarity(a, b, dim=1).item()
                cos_sims[token_pos].append(cos_val)

        # Create plot
        plt.figure(figsize=(12, 8))
        markers = ['o', '*', 'x']  # dot, star, cross for accessibility
        for token_pos in range(self.args.num_last_tokens):
            if self.args.num_last_tokens == 3:
                label = {
                    0: "third‐last token",
                    1: "second‐last token", 
                    2: "last token"
                }[token_pos]
            else:
                label = f"token position -{self.args.num_last_tokens - token_pos}"
            
            marker = markers[token_pos] if token_pos < len(markers) else 'o'
            plt.plot(layer_indices, cos_sims[token_pos], label=label, marker=marker, linewidth=2, markersize=8)

        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Cosine Similarity", fontsize=12)
        plt.title(f"Cosine Similarity Between Harmful & Harmless Mean Vectors\n"
                 f"Model: {self.args.model_name} | Samples: {self.args.target_size:,}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.nb_model_dir, "cosine_similarity_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved plot → {plot_path}")
        
        # Save data as CSV
        import pandas as pd
        data_rows = []
        for layer_idx in layer_indices:
            for token_pos in range(self.args.num_last_tokens):
                data_rows.append({
                    'layer_idx': layer_idx,
                    'token_position': token_pos,
                    'token_label': f"pos-{self.args.num_last_tokens - token_pos}",
                    'cosine_similarity': cos_sims[token_pos][layer_indices.index(layer_idx)]
                })
        
        df = pd.DataFrame(data_rows)
        csv_path  = os.path.join(self.nb_model_dir, "cosine_similarity_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"   Saved data → {csv_path}")
        
    def create_summary_report(self):
        """Create a summary report"""
        report_path = os.path.join(self.nb_model_dir, "analysis_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("REFUSAL DIRECTION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Target samples: {self.args.target_size:,}\n")
            f.write(f"Batch size: {self.args.batch_size}\n")
            f.write(f"Last tokens analyzed: {self.args.num_last_tokens}\n")
            f.write(f"Device: {self.args.device}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write(f"- Harmful activations: {self.args.model_name}/harmful_prompt_activations/\n")
            f.write(f"- Harmless activations: {self.args.model_name}/harmless_prompt_activations/\n")
            f.write(f"- Harmful mean: {self.args.model_name}/harmful_mean.pt\n")
            f.write(f"- Harmless mean: {self.args.model_name}/harmless_mean.pt\n")
            f.write(f"- Refusal vector: {self.args.model_name}/refusal_vector.pt\n")
            f.write(f"- Cosine similarity plot: {self.args.model_name}/cosine_similarity_analysis.png\n")
            f.write(f"- Cosine similarity data: {self.args.model_name}/cosine_similarity_data.csv\n")
            
        print(f"📝 Summary report saved → {report_path}")
        
    def run(self):
        """Run the complete pipeline"""
        try:
            # Load model and datasets
            self.load_model()
            self.load_datasets()
            
            # Check if we should skip extraction
            harmful_exists = len(list(Path(self.harmful_dir).glob("*.pt"))) > 0
            harmless_exists = len(list(Path(self.harmless_dir).glob("*.pt"))) > 0
            
            if not self.args.skip_extraction or not (harmful_exists and harmless_exists) or self.args.force_recompute:
                # Setup hooks and extract activations
                self.setup_hooks()
                
                # Extract harmful activations
                self.extract_activations(self.harmful_prompts, self.harmful_dir, "harmful")
                
                # Extract harmless activations  
                self.extract_activations(self.harmless_prompts, self.harmless_dir, "harmless")
                
                self.remove_hooks()
            else:
                print("⏭️ Skipping activation extraction (files exist)")
            
            # Compute statistics
            means_harmful, means_harmless, refusal_vector = self.compute_statistics()
            
            # Create visualizations
            self.create_visualizations(means_harmful, means_harmless)
            
            # Create summary report
            self.create_summary_report()
            
            print("\n🎉 Analysis complete!")
            print(f"Results saved in: {os.path.join(self.output_base, self.args.model_name)}/")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            if hasattr(self, 'hooks'):
                self.remove_hooks()
            raise

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not Path(f"../llm_models/{args.model_name}").exists():
        print(f"❌ Model not found: ../llm_models/{args.model_name}")
        print("Run setup_models_datasets.py first to download the model")
        sys.exit(1)
        
    if not Path("../dataset_folder/PKU-Alignment/PKU-SafeRLHF-prompt").exists():
        print("❌ Harmful dataset not found")
        print("Run setup_models_datasets.py first to download datasets")
        sys.exit(1)
        
    if not Path("../dataset_folder/tatsu-lab/alpaca").exists():
        print("❌ Harmless dataset not found")
        print("Run setup_models_datasets.py first to download datasets")
        sys.exit(1)
    
    # Run pipeline
    pipeline = RefusalAnalysisPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()