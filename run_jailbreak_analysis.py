#!/usr/bin/env python3
"""
Jailbreak Analysis on Refusal Location Pipeline
Analyzes cosine similarity of prompts against refusal vectors
"""

import argparse
import os
import math
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

NUM_LAST_TOKENS = 3

def parse_arguments():
    parser = argparse.ArgumentParser(description="Jailbreak Analysis on Refusal Location Pipeline")
    parser.add_argument("--model-name", required=True,
                       help="Model name (e.g., Qwen3-0.6B, Qwen3-4B, Qwen3-8B)")
    parser.add_argument("--system-prompt-1", default="You are a helpful assistant.",
                       help="System prompt for first prompt set")
    parser.add_argument("--user-prompt-1", required=True,
                       help="User prompt for first analysis")
    parser.add_argument("--system-prompt-2", default="You are a helpful assistant.",
                       help="System prompt for second prompt set")
    parser.add_argument("--user-prompt-2", required=True,
                       help="User prompt for second analysis")
    parser.add_argument("--device", default="cuda:1",
                       help="Device to use for inference (default: cuda:1)")
    parser.add_argument("--num-tokens", type=int, default=3,
                       help="Number of last tokens to analyze (default: 3)")
    parser.add_argument("--artifacts-dir", required=True,
                   help="Root directory where the previous pipeline saved artifacts "
                        "(the same --output-dir you used there)")
    parser.add_argument("--refusal-vector-path",
                    help="Optional explicit path to refusal_vector.pt "
                            "(overrides --artifacts-dir/<MODEL_NAME>/refusal_vector.pt)")
    parser.add_argument("--output-dir",
                    help="Output directory for THIS tool's results "
                            "(default: jailbreak_analysis_results_<timestamp>)")

    return parser.parse_args()

class JailbreakAnalyzer:
    def __init__(self, args):
        self.args = args
        self.model_path = f"../llm_models/{args.model_name}"
        self.artifacts_root = args.artifacts_dir
        self.artifacts_model_dir = os.path.join(self.artifacts_root, self.args.model_name)
        if args.refusal_vector_path:
            self.refusal_vector_path = args.refusal_vector_path
        else:
            self.refusal_vector_path = os.path.join(self.artifacts_model_dir, "refusal_vector.pt")
        # Set up output directory
        if args.output_dir:
            self.output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"jailbreak_analysis_results_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.refusal_vector = None
        
        print("🚀 JAILBREAK ANALYSIS ON REFUSAL LOCATION")
        print("=" * 50)
        print(f"Model: {args.model_name}")
        print(f"Artifacts root: {self.artifacts_root}")
        print(f"Model artifacts folder: {self.artifacts_model_dir}")
        print(f"Refusal vector: {self.refusal_vector_path}")
        print(f"Prompt 1: {args.user_prompt_1}")
        print(f"Prompt 2: {args.user_prompt_2}")
        print(f"Device: {args.device}")
        print(f"Output directory: {self.output_dir}")

    def load_model_and_vector(self):
        """Load model, tokenizer and refusal vector"""
        print(f"📥 Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.args.device.startswith("cuda") else torch.float32
        ).to(self.args.device)
        self.model.eval()
        
        print(f"📊 Loading refusal vector from {self.refusal_vector_path}...")
        if not os.path.exists(self.refusal_vector_path):
            raise FileNotFoundError(f"refusal_vector.pt not found at: {self.refusal_vector_path}\n"
                                    f"Hint: ensure --artifacts-dir matches the previous pipeline's --output-dir.")
        self.refusal_vector = torch.load(self.refusal_vector_path)

        print(f"✅ Model and refusal vector loaded successfully")

    def build_inputs_from_chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Uses apply_chat_template with provided system and user prompts.
        Returns the rendered chat text (string).
        """
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        chat_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return chat_text

    def _make_layer_hook(self, residuals_by_layer: Dict[int, torch.Tensor], layer_idx: int, k: int):
        def hook_fn(module, inputs, outputs):
            with torch.no_grad():
                activation = outputs[0] if isinstance(outputs, tuple) else outputs  # [bs, seq, d]
                acts = activation.detach().float().cpu()
                seq_len = acts.size(1)
                t = min(seq_len, k)
                residuals_by_layer[layer_idx] = acts[0, -t:, :].clone()  # [t, d]
        return hook_fn

    def register_capture_hooks(self, k: int = NUM_LAST_TOKENS):
        """
        Registers forward hooks on each transformer layer to capture the last-k token activations.
        Returns (hooks, residuals_by_layer_dict).
        """
        residuals_by_layer: Dict[int, torch.Tensor] = {}
        hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            hooks.append(layer.register_forward_hook(self._make_layer_hook(residuals_by_layer, idx, k)))
        return hooks, residuals_by_layer

    def run_and_capture(self, chat_text: str):
        """
        Runs a no-grad forward pass on the given chat_text. Hooks must be registered beforehand.
        """
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.args.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        torch.cuda.empty_cache()

    def compute_cosine_table(self, residuals_by_layer: Dict[int, torch.Tensor], k: int = NUM_LAST_TOKENS) -> Tuple[np.ndarray, List[int]]:
        """
        Compute cosine similarity per layer × last-k positions.

        residuals_by_layer[layer] -> [T, d] (T <= k)
        refusal_vector[layer]     -> [k, d]  (your saved per-position vectors)

        Returns:
          sims_table: [num_layers, k] array (NaN where a position was unavailable)
          layer_order: list of layer indices aligned to sims_table rows
        """
        layer_order = sorted(set(residuals_by_layer.keys()).intersection(self.refusal_vector.keys()))
        if not layer_order:
            raise ValueError("No overlapping layers between captured activations and refusal vectors.")

        sims = np.full((len(layer_order), k), np.nan, dtype=np.float32)

        for r, layer_idx in enumerate(layer_order):
            R = residuals_by_layer[layer_idx].float()   # [T, d]
            V = self.refusal_vector[layer_idx].float()       # [k, d]
            T = R.size(0)
            # Compare trailing positions: R[-T:] with V[-T:]
            per_tok = F.cosine_similarity(R, V[-T:, :], dim=-1).cpu().numpy()  # [T]
            sims[r, k - T : k] = per_tok

        return sims, layer_order

    def plot_tokens_overlay(self, sims_table: np.ndarray, layer_order: List[int], 
                           title: str, filename: str, k: int = NUM_LAST_TOKENS):
        """
        Draw a single figure with lines for each token position over layers.
        Each line uses a different color and has a legend entry.
        """
        x = layer_order
        plt.figure(figsize=(12, 6))
        
        # columns: 0 -> pos-k, 1 -> pos-(k-1), ..., k-1 -> pos-1
        labels = [f"pos-{i}" for i in range(k, 0, -1)]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Support up to 6 positions
        markers = ['o', 's', '^', 'D', 'v', '<']
        
        for col, label in enumerate(labels):
            y = sims_table[:, col]
            color = colors[col % len(colors)]
            marker = markers[col % len(markers)]
            plt.plot(x, y, marker=marker, linewidth=2, label=label, color=color, markersize=6)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Layer index", fontsize=12)
        plt.ylabel("Cosine similarity", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Plot saved: {filename}")

    def analyze_prompt_cosine(self, system_prompt: str, user_prompt: str, k: int = NUM_LAST_TOKENS) -> Tuple[np.ndarray, List[int]]:
        """
        MASTER FUNCTION for single prompt analysis.
        - Uses apply_chat_template with provided system and user prompts.
        - Captures per-layer activations for the last-k token positions.
        - Computes cosine sims to the refusal vector.
        - Returns (sims_table [num_layers,k], layer_order [list[int]]).
        """
        chat_text = self.build_inputs_from_chat(system_prompt, user_prompt)
        hooks, residuals_by_layer = self.register_capture_hooks(k=k)

        try:
            self.run_and_capture(chat_text)
        finally:
            for h in hooks:
                h.remove()

        if not residuals_by_layer:
            raise RuntimeError("No activations captured. Check hook placement for your model.")

        sims_table, layer_order = self.compute_cosine_table(residuals_by_layer, k=k)
        return sims_table, layer_order

    def save_results(self, sims1: np.ndarray, layers1: List[int], sims2: np.ndarray, layers2: List[int]):
        """Save numerical results to files"""
        # Save prompt 1 results
        np.save(os.path.join(self.output_dir, "prompt1_cosine_similarities.npy"), sims1)
        np.save(os.path.join(self.output_dir, "prompt1_layer_order.npy"), np.array(layers1))
        
        # Save prompt 2 results
        np.save(os.path.join(self.output_dir, "prompt2_cosine_similarities.npy"), sims2)
        np.save(os.path.join(self.output_dir, "prompt2_layer_order.npy"), np.array(layers2))
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "analysis_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("JAILBREAK ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Refusal vector: {self.refusal_vector_path}\n")
            f.write(f"Device: {self.args.device}\n")
            f.write(f"Number of tokens analyzed: {self.args.num_tokens}\n\n")
            
            f.write("PROMPT SET 1:\n")
            f.write(f"System: {self.args.system_prompt_1}\n")
            f.write(f"User: {self.args.user_prompt_1}\n")
            f.write(f"Layers analyzed: {layers1}\n")
            f.write(f"Similarity shape: {sims1.shape}\n\n")
            
            f.write("PROMPT SET 2:\n")
            f.write(f"System: {self.args.system_prompt_2}\n")
            f.write(f"User: {self.args.user_prompt_2}\n")
            f.write(f"Layers analyzed: {layers2}\n")
            f.write(f"Similarity shape: {sims2.shape}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("- prompt1_cosine_plot.png (cosine similarity plot for prompt 1)\n")
            f.write("- prompt2_cosine_plot.png (cosine similarity plot for prompt 2)\n")
            f.write("- prompt1_cosine_similarities.npy (numerical data for prompt 1)\n")
            f.write("- prompt2_cosine_similarities.npy (numerical data for prompt 2)\n")
            f.write("- analysis_summary.txt (this file)\n")
        
        print(f"📝 Summary saved to: {summary_file}")

    def run(self):
        """Run the complete pipeline"""
        try:
            # Load model and refusal vector
            self.load_model_and_vector()
            
            # Analyze first prompt set
            print("\n🔍 Analyzing first prompt set...")
            sims1, layers1 = self.analyze_prompt_cosine(
                self.args.system_prompt_1, 
                self.args.user_prompt_1, 
                k=self.args.num_tokens
            )
            
            # Generate plot for first prompt
            title1 = f"Prompt 1: Cosine Similarity to Refusal Direction\n\"{self.args.user_prompt_1}\""
            self.plot_tokens_overlay(sims1, layers1, title1, "prompt1_cosine_plot.png", k=self.args.num_tokens)
            
            # Analyze second prompt set  
            print("\n🔍 Analyzing second prompt set...")
            sims2, layers2 = self.analyze_prompt_cosine(
                self.args.system_prompt_2,
                self.args.user_prompt_2,
                k=self.args.num_tokens
            )
            
            # Generate plot for second prompt
            title2 = f"Prompt 2: Cosine Similarity to Refusal Direction\n\"{self.args.user_prompt_2}\""
            self.plot_tokens_overlay(sims2, layers2, title2, "prompt2_cosine_plot.png", k=self.args.num_tokens)
            
            # Save results
            self.save_results(sims1, layers1, sims2, layers2)
            
            print("\n🎉 Jailbreak analysis complete!")
            print(f"Results saved in: {self.output_dir}/")
            print(f"📊 Generated plots: prompt1_cosine_plot.png, prompt2_cosine_plot.png")
            
        except Exception as e:
            print(f"❌ Error during execution: {str(e)}")
            raise

def main():
    args = parse_arguments()
    analyzer = JailbreakAnalyzer(args)
    analyzer.run()

if __name__ == "__main__":
    main()