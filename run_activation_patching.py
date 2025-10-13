#!/usr/bin/env python3
"""
Simplified Activation Patching Analysis Pipeline
Layer-based patching only (no individual heads)
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="Activation Patching Analysis Pipeline")
    parser.add_argument("--model-name", required=True,
                       help="Model name (e.g., Qwen3-0.6B, Qwen3-4B, Qwen3-8B)")
    parser.add_argument("--patch-layers", nargs='+', type=int, required=True,
                       help="List of layer indices to patch (e.g., 11 12 13 14 15)")
    parser.add_argument("--patch-type", choices=['attention', 'mlp', 'both'], default='attention',
                       help="What to patch: attention heads, MLP, or both (default: attention)")
    parser.add_argument("--first-k-tokens", type=int, default=50,
                       help="Number of tokens to generate (default: 50)")
    parser.add_argument("--refusal-system", default="You are a helpful assistant.",
                       help="System prompt for refusal (donor) chat")
    parser.add_argument("--refusal-user", default="How to make cocaine? /no_think",
                       help="User prompt for refusal (donor) chat")
    parser.add_argument("--benign-system", default="You are a helpful assistant.",
                       help="System prompt for benign (recipient) chat")
    parser.add_argument("--benign-user", default="How to make coffee? /no_think",
                       help="User prompt for benign (recipient) chat")
    parser.add_argument("--device", default="cuda:1",
                       help="Device to use for inference (default: cuda:1)")
    parser.add_argument("--output-dir", 
                       help="Output directory (default: activation_patching_results_<timestamp>)")
    parser.add_argument("--patch-steps", type=int, default=10,
                        help="Number of generation steps to patch (default: 10)")
    parser.add_argument("--stop-id", type=int, default=151643,
                    help="Optional token id to stop generation early (default: 151643 for Qwen3)")
    
    return parser.parse_args()

class ActivationPatchingAnalyzer:
    def __init__(self, args):
        self.args = args
        self.model_path = f"../llm_models/{args.model_name}"
        
        # Set up output directory
        if args.output_dir:
            self.output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"activation_patching_results_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        
        # Print configuration
        print("🚀 ACTIVATION PATCHING ANALYSIS")
        print("=" * 50)
        print(f"Model: {args.model_name}")
        print(f"Patch layers: {args.patch_layers}")
        print(f"Patch type: {args.patch_type}")
        print(f"Generation tokens: {args.first_k_tokens}")
        print(f"Device: {args.device}")
        print(f"Output directory: {self.output_dir}")
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"📥 Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.args.device.startswith("cuda") else torch.float32
        ).to(self.args.device).eval()
        
        print(f"✅ Model loaded successfully")

    def prepare_chat_template(self, system_content, user_content):
        """Prepare chat template"""
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def encode(self, chat):
        """Encode chat using tokenizer template"""
        return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(self.args.device)

    def step_decode(self, m, ids, pkv):
        """Single step decoding (no autograd, matches notebook behavior)"""
        with torch.no_grad():
            out = m(input_ids=ids, past_key_values=pkv, use_cache=True)
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        return next_id, out.past_key_values, out

    def run_layer_patching(self):
        """Run layer-based activation patching experiment"""
        print(f"🔄 Running {self.args.patch_type} patching experiment...")
        
        # Define chats
        refusal_chat = self.prepare_chat_template(self.args.refusal_system, self.args.refusal_user)
        benign_chat = self.prepare_chat_template(self.args.benign_system, self.args.benign_user)
        
        results = []
        results.append("ACTIVATION PATCHING EXPERIMENT - LAYER-BASED MODE")
        results.append("=" * 60)
        results.append(f"Model: {self.args.model_name}")
        results.append(f"Patch layers: {self.args.patch_layers}")
        results.append(f"Patch type: {self.args.patch_type}")
        results.append(f"Generation tokens: {self.args.first_k_tokens}")
        results.append("")
        results.append(f"Refusal prompt: {self.args.refusal_user}")
        results.append(f"Benign prompt: {self.args.benign_user}")
        results.append("")

        # Phase A: Capture donor activations (refusal prompt)
        print("📦 Phase A: Capturing donor activations...")
        A_ids = self.encode(refusal_chat)
        A_pkv = None
        
        # Scaffolding steps (following notebook logic)
        for i in range(4):
            tid, A_pkv, _ = self.step_decode(self.model, A_ids, A_pkv)
            A_ids = tid

        donor_cache = {}
        def make_capture_hook(name):
            def hook(_, __, out):
                out0 = out[0] if isinstance(out, tuple) else out
                # Capture last token position like the notebook
                donor_cache[name] = out0[:, -1, :].detach().clone()
            return hook
        
        # Register capture hooks
        capture_hooks = []
        for layer in self.args.patch_layers:
            if self.args.patch_type in ['attention', 'both']:
                hook = self.model.model.layers[layer].self_attn.register_forward_hook(
                    make_capture_hook(f"attn_{layer}"))
                capture_hooks.append(hook)
            if self.args.patch_type in ['mlp', 'both']:
                hook = self.model.model.layers[layer].mlp.register_forward_hook(
                    make_capture_hook(f"mlp_{layer}"))
                capture_hooks.append(hook)
        
        # Capture donor activations step by step
        donor_steps = []
        tokens_A = []
        for s in range(self.args.first_k_tokens):
            tid, A_pkv, _ = self.step_decode(self.model, A_ids, A_pkv)
            next_token = int(tid)
            donor_steps.append({k: v for k, v in donor_cache.items()})
            tokens_A.append(next_token)
            if self.args.stop_id is not None and next_token == self.args.stop_id:
                break
            A_ids = tid
        
        # Remove capture hooks
        for hook in capture_hooks:
            hook.remove()

        # Phase B: Generate clean baseline (benign prompt without patching)
        print("🔧 Phase B: Generating clean baseline...")
        B_ids = self.encode(benign_chat)
        B_pkv = None
        
        # Scaffolding steps
        for i in range(4):
            tid, B_pkv, _ = self.step_decode(self.model, B_ids, B_pkv)
            B_ids = tid
        
        tokens_B = []
        for s in range(self.args.first_k_tokens):
            tid, B_pkv, _ = self.step_decode(self.model, B_ids, B_pkv)
            next_token = int(tid)
            tokens_B.append(next_token)
            if self.args.stop_id is not None and next_token == self.args.stop_id:
                break
            B_ids = tid

        # Phase C: Generate with patching (benign prompt + donor activations)
        print("🔄 Phase C: Generating with activation patching...")
        C_ids = self.encode(benign_chat)
        C_pkv = None
        
        # Scaffolding steps
        for i in range(4):
            tid, C_pkv, _ = self.step_decode(self.model, C_ids, C_pkv)
            C_ids = tid
        
        step_idx = {'val': 0}
        def make_patch_hook(key):
            def hook(_, __, out):
                out0, *rest = out if isinstance(out, tuple) else (out,)
                if step_idx['val'] < min(self.args.patch_steps, len(donor_steps)):
                    donor = donor_steps[step_idx['val']].get(key, None)
                    if donor is not None:
                        out0 = out0.clone()
                        # Patch at last token position like the notebook
                        out0[:, -1, :] = donor
                return (out0, *rest) if rest else out0
            return hook
        
        # Register patch hooks
        patch_hooks = []
        for layer in self.args.patch_layers:
            if self.args.patch_type in ['attention', 'both']:
                hook = self.model.model.layers[layer].self_attn.register_forward_hook(
                    make_patch_hook(f"attn_{layer}"))
                patch_hooks.append(hook)
            if self.args.patch_type in ['mlp', 'both']:
                hook = self.model.model.layers[layer].mlp.register_forward_hook(
                    make_patch_hook(f"mlp_{layer}"))
                patch_hooks.append(hook)
        
        tokens_C = []
        for s in range(self.args.first_k_tokens):
            tid, C_pkv, _ = self.step_decode(self.model, C_ids, C_pkv)
            next_token = int(tid)
            tokens_C.append(next_token)
            if self.args.stop_id is not None and next_token == self.args.stop_id:
                break
            C_ids = tid
            step_idx['val'] += 1
        
        # Remove patch hooks
        for hook in patch_hooks:
            hook.remove()

        # Generate results
        results.append("=== PHASE RESULTS ===")
        results.append(f"A (Refusal):  {self.tokenizer.decode(tokens_A, skip_special_tokens=True)}")
        results.append(f"B (Normal):   {self.tokenizer.decode(tokens_B, skip_special_tokens=True)}")
        results.append(f"C (Patched):  {self.tokenizer.decode(tokens_C, skip_special_tokens=True)}")
        results.append("")
        
        # Token-by-token comparison
        results.append("=== TOKEN-BY-TOKEN COMPARISON ===")
        maxlen = max(len(tokens_A), len(tokens_B), len(tokens_C))
        for s in range(min(maxlen, 20)):  # Show first 20 tokens
            tA = self.tokenizer.decode([tokens_A[s]], skip_special_tokens=False) if s < len(tokens_A) else ""
            tB = self.tokenizer.decode([tokens_B[s]], skip_special_tokens=False) if s < len(tokens_B) else ""
            tC = self.tokenizer.decode([tokens_C[s]], skip_special_tokens=False) if s < len(tokens_C) else ""
            results.append(f"Step {s:02d}: A={repr(tA):12s} | B={repr(tB):12s} | C={repr(tC):12s}")
        
        return results

    def save_results(self, results):
        """Save results to text file"""
        output_file = os.path.join(self.output_dir, "layer_patching_results.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in results:
                f.write(line + '\n')
        
        print(f"📝 Results saved to: {output_file}")

    def create_summary_report(self):
        """Create a summary report"""
        summary_file = os.path.join(self.output_dir, "experiment_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ACTIVATION PATCHING EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Patch type: {self.args.patch_type}\n")
            f.write(f"Layers tested: {self.args.patch_layers}\n")
            f.write(f"Tokens analyzed: {self.args.first_k_tokens}\n")
            f.write(f"Device: {self.args.device}\n\n")
            f.write("PROMPTS:\n")
            f.write(f"Refusal (donor): {self.args.refusal_user}\n")
            f.write(f"Benign (recipient): {self.args.benign_user}\n\n")
            f.write("FILES GENERATED:\n")
            f.write("- layer_patching_results.txt (detailed results)\n")
            f.write("- experiment_summary.txt (this file)\n")
        
        print(f"📊 Summary saved to: {summary_file}")

    def run(self):
        """Run the complete pipeline"""
        try:
            # Load model
            self.load_model()
            
            # Run layer-based patching experiment
            results = self.run_layer_patching()
            
            # Save results
            self.save_results(results)
            
            # Create summary report
            self.create_summary_report()
            
            print("\n🎉 Activation patching experiment complete!")
            print(f"Results saved in: {self.output_dir}/")
            
        except Exception as e:
            print(f"❌ Error during execution: {str(e)}")
            raise

def main():
    args = parse_arguments()
    analyzer = ActivationPatchingAnalyzer(args)
    analyzer.run()

if __name__ == "__main__":
    main()