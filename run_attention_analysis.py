#!/usr/bin/env python3
"""
Attention Visualization Analysis - CLI Script
Automated pipeline for comparing attention patterns between different prompts
Based on attention_visualization.ipynb - EXACT functionality match
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import csv
import glob
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser(description="Attention Visualization Analysis Pipeline")
    parser.add_argument("--model-name", required=True,
                       help="Model name (e.g., Qwen3-0.6B, Qwen3-4B, Qwen3-8B)")
    parser.add_argument("--system-prompt-a", default="You are a helpful assistant.",
                       help="System prompt for comparison A")
    parser.add_argument("--user-prompt-a", required=True,
                       help="User prompt for comparison A")
    parser.add_argument("--system-prompt-b", default="You are a helpful assistant.",
                       help="System prompt for comparison B")
    parser.add_argument("--user-prompt-b", required=True,
                       help="User prompt for comparison B")
    parser.add_argument("--output-dir", default="attention_analysis_results",
                       help="Output directory for results")
    parser.add_argument("--comparison-name", default="prompt_comparison",
                       help="Name for this comparison (used in file names)")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                       help="Maximum new tokens to generate")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use")
    
    return parser.parse_args()

# ============================================================================
# EXACT NOTEBOOK FUNCTIONS - Copy from attention_visualization.ipynb
# ============================================================================

class AttentionVisualizationPipeline:
    def __init__(self, args):
        self.args = args
        self.model_path = f"../llm_models/{args.model_name}"
        self.output_base = args.output_dir
        self.comparison_dir = f"{self.output_base}/{args.model_name}/{args.comparison_name}"
        
        # Create output directories
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        print(f"🚀 Attention Visualization Analysis Pipeline")
        print(f"Model: {args.model_name}")
        print(f"Device: {args.device}")
        print(f"Output directory: {self.comparison_dir}")
        
    def load_model(self):
        """Load model and tokenizer with attention capture enabled"""
        print(f"📥 Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.args.device.startswith("cuda") else torch.float32,
            device_map="auto" if self.args.device.startswith("cuda") else None,
            attn_implementation="eager",   # critical: return attention weights
        ).eval()
        
        self.model.config.output_attentions = True
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Model loaded successfully")
        
    def generate_text(self, user_prompt, system_prompt="You are a helpful assistant."):
        """Generate text while capturing attention weights"""
        # Create chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template or fallback
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.args.device)
        
        # Generate with attention
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=0.7,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return outputs, generated_text, inputs.input_ids[0]

    def get_attention_matrix(self, outputs, layer_idx=-1):
        """Get attention matrix from model outputs"""
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            raise ValueError("No attentions found in outputs (attentions=None).")

        step_mats = []
        for step_attn in outputs.attentions:
            if step_attn is None:
                continue
            mat = step_attn[layer_idx] if isinstance(step_attn, (list, tuple)) else step_attn
            # mat: (1, H, q_len, kv_len)
            mat = mat.squeeze(0)            # (H, q_len, kv_len)
            mat = mat.mean(dim=0)           # (q_len, kv_len)
            step_mats.append(mat.cpu().numpy())

        if not step_mats:
            raise ValueError("All attention steps are None. Ensure attn_implementation='eager' and output_attentions=True.")

        I = step_mats[0].shape[0]
        N = len(step_mats) - 1
        L = I + N
        full = np.zeros((L, L), dtype=step_mats[0].dtype)
        full[:I, :I] = step_mats[0]
        for k, mat in enumerate(step_mats[1:], start=1):
            q_len, kv_len = mat.shape
            assert q_len == 1, "Each generation step must have q_len=1"
            row_idx = I + k - 1
            full[row_idx, :kv_len] = mat[0]
        return full

    def find_token_spans(self, tokens):
        """Locate the three <|im_start|> markers and define spans"""
        start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_id   = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        starts = [i for i, t in enumerate(tokens.tolist()) if t == start_id]
        ends   = [i for i, t in enumerate(tokens.tolist()) if t == end_id]
        if len(starts) != 3:
            raise ValueError(f"Expected 3 <|im_start|> markers, got {len(starts)}")

        spans = {}
        roles = ["system", "user", "assistant"]
        for idx, role in enumerate(roles):
            s = starts[idx] + 1
            e = ends[idx] if idx < len(ends) else len(tokens)
            spans[role] = (s, e)
        return spans

    def calculate_attention_scores(self, attn, spans):
        """Compute both standard and GCG‐style scores"""
        s0, s1 = spans["system"]
        u0, u1 = spans["user"]
        a0, a1 = spans["assistant"]

        sys_to_user = attn[s0:s1, u0:u1].mean()
        user_to_sys = attn[u0:u1, s0:s1].mean()
        user_self  = attn[u0:u1, u0:u1].mean()
        sys_self   = attn[s0:s1, s0:s1].mean()

        # GCG: proportion of assistant attention on user vs system
        if a1 > a0:
            block = attn[a0:a1, :]
            total = block.sum()
            gcg_user   = block[:, u0:u1].sum() / total if total > 0 else 0.0
            gcg_system = block[:, s0:s1].sum() / total if total > 0 else 0.0
        else:
            gcg_user = gcg_system = 0.0

        return {
            "system_to_user":        float(sys_to_user),
            "user_to_system":        float(user_to_sys),
            "user_self_attention":   float(user_self),
            "system_self_attention": float(sys_self),
            "gcg_user_attention":    float(gcg_user),
            "gcg_system_attention":  float(gcg_system),
        }

    def find_important_tokens(self, attn, spans, tokens, span_name, top_k=10):
        """Rank tokens in the given span by total attention received"""
        start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_id   = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        s, e = spans[span_name]
        col_sums = attn[:, s:e].sum(axis=0)

        # filter out any markers
        candidates = [
            (float(col_sums[i]), i)
            for i in range(e - s)
            if tokens[s + i].item() not in (start_id, end_id)
        ]
        top = sorted(candidates, key=lambda x: x[0], reverse=True)[:top_k]

        result = []
        for score, rel in top:
            gid = s + rel
            txt = self.tokenizer.decode([int(tokens[gid])])
            result.append((txt, score, gid))
        return result

    def analyze_prompt(self, system_prompt, user_prompt, prompt_type="A"):
        """Run analysis on a single prompt"""
        print(f"\n=== {prompt_type} Prompt Analysis ===")
        print(f"System prompt: {system_prompt}")
        print(f"User prompt:   {user_prompt[:100]}...")

        # generate + raw attentions
        outputs, gen_text, tokens = self.generate_text(user_prompt, system_prompt)

        # full attention matrix
        attn = self.get_attention_matrix(outputs)

        # original prompt spans
        spans = self.find_token_spans(tokens)
        # extend assistant span through all generated tokens
        spans["assistant"] = (spans["assistant"][0], attn.shape[0])

        # compute scores
        scores = self.calculate_attention_scores(attn, spans)
        top_user   = self.find_important_tokens(attn, spans, tokens, "user")
        top_system = self.find_important_tokens(attn, spans, tokens, "system")

        # print results
        print("Standard Attention Scores:")
        print(f"  System→User: {scores['system_to_user']:.4f}")
        print(f"  User→System: {scores['user_to_system']:.4f}")
        print(f"  User self:   {scores['user_self_attention']:.4f}")
        print(f"  Sys self:    {scores['system_self_attention']:.4f}")

        print("\nGCG-Style Attention Scores:")
        print(f"  Assistant→User proportion:   {scores['gcg_user_attention']:.4f}")
        print(f"  Assistant→System proportion: {scores['gcg_system_attention']:.4f}")

        print("\nTop 10 user tokens by attention received:")
        for i, (tok, score, idx) in enumerate(top_user, 1):
            print(f"  {i:2d}. '{tok}' (score {score:.4f}, idx {idx})")

        print("\nTop 10 system tokens by attention received:")
        for i, (tok, score, idx) in enumerate(top_system, 1):
            print(f"  {i:2d}. '{tok}' (score {score:.4f}, idx {idx})")

        return {
            'attention_matrix':   attn,
            'spans':              spans,
            'scores':             scores,
            'top_user_tokens':    top_user,
            'top_system_tokens':  top_system,
            'tokens':             tokens,
            'generated_text':     gen_text
        }

    def compare_prompts(self):
        """Compare two prompt pairs A vs B"""
        data_a = self.analyze_prompt(self.args.system_prompt_a, self.args.user_prompt_a, prompt_type="A")
        data_b = self.analyze_prompt(self.args.system_prompt_b, self.args.user_prompt_b, prompt_type="B")

        # Standard & GCG comparisons
        s2u_a = data_a['scores']['system_to_user']
        u2s_a = data_a['scores']['user_to_system']
        s2u_b = data_b['scores']['system_to_user']
        u2s_b = data_b['scores']['user_to_system']

        g_u_a = data_a['scores']['gcg_user_attention']
        g_s_a = data_a['scores']['gcg_system_attention']
        g_u_b = data_b['scores']['gcg_user_attention']
        g_s_b = data_b['scores']['gcg_system_attention']

        print("\n" + "="*60)
        print("COMPARISON A vs B")
        print("="*60)

        print("📊 Standard Attention Scores:")
        print(f"  System→User: A={s2u_a:.4f}, B={s2u_b:.4f}, Δ={s2u_b-s2u_a:.4f}")
        print(f"  User→System: A={u2s_a:.4f}, B={u2s_b:.4f}, Δ={u2s_b-u2s_a:.4f}")

        print("\n🎯 GCG-Style Attention Scores:")
        print(f"  Assistant→User:   A={g_u_a:.4f}, B={g_u_b:.4f}, Δ={g_u_b-g_u_a:.4f}")
        print(f"  Assistant→System: A={g_s_a:.4f}, B={g_s_b:.4f}, Δ={g_s_b-g_s_a:.4f}")

        # Top tokens comparison
        print("\n📝 Top User Tokens A vs B:")
        for i, ((toka, sa, _), (tokb, sb, _)) in enumerate(zip(data_a['top_user_tokens'], data_b['top_user_tokens']), 1):
            print(f"  {i:2d}. A='{toka}'({sa:.4f})  B='{tokb}'({sb:.4f})")

        print("\n📝 Top System Tokens A vs B:")
        for i, ((toka, sa, _), (tokb, sb, _)) in enumerate(zip(data_a['top_system_tokens'], data_b['top_system_tokens']), 1):
            print(f"  {i:2d}. A='{toka}'({sa:.4f})  B='{tokb}'({sb:.4f})")

        return data_a, data_b

    def save_csv_data(self, data, label, span_name):
        """Save token-level attention scores to CSV"""
        start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        A = data["attention_matrix"]
        spans = data["spans"]
        ids = data["tokens"]
        
        s, e = spans[span_name]
        std_recv = A[:, s:e].sum(axis=0)
        
        a0, a1 = spans["assistant"]
        if a1 > a0:
            asst_recv = A[a0:a1, s:e].sum(axis=0)
            total_asst = A[a0:a1, :].sum()
        else:
            asst_recv = np.zeros(e - s)
            total_asst = 0.0
            
        path = os.path.join(self.comparison_dir, f"{span_name}_scores_{label}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "standard_attention", "assistant_attention", "gcg_proportion"])
            
            for rel in range(e - s):
                tok_id = ids[s + rel].item()
                if tok_id in (start_id, end_id):
                    continue
                txt = self.tokenizer.decode([tok_id])
                std = std_recv[rel]
                assn = asst_recv[rel]
                prop = (assn / total_asst) if total_asst > 0 else 0.0
                writer.writerow([txt, f"{std:.6f}", f"{assn:.6f}", f"{prop:.6f}"])
        
        print(f"→ Wrote {span_name} scores ({label}) to {path}")
        
    def _dump_span_scores_raw(self, data, label, span_name):
        A       = data["attention_matrix"]
        spans   = data["spans"]
        tokens  = data["tokens"]

        s0, s1  = spans[span_name]
        a0, a1  = spans["assistant"]

        L              = A.shape[0]
        n_asst_rows    = a1 - a0
        std_span_sum   = A[:, s0:s1].sum()
        asst_span_sum  = A[a0:a1, s0:s1].sum()

        std_recv   = A[:,  s0:s1].sum(axis=0)
        asst_recv  = A[a0:a1, s0:s1].sum(axis=0)

        start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_id   = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        path = os.path.join(self.comparison_dir, f"{span_name}_token_scores_{label}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([f"# total_rows={L}",
                        f"assistant_rows={n_asst_rows}",
                        f"span_std_sum={float(std_span_sum):.6f}",
                        f"span_asst_sum={float(asst_span_sum):.6f}"])
            w.writerow(["global_idx","token","standard_sum","assistant_sum",
                        "assistant_prop","std_per_row","asst_per_asstrow"])

            for rel in range(s1 - s0):
                gid     = s0 + rel
                tok_id  = int(tokens[gid])
                if tok_id in (start_id, end_id):
                    continue
                txt   = self.tokenizer.decode([tok_id])
                std   = float(std_recv[rel])
                asst  = float(asst_recv[rel])
                prop  = asst / float(asst_span_sum) if asst_span_sum > 0 else 0.0
                w.writerow([gid, txt,
                            f"{std:.6f}", f"{asst:.6f}",
                            f"{prop:.6f}",
                            f"{std/L:.6f}",
                            f"{asst/n_asst_rows:.6f}" if n_asst_rows else 0.0])

        print(f"→ wrote raw scores for '{span_name}' ({label}) to {path}")

    def create_visualizations(self, data_a, data_b):
        """Create all visualizations and save them"""
        print("📈 Creating visualizations...")
        
        for lbl, d in [("A", data_a), ("B", data_b)]:
            self._dump_span_scores_raw(d, lbl, "user")
            self._dump_span_scores_raw(d, lbl, "system")
        
        # Save CSV data
        for lbl, d in [("A", data_a), ("B", data_b)]:
            self.save_csv_data(d, lbl, "user")
            self.save_csv_data(d, lbl, "system")

        # Standard attention comparison
        cats = ["Sys→User", "User→Sys", "Sys Self", "User Self"]
        valsA = [data_a["scores"][k] for k in
                ["system_to_user", "user_to_system", "system_self_attention", "user_self_attention"]]
        valsB = [data_b["scores"][k] for k in
                ["system_to_user", "user_to_system", "system_self_attention", "user_self_attention"]]
        x = np.arange(len(cats))

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.bar(x-0.2, valsA, width=0.4, label="A", alpha=0.8)
        ax1.bar(x+0.2, valsB, width=0.4, label="B", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cats, rotation=45, ha="right")
        ax1.set_ylabel("Attention Score")
        ax1.set_title("Standard Attention Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(os.path.join(self.comparison_dir, "standard_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # GCG-style attention comparison
        gcg_cats = ["GCG User", "GCG System"]
        gcgA = [data_a["scores"]["gcg_user_attention"],
                data_a["scores"]["gcg_system_attention"]]
        gcgB = [data_b["scores"]["gcg_user_attention"],
                data_b["scores"]["gcg_system_attention"]]
        x2 = np.arange(len(gcg_cats))

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.bar(x2-0.2, gcgA, width=0.4, label="A", alpha=0.8)
        ax2.bar(x2+0.2, gcgB, width=0.4, label="B", alpha=0.8)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(gcg_cats)
        ax2.set_ylabel("Proportion")
        ax2.set_title("GCG-Style Attention Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(self.comparison_dir, "gcg_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # Delta standard attention
        deltas_std = [b - a for a, b in zip(valsA, valsB)]
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        colors = ["green" if d > 0 else "red" for d in deltas_std]
        ax3.bar(cats, deltas_std, color=colors, alpha=0.7)
        ax3.set_ylabel("Δ Attention (B−A)")
        ax3.set_title("Delta Standard Attention")
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        fig3.tight_layout()
        fig3.savefig(os.path.join(self.comparison_dir, "standard_delta.png"), dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # Delta GCG attention
        deltas_gcg = [b - a for a, b in zip(gcgA, gcgB)]
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        colors_gcg = ["green" if d > 0 else "red" for d in deltas_gcg]
        ax4.bar(gcg_cats, deltas_gcg, color=colors_gcg, alpha=0.7)
        ax4.set_ylabel("Δ Proportion (B−A)")
        ax4.set_title("Delta GCG Attention")
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(os.path.join(self.comparison_dir, "gcg_delta.png"), dpi=300, bbox_inches='tight')
        plt.close(fig4)

        print(f"   Saved comparison charts to {self.comparison_dir}")
        
        # Add the missing token line plots from the notebook
        self._plot_single_span_tok_attn(data_a, "user", os.path.join(self.comparison_dir, "token_line_user_A.png"))
        self._plot_single_span_tok_attn(data_a, "system", os.path.join(self.comparison_dir, "token_line_system_A.png"))
        self._plot_combined_tok_attn(data_a, os.path.join(self.comparison_dir, "token_line_combined_A.png"))
        
        self._plot_single_span_tok_attn(data_b, "user", os.path.join(self.comparison_dir, "token_line_user_B.png"))
        self._plot_single_span_tok_attn(data_b, "system", os.path.join(self.comparison_dir, "token_line_system_B.png"))
        self._plot_combined_tok_attn(data_b, os.path.join(self.comparison_dir, "token_line_combined_B.png"))
        
        print(f"   Saved token line plots to {self.comparison_dir}")

    def _plot_single_span_tok_attn(self, data, span_name="user", save_path=None):
        """Line graph for tokens in the requested span - EXACT from notebook"""
        A = data["attention_matrix"]
        spans = data["spans"]
        ids = data["tokens"]

        # row/col ranges
        p0, p1 = spans[span_name]
        a0, a1 = spans["assistant"]

        # per-token aggregates
        std_recv = A[:, p0:p1].sum(axis=0)
        asst_block = A[a0:a1, p0:p1]
        tot_asst = asst_block.sum()
        gcg_recv = asst_block.sum(axis=0) / tot_asst if tot_asst > 0 else np.zeros_like(std_recv)

        # strip chat-template markers
        start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        toks, std_y, gcg_y = [], [], []
        for rel in range(p1 - p0):
            tok_id = int(ids[p0 + rel].item())
            if tok_id in (start_id, end_id):
                continue
            toks.append(self.tokenizer.decode([tok_id]))
            std_y.append(float(std_recv[rel]))
            gcg_y.append(float(gcg_recv[rel]))

        # plot
        x = range(len(toks))
        plt.figure(figsize=(max(6, 0.6 * len(toks)), 3.2))
        plt.plot(x, std_y, marker="o", label="Standard")
        plt.plot(x, gcg_y, marker="o", label="GCG prop.")
        plt.xticks(x, toks, rotation=45, ha="right")
        plt.ylabel("Attention")
        plt.title(f"{span_name.capitalize()} tokens")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    def _plot_combined_tok_attn(self, data, save_path=None):
        """Combined user + system token plot - EXACT from notebook"""
        A = data["attention_matrix"]
        spans = data["spans"]
        ids = data["tokens"]

        u0, u1 = spans["user"]
        s0, s1 = spans["system"]
        a0, a1 = spans["assistant"]

        start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        records = []  # (global_idx, span_tag, std_attn, gcg_attn, tok_text)

        def _collect(tag, p0, p1):
            std = A[:, p0:p1].sum(axis=0)
            blk = A[a0:a1, p0:p1]
            tot = blk.sum()
            gcg = blk.sum(axis=0) / tot if tot > 0 else np.zeros_like(std)
            for rel in range(p1 - p0):
                gid = p0 + rel
                tok_id = int(ids[gid].item())
                if tok_id in (start_id, end_id):
                    continue
                records.append((gid, tag, float(std[rel]), float(gcg[rel]),
                                self.tokenizer.decode([tok_id])))

        _collect("user", u0, u1)
        _collect("system", s0, s1)

        # sort by appearance in prompt
        records.sort(key=lambda r: r[0])

        x = range(len(records))
        labels = [r[4] for r in records]
        usr_std = [r[2] if r[1] == "user" else np.nan for r in records]
        sys_std = [r[2] if r[1] == "system" else np.nan for r in records]
        usr_gcg = [r[3] if r[1] == "user" else np.nan for r in records]
        sys_gcg = [r[3] if r[1] == "system" else np.nan for r in records]

        plt.figure(figsize=(max(8, 0.6 * len(labels)), 4))
        plt.plot(x, usr_std, marker="o", label="User-Standard")
        plt.plot(x, sys_std, marker="o", label="System-Standard")
        plt.plot(x, usr_gcg, marker="x", linestyle="--", label="User-GCG")
        plt.plot(x, sys_gcg, marker="x", linestyle="--", label="System-GCG")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Attention")
        plt.title("User + System tokens")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    def create_summary_report(self, data_a, data_b):
        """Create a summary report"""
        report_path = os.path.join(self.comparison_dir, "analysis_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("ATTENTION VISUALIZATION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Comparison: {self.args.comparison_name}\n")
            f.write(f"Max new tokens: {self.args.max_new_tokens}\n")
            f.write(f"Device: {self.args.device}\n\n")
            
            f.write("PROMPT A:\n")
            f.write(f"  System: {self.args.system_prompt_a}\n")
            f.write(f"  User: {self.args.user_prompt_a[:100]}...\n\n")
            
            f.write("PROMPT B:\n")
            f.write(f"  System: {self.args.system_prompt_b}\n")
            f.write(f"  User: {self.args.user_prompt_b[:100]}...\n\n")
            
            f.write("STANDARD ATTENTION SCORES:\n")
            for key in ["system_to_user", "user_to_system", "system_self_attention", "user_self_attention"]:
                f.write(f"  {key}: A={data_a['scores'][key]:.4f}, B={data_b['scores'][key]:.4f}, Δ={data_b['scores'][key]-data_a['scores'][key]:.4f}\n")
            
            f.write("\nGCG ATTENTION SCORES:\n")
            for key in ["gcg_user_attention", "gcg_system_attention"]:
                f.write(f"  {key}: A={data_a['scores'][key]:.4f}, B={data_b['scores'][key]:.4f}, Δ={data_b['scores'][key]-data_a['scores'][key]:.4f}\n")
            
            f.write("\nFILES GENERATED:\n")
            f.write(f"- Standard comparison: standard_comparison.png\n")
            f.write(f"- GCG comparison: gcg_comparison.png\n")
            f.write(f"- Standard delta: standard_delta.png\n")
            f.write(f"- GCG delta: gcg_delta.png\n")
            f.write(f"- CSV data: user_scores_A.csv, user_scores_B.csv, system_scores_A.csv, system_scores_B.csv\n")
            
        print(f"📝 Summary report saved → {report_path}")

    def run(self):
        """Run the complete pipeline"""
        try:
            # Load model
            self.load_model()
            
            # Compare prompts
            data_a, data_b = self.compare_prompts()
            
            # Create visualizations
            self.create_visualizations(data_a, data_b)
            
            # Create summary report
            self.create_summary_report(data_a, data_b)
            
            print("\n🎉 Analysis complete!")
            print(f"Results saved in: {self.comparison_dir}/")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not Path(f"../llm_models/{args.model_name}").exists():
        print(f"❌ Model not found: ../llm_models/{args.model_name}")
        print("Run setup_models_datasets.py first to download the model")
        sys.exit(1)
    
    # Run pipeline
    pipeline = AttentionVisualizationPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()