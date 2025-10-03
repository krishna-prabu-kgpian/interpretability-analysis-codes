# LLM Safety Research - Mechanistic Analysis Notebooks

Three research notebooks for analyzing internal mechanisms of Large Language Models, specifically investigating how models process harmful content and implementing activation patching techniques.

## Notebooks

### `identify_refusal_location.ipynb`
Extracts model activations to identify where refusal mechanisms are located within transformer layers.

**Process:**
1. Load PKU-SafeRLHF (harmful) and Alpaca (harmless) datasets (20,000 samples each)
2. Extract activations from last 3 tokens during forward pass
3. Compute mean activations for harmful vs harmless categories
4. Create refusal vectors (harmful_mean - harmless_mean)
5. Test refusal vector injection effects on model behavior

**Outputs:** Activation files, mean vectors, refusal vectors, cosine similarity analysis

### `attention_visualization.ipynb`
Analyzes attention patterns when processing base prompts vs jailbreak attempts.

**Process:**
1. Generate text while capturing attention weights across all layers
2. Compare base prompt ("How to make cocaine?") vs jailbreak variants
3. Build attention matrices combining prompt + generated tokens
4. Compute standard and GCG-style attention metrics
5. Export token-level attention scores and generate visualizations

**Outputs:** Attention heatmaps, comparison charts, CSV files with per-token scores

### `attention_patching.ipynb`
Demonstrates activation patching to transfer refusal behaviors between prompts.

**Process:**
1. **Phase A:** Capture activations from harmful prompt during generation
2. **Phase B:** Generate baseline response from benign prompt
3. **Phase C:** Patch harmful activations into benign prompt generation
4. Compare three outputs: harmful, benign, and patched

**Outputs:** Side-by-side token comparisons, full text outputs showing behavior transfer

## Setup

**Requirements:**
- CUDA GPU (configured for `cuda:1`)
- Models in `../llm_models/Qwen3-0.6B/`
- Python packages: `torch`, `transformers`, `datasets`, `matplotlib`, `numpy`, `pandas`

**Installation:**
```bash
pip install torch transformers datasets matplotlib numpy pandas seaborn
```

## Usage

1. **Refusal Location:** Run extraction twice (harmful mode, then harmless mode) before analysis
2. **Attention Analysis:** Execute cells sequentially, modify prompts as needed
3. **Activation Patching:** Run all three phases to see behavior transfer effects

## Key Findings

These notebooks enable investigation of:
- Where safety mechanisms are encoded in transformer layers
- How attention patterns differ between safe and jailbreak prompts  
- Which model components causally control refusal behavior
- How activations can be transferred between different contexts

---