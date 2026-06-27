# Plot2Code: Fine-Tuning a Vision-Language Model for Plot-to-Code Generation

**Author:** Dhruv Chandra

## Table of Contents

- [Overview](#overview)
- [Objective](#objective)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model & Fine-Tuning Configuration](#model--fine-tuning-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
  - [Training Loss Curve](#training-loss-curve)
  - [Zero-Shot vs Fine-Tuned Comparison](#zero-shot-vs-fine-tuned-comparison)
  - [What Improved (Strengths)](#what-improved-strengths)
  - [What Regressed (Weaknesses)](#what-regressed-weaknesses)
  - [Analysis & Takeaways](#analysis--takeaways)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

**Plot2Code** is a research project that fine-tunes a **vision-language model (VLM)** to automatically generate code from plot/chart images. Given a static image of a data visualization (e.g., a matplotlib scatter plot, bar chart, line graph, heatmap, etc.), the model generates either:

1. **Vega-Lite JSON specification** — a declarative grammar for interactive visualizations, or
2. **Executable Python (matplotlib) code** — that directly recreates the plot.

The project uses a dual-task instruction approach where the model learns to produce the appropriate output format based on the type of input plot.

---

## Objective

The core research question is:

> **Can a small, quantized vision-language model be fine-tuned on a modest dataset to reliably convert plot images into executable code or declarative specifications?**

This project explores the feasibility, measures improvements over zero-shot performance, and identifies remaining challenges.

---

## Architecture

### Base Model

- **Model:** [`Qwen3-VL-2B-Instruct`](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit) (2 billion parameters)
- **Quantization:** 4-bit quantization via `bitsandbytes` (BnB) for memory-efficient training
- **Framework:** [Unsloth](https://github.com/unslothai/unsloth) for fast vision model patching + [TRL](https://github.com/huggingface/trl) `SFTTrainer` for supervised fine-tuning

### Fine-Tuning Method: LoRA (Low-Rank Adaptation)

Instead of full fine-tuning (which would require far more compute), the project uses **LoRA** adapters:

| LoRA Parameter           | Value    |
|--------------------------|----------|
| Rank (`r`)               | 16       |
| Alpha (`lora_alpha`)     | 32       |
| Dropout (`lora_dropout`) | 0.05     |
| Bias                     | None     |
| `use_rslora`             | True     |
| Vision layers fine-tuned | ✅ Yes   |
| Language layers fine-tuned| ✅ Yes   |
| Attention modules        | ✅ Yes   |
| MLP modules              | ❌ No    |

### Dual-Task Instruction Design

The model receives one of two system prompts depending on the training sample:

- **Vega-Lite Task:** Image → Vega-Lite JSON specification (for supported chart types: line, scatter, bar, histogram, heatmap, boxplot, area, multi-line, grouped bar)
- **Matplotlib Task:** Image → Executable Python/matplotlib code (for complex plot types not easily represented in Vega-Lite, e.g., quiver, contour, 3D plots)

The task type is determined by whether the sample has a valid `vega_spec` field.

### Hardware

- **GPU:** 2× NVIDIA RTX A4000 (15.7 GB VRAM each)
- **Precision:** BFloat16
- **Platform:** Linux

---

## Dataset

### Data Source

Training data is derived from the official **matplotlib** and **seaborn** gallery examples:

- Python source files are scraped from the matplotlib/seaborn galleries
- Each script is executed to render a PNG plot image
- The `(image, code)` pairs form the base dataset
- A Vega-Lite specification is automatically generated for compatible plot types

### Data Augmentation

The `augment.py` module performs code-level augmentation to increase diversity:

- **Color mutations** (random from a safe color pool)
- **Marker style swaps**
- **Line width / alpha / line-style perturbations**
- **Colormap changes**
- **Matplotlib style variations**
- Each base example generates up to 5 augmented variants
- Augmented code is re-executed to produce new plot images
- Image quality checks (corruption, blank, near-blank, minimum size) filter bad samples

### Dataset Statistics

| Split    | Examples |
|----------|----------|
| Training | 1,870    |
| Test     | 82       |

### Data Format

Each sample is stored as a JSONL record with the following fields:

```json
{
  "image": "/path/to/plot_image.png",
  "code": "import matplotlib.pyplot as plt\n...\nplt.show()",
  "vega_spec": { "data": {...}, "mark": "bar", "encoding": {...} }
}
```

- `vega_spec` is `null` for plot types that don't map to Vega-Lite
- The training pipeline uses `vega_spec` presence to select the appropriate instruction

### Data Filtering

Before training, samples are filtered by:

- **Code character length:** Max 3,000 characters (for Python code samples; Vega-Lite samples are exempt)
- **Token length:** Enforced via processor tokenization to fit within `max_seq_length = 2048`

---

## Model & Fine-Tuning Configuration

| Hyperparameter                  | Value                     |
|---------------------------------|---------------------------|
| Max sequence length             | 2,048                     |
| Per-device train batch size     | 2                         |
| Per-device eval batch size      | 1                         |
| Gradient accumulation steps     | 4                         |
| Effective batch size            | 8                         |
| Number of epochs                | 6                         |
| Learning rate                   | 5e-6                      |
| LR scheduler                   | Cosine                    |
| Warmup ratio                   | 0.04                      |
| Max gradient norm               | 0.5                       |
| Optimizer                       | `paged_adamw_32bit`       |
| Weight decay                    | 0.01                      |
| Precision                       | BFloat16                  |
| Evaluation strategy             | Every 50 steps            |
| Checkpointing strategy          | Every 100 steps           |
| Save total limit                | 2                         |
| Best model selection metric     | `eval_loss` (lower is better) |
| Seed                            | 3407                      |
| `torch_compile`                 | Disabled                  |

---

## Evaluation Metrics

The evaluation pipeline uses a comprehensive suite of metrics comparing the generated output against the reference:

### Visual Metric

- **SSIM (Structural Similarity Index):** Compares the rendered plot image from generated code against the original reference image. Measures perceptual visual similarity (range: 0–1, higher is better).

### Code-Level Metrics

- **BLEU (Bilingual Evaluation Understudy):** Token-level overlap between generated and reference code using smoothed BLEU-4.
- **Edit Similarity:** Sequence-matcher-based edit distance ratio between the generated and reference code strings.
- **API Overlap:** Jaccard similarity of matplotlib API function calls used in the reference vs. generated code (e.g., `scatter`, `subplots`, `set_xlabel`).

### Composite Scores

- **Code Score:** `0.4 × BLEU + 0.3 × Edit Similarity + 0.3 × API Overlap`
- **Overall Score:** `0.5 × SSIM + 0.5 × Code Score`

### Vega-Lite IR Metrics (for Vega-Lite task only)

- **IR Exact Match:** Whether the generated spec exactly matches the reference
- **IR Field F1:** Precision/recall of Vega-Lite specification fields
- **IR Tree Similarity:** Structural tree similarity of the specification

### Execution Success Rate

- Percentage of generated code snippets that execute without errors and produce a valid plot image.

---

## Results

### Training Loss Curve

The model was trained for **900 steps** (6 epochs). The training loss shows strong convergence:

![Training Loss Curve](25_05/training_loss_curve.png)

| Metric             | Value                 |
|--------------------|-----------------------|
| Initial loss       | 2.0516                |
| Final loss         | 0.4631                |
| Minimum loss       | 0.3614 (at step 700)  |
| Loss reduction     | 1.5886 (77.4%)        |
| Total logged steps | 36                    |

The loss drops sharply in the first 200 steps (from ~2.05 to ~0.60), then gradually plateaus around ~0.43–0.46 with minor fluctuations. The polynomial trend line confirms healthy convergence without signs of divergence.

---

### Zero-Shot vs Fine-Tuned Comparison

The fine-tuned model was compared against the same base model in zero-shot mode on an 82-example test set:

![Comparison Chart](25_05/comparison_zeroshot_vs_finetuned.png)

| Metric                     | Zero-Shot        | Fine-Tuned       | Absolute Δ | Relative Δ |
|----------------------------|------------------|------------------|------------|------------|
| **SSIM (Visual)**          | 0.585 ± 0.167   | 0.634 ± 0.070   | **+0.048** | **+8.2%**  |
| **BLEU (Text)**            | 0.086 ± 0.067   | 0.075 ± 0.106   | −0.011     | −12.9%     |
| **Edit Similarity**        | 0.134 ± 0.111   | 0.132 ± 0.185   | −0.002     | −1.8%      |
| **API Overlap**            | 0.458 ± 0.264   | 0.668 ± 0.211   | **+0.210** | **+45.9%** |
| **Code Score**             | 0.212 ± 0.122   | 0.270 ± 0.132   | **+0.058** | **+27.3%** |
| **Overall Score**          | 0.399 ± 0.116   | 0.452 ± 0.064   | **+0.053** | **+13.3%** |
| **Execution Success Rate** | 50.0%            | 12.2%            | **−37.8%** | **−75.6%** |

---

### What Improved (Strengths) ✅

1. **API Overlap (+45.9%):** The most dramatic improvement. After fine-tuning, the model learned to use the *correct* matplotlib API calls (e.g., `scatter`, `bar`, `subplots`, `set_xlabel`) far more consistently. This means the model acquired a much better understanding of *which* plotting functions to use for different chart types.

2. **SSIM / Visual Similarity (+8.2%):** The rendered plots from the fine-tuned model are visually closer to the originals. Notably, the standard deviation dropped from ±0.167 to ±0.070, indicating **more consistent** output quality (less variance between good and bad attempts).

3. **Code Score (+27.3%):** The composite code quality metric improved substantially, driven primarily by the API overlap gains.

4. **Overall Score (+13.3%):** The combined visual + code metric improved, with significantly tighter confidence intervals (±0.064 vs ±0.116), showing the fine-tuned model is more predictable and reliable.

---

### What Regressed (Weaknesses) ❌

1. **Execution Success Rate (−75.6%):** This is the most significant regression. The fine-tuned model's generated code executes successfully only **12.2%** of the time compared to **50.0%** for zero-shot. This likely stems from:
   - The model learning to produce Vega-Lite JSON (which then requires a conversion step to matplotlib code), and truncation/malformation in the JSON specs
   - Truncated outputs due to the `max_seq_length = 2048` token limit cutting off longer code
   - The model sometimes producing syntactically incomplete code

2. **BLEU (−12.9%):** A slight decrease in token-level overlap. This is expected because the fine-tuned model generates code in a *different style* from the reference — it may use correct API calls but with different variable names, data structures, or code organization.

3. **Edit Similarity (−1.8%):** Essentially flat — the character-level similarity is roughly unchanged.

---

### Analysis & Takeaways

1. **The model learned *what to do* but struggles with *how to do it completely*.** It correctly identifies chart types and selects appropriate APIs, but often fails to produce fully executable code. This is the classic tension between semantic understanding and syntactic correctness.

2. **The Vega-Lite intermediate representation (IR) approach shows promise.** By training the model to output a structured JSON spec, the generated output is more constrained and semantically meaningful. However, the JSON-to-matplotlib conversion pipeline introduces fragility.

3. **Token budget is a bottleneck.** With `max_seq_length = 2048`, many complex plots require code that exceeds the token limit, leading to truncation. Increasing the sequence length (with corresponding GPU memory) would likely improve execution rates.

4. **Variance reduction is a key win.** The fine-tuned model is far more consistent (lower standard deviations across all metrics), which is valuable for production use cases where unpredictable output is unacceptable.

5. **Data augmentation was essential.** The augmentation pipeline expanded the dataset from ~379 base examples to 1,870 training samples, providing the diversity needed for the model to generalize.

#### Potential Improvements

- **Increase `max_seq_length`** (e.g., 4096) to avoid output truncation
- **More training data** through additional augmentation strategies or sourcing from Plotly/Altair galleries
- **Post-processing / repair pipeline:** The existing `repair_truncated_code()` function could be extended to handle more failure modes
- **Larger base model:** Scaling from the 2B to 8B parameter variant (also available via Unsloth) could improve code correctness
- **Reinforcement learning from execution feedback:** Using code execution success as a reward signal (RLHF/RLAIF style) could directly optimize for runnability

---

## Project Structure

```
plot2code/
├── README.md                          # This file
├── prepare_data.ipynb                 # Dataset generation: scrape gallery, augment, create JSONL
├── plot2code_pipeline_US_train.ipynb  # Main training & evaluation notebook
├── common_utils.py                   # Shared utilities: code execution, metrics, evaluation
├── train_utils.py                    # Training data conversion & filtering
├── vegalite_utils.py                 # Vega-Lite parsing, conversion, IR evaluation
├── augment.py                        # Data augmentation (color, marker, style mutations)
├── plot2code_train.jsonl             # Training dataset (1,870 samples)
├── plot2code_test.jsonl              # Test dataset (82 samples)
├── 25_05/                            # Output directory (run dated 25 May)
│   ├── plot2code_lora_final/         # Saved LoRA adapter checkpoints
│   ├── zero_shot_*/                  # Zero-shot evaluation results
│   ├── fine_tuned_*/                 # Fine-tuned evaluation results
│   ├── training_loss_curve.png       # Training loss visualization
│   ├── training_loss_history_*.csv   # Raw loss data
│   ├── comparison_zeroshot_vs_finetuned.png  # Side-by-side comparison chart
│   └── comparison_zeroshot_vs_finetuned_*.csv # Comparison metrics CSV
└── data/                             # Raw source data (matplotlib/seaborn galleries)
    ├── matplotlib/
    └── seaborn/
```

---

## Setup & Usage

### Prerequisites

- **Python** 3.10+
- **CUDA** 12.x compatible GPU (tested with NVIDIA RTX A4000 × 2)
- ~16 GB+ VRAM per GPU recommended

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd plot2code

# Install dependencies
pip install unsloth trl torch transformers accelerate bitsandbytes
pip install matplotlib numpy pandas pillow scikit-image nltk tqdm
```

### Step 1: Prepare the Dataset

Run `prepare_data.ipynb` to:

1. Scrape matplotlib/seaborn gallery Python examples
2. Execute each script to render plot PNG images
3. Augment the dataset with style/color/marker variations
4. Generate Vega-Lite specs for compatible plot types
5. Output `plot2code_train.jsonl` and `plot2code_test.jsonl`

### Step 2: Train & Evaluate

Run `plot2code_pipeline_US_train.ipynb` to:

1. Load the quantized Qwen3-VL-2B model with LoRA adapters
2. Prepare training data from the JSONL files
3. Fine-tune the model (900 steps, ~6 epochs)
4. Run zero-shot evaluation on the test set
5. Run fine-tuned evaluation on the test set
6. Generate comparison metrics and visualizations

### Inference (Quick Start)

```python
from unsloth import FastVisionModel
from PIL import Image

# Load the fine-tuned model
model, processor = FastVisionModel.from_pretrained(
    model_name="./25_05/plot2code_lora_final",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

# Prepare input
image = Image.open("path/to/chart.png").convert("RGB")
instruction = (
    "You are an expert data visualization engineer.\n"
    "Return ONLY a valid Vega-Lite JSON specification that recreates the given plot.\n"
    # ... (full instruction from VEGA_LITE_INSTRUCTION in vegalite_utils.py)
)

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": instruction},
]}]

# Generate
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=1200, do_sample=False)

generated = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
)[0]
print(generated)
```

---

## Dependencies

| Package           | Purpose                                          |
|-------------------|--------------------------------------------------|
| `unsloth`         | Fast LoRA patching for vision-language models    |
| `trl`             | SFTTrainer for supervised fine-tuning            |
| `torch`           | PyTorch backend                                  |
| `transformers`    | Hugging Face model/tokenizer infrastructure      |
| `accelerate`      | Multi-GPU / mixed-precision training support     |
| `bitsandbytes`    | 4-bit quantization                               |
| `matplotlib`      | Plot rendering and reference code execution      |
| `numpy`           | Numerical operations                             |
| `pandas`          | Data manipulation and CSV I/O                    |
| `Pillow`          | Image loading, conversion, and quality checks    |
| `scikit-image`    | SSIM computation for visual similarity           |
| `nltk`            | BLEU score computation                           |
| `tqdm`            | Progress bars                                    |

---

## License

This project is for research and educational purposes.
