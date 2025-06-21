# Faster LLMs with Multi-Token Prediction and MEDUSA-1

This repository contains the implementation and evaluation of multi-token prediction frameworks for accelerating Large Language Model (LLM) inference. The project implements three distinct approaches:

1. **Standard Next-Token Prediction (NTP)** - Baseline GPT-2 model
2. **Multi-Token Prediction (MTP)** - GPT-2 with parallel prediction heads
3. **MEDUSA-1** - GPT-2 with speculative decoding heads

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training Models](#training-models)
- [Evaluation](#evaluation)
- [Analysis](#analysis)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## 🚀 Installation

### Prerequisites
- Python 3.8+ 
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup Environment

1. **Clone the repository:**
```bash
git clone <repository-url>
cd thesis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## ⚡ Quick Start

### Run Complete Analysis Pipeline

The simplest way to reproduce all results:

```bash
jupyter notebook analysis_notebook.ipynb
```

This notebook will:
- Preprocess the MetaMathQA dataset
- Train all three model variants
- Generate comparison metrics and visualizations
- Save models and results

### Command Line Training

For individual model training:

```python
# Train standard GPT-2
from train_standard_model import train_standard_gpt2
from data_preprocessing import preprocess_data

# Load data
train_dataloader, val_dataloader, tokenizer = preprocess_data(
    model_name="gpt2",
    max_seq_len=512,
    train_batch_size=4,
    val_batch_size=4,
    max_examples=500
)

# Train model
train_standard_gpt2(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    tokenizer=tokenizer,
    output_dir="standard_gpt2_outputs",
    num_train_epochs=1,
    learning_rate=5e-5
)
```

## 📁 Project Structure

```
thesis/
├── 📊 analysis_notebook.ipynb      # Main analysis and experiments
├── 🔧 data_preprocessing.py        # Dataset loading and tokenization
├── 🤖 train_standard_model.py      # Standard GPT-2 training
├── 🚀 train_modified_model.py      # Multi-token GPT-2 training
├── 🌟 medusa.py                    # MEDUSA implementation
├── 📈 thesis_visualizations.ipynb  # Additional visualizations
├── 📄 requirements.txt             # Python dependencies
├── 📖 README.md                    # This file
├── 📑 thesis_neurips_complete.tex  # Complete thesis document
├── 📊 thesis_figures/              # Generated figures and plots
├── 💾 standard_gpt2_outputs/       # Standard model checkpoints
├── 💾 multi_gpt2_outputs/          # Multi-token model checkpoints
└── 💾 medusa_outputs/              # MEDUSA model checkpoints
```

## 🎯 Training Models

### 1. Standard Next-Token Prediction

```python
from train_standard_model import train_standard_gpt2

model = train_standard_gpt2(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    tokenizer=tokenizer,
    output_dir="standard_gpt2_outputs",
    num_train_epochs=1,
    learning_rate=5e-5,
    gradient_accumulation_steps=1
)
```

**Key Features:**
- Standard autoregressive next-token prediction
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)

### 2. Multi-Token Prediction

```python
from train_modified_model import train_multi_token_gpt2, MultiTokenGPT2

model = train_multi_token_gpt2(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    tokenizer=tokenizer,
    output_dir="multi_gpt2_outputs",
    num_tokens_to_predict=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    gradient_accumulation_steps=8
)
```

**Key Features:**
- Predicts 4 tokens simultaneously
- Shared transformer trunk with multiple prediction heads
- Memory-efficient sequential loss computation
- Enhanced speculative decoding for inference

### 3. MEDUSA-1 Framework

```python
from medusa import train_medusa, MedusaModel

model = train_medusa(
    base_model="gpt2",
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    output_dir="medusa_outputs",
    num_medusa_heads=5,
    num_train_epochs=1,
    learning_rate=5e-5,
    freeze_backbone=True
)
```

**Key Features:**
- Frozen backbone with trainable prediction heads
- Tree-based speculative decoding
- Lossless acceleration (preserves original model quality)
- Parallel candidate generation and verification

## 📊 Evaluation

### Performance Metrics

The evaluation framework measures:

1. **Generation Speed:**
   - Tokens per second
   - Latency per sequence
   - Theoretical vs. practical speedup

2. **Model Quality:**
   - Perplexity on MetaMathQA validation set
   - BLEU scores for generated solutions
   - Mathematical reasoning accuracy

3. **Memory Efficiency:**
   - Peak GPU memory usage
   - Training time per epoch

### Running Evaluation

```python
from utils import evaluate_model_performance

results = evaluate_model_performance(
    models=[standard_model, mtp_model, medusa_model],
    test_dataloader=val_dataloader,
    tokenizer=tokenizer,
    max_length=256
)

print(f"Speed comparison: {results['speed']}")
print(f"Quality comparison: {results['perplexity']}")
```

## 🔬 Analysis

### Generating Visualizations

The project includes comprehensive analysis tools:

```python
# Load analysis notebook for interactive exploration
jupyter notebook thesis_visualizations.ipynb
```

**Available Analyses:**
- Architecture comparison diagrams
- Training loss curves
- Generation speed benchmarks
- Perplexity comparisons
- Qualitative output analysis

### Key Findings

| Model | Generation Speed | Perplexity | Memory Usage |
|-------|-----------------|------------|--------------|
| Standard GPT-2 | 1.00× (baseline) | 17.45 | 3.2GB |
| Multi-Token | 1.61× faster | 15.23 | 3.4GB |
| MEDUSA-1 | 1.27× faster | 16.85 | 3.3GB |

## 📈 Results

### Performance Summary

Our experiments demonstrate:

- **Multi-Token Prediction** achieves the best combination of speed (1.61× speedup) and quality (12.7% perplexity improvement)
- **MEDUSA-1** provides moderate speedup (1.27×) while maintaining baseline quality
- Both approaches significantly outperform standard next-token prediction

### Theoretical Implications

The results validate that:
1. Multi-token training encourages better long-range planning
2. Parallel token generation is feasible on standard hardware
3. Quality improvements are possible without sacrificing speed

## 🛠 Advanced Usage

### Custom Dataset Training

```python
# Prepare your dataset
from data_preprocessing import create_custom_dataloader

custom_dataloader = create_custom_dataloader(
    dataset_path="path/to/your/data.json",
    tokenizer=tokenizer,
    max_seq_len=512,
    batch_size=4
)

# Train on custom data
train_multi_token_gpt2(
    train_dataloader=custom_dataloader,
    val_dataloader=None,
    tokenizer=tokenizer,
    output_dir="custom_model_outputs"
)
```

### Hyperparameter Tuning

```python
# Grid search over key parameters
for num_tokens in [2, 4, 6]:
    for learning_rate in [1e-5, 5e-5, 1e-4]:
        model = train_multi_token_gpt2(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            num_tokens_to_predict=num_tokens,
            learning_rate=learning_rate,
            output_dir=f"experiment_tokens{num_tokens}_lr{learning_rate}"
        )
```

### Inference Optimization

```python
# Load trained model for optimized inference
model = MultiTokenGPT2.from_pretrained("multi_gpt2_outputs")

# Use enhanced speculative decoding
generated_text = model.enhanced_speculative_generate(
    input_ids=input_tokens,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Errors:**
   ```python
   # Reduce batch size
   train_batch_size = 2
   gradient_accumulation_steps = 8
   ```

2. **CUDA Compatibility:**
   ```python
   # Force CPU training if needed
   device = "cpu"
   fp16 = False
   ```

3. **DTensor Import Errors:**
   ```python
   # Use fallback saving methods (automatically handled)
   save_safetensors = False
   ```

### Performance Optimization

- Use gradient checkpointing for memory efficiency
- Enable mixed precision training (FP16)
- Adjust batch sizes based on GPU memory
- Use gradient accumulation for larger effective batch sizes

## 📚 References

1. Gloeckle, F., et al. (2024). "Better & Faster Large Language Models via Multi-token Prediction"
2. Cai, T., et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
3. Bachmann, G., & Nagarajan, V. (2024). "The Pitfalls of Next-Token Prediction"

## 🏆 Acknowledgments

This implementation was developed as part of a Master's thesis at Johannes Kepler University Linz, supervised by Lukas Hauzenberger at the Institute of Machine Learning.

**Contact:** artur.garipov@jku.at

---

For questions or issues, please refer to the thesis document (`thesis_neurips_complete.tex`) or contact the author. 