# Faster LLMs with Multi-Token Prediction and MEDUSA-1

This repository contains the implementation and evaluation of multi-token prediction frameworks for accelerating Large Language Model (LLM) inference. The project implements three distinct approaches:

1. **Standard Next-Token Prediction (NTP)** - Baseline GPT-2 model
2. **Multi-Token Prediction (MTP)** - GPT-2 with parallel prediction heads
3. **MEDUSA-1** - GPT-2 with speculative decoding heads

## Installation

### Prerequisites
- Python 3.8+ 
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup Environment

1. **Clone the repository:**
```bash
git clone https://github.com/Goldenwert/thesis_iml.git
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

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

## Project Structure

```
├── analysis_notebook.ipynb      # Main analysis and experiments
├── data_preprocessing.py        # Dataset loading and tokenization
├── train_standard_model.py      # Standard GPT-2 training
├── train_modified_model.py      # Multi-token GPT-2 training
├── medusa.py                    # MEDUSA implementation
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── 25SS-K12146699-Garipov_Artur-Thesis_BSc-v1-Multi_Token_Prediction.pdf  # Complete thesis document
```

## Training Models

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

## Evaluation

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


## Acknowledgments

This implementation was developed as part of a Bachelor's thesis at Johannes Kepler University Linz.
