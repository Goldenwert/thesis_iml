# data_preprocessing.py

import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import TokenizerMetaMath, DataCollator

def preprocess_data(model_name: str = "gpt2",
                    max_seq_len: int = 256,
                    train_batch_size: int = 2,
                    val_batch_size: int = 2,
                    split_ratio: float = 0.1,
                    seed: int = 42,
                    use_gsm8k_backward: bool = False,
                    max_examples: int = None,
                    num_tokens_to_predict: int = 1,
                    cache_dir: str = None):
    """
    Preprocess the MetamathQA dataset and optionally the GSM8K_Backward test set.
    
    Args:
        model_name: Model name/path for tokenizer
        max_seq_len: Maximum sequence length
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        split_ratio: Ratio of training data to use for validation
        seed: Random seed for reproducibility
        use_gsm8k_backward: Whether to load GSM8K_Backward for testing
        max_examples: Maximum number of examples to use (None for all)
        num_tokens_to_predict: Number of tokens to predict (for multi-token model)
        cache_dir: Cache directory for datasets
        
    Returns:
        train_dataloader, val_dataloader, test_dataloader (if use_gsm8k_backward=True), tokenizer
    """
    datasets = get_tokenized_datasets(
        model_name=model_name,
        split_ratio=split_ratio,
        seed=seed,
        use_gsm8k_backward=use_gsm8k_backward,
        max_examples=max_examples,
        cache_dir=cache_dir
    )
    
    train_ds = datasets["train"]
    val_ds = datasets["validation"]
    test_ds = datasets["test"]
    tokenizer = datasets["tokenizer"]

    data_collator = DataCollator(
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_seq_len,
        num_tokens_to_predict=num_tokens_to_predict
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    val_dataloader = None
    if val_ds is not None:
        val_dataloader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            collate_fn=data_collator,
            shuffle=False
        )
    
    test_dataloader = None
    if test_ds is not None:
        test_dataloader = DataLoader(
            test_ds,
            batch_size=val_batch_size,
            collate_fn=data_collator,
            shuffle=False
        )

    if use_gsm8k_backward:
        return train_dataloader, val_dataloader, test_dataloader, tokenizer
    else:
        return train_dataloader, val_dataloader, tokenizer

def get_tokenized_datasets(model_name: str = "gpt2", 
                          split_ratio: float = 0.1, 
                          seed: int = 42, 
                          use_gsm8k_backward: bool = False, 
                          max_examples: int = None,
                          cache_dir: str = None):
    """
    Get tokenized datasets without creating DataLoaders.
    Useful for experimenting with different batch sizes or collation strategies.
    """
    train_dataset = load_dataset("meta-math/MetaMathQA", split="train", cache_dir=cache_dir)

    if max_examples and max_examples < len(train_dataset):
        print(f"Using {max_examples} examples out of {len(train_dataset)}")
        train_dataset = train_dataset.select(range(max_examples))

    if split_ratio > 0:
        dataset_split = train_dataset.train_test_split(test_size=split_ratio, seed=seed)
        train_ds = dataset_split["train"]
        val_ds = dataset_split["test"]
    else:
        train_ds = train_dataset
        val_ds = None

    test_ds = None
    if use_gsm8k_backward:
        try:
            test_ds = load_dataset("meta-math/GSM8K_Backward", split="test")
        except Exception as e:
            print(f"Error loading GSM8K_Backward: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer_fn = TokenizerMetaMath(tokenizer_path=model_name)
    
    train_ds = train_ds.map(
        tokenizer_fn, 
        batched=True,
        remove_columns=train_ds.column_names
    )
    
    if val_ds is not None:
        val_ds = val_ds.map(
            tokenizer_fn, 
            batched=True,
            remove_columns=val_ds.column_names
        )
    
    if test_ds is not None:
        if "question" in test_ds.column_names and "answer" in test_ds.column_names:
            test_ds = test_ds.rename_column("question", "query")
            test_ds = test_ds.rename_column("answer", "response")
        
        test_ds = test_ds.map(
            tokenizer_fn, 
            batched=True,
            remove_columns=test_ds.column_names
        )
    
    #format to PyTorch tensors
    train_ds.set_format(type="torch")
    if val_ds is not None:
        val_ds.set_format(type="torch")
    if test_ds is not None:
        test_ds.set_format(type="torch")
    
    return {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
        "tokenizer": tokenizer
    }

def explore_dataset(dataset_name: str = "meta-math/MetaMathQA"):
    """
    Explore the dataset structure and content.
    """
    dataset = load_dataset(dataset_name)
    
    print(f"Dataset: {dataset_name}")
    print(f"Splits: {dataset.keys()}")
    
    for split in dataset.keys():
        print(f"\nSplit: {split}")
        print(f"Number of examples: {len(dataset[split])}")
        print(f"Features: {dataset[split].features}")
        print(f"Column names: {dataset[split].column_names}")
        
        # Show a sample
        print("\nSample example:")
        sample = dataset[split][0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:100]}...")
            else:
                print(f"{key}: {value}")

def validate_tokenization(tokenized_dataset, tokenizer, num_samples=5):
    """
    Validate that tokenization works as expected by decoding some samples.
    """
    print("Validating tokenization...")
    
    for i in range(min(num_samples, len(tokenized_dataset))):
        sample = tokenized_dataset[i]
        input_ids = sample["input_ids"]

        decoded = tokenizer.decode(input_ids)
        
        #prompt_length is correct
        prompt_length = sample["prompt_length"]
        prompt_ids = input_ids[:prompt_length]
        decoded_prompt = tokenizer.decode(prompt_ids)
        
        print(f"\nSample {i+1}:")
        print(f"Prompt length: {prompt_length}")
        print(f"Decoded prompt: {decoded_prompt[:100]}...")
        print(f"Full decoded text: {decoded[:100]}...")

        if "labels" in sample:
            labels = sample["labels"]
            valid_labels = labels[labels != -100]
            print(f"Number of valid labels: {len(valid_labels)}")
            if len(valid_labels) > 0:
                decoded_labels = tokenizer.decode(valid_labels)
                print(f"Decoded labels: {decoded_labels[:100]}...")

if __name__ == "__main__":
    print("Exploring MetaMathQA dataset:")
    explore_dataset("meta-math/MetaMathQA")
    
    try:
        print("\nExploring GSM8K_Backward dataset:")
        explore_dataset("meta-math/GSM8K_Backward")
    except Exception as e:
        print(f"Error exploring GSM8K_Backward: {e}")
    
    print("\nGetting tokenized datasets:")
    datasets = get_tokenized_datasets(use_gsm8k_backward=True)
    
    print("\nValidating tokenization for training set:")
    validate_tokenization(datasets["train"], datasets["tokenizer"])
    
    if datasets["test"] is not None:
        print("\nValidating tokenization for test set:")
        validate_tokenization(datasets["test"], datasets["tokenizer"])
    
    print("\nCreating dataloaders:")
    train_dl, val_dl, test_dl, tokenizer = preprocess_data(use_gsm8k_backward=True)
    
    print(f"Train dataloader batch size: {train_dl.batch_size}")
    print(f"Number of training batches: {len(train_dl)}")
    
    if val_dl is not None:
        print(f"Validation dataloader batch size: {val_dl.batch_size}")
        print(f"Number of validation batches: {len(val_dl)}")
    
    if test_dl is not None:
        print(f"Test dataloader batch size: {test_dl.batch_size}")
        print(f"Number of test batches: {len(test_dl)}")
