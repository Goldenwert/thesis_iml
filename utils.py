import torch
from transformers import AutoTokenizer



class TokenizerMetaMath:
    PROMPT_NO_INPUT = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response: "
    )
    PROMPT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Input:\n{input}\n\n### Response: "
    )

    def format_prompt(self, query):
        query = query.split("\n", 1)
        if len(query) == 1 or query[1].strip("\n") == "":
            return self.PROMPT_NO_INPUT.format(query=query[0])
        else:
            return self.PROMPT.format(query=query[0], input=query[1])

    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, examples):
        prompts = [self.format_prompt(text) for text in examples["query"]]
        completions = examples["response"]
        return self._tokenize_fn(prompts, completions)

    def _tokenize_fn(self, prompts, completions):
        prompt_tokens = self.tokenizer(prompts, add_special_tokens=False, truncation=True, max_length=1024)["input_ids"]
        combined_texts = [x + y for x, y in zip(prompts, completions)]
        input_tokens = self.tokenizer(combined_texts, add_special_tokens=False, truncation=True, max_length=1024)["input_ids"]

        input_tokens = [[self.tokenizer.bos_token_id] + x + [self.tokenizer.eos_token_id] for x in input_tokens]
        prompt_length = [len(x) + 1 for x in prompt_tokens]  # +1 for the bos token
        input_length = [len(x) for x in input_tokens]
        return {"input_ids": input_tokens, "prompt_length": prompt_length, "input_length": input_length}


class DataCollator:
    def __init__(self, eos_token_id, max_length=None, num_tokens_to_predict=1):
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.num_tokens_to_predict = num_tokens_to_predict

    def __call__(self, batch):
        # Extract all keys from each item in the batch
        batch = {k: [item[k] for item in batch if k in item] for k in batch[0].keys()}
        
        # Handle missing input_length - calculate it from input_ids if needed
        if "input_length" not in batch or not batch["input_length"]:
            if "input_ids" in batch:
                batch["input_length"] = [len(ids) for ids in batch["input_ids"]]
            else:
                # If no input_ids either, we can't proceed
                raise ValueError("Neither input_length nor input_ids found in batch")
        
        # Handle missing prompt_length - use default of 1 if needed
        if "prompt_length" not in batch or not batch["prompt_length"]:
            if "input_ids" in batch:
                # Default to 1/3 of the input length as a reasonable fallback
                batch["prompt_length"] = [max(1, len(ids) // 3) for ids in batch["input_ids"]]
            else:
                # Default to 1 if we can't determine
                batch["prompt_length"] = [1] * len(batch["input_length"])
        
        input_lengths = torch.tensor(batch["input_length"])
        prompt_lengths = torch.tensor(batch["prompt_length"])
        
        # Pad the input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=self.eos_token_id
        )
        
        # Create attention mask
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.arange(seq_length).unsqueeze(0) < input_lengths.unsqueeze(1)
        
        # Create standard labels where:
        # - Prompt tokens are masked with -100
        # - Response tokens are predicted
        # This creates proper labels for the first head (next-token prediction)
        label_mask = torch.arange(seq_length).unsqueeze(0) < prompt_lengths.unsqueeze(1)
        labels = input_ids.masked_fill(label_mask, -100)
        
        # For multi-token prediction, we may need to prepare additional information
        # that can be used during training to create properly shifted labels for each head
        if self.num_tokens_to_predict > 1:
            # Store the prompt lengths to help with shifting labels during training
            # Each head i will predict token at position t+i+1 using context up to position t
            prompt_lengths_tensor = prompt_lengths.clone()
        
        # Apply length constraints if specified
        if self.max_length is not None:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            labels = labels[:, :self.max_length]
            
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        if self.num_tokens_to_predict > 1:
            result["prompt_lengths"] = prompt_lengths_tensor
            
        return result