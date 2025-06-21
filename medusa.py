import torch
import torch.nn as nn
import torch.amp as amp
from copy import deepcopy
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import numpy as np
import os
import time
import torch.nn.functional as F  # Add F import which may be needed elsewhere
try:
    from accelerate import Accelerator
except ImportError:
    print("Warning: accelerate package not found. If using mixed precision training, please install it with: pip install accelerate")
    # Define a fallback for systems without accelerate
    class Accelerator:
        def backward(self, loss):
            loss.backward()

class MedusaModel(nn.Module):
    #medusa model with multiple prediction heads for parallel token generation
    def __init__(self, base_model_name="gpt2", num_medusa_heads=5):
        super().__init__()
        self.num_medusa_heads = num_medusa_heads
        
        #load base model
        self.backbone = GPT2LMHeadModel.from_pretrained(base_model_name)
        
        #memory optimizations
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable()
        
        self.config = self.backbone.config
        hidden_size = self.config.n_embd
        vocab_size = self.config.vocab_size
        
        #medusa heads for next token predictions
        self.medusa_heads = nn.ModuleList()
        for k in range(num_medusa_heads):
            #feedforward layer
            ff = nn.Linear(hidden_size, hidden_size, bias=True)

            nn.init.zeros_(ff.weight)
            nn.init.zeros_(ff.bias)
            
            #projection layer
            proj = nn.Linear(hidden_size, vocab_size, bias=False)
            #init with backbone weights
            proj.weight = nn.Parameter(self.backbone.lm_head.weight.clone())
            
            self.medusa_heads.append(nn.ModuleDict({"ff": ff, "proj": proj}))
    
    def freeze_backbone(self):
        #freeze backbone for medusa-1 training
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, labels=None, compute_only_backbone=False, past_key_values=None):
        #forward pass with medusa heads
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        #get backbone outputs
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        #backbone only mode
        if compute_only_backbone:
            return outputs
        
        #extract hidden states and logits
        hidden_states = outputs.hidden_states[-1]
        backbone_logits = outputs.logits
        
        #compute medusa head logits
        medusa_logits = []
        for i, head in enumerate(self.medusa_heads):
            #medusa-1 formula: ff -> silu -> residual -> projection
            ff_output = head["ff"](hidden_states)
            activated = torch.nn.functional.silu(ff_output)
            residual_output = activated + hidden_states
            logits_i = head["proj"](residual_output)
            
            medusa_logits.append(logits_i.unsqueeze(1))
        
        #combine all logits
        all_logits = torch.cat([backbone_logits.unsqueeze(1)] + medusa_logits, dim=1)
        
        #compute loss during training
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            total_loss = 0.0
            
            #backbone loss
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = backbone_logits[..., :-1, :].contiguous()
            backbone_loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)),
                                     shift_labels.reshape(-1))
            total_loss += backbone_loss
            
            #medusa head losses with exponential weighting
            head_weights = [0.8**k for k in range(1, self.num_medusa_heads + 1)]
            
            for k, head in enumerate(self.medusa_heads):
                #head k predicts token k+1 steps ahead
                head_logits = medusa_logits[k].squeeze(1)
                
                if head_logits.size(1) <= k + 1:
                    continue
                    
                prediction_logits = head_logits[:, :-(k+1), :]
                target_labels = labels[:, k+1:]
                
                #ensure matching shapes
                min_len = min(prediction_logits.size(1), target_labels.size(1))
                if min_len <= 0:
                    continue
                    
                prediction_logits = prediction_logits[:, :min_len, :]
                target_labels = target_labels[:, :min_len]
                
                #compute weighted loss
                if target_labels.numel() > 0 and (target_labels != -100).any():
                    head_loss = loss_fct(prediction_logits.reshape(-1, prediction_logits.size(-1)),
                                        target_labels.reshape(-1))
                    total_loss += head_loss * head_weights[k]
            
            loss = total_loss
        
        return {
            "loss": loss,
            "logits": all_logits,
            "backbone_logits": backbone_logits,
            "past_key_values": outputs.past_key_values
        }
    
    def generate_with_medusa(self, input_ids, attention_mask=None, max_new_tokens=100, 
                             tree_branching=(5, 5, 3, 3, 2), temperature=0.0, top_p=1.0,
                             typical_threshold=0.2, typical_epsilon=1e-4):
        #generate text using medusa tree attention
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        #limit branches to available heads
        tree_branching = tree_branching[:self.num_medusa_heads]
        
        #initialize generation
        generated_tokens = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        past_key_values = None
        
        #track generation stats
        total_steps = 0
        total_tokens_accepted = 0
        
        for _ in range(0, max_new_tokens, self.num_medusa_heads + 1):
            total_steps += 1
            
            # Get predictions from backbone and Medusa heads
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_tokens,
                    attention_mask=current_attention_mask,
                    compute_only_backbone=False,
                    past_key_values=None
                )
            
            backbone_logits = outputs["backbone_logits"][:, -1, :]
            medusa_logits = [outputs["logits"][:, i+1, -1, :] for i in range(self.num_medusa_heads)]
            
            # TREE CONSTRUCTION: Build candidate tree from predictions
            tree_candidates = self._construct_candidate_tree(
                backbone_logits, medusa_logits, tree_branching, temperature, top_p
            )
            
            if not tree_candidates:
                # Fallback to single token generation
                if temperature > 0:
                    probs = torch.softmax(backbone_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(backbone_logits, dim=-1, keepdim=True)
                
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones(1, 1, device=input_ids.device)], dim=1)
                total_tokens_accepted += 1
                continue
            
            # TREE ATTENTION: Process all candidates in single forward pass
            tree_logits = self._process_tree_with_attention(
                generated_tokens, current_attention_mask, tree_candidates
            )
            
            # VERIFICATION: Find longest acceptable candidate
            accepted_candidate = self._verify_tree_candidates(
                tree_candidates, tree_logits, typical_threshold, typical_epsilon
            )
            
            # Update generated sequence with accepted candidate
            if accepted_candidate and len(accepted_candidate) > 0:
                new_tokens = torch.tensor(accepted_candidate, device=input_ids.device).unsqueeze(0)
                generated_tokens = torch.cat([generated_tokens, new_tokens], dim=1)
                new_mask = torch.ones(1, len(accepted_candidate), device=input_ids.device)
                current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
                total_tokens_accepted += len(accepted_candidate)
            else:
                # Fallback: accept first token only
                if temperature > 0:
                    probs = torch.softmax(backbone_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(backbone_logits, dim=-1, keepdim=True)
                
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones(1, 1, device=input_ids.device)], dim=1)
                total_tokens_accepted += 1
            
            # Check if we've generated enough tokens
            if generated_tokens.size(1) - input_ids.size(1) >= max_new_tokens:
                break
        
        return generated_tokens
    
    def _construct_candidate_tree(self, backbone_logits, medusa_logits, tree_branching, temperature, top_p):
        """Construct tree of candidate continuations from head predictions"""
        device = backbone_logits.device
        
        # Get top-k predictions from backbone (first position)
        if temperature > 0:
            backbone_probs = torch.softmax(backbone_logits / temperature, dim=-1)
            if top_p < 1.0:
                backbone_probs = self._apply_top_p(backbone_probs, top_p)
            backbone_candidates = torch.multinomial(backbone_probs, num_samples=min(tree_branching[0], backbone_probs.size(-1)))
        else:
            backbone_candidates = torch.topk(backbone_logits, k=min(tree_branching[0], backbone_logits.size(-1)), dim=-1).indices
        
        # Build tree structure using Cartesian product approach
        tree_paths = []
        
        # Start with backbone predictions
        for backbone_token in backbone_candidates[0]:
            current_path = [backbone_token.item()]
            tree_paths.append(current_path)
        
        # Extend paths with Medusa head predictions
        for head_idx, (head_logits, branching) in enumerate(zip(medusa_logits, tree_branching[1:])):
            if temperature > 0:
                head_probs = torch.softmax(head_logits / temperature, dim=-1)
                if top_p < 1.0:
                    head_probs = self._apply_top_p(head_probs, top_p)
                head_candidates = torch.multinomial(head_probs, num_samples=min(branching, head_probs.size(-1)))
            else:
                head_candidates = torch.topk(head_logits, k=min(branching, head_logits.size(-1)), dim=-1).indices
            
            # Extend each existing path with head predictions
            new_paths = []
            for path in tree_paths:
                for head_token in head_candidates[0]:
                    new_path = path + [head_token.item()]
                    new_paths.append(new_path)
            tree_paths = new_paths
            
            # Limit total number of paths to prevent explosion
            if len(tree_paths) > 64:  # Reasonable limit
                tree_paths = tree_paths[:64]
        
        return tree_paths
    
    def _process_tree_with_attention(self, generated_tokens, attention_mask, tree_candidates):
        """Process tree candidates using tree attention in single forward pass"""
        device = generated_tokens.device
        batch_size = generated_tokens.size(0)
        
        if not tree_candidates:
            return []
        
        # Construct tree input tensor
        max_candidate_len = max(len(candidate) for candidate in tree_candidates)
        tree_input_ids = []
        tree_attention_masks = []
        tree_position_ids = []
        
        base_seq_len = generated_tokens.size(1)
        
        for candidate in tree_candidates:
            # Create full sequence: original + candidate
            candidate_tensor = torch.tensor(candidate, device=device)
            full_seq = torch.cat([generated_tokens[0], candidate_tensor])
            
            # Pad to consistent length
            if len(candidate) < max_candidate_len:
                pad_len = max_candidate_len - len(candidate)
                padding = torch.zeros(pad_len, device=device, dtype=candidate_tensor.dtype)
                full_seq = torch.cat([full_seq, padding])
            
            tree_input_ids.append(full_seq)
            
            # Create tree attention mask - only attend to predecessors in the tree
            seq_len = base_seq_len + len(candidate)
            tree_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            
            # Pad attention mask
            if seq_len < base_seq_len + max_candidate_len:
                pad_len = base_seq_len + max_candidate_len - seq_len
                padding = torch.zeros(seq_len, pad_len, device=device)
                tree_mask = torch.cat([tree_mask, padding], dim=1)
                bottom_padding = torch.zeros(pad_len, base_seq_len + max_candidate_len, device=device)
                tree_mask = torch.cat([tree_mask, bottom_padding], dim=0)
            
            tree_attention_masks.append(tree_mask)
            
            # Position IDs for tree structure
            pos_ids = torch.arange(base_seq_len + max_candidate_len, device=device)
            tree_position_ids.append(pos_ids)
        
        # Stack into batch tensors
        tree_input_batch = torch.stack(tree_input_ids)  # [num_candidates, seq_len]
        tree_mask_batch = torch.stack(tree_attention_masks)  # [num_candidates, seq_len, seq_len]
        tree_pos_batch = torch.stack(tree_position_ids)  # [num_candidates, seq_len]
        
        # Single forward pass through backbone with tree attention
        with torch.no_grad():
            # Process tree in backbone model
            tree_outputs = self.backbone(
                input_ids=tree_input_batch,
                attention_mask=tree_mask_batch[:, :, 0, :].squeeze(2),  # Convert to 2D mask
                position_ids=tree_pos_batch,
                use_cache=False,
                return_dict=True
            )
        
        return tree_outputs.logits  # [num_candidates, seq_len, vocab_size]
    
    def _verify_tree_candidates(self, tree_candidates, tree_logits, typical_threshold, typical_epsilon):
        """Verify tree candidates and return longest acceptable sequence"""
        if not tree_candidates or tree_logits is None:
            return None
        
        best_candidate = None
        best_length = 0
        
        for i, candidate in enumerate(tree_candidates):
            if i >= tree_logits.size(0):
                break
                
            candidate_logits = tree_logits[i]  # [seq_len, vocab_size]
            accepted_length = 0
            
            # Verify each token in the candidate
            for pos, token in enumerate(candidate):
                if pos >= candidate_logits.size(0) - 1:  # -1 because we predict next token
                    break
                
                # Get prediction at this position
                pred_logits = candidate_logits[-(len(candidate) - pos + 1)]  # Correct indexing
                pred_probs = torch.softmax(pred_logits, dim=-1)
                
                # Apply typical acceptance criterion
                token_prob = pred_probs[token].item()
                entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10)).item()
                
                threshold = max(typical_epsilon, typical_threshold * torch.exp(torch.tensor(-entropy)).item())
                
                if token_prob >= threshold:
                    accepted_length += 1
                else:
                    break
            
            # Keep track of best (longest) accepted candidate
            if accepted_length > best_length:
                best_length = accepted_length
                best_candidate = candidate[:accepted_length]
        
        return best_candidate
    
    def _apply_top_p(self, probs, top_p):
        """Apply top-p (nucleus) sampling"""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for tokens to keep
        mask = cumsum_probs <= top_p
        mask[..., 0] = True  # Always keep the top token
        
        # Zero out probabilities outside the nucleus
        sorted_probs[~mask] = 0
        
        # Scatter back to original order
        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
        
        # Renormalize
        probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)
        
        return probs_filtered

def train_medusa(
    base_model,
    train_dataloader,
    val_dataloader=None,
    output_dir="medusa_model",
    num_medusa_heads=5,
    num_train_epochs=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    gradient_accumulation_steps=1,
    fp16=True,
    freeze_backbone=True
):
    """
    Train a Medusa model on top of a pre-trained LLM
    
    Args:
        base_model: Base model name or path
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        output_dir: Directory to save the model
        num_medusa_heads: Number of Medusa heads to train
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        gradient_accumulation_steps: Number of gradient accumulation steps
        fp16: Whether to use mixed precision training
        freeze_backbone: Whether to freeze the backbone model
    """
    from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
    
    # Try to import AdamW from different locations based on transformers version
    try:
        # For newer transformers versions
        from torch.optim import AdamW
    except ImportError:
        try:
            # For older transformers versions
            from transformers.optimization import AdamW
        except ImportError:
            try:
                # For very old transformers versions
                from transformers import AdamW
            except ImportError:
                raise ImportError("Could not import AdamW from torch.optim, transformers.optimization, or transformers. Please check your installation.")
    
    # Create Medusa model
    model = MedusaModel(base_model_name=base_model, num_medusa_heads=num_medusa_heads)
    
    #backbone freezing
    model.freeze_backbone()
    print("Training with frozen backbone (MEDUSA-1)")

    optimizer_to_use = None

    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_dataloader.batch_size,
        "per_device_eval_batch_size": val_dataloader.batch_size if val_dataloader else 4,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 100,
        "load_best_model_at_end": val_dataloader is not None,
        "fp16": fp16,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": "none",
        "save_safetensors": False,
        "evaluation_strategy": "epoch" if val_dataloader else "no",
        "save_strategy": "epoch",
        "save_steps": 99999999,
        "save_total_limit": 1,
    }

    try:
        training_args = TrainingArguments(**training_args_dict)
    except TypeError as e:
        print(f"Warning: Some training arguments not supported: {e}")
        # Try with minimal args if the above fails
        basic_args = {
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": train_dataloader.batch_size,
            "per_device_eval_batch_size": val_dataloader.batch_size if val_dataloader else 4,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "logging_dir": f"{output_dir}/logs",
            "save_steps": 99999999,
            "report_to": "none",
            "save_safetensors": False,
        }
        training_args = TrainingArguments(**basic_args)
    
    # Define Trainer class
    class MedusaTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Initialize accelerator if we're using mixed precision and accelerate is available
            if self.args.fp16 and 'Accelerator' in globals():
                try:
                    self.accelerator = Accelerator(fp16=True)
                except Exception as e:
                    print(f"Warning: Could not initialize Accelerator: {e}")
                    # Fallback - just directly use backward
                    self.accelerator = None
        
        def autocast_smart_context_manager(self):
            """Return a context manager for autocast or no-op context for mixed precision training"""
            if hasattr(self, '_enable_autocast') and self._enable_autocast:
                # Default HF Trainer method handles this correctly
                return super().autocast_smart_context_manager() 
            elif self.args.fp16:
                return torch.cuda.amp.autocast()
            else:
                import contextlib
                return contextlib.nullcontext()
        
        def compute_loss_context_manager(self):
            """Return appropriate context manager for loss computation"""
            # Check if parent class method exists (newer HF versions)
            if hasattr(super(), 'compute_loss_context_manager'):
                return super().compute_loss_context_manager()
            
            #fallback implementation
            import contextlib
            return contextlib.nullcontext()
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Remove any keys that aren't expected by the model
            clean_inputs = {}
            for k in ["input_ids", "attention_mask", "labels"]:
                if k in inputs:
                    clean_inputs[k] = inputs[k]
            
            # Use autocast context manager for mixed precision
            with self.autocast_smart_context_manager():
                outputs = model(
                    input_ids=clean_inputs["input_ids"],
                    attention_mask=clean_inputs["attention_mask"],
                    labels=clean_inputs["labels"]
                )
            
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
            
        def training_step(self, model, inputs, num_items_in_batch=None):
            """Override the training step to use our custom loss computation"""
            model.train()
            inputs = self._prepare_inputs(inputs)
            
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
                
            # Handle mixed precision backward pass correctly
            if hasattr(self, 'accelerator') and self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                # Fallback if accelerator is not available
                loss.backward()
            
            return loss
            
        def _save_checkpoint(self, model, trial, *args, **kwargs):
            """Override to handle DTensor import issues"""
            try:
                # Call the parent method with all arguments passed through
                super()._save_checkpoint(model, trial, *args, **kwargs)
            except ImportError as e:
                if "DTensor" in str(e):
                    print(f"Warning: Skipping checkpoint save due to DTensor compatibility issue: {e}")
                    print("Model will still be saved at the end of training.")
                else:
                    raise e
            except TypeError as e:
                # Handle signature mismatch - try different argument combinations
                print(f"Warning: Checkpoint save signature mismatch, trying different approaches: {e}")
                try:
                    # Try without extra args
                    super()._save_checkpoint(model, trial)
                except ImportError as e2:
                    if "DTensor" in str(e2):
                        print(f"Warning: Skipping checkpoint save due to DTensor compatibility issue: {e2}")
                        print("Model will still be saved at the end of training.")
                    else:
                        raise e2
                except Exception as e3:
                    print(f"Warning: Could not save checkpoint: {e3}")
                    print("Model will still be saved at the end of training.")
        
        def _save(self, output_dir, state_dict=None):
            """Override _save to avoid DTensor NameError issues"""
            try:
                super()._save(output_dir, state_dict)
            except NameError as e:
                if "DTensor" in str(e):
                    print(f"Warning: DTensor NameError in _save, using manual save approach")
                    # Manual save approach to avoid DTensor issues
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save model weights
                    if state_dict is None:
                        state_dict = self.model.state_dict()
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    
                    # Save config (for MEDUSA model this includes head configs)
                    if hasattr(self.model, 'config'):
                        self.model.config.save_pretrained(output_dir)
                    
                    # Save tokenizer if available
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        try:
                            self.tokenizer.save_pretrained(output_dir)
                        except:
                            pass
                    elif hasattr(self, 'processing_class') and self.processing_class is not None:
                        try:
                            self.processing_class.save_pretrained(output_dir)
                        except:
                            pass
                    
                    print(f"MEDUSA model manually saved to {output_dir}")
                else:
                    raise e
            except ImportError as e:
                if "DTensor" in str(e):
                    print(f"Warning: DTensor ImportError in _save, using manual save approach")
                    # Same fallback as above
                    os.makedirs(output_dir, exist_ok=True)
                    
                    if state_dict is None:
                        state_dict = self.model.state_dict()
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    
                    if hasattr(self.model, 'config'):
                        self.model.config.save_pretrained(output_dir)
                    
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        try:
                            self.tokenizer.save_pretrained(output_dir)
                        except:
                            pass
                    elif hasattr(self, 'processing_class') and self.processing_class is not None:
                        try:
                            self.processing_class.save_pretrained(output_dir)
                        except:
                            pass
                    
                    print(f"MEDUSA model manually saved to {output_dir}")
                else:
                    raise e

        def save_model(self, output_dir=None, _internal_call=False):
            """Override to handle DTensor import issues during saving"""
            if output_dir is None:
                output_dir = self.args.output_dir
                
            try:
                super().save_model(output_dir, _internal_call)
            except ImportError as e:
                if "DTensor" in str(e):
                    print(f"Warning: Using fallback save method due to DTensor compatibility issue")
                    # Fallback: manually save model weights
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    print(f"Model saved to {output_dir} (fallback method)")
                else:
                    raise e
            except NameError as e:
                if "DTensor" in str(e):
                    print(f"Warning: DTensor NameError encountered, using fallback save method")
                    # Fallback: manually save model weights
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    print(f"Model saved to {output_dir} (fallback method)")
                else:
                    raise e
            
        def get_batch_samples(self, epoch_iterator, num_batches, device=None):
            """
            Override the get_batch_samples method to handle missing fields in batch.
            This makes the code more robust against KeyErrors.
            """
            batch_samples = []
            total_items = 0
            
            for _ in range(num_batches):
                try:
                    # Get the next batch
                    batch = next(epoch_iterator)
                    
                    # We only need input_ids, attention_mask, and labels for the model
                    # Filter out any other keys that might cause issues
                    filtered_batch = {}
                    for k in ["input_ids", "attention_mask", "labels"]:
                        if k in batch:
                            filtered_batch[k] = batch[k]
                    
                    # Add to our batch samples
                    batch_samples.append(filtered_batch)
                    
                    # Count the number of items
                    if "input_ids" in filtered_batch:
                        total_items += filtered_batch["input_ids"].size(0)
                    
                except StopIteration:
                    break
                
            return batch_samples, total_items

        def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
            """Override to handle DTensor import issues during evaluation"""
            try:
                super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate)
            except ImportError as e:
                if "DTensor" in str(e):
                    print(f"Warning: Skipping save/evaluate due to DTensor compatibility issue: {e}")
                    print("Model will still be saved at the end of training.")
                else:
                    raise e

    # Fix for deprecated tokenizer parameter
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataloader.dataset,
        "eval_dataset": val_dataloader.dataset if val_dataloader else None,
        "data_collator": train_dataloader.collate_fn
    }
    
    # Use processing_class for newer versions, tokenizer for older versions  
    # Note: MEDUSA doesn't use tokenizer in trainer, so we can skip this
    trainer = MedusaTrainer(**trainer_kwargs)

    trainer.train()

    # Save model with error handling
    try:
        trainer.save_model(output_dir)
    except ImportError as e:
        if "DTensor" in str(e):
            print(f"Warning: Using fallback save method due to DTensor compatibility issue")
            # Fallback: manually save model weights
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            print(f"Model saved to {output_dir} (fallback method)")
        else:
            raise e

    return model

def generate_text_with_medusa(model, tokenizer, prompt, max_new_tokens=100, tree_branching=(5,5,3,3,2), temperature=0.0, top_p=1.0, typical_threshold=0.2, typical_epsilon=1e-4):
    """Generate text with MEDUSA model using tree-based verification.
    
    Args:
        model: The MedusaModel
        tokenizer: Tokenizer to use
        prompt: Text prompt to continue
        max_new_tokens: Maximum number of new tokens to generate
        tree_branching: Number of top predictions to consider for each head
        temperature: Sampling temperature (0 for greedy decoding)
        top_p: Top-p sampling parameter
        typical_threshold: Threshold for typical acceptance (delta in paper)
        typical_epsilon: Minimum threshold for typical acceptance (epsilon in paper)
    """
    device = next(model.parameters()).device
    
    # Ensure pad_token_id is set to eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Initialize sequence with input
    sequence = input_ids
    current_attention_mask = attention_mask.clone()
    
    # Count tokens for statistics
    tokens_accepted = 0
    steps = 0
    
    # Generate tokens one by one (with tree-based verification)
    while sequence.size(1) < input_ids.size(1) + max_new_tokens:
        steps += 1
        with torch.no_grad():
            # Forward pass through the model
            # Note: Skip KV caching for compatibility with newer transformers
            outputs = model(
                input_ids=sequence,
                attention_mask=current_attention_mask,
                compute_only_backbone=False,
                past_key_values=None
            )
            
            # Get logits from backbone and Medusa heads
            backbone_logits = outputs["backbone_logits"][:, -1, :]
            medusa_logits = [outputs["logits"][:, i+1, -1, :] for i in range(model.num_medusa_heads)]
            
            # Get predictions
            if temperature > 0:
                # Apply temperature and sample
                backbone_probs = torch.softmax(backbone_logits / temperature, dim=-1)
                
                # Apply top-p sampling if specified
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(backbone_probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs > top_p
                    mask[..., 0] = False  # Keep at least one token
                    backbone_probs.scatter_(dim=-1, index=sorted_indices[mask], src=torch.zeros_like(backbone_probs))
                    backbone_probs = backbone_probs / backbone_probs.sum(dim=-1, keepdim=True)
                
                backbone_next_token = torch.multinomial(backbone_probs, num_samples=1)
                
                medusa_next_tokens = []
                for i, logits in enumerate(medusa_logits):
                    if i < len(tree_branching):
                        probs = torch.softmax(logits / temperature, dim=-1)
                        
                        # Apply top-p sampling if specified
                        if top_p < 1.0:
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                            mask = cumsum_probs > top_p
                            mask[..., 0] = False  # Keep at least one token
                            probs.scatter_(dim=-1, index=sorted_indices[mask], src=torch.zeros_like(probs))
                            probs = probs / probs.sum(dim=-1, keepdim=True)
                        
                        next_tokens = torch.multinomial(
                            probs, num_samples=min(tree_branching[i], probs.size(-1))
                        )
                        medusa_next_tokens.append(next_tokens)
            else:
                # Greedy decoding
                backbone_next_token = torch.argmax(backbone_logits, dim=-1, keepdim=True)
                
                medusa_next_tokens = []
                for i, logits in enumerate(medusa_logits):
                    if i < len(tree_branching):
                        next_tokens = torch.topk(
                            logits, k=min(tree_branching[i], logits.size(-1)), dim=-1
                        ).indices
                        medusa_next_tokens.append(next_tokens)
            
            # Implement Tree Verification:
            # Create tree candidates based on branching factors
            branch_indices = [[backbone_next_token]]
            
            # Generate the tree structure from medusa head predictions
            for depth, tokens in enumerate(medusa_next_tokens):
                # Create branches for each previous candidate path
                new_branches = []
                for prev_branch in branch_indices:
                    # For each token in the current prediction
                    for token_idx in range(tokens.size(1)):
                        # Add the new token to the branch
                        new_branch = prev_branch + [tokens[:, token_idx:token_idx+1]]
                        new_branches.append(new_branch)
                
                branch_indices = new_branches
                # Limit the number of branches if too many
                if len(branch_indices) > 1000:  # Safety limit
                    branch_indices = branch_indices[:1000]
            
            # Create candidate sequences
            candidate_seqs = []
            for branch in branch_indices:
                # Concatenate tokens in this branch
                branch_tokens = torch.cat(branch, dim=1)
                # Add only the branch tokens, not the full sequence
                candidate_seqs.append(branch_tokens)
            
            # Batch verification with tree attention
            if len(candidate_seqs) > 0:
                # Create batched input
                batched_candidates = torch.cat(candidate_seqs, dim=0)
                
                # Create attention mask for verification
                batch_size = len(candidate_seqs)
                seq_len = batched_candidates.size(1)
                
                # Create proper 2D attention mask for HF transformers
                # Each sequence gets its own causal mask
                cand_attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
                
                # Note: We skip KV cache expansion for compatibility with newer transformers
                # Tree verification works by processing full candidate sequences
                
                # Now verify the candidates
                # Note: Skip past_key_values to avoid compatibility issues with newer transformers
                verify_outputs = model.backbone(
                    input_ids=batched_candidates,
                    attention_mask=cand_attention_mask,
                    use_cache=False,
                    return_dict=True
                )
                
                verify_logits = verify_outputs.logits
                
                # Calculate entropy for typical sampling
                log_probs = F.log_softmax(verify_logits, dim=-1)
                probs = torch.exp(log_probs)
                entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: [batch_size, seq_len]
                
                # Prepare to find the longest verified branch
                best_branch_idx = 0
                best_accept_len = 0
                
                # Evaluate each branch
                for b_idx, branch in enumerate(branch_indices):
                    branch_len = len(branch)
                    accept_len = 0
                    
                    # Extract logits for this branch
                    branch_logits = verify_logits[b_idx]
                    
                    # Check each position in the branch
                    for i in range(branch_len):
                        # Position in the verification sequence
                        pos = i
                        if pos >= branch_logits.size(0):
                            break
                            
                        pred_token = batched_candidates[b_idx, pos]
                        
                        # If we're at position 0, we already know it's accepted (it's the backbone token)
                        if i == 0:
                            accept_len += 1
                            continue
                        
                        # Get the model's prediction at this position
                        prev_pos_logits = branch_logits[pos-1]
                        pos_log_probs = F.log_softmax(prev_pos_logits.unsqueeze(0), dim=-1)
                        pred_log_prob = pos_log_probs[0, pred_token]
                        
                        # Calculate typical acceptance threshold
                        pos_entropy = entropy[b_idx, pos-1]
                        threshold = max(typical_epsilon, typical_threshold * torch.exp(-pos_entropy))
                        
                        # Check if prediction is accepted
                        if torch.exp(pred_log_prob) >= threshold:
                            accept_len += 1
                        else:
                            break
                    
                    # Update best branch if this one is better
                    if accept_len > best_accept_len:
                        best_branch_idx = b_idx
                        best_accept_len = accept_len
                
                # Use the best branch
                if best_accept_len > 0:
                    best_branch = branch_indices[best_branch_idx]
                    accepted_tokens = torch.cat(best_branch[:best_accept_len], dim=1)
                else:
                    # Fallback to just the backbone prediction if no branches were accepted
                    accepted_tokens = backbone_next_token
                    best_accept_len = 1
            else:
                # If no candidates (shouldn't happen), just use backbone prediction
                accepted_tokens = backbone_next_token
                best_accept_len = 1
            
            # Update sequence
            sequence = torch.cat([sequence, accepted_tokens], dim=1)
            new_mask = torch.ones_like(accepted_tokens)
            current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
            
            # Update statistics
            tokens_accepted += best_accept_len
            
            # Check for max tokens or EOS
            if sequence.size(1) - input_ids.size(1) >= max_new_tokens:
                # Truncate to exactly max_new_tokens
                sequence = sequence[:, :input_ids.size(1) + max_new_tokens]
                break
            
            # Check if EOS token was generated
            if sequence[0, -1].item() == tokenizer.eos_token_id:
                break
    
    # Print statistics
    print(f"MEDUSA generated {tokens_accepted} tokens in {steps} steps")
    print(f"Average acceptance: {tokens_accepted/steps:.2f} tokens per step")
    
    # Return the token IDs
    return sequence 

def test_medusa_tree_attention():
    """Test function for the new MEDUSA tree attention"""
    print("Testing MEDUSA Tree Attention...")
    
    # Load model and tokenizer
    model_name = "gpt2"
    model = MedusaModel(model_name, num_medusa_heads=4)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompt
    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"Prompt: {prompt}")
    print("Generating with tree attention...")
    
    # Generate with tree attention
    start_time = time.time()
    generated_ids = model.generate_with_medusa(
        input_ids=input_ids,
        max_new_tokens=50,
        tree_branching=(4, 4, 3, 2),
        temperature=0.8,
        top_p=0.9,
        typical_threshold=0.2,
        typical_epsilon=1e-4
    )
    generation_time = time.time() - start_time
    
    # Decode result
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print(f"Generation time: {generation_time:.2f} seconds")
    
    return generated_text

if __name__ == "__main__":
    test_medusa_tree_attention() 