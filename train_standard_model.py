# train_standard_model.py

import torch
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import torch.amp as amp
import os
import inspect

def train_standard_gpt2(train_dataloader,
                        val_dataloader,
                        tokenizer,
                        output_dir: str = "standard_gpt2_outputs",
                        num_train_epochs: int = 1,
                        learning_rate: float = 5e-5,
                        gradient_accumulation_steps: int = 1):
    #train standard gpt2 model with next-token prediction

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    #memory optimizations
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    #training arguments
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_dataloader.batch_size,
        "per_device_eval_batch_size": val_dataloader.batch_size if val_dataloader else 4,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 3950,
        "remove_unused_columns": False,
        "load_best_model_at_end": False,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "dataloader_num_workers": 0,
        "fp16": torch.cuda.is_available(),
        "fp16_opt_level": "O1",
        "report_to": "none",
        "save_safetensors": False,
        "evaluation_strategy": "no",
        "save_strategy": "epoch",
        "save_steps": 99999999,
        "save_total_limit": 1,
    }
    
    #setup training arguments with fallback
    try:
        training_args = TrainingArguments(**training_args_dict)
    except TypeError as e:
        print(f"Warning: Some training arguments not supported: {e}")
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

    #custom trainer
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            #clean inputs for model
            clean_inputs = {}
            for k in ["input_ids", "attention_mask", "labels"]:
                if k in inputs:
                    clean_inputs[k] = inputs[k]
            
            #forward pass
            with self.autocast_smart_context_manager():
                outputs = model(**clean_inputs)
                loss = outputs.loss
            
            return (loss, outputs) if return_outputs else loss
        
        def training_step(self, model, inputs, num_items_in_batch=None):
            """Override the training step to use our custom loss computation"""
            model.train()
            inputs = self._prepare_inputs(inputs)
            
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
                
            if self.args.fp16 and hasattr(self, 'use_apex') and self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                
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
                    # Save config
                    if hasattr(self.model, 'config'):
                        self.model.config.save_pretrained(output_dir)
                    print(f"Model saved to {output_dir} (fallback method)")
                else:
                    raise e
            except NameError as e:
                if "DTensor" in str(e):
                    print(f"Warning: DTensor NameError encountered, using fallback save method")
                    # Fallback: manually save model weights
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    # Save config
                    if hasattr(self.model, 'config'):
                        self.model.config.save_pretrained(output_dir)
                    print(f"Model saved to {output_dir} (fallback method)")
                else:
                    raise e
        
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
                    
                    # Save config
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
                    
                    print(f"Model manually saved to {output_dir}")
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
                    
                    print(f"Model manually saved to {output_dir}")
                else:
                    raise e
        
        # Add a custom get_batch_samples method to handle missing 'input_length'
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

    # Fix for deprecated tokenizer parameter
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataloader.dataset,
        "eval_dataset": val_dataloader.dataset if val_dataloader else None,
        "data_collator": train_dataloader.collate_fn
    }
    
    # Use processing_class for newer versions, tokenizer for older versions
    try:
        trainer_kwargs["processing_class"] = tokenizer
        trainer = CustomTrainer(**trainer_kwargs)
    except TypeError:
        # Fallback to tokenizer for older versions
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = CustomTrainer(**trainer_kwargs)

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
            # Save config
            if hasattr(model, 'config'):
                model.config.save_pretrained(output_dir)
            print(f"Model saved to {output_dir} (fallback method)")
        else:
            raise e

    return model
