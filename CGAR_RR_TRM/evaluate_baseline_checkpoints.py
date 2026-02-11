#!/usr/bin/env python3
"""
Evaluate baseline TRM checkpoints on test set.
Modified from evaluate_checkpoints.py to work with standard TRM (not CGAR).
"""

import torch
import sys
import json
from pathlib import Path
from omegaconf import OmegaConf
import tqdm as tqdm_module

# Add project root to path
sys.path.insert(0, '/data/TRM')

# Import from pretrain (includes all necessary utilities)
from pretrain import *

# Import model and loss classes - BASELINE versions (not CGAR)
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.losses import ACTLossHead

# Explicitly assign tqdm to avoid conflicts
tqdm = tqdm_module.tqdm


def load_checkpoint(checkpoint_path, config, test_metadata):
    """Load a checkpoint and return the model."""
    print(f"\nLoading checkpoint: {checkpoint_path}", flush=True)
    
    # Convert OmegaConf to dict
    arch_dict = OmegaConf.to_container(config.arch, resolve=True)
    loss_dict = OmegaConf.to_container(config.arch.loss, resolve=True)
    
    # Add required metadata from test set
    arch_dict['vocab_size'] = test_metadata.vocab_size
    arch_dict['seq_len'] = test_metadata.seq_len
    arch_dict['num_puzzle_identifiers'] = test_metadata.num_puzzle_identifiers
    arch_dict['batch_size'] = config.global_batch_size
    arch_dict['causal'] = False
    
    # Filter out 'name' from loss_dict (it's metadata, not a constructor arg)
    loss_params = {k: v for k, v in loss_dict.items() if k != 'name'}
    
    # Create BASELINE model (not CGAR) - with torch.device("cuda") to ensure all buffers are on GPU
    print(f"  Creating baseline TRM model...", flush=True)
    with torch.device("cuda"):
        model = TinyRecursiveReasoningModel_ACTV1(arch_dict)
        loss_head = ACTLossHead(model, **loss_params)
    
    # Load checkpoint
    print(f"  Loading state dict...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Remove _orig_mod. prefix if present (from torch.compile)
    new_checkpoint = {}
    for key, value in checkpoint.items():
        new_key = key.replace('_orig_mod.', '')
        new_checkpoint[new_key] = value
    
    loss_head.load_state_dict(new_checkpoint)
    loss_head.eval()
    
    print(f"  ✅ Checkpoint loaded successfully", flush=True)
    return loss_head


def evaluate_checkpoint(model, test_loader, config):
    """Evaluate a single checkpoint on the test set."""
    print(f"\n{'='*60}", flush=True)
    print(f"Starting evaluation...", flush=True)
    print(f"{'='*60}", flush=True)
    
    total_correct = 0
    total_tokens = 0
    total_exact_correct = 0
    total_puzzles = 0
    processed_batches = 0
    
    return_keys = ["preds"]  # What we want back from the model
    
    with torch.no_grad():
        for set_name, batch, global_batch_size in tqdm(test_loader, desc="Evaluating"):
            processed_batches += 1
            
            # Print progress every 10 batches
            if processed_batches % 10 == 0:
                print(f"  Processing batch {processed_batches}...", flush=True)
            
            # Move batch to GPU (matching CGAR evaluation)
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Initialize carry state with torch.device context
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            
            # Forward - loop until all_finish (same as CGAR evaluation)
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                
                if all_finish:
                    break
                
                if inference_steps > config.arch.halt_max_steps:
                    print(f"Warning: Exceeded max steps ({config.arch.halt_max_steps})", flush=True)
                    break
            
            # Extract predictions and labels
            predictions = preds["preds"]  # [batch, seq_len] - already token IDs
            labels = batch['labels']  # [batch, seq_len]
            
            # Debug shapes on first batch
            if processed_batches == 1:
                print(f"  Debug: predictions shape={predictions.shape}, dtype={predictions.dtype}", flush=True)
                print(f"  Debug: labels shape={labels.shape}, dtype={labels.dtype}", flush=True)
            
            # Calculate token accuracy
            mask = labels != 0  # Non-padding tokens
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Calculate exact accuracy (entire sequence must match)
            per_example_correct = ((predictions == labels) | (~mask)).all(dim=1)
            total_exact_correct += per_example_correct.sum().item()
            total_puzzles += predictions.shape[0]
            
            # Clean up memory
            del carry, loss, preds, batch
    
    print(f"\n  Processed {processed_batches} batches, {total_puzzles} puzzles", flush=True)
    
    # Calculate final metrics
    token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    exact_accuracy = total_exact_correct / total_puzzles if total_puzzles > 0 else 0.0
    
    return {
        'token_accuracy': token_accuracy,
        'exact_accuracy': exact_accuracy,
        'total_tokens': total_tokens,
        'total_puzzles': total_puzzles,
        'total_correct_tokens': total_correct,
        'total_exact_correct': total_exact_correct
    }


def main():
    print("\n" + "="*80, flush=True)
    print("BASELINE TRM CHECKPOINT EVALUATION", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Find baseline checkpoint directory
    checkpoint_base = Path("/data/TRM/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch")
    baseline_dirs = list(checkpoint_base.glob("baseline_trm_*"))
    
    if not baseline_dirs:
        raise RuntimeError(f"No baseline checkpoint directory found in {checkpoint_base}")
    
    # Use the most recent one
    checkpoint_dir = sorted(baseline_dirs)[-1]
    print(f"Using checkpoint directory: {checkpoint_dir}", flush=True)
    
    # Load config
    config_path = checkpoint_dir / "all_config.yaml"
    print(f"Loading config from: {config_path}", flush=True)
    config = OmegaConf.load(config_path)
    
    # Create test dataloader (same as in evaluate_checkpoints.py)
    print(f"\nCreating test dataloader...", flush=True)
    test_loader, test_metadata = create_dataloader(
        config, 
        "test",  # Use test set
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1
    )
    print(f"Test metadata: vocab_size={test_metadata.vocab_size}, seq_len={test_metadata.seq_len}", flush=True)
    print(f"✅ Test dataloader created", flush=True)
    
    # Find all checkpoint files
    checkpoint_files = sorted(checkpoint_dir.glob("step_*"))
    print(f"\nFound {len(checkpoint_files)} checkpoints to evaluate", flush=True)
    
    # Evaluate each checkpoint
    results = {}
    
    for i, ckpt_path in enumerate(checkpoint_files, 1):
        step = ckpt_path.name
        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating checkpoint {i}/{len(checkpoint_files)}: {step}", flush=True)
        print(f"{'='*80}", flush=True)
        
        try:
            # Load checkpoint
            model = load_checkpoint(ckpt_path, config, test_metadata)
            
            # Evaluate
            metrics = evaluate_checkpoint(model, test_loader, config)
            
            # Store results
            results[step] = metrics
            
            print(f"\n📊 Results for {step}:", flush=True)
            print(f"  Token Accuracy: {metrics['token_accuracy']*100:.2f}%", flush=True)
            print(f"  Exact Accuracy: {metrics['exact_accuracy']*100:.2f}%", flush=True)
            print(f"  Total Puzzles: {metrics['total_puzzles']}", flush=True)
            print(f"  Total Tokens: {metrics['total_tokens']}", flush=True)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error evaluating {step}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_path = checkpoint_dir / "test_evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}", flush=True)
    print(f"✅ EVALUATION COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"\nResults saved to: {output_path}", flush=True)
    
    # Print summary
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY - BASELINE TRM", flush=True)
    print(f"{'='*80}", flush=True)
    
    for step in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        metrics = results[step]
        epoch = int(step.split('_')[1]) * 50000 // 65104  # Convert step to epoch
        print(f"Epoch {epoch:5d} ({step:12s}): "
              f"Token={metrics['token_accuracy']*100:5.2f}%, "
              f"Exact={metrics['exact_accuracy']*100:5.2f}%", flush=True)
    
    # Final checkpoint
    final_step = sorted(results.keys(), key=lambda x: int(x.split('_')[1]))[-1]
    final_metrics = results[final_step]
    print(f"\n🎯 Final Test Accuracy: {final_metrics['exact_accuracy']*100:.2f}%", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"Compare with CGAR Test Accuracy: 86.02%", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Handle case where no checkpoints were successfully evaluated
    if not results:
        print("❌ No checkpoints were successfully evaluated!", flush=True)
        return


if __name__ == "__main__":
    main()

