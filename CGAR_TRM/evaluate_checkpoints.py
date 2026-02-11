#!/usr/bin/env python3
"""
Evaluate CGAR checkpoints on test set
"""
import torch
import os
import sys
import json

# Add project root to path
sys.path.insert(0, '/data/TRM')

from pretrain import *
from models.recursive_reasoning.trm_cgar import TinyRecursiveReasoningModel_ACTV1_CGAR
from models.losses_cgar import ACTLossHead_CGAR

def load_checkpoint(checkpoint_path, config, test_metadata):
    """Load model from checkpoint - based on create_model in pretrain.py"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    from omegaconf import OmegaConf
    
    # Prepare model config (same as create_model in pretrain.py)
    model_cfg = dict(
        **{k: v for k, v in OmegaConf.to_container(config.arch, resolve=True).items() 
           if k not in ['loss', 'name']},
        batch_size=config.global_batch_size,
        vocab_size=test_metadata.vocab_size,
        seq_len=test_metadata.seq_len,
        num_puzzle_identifiers=test_metadata.num_puzzle_identifiers,
        causal=False
    )
    
    # Prepare loss config
    loss_config = {k: v for k, v in OmegaConf.to_container(config.arch.loss, resolve=True).items() 
                   if k not in ['name']}
    
    # Create model and loss head
    with torch.device("cuda"):
        model = TinyRecursiveReasoningModel_ACTV1_CGAR(model_cfg)
        loss_head = ACTLossHead_CGAR(model, **loss_config)
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Remove _orig_mod. prefix from torch.compile
    new_checkpoint = {}
    for key, value in checkpoint.items():
        new_key = key.replace('_orig_mod.', '')
        new_checkpoint[new_key] = value
    
    loss_head.load_state_dict(new_checkpoint)
    loss_head.eval()
    
    return loss_head

def evaluate_checkpoint(model, test_loader, config):
    """Evaluate model on test set - based on evaluate() in pretrain.py"""
    total_correct = 0
    total_samples = 0
    total_exact_correct = 0
    total_puzzles = 0
    
    print("Starting evaluation...")
    
    with torch.inference_mode():
        return_keys = ["preds"]
        
        processed_batches = 0
        
        for set_name, batch, global_batch_size in test_loader:
            processed_batches += 1
            if processed_batches % 10 == 0:
                print(f"  Processing batch {processed_batches}...", flush=True)
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            
            # Forward - loop until all_finish
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                
                if all_finish:
                    break
                
                if inference_steps > config.arch.halt_max_steps:
                    print(f"Warning: Exceeded max steps ({config.arch.halt_max_steps})")
                    break
            
            # Get predictions - preds["preds"] is already argmax'd (int64)
            predictions = preds["preds"]  # [batch, seq_len] - already token IDs
            labels = batch['labels']  # [batch, seq_len]
            
            if processed_batches == 1:
                print(f"  Debug: predictions shape={predictions.shape}, dtype={predictions.dtype}")
                print(f"  Debug: labels shape={labels.shape}, dtype={labels.dtype}")
                sys.stdout.flush()
            
            # Calculate token accuracy
            mask = labels != 0  # Ignore padding
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_samples += mask.sum().item()
            
            # Calculate exact accuracy (all tokens correct per puzzle)
            per_example_correct = ((predictions == labels) | (~mask)).all(dim=1)
            total_exact_correct += per_example_correct.sum().item()
            total_puzzles += predictions.shape[0]
            
            del carry, loss, preds, batch
    
    print(f"\n  Processed {processed_batches} batches, {total_puzzles} puzzles", flush=True)
    
    # Calculate final metrics
    token_accuracy = total_correct / total_samples if total_samples > 0 else 0
    exact_accuracy = total_exact_correct / total_puzzles if total_puzzles > 0 else 0
    
    return {
        'token_accuracy': token_accuracy,
        'exact_accuracy': exact_accuracy,
        'total_samples': total_samples,
        'total_puzzles': total_puzzles
    }

def main():
    print("=" * 80)
    print("CGAR Checkpoint Evaluation on Test Set")
    print("=" * 80)
    print()
    
    # Configuration
    checkpoint_dir = "/data/TRM/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/cgar_full_50k_20251016_103736"
    config_path = f"{checkpoint_dir}/all_config.yaml"
    
    # Load configuration
    print("Loading configuration...")
    from omegaconf import OmegaConf
    config_dict = OmegaConf.load(config_path)
    config = OmegaConf.create(config_dict)
    
    print(f"Dataset: {config.data_paths}")
    print(f"Architecture: {config.arch.name}")
    print()
    
    # Create test dataloader
    print("Creating test dataloader...")
    test_loader, test_metadata = create_dataloader(
        config, 
        "test",  # Use test set
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1
    )
    print(f"Test metadata: vocab_size={test_metadata.vocab_size}, seq_len={test_metadata.seq_len}")
    print(f"Total test samples: {test_metadata.total_puzzles if hasattr(test_metadata, 'total_puzzles') else 'unknown'}")
    print()
    
    # Find all checkpoints
    checkpoint_files = sorted([
        f for f in os.listdir(checkpoint_dir) 
        if f.startswith('step_') and not f.endswith('.yaml')
    ])
    
    if not checkpoint_files:
        print("❌ No checkpoints found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints:")
    for cf in checkpoint_files:
        print(f"  - {cf}")
    print()
    
    # Evaluate each checkpoint
    results = {}
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        epoch = checkpoint_file.replace('step_', '')
        
        print(f"\n{'='*80}")
        print(f"Evaluating Checkpoint: {checkpoint_file}")
        print(f"{'='*80}")
        
        try:
            # Load model
            model = load_checkpoint(checkpoint_path, config, test_metadata)
            
            # Evaluate
            metrics = evaluate_checkpoint(model, test_loader, config)
            
            # Store results
            results[epoch] = metrics
            
            # Print results
            print(f"\n📊 Results for {checkpoint_file}:")
            print(f"  Token Accuracy:  {metrics['token_accuracy']*100:.2f}%")
            print(f"  Exact Accuracy:  {metrics['exact_accuracy']*100:.2f}%")
            print(f"  Total Puzzles:   {metrics['total_puzzles']}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_file = f"{checkpoint_dir}/test_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {results_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<15} {'Token Acc':<12} {'Exact Acc':<12}")
    print(f"{'-'*40}")
    
    for epoch in sorted(results.keys(), key=lambda x: int(x)):
        metrics = results[epoch]
        print(f"step_{epoch:<10} {metrics['token_accuracy']*100:>10.2f}% {metrics['exact_accuracy']*100:>10.2f}%")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
