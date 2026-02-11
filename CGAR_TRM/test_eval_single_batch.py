#!/usr/bin/env python3
"""
Test evaluation on a single batch to debug
"""
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, '/data/TRM')

from pretrain import *
from models.recursive_reasoning.trm_cgar import TinyRecursiveReasoningModel_ACTV1_CGAR
from models.losses_cgar import ACTLossHead_CGAR

def main():
    print("="*80, flush=True)
    print("SINGLE BATCH TEST", flush=True)
    print("="*80, flush=True)
    
    checkpoint_dir = "/data/TRM/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/cgar_full_50k_20251016_103736"
    checkpoint_path = f"{checkpoint_dir}/step_65100"  # Test with final checkpoint
    config_path = f"{checkpoint_dir}/all_config.yaml"
    
    print("\n1. Loading configuration...", flush=True)
    from omegaconf import OmegaConf
    config = OmegaConf.create(OmegaConf.load(config_path))
    
    print("2. Creating test dataloader...", flush=True)
    test_loader, test_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1
    )
    print(f"   Test metadata: vocab={test_metadata.vocab_size}, seq_len={test_metadata.seq_len}", flush=True)
    
    print("\n3. Loading model...", flush=True)
    model_cfg = dict(
        **{k: v for k, v in OmegaConf.to_container(config.arch, resolve=True).items() 
           if k not in ['loss', 'name']},
        batch_size=config.global_batch_size,
        vocab_size=test_metadata.vocab_size,
        seq_len=test_metadata.seq_len,
        num_puzzle_identifiers=test_metadata.num_puzzle_identifiers,
        causal=False
    )
    loss_config = {k: v for k, v in OmegaConf.to_container(config.arch.loss, resolve=True).items() 
                   if k not in ['name']}
    
    with torch.device("cuda"):
        model = TinyRecursiveReasoningModel_ACTV1_CGAR(model_cfg)
        loss_head = ACTLossHead_CGAR(model, **loss_config)
    
    print("4. Loading checkpoint weights...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    new_checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    loss_head.load_state_dict(new_checkpoint)
    loss_head.eval()
    print("   Model loaded successfully!", flush=True)
    
    print("\n5. Getting first batch...", flush=True)
    with torch.inference_mode():
        for set_name, batch, global_batch_size in test_loader:
            print(f"   Got batch: set={set_name}, batch_size={list(batch.values())[0].shape[0]}", flush=True)
            
            # Move to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            print(f"   Batch keys: {list(batch.keys())}", flush=True)
            print(f"   Labels shape: {batch['labels'].shape}", flush=True)
            
            print("\n6. Running inference...", flush=True)
            with torch.device("cuda"):
                carry = loss_head.initial_carry(batch)
            print(f"   Initial carry created", flush=True)
            
            inference_steps = 0
            while True:
                print(f"   Inference step {inference_steps+1}...", flush=True)
                carry, loss, metrics, preds, all_finish = loss_head(
                    carry=carry, batch=batch, return_keys=["preds"]
                )
                inference_steps += 1
                
                if all_finish:
                    print(f"   All finished after {inference_steps} steps!", flush=True)
                    break
                    
                if inference_steps > config.arch.halt_max_steps:
                    print(f"   Exceeded max steps ({config.arch.halt_max_steps})", flush=True)
                    break
            
            print(f"\n7. Checking predictions...", flush=True)
            print(f"   preds dict keys: {list(preds.keys())}", flush=True)
            print(f"   preds['preds'] shape: {preds['preds'].shape}", flush=True)
            print(f"   preds['preds'] dtype: {preds['preds'].dtype}", flush=True)
            print(f"   preds['preds'] ndim: {preds['preds'].ndim}", flush=True)
            
            if preds['preds'].ndim == 3:
                print(f"   Shape is [batch={preds['preds'].shape[0]}, seq={preds['preds'].shape[1]}, vocab={preds['preds'].shape[2]}]", flush=True)
                predictions = preds['preds'].argmax(dim=-1)
                print(f"   After argmax: {predictions.shape}", flush=True)
            else:
                predictions = preds['preds']
                print(f"   Using preds as-is: {predictions.shape}", flush=True)
            
            labels = batch['labels']
            print(f"   Labels shape: {labels.shape}", flush=True)
            
            if predictions.shape == labels.shape:
                print(f"\n✅ SHAPES MATCH!", flush=True)
                mask = labels != 0
                correct = (predictions == labels) & mask
                token_acc = correct.sum().item() / mask.sum().item()
                exact_acc = ((predictions == labels) | (~mask)).all(dim=1).sum().item() / predictions.shape[0]
                print(f"   Token Accuracy: {token_acc*100:.2f}%", flush=True)
                print(f"   Exact Accuracy: {exact_acc*100:.2f}%", flush=True)
            else:
                print(f"\n❌ SHAPE MISMATCH!", flush=True)
                print(f"   predictions: {predictions.shape}", flush=True)
                print(f"   labels: {labels.shape}", flush=True)
            
            # Only test one batch
            break
    
    print("\n" + "="*80, flush=True)
    print("TEST COMPLETE", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()




