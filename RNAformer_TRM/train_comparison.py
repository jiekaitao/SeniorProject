"""Train baseline RNAformer vs RNAformer_TRM side by side.

Usage:
    # Baseline (no TRM features)
    python3 train_comparison.py --variant baseline

    # TRM (dual-state + CGAR + ACT + RR losses)
    python3 train_comparison.py --variant trm

    # Run both concurrently:
    python3 train_comparison.py --variant baseline &
    python3 train_comparison.py --variant trm &
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'RNAformer'))

from RNAformer.utils.configuration import Config
from RNAformer.pl_module.datamodule_rna import DataModuleRNA


class CosineWarmupLambda:
    """Picklable cosine warmup schedule (replaces closure in lr_schedule.py)."""
    def __init__(self, num_warmup_steps, num_training_steps, decay_factor=0.1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.decay_factor = decay_factor

    def __call__(self, current_step):
        import math
        training_steps = self.num_training_steps - self.num_warmup_steps
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        cosine_decay = max(0.0, (1 + math.cos(
            math.pi * (current_step - self.num_warmup_steps) / float(max(1, training_steps)))) / 2)
        return self.decay_factor + (1 - self.decay_factor) * cosine_decay


def make_config(variant: str, args) -> Config:
    """Build training config for baseline or TRM variant."""
    project_root = os.path.dirname(os.path.abspath(__file__))

    config_dict = {
        'experiment': {
            'project_name': 'RNAformer_comparison',
            'session_name': f'{variant}',
            'experiment_name': f'{variant}_run',
        },
        'resume_training': False,
        'rna_data': {
            'dataframe_path': os.path.join(project_root, 'RNAformer', 'data', 'bprna_data.plk'),
            'num_cpu_worker': min(4, os.cpu_count() or 2),
            'min_len': args.min_len,
            'max_len': args.max_len,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'batch_by_token_size': False,
            'batch_token_size': None,
            'shuffle_pool_size': 1000,
            'cache_dir': os.path.join(project_root, 'cache', variant),
            'oversample_pdb': 1,
            'random_ignore_mat': False,
            'random_crop_mat': False,
            'valid_sets': ['bprna_ts0'],
            'test_sets': [],
        },
        'trainer': {
            '_target_': 'pytorch_lightning.Trainer',
            'max_epochs': args.max_epochs,
            'accelerator': 'gpu',
            'devices': 1,
            'precision': 'bf16-mixed',
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': args.accumulate_grad,
            'val_check_interval': args.val_check_interval,
            'log_every_n_steps': 10,
            'num_nodes': 1,
            'enable_checkpointing': True,
        },
        'train': {
            'seed': args.seed,
            'optimizer': {
                '_target_': 'torch.optim.AdamW',
                'lr': args.lr,
                'weight_decay': 0.01,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'scheduler_mult_factor': None,
            },
            'optimizer_param_grouping': {
                'normalization_regularization': False,
                'bias_regularization': False,
            },
            'scheduler': {
                'schedule': 'cosine',
                'num_warmup_steps': args.warmup_steps,
                'num_training_steps': args.max_epochs * 5000,  # approx steps/epoch * epochs
                'decay_factor': 0.1,
            },
        },
        'deepspeed': {},
        'RNAformer': {
            # Core architecture
            'model_dim': args.model_dim,
            'num_head': args.num_head,
            'n_layers': args.n_layers,
            'key_dim': args.model_dim // args.num_head,
            'ff_factor': 4,
            'ff_kernel': 3,
            'max_len': 200 if args.max_len <= 200 else 500,
            'initializer_range': 0.02,
            'seq_vocab_size': 5,   # overwritten by data module
            'trg_vocab_size': 2,   # overwritten by data module
            # Embedding
            'pos_embedding': True,
            'rel_pos_enc': True,
            # Attention
            'flash_attn': True,
            'rotary_emb': True,
            'rotary_emb_fraction': 0.5,
            'softmax_scale': None,
            'use_bias': True,
            'use_glu': False,
            # Normalization
            'ln_eps': 1e-5,
            'learn_ln': True,
            'zero_init': True,
            # Dropout
            'resi_dropout': 0.1,
            # Model flags
            'pdb_flag': True,
            'binary_output': True,
            'precision': 'bf16',
            # Cycling
            'cycling': args.cycling,
            # Phase 1: CGAR
            'cgar_enabled': False,
            'input_reinjection': False,
            'supervision_decay': 0.7,
            # Phase 2: Dual-state
            'dual_state': False,
            'H_cycles': 1,
            'L_cycles': 1,
            # Phase 3: ACT
            'act_enabled': False,
            'enable_brier_halting': False,
            'enable_monotonicity': False,
            'enable_smoothness': False,
        },
    }

    # Apply variant-specific settings
    if variant == 'trm':
        config_dict['RNAformer'].update({
            'cycling': args.cycling,
            'cgar_enabled': True,
            'input_reinjection': True,
            'dual_state': True,
            'H_cycles': args.h_cycles,
            'L_cycles': args.l_cycles,
            'act_enabled': True,
            'enable_brier_halting': True,
            'enable_monotonicity': True,
            'enable_smoothness': True,
        })

    return Config(config_dict=config_dict)


def main():
    parser = argparse.ArgumentParser(description='RNAformer comparison training')
    parser.add_argument('--variant', type=str, required=True, choices=['baseline', 'trm'],
                        help='Which variant to train')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate-grad', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accumulate)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--model-dim', type=int, default=128)
    parser.add_argument('--num-head', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--cycling', type=int, default=6)
    parser.add_argument('--h-cycles', type=int, default=2)
    parser.add_argument('--l-cycles', type=int, default=3)
    parser.add_argument('--min-len', type=int, default=16)
    parser.add_argument('--max-len', type=int, default=200)
    parser.add_argument('--val-check-interval', type=float, default=0.5,
                        help='Run validation every N fraction of epoch')
    parser.add_argument('--wandb-project', type=str, default='RNAformer-TRM-comparison')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(f'train_{args.variant}')

    # Set seeds
    pl.seed_everything(args.seed)

    # Build config
    cfg = make_config(args.variant, args)

    logger.info(f"=== Training {args.variant.upper()} variant ===")
    logger.info(f"Model: dim={args.model_dim}, heads={args.num_head}, layers={args.n_layers}")
    if args.variant == 'trm':
        logger.info(f"TRM: dual_state=True, H={args.h_cycles}, L={args.l_cycles}, "
                     f"CGAR=True, ACT=True, cycling={args.cycling}")
    else:
        logger.info(f"Baseline: cycling={args.cycling}, no TRM features")

    # Data module
    logger.info("Loading data...")
    data_module = DataModuleRNA(**cfg.rna_data, logger=logger)

    # Update model config from data
    cfg.RNAformer.seq_vocab_size = data_module.seq_vocab_size

    # Build trainer module (baseline or TRM)
    if args.variant == 'trm':
        from RNAformer_TRM.trainer.rna_trm_trainer import RNATRMTrainer
        model_module = RNATRMTrainer(
            cfg_train=cfg.train,
            cfg_model=cfg.RNAformer,
            py_logger=logger,
            data_module=data_module,
        )
    else:
        from RNAformer.pl_module.rna_folding_trainer import RNAFoldingTrainer
        model_module = RNAFoldingTrainer(
            cfg_train=cfg.train,
            cfg_model=cfg.RNAformer,
            py_logger=logger,
            data_module=data_module,
        )

    param_count = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {param_count:,}")

    # Override configure_optimizers with picklable scheduler
    _orig_configure = model_module.configure_optimizers
    def _configure_optimizers_picklable():
        from RNAformer.utils.group_parameters import group_parameters_for_optimizer
        from RNAformer.utils import instantiate
        parameters = group_parameters_for_optimizer(
            model_module.model, cfg.train.optimizer,
            normalization_regularization=cfg.train.optimizer_param_grouping.normalization_regularization,
            bias_regularization=cfg.train.optimizer_param_grouping.bias_regularization)
        optimizer = instantiate(cfg.train.optimizer, parameters)
        lr_lambda = CosineWarmupLambda(
            num_warmup_steps=cfg.train.scheduler.num_warmup_steps,
            num_training_steps=cfg.train.scheduler.num_training_steps,
            decay_factor=cfg.train.scheduler.decay_factor,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], {'scheduler': lr_scheduler, 'interval': 'step'}
    model_module.configure_optimizers = _configure_optimizers_picklable

    # wandb logger
    if args.no_wandb:
        training_logger = None
    else:
        training_logger = WandbLogger(
            project=args.wandb_project,
            name=f"{args.variant}_dim{args.model_dim}_L{args.n_layers}",
            tags=[args.variant, f'dim{args.model_dim}', f'layers{args.n_layers}'],
            config={
                'variant': args.variant,
                'model_dim': args.model_dim,
                'num_head': args.num_head,
                'n_layers': args.n_layers,
                'cycling': args.cycling,
                'batch_size': args.batch_size,
                'accumulate_grad': args.accumulate_grad,
                'effective_batch_size': args.batch_size * args.accumulate_grad,
                'lr': args.lr,
                'max_len': args.max_len,
                'seed': args.seed,
                **(
                    {'h_cycles': args.h_cycles, 'l_cycles': args.l_cycles,
                     'cgar': True, 'act': True, 'dual_state': True,
                     'input_reinjection': True, 'brier_halting': True,
                     'monotonicity': True, 'smoothness': True}
                    if args.variant == 'trm' else
                    {'cgar': False, 'act': False, 'dual_state': False}
                ),
            },
        )

    # Callbacks
    project_root = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(project_root, 'checkpoints', args.variant)
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f'{args.variant}' + '-{epoch:02d}-{val/bprna_ts0/f1_score:.4f}',
            monitor='val/bprna_ts0/f1_score',
            mode='max',
            save_top_k=3,
            save_last=True,
            save_weights_only=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=1,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=training_logger,
        enable_progress_bar=True,
        num_sanity_val_steps=1,
    )

    logger.info("Starting training...")
    trainer.fit(model=model_module, datamodule=data_module)

    logger.info("Training complete!")
    if training_logger:
        logger.info(f"wandb run: {training_logger.experiment.url}")


if __name__ == '__main__':
    main()
