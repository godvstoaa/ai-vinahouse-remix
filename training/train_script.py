"""
Training Script for DiT Audio Model
Sử dụng HuggingFace Accelerate để tối ưu VRAM 8GB
"""

import os
import sys
import yaml
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from tqdm import tqdm
import argparse

# Accelerate for multi-GPU, mixed precision, gradient accumulation
from accelerate import Accelerator
from accelerate.utils import set_seed

# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dit_model_architecture import DiT, DiTConfig, GaussianDiffusion, create_model
from training.data_prep_pipeline import StreamingDataset


class Trainer:
    """
    DiT Trainer with Accelerate integration.
    Handles training loop, checkpointing, and logging.
    """
    
    def __init__(
        self,
        config_path: str = "config/train_config.yaml",
        output_dir: str = "checkpoints/"
    ):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config.get('mixed_precision', 'fp16'),
            gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 16),
            log_with="tensorboard",
            project_dir=self.output_dir / "logs"
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Device
        self.device = self.accelerator.device
        print(f"Using device: {self.device}")
        print(f"Mixed precision: {self.accelerator.mixed_precision}")
        
        # Build model
        self._build_model()
        
        # Build diffusion
        self._build_diffusion()
        
        # Build optimizer
        self._build_optimizer()
        
        # Build data loader
        self._build_dataloader()
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=self.output_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def _build_model(self):
        """Build DiT model"""
        model_config = DiTConfig(
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            mlp_ratio=self.config['model']['mlp_ratio'],
            latent_dim=self.config['model']['latent_dim'],
            context_length=self.config['model']['context_length'],
            bpm_bins=self.config['model']['conditioning']['bpm_bins'],
            key_classes=self.config['model']['conditioning']['key_classes'],
            energy_levels=self.config['model']['conditioning']['energy_levels'],
            num_timesteps=self.config['diffusion']['num_timesteps'],
            use_checkpoint=True  # Gradient checkpointing for memory
        )
        
        self.model = create_model(model_config)
        
        # Print model info
        params = self.model.get_param_count()
        print(f"\nModel Parameters:")
        print(f"  Total: {params['total_M']:.2f}M")
        print(f"  Trainable: {params['trainable_M']:.2f}M")
    
    def _build_diffusion(self):
        """Build Gaussian diffusion"""
        self.diffusion = GaussianDiffusion(DiTConfig(
            num_timesteps=self.config['diffusion']['num_timesteps']
        ))
    
    def _build_optimizer(self):
        """Build optimizer and scheduler"""
        train_config = self.config['training']
        
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=train_config['learning_rate'],
            total_steps=train_config['max_train_steps'],
            pct_start=train_config['warmup_steps'] / train_config['max_train_steps'],
            anneal_strategy='cos'
        )
    
    def _build_dataloader(self):
        """Build data loader"""
        # Check if processed data exists
        manifest_path = Path(self.config['data']['metadata_dir']) / 'manifest.json'
        
        if manifest_path.exists():
            print(f"\nLoading dataset from {manifest_path}")
            self.dataset = StreamingDataset(
                manifest_path=str(manifest_path),
                cache_dir=self.config['data']['cache_dir'],
                sample_rate=self.config['audio']['sample_rate']
            )
            
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
        else:
            print(f"\nWarning: No processed data found at {manifest_path}")
            print("Run data_prep_pipeline.py first to process your audio files.")
            self.dataset = None
            self.dataloader = None
    
    def _prepare_accelerator(self):
        """Prepare model and optimizer with Accelerator"""
        if self.dataloader:
            self.model, self.optimizer, self.dataloader, self.lr_scheduler = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.dataloader, self.lr_scheduler
                )
        else:
            self.model, self.optimizer = self.accelerator.prepare(
                self.model, self.optimizer
            )
    
    def _encode_condition(self, batch: Dict) -> tuple:
        """Encode BPM, key, energy, set_phase from batch to tensor indices"""
        # BPM to bin (100-180 -> 0-39)
        bpm = batch.get('bpm', torch.tensor([128] * len(batch.get('mel', [1]))))
        if isinstance(bpm, list):
            bpm = torch.tensor(bpm)
        bpm_bins = ((bpm.float() - 100) / 2).long().clamp(0, 39)
        
        # Key to index (default: C major = 0)
        key = batch.get('key', torch.tensor([0] * len(bpm_bins)))
        if isinstance(key, list):
            key = torch.tensor(key)
        else:
            key = torch.zeros(len(bpm_bins), dtype=torch.long)
        
        # Energy to index
        energy_map = {'low': 0, 'medium': 1, 'high': 2, 'build': 3, 'drop': 4}
        energy_labels = batch.get('energy_label', ['medium'] * len(bpm_bins))
        if isinstance(energy_labels, str):
            energy_labels = [energy_labels]
        energy = torch.tensor([energy_map.get(e, 1) for e in energy_labels])
        
        # Set phase to index (Macro-Conditioning from Long Audio)
        phase_map = {'warmup': 0, 'buildup': 1, 'peak': 2, 'cooldown': 3, 'main': 2}
        phase_labels = batch.get('set_phase', ['peak'] * len(bpm_bins))
        if isinstance(phase_labels, str):
            phase_labels = [phase_labels]
        set_phase = torch.tensor([phase_map.get(p, 2) for p in phase_labels])
        
        return bpm_bins, key, energy, set_phase
    
    def train_step(self, batch: Dict) -> float:
        """Single training step"""
        with self.accelerator.accumulate(self.model):
            # Get inputs
            mel = batch['mel'].to(self.device)
            if len(mel.shape) == 3:
                mel = mel.unsqueeze(1)  # Add channel dim
            
            # Encode conditioning
            bpm, key, energy, set_phase = self._encode_condition(batch)
            bpm, key, energy, set_phase = (
                bpm.to(self.device), key.to(self.device), 
                energy.to(self.device), set_phase.to(self.device)
            )
            
            # Sample timesteps
            batch_size = mel.size(0)
            t = torch.randint(
                0, self.config['diffusion']['num_timesteps'],
                (batch_size,), device=self.device
            )
            
            # Forward diffusion + prediction
            loss = self.diffusion.p_losses(
                self.model, mel, t, bpm, key, energy, set_phase
            )
            
            # Backward
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            return loss.item()
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint"""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.accelerator.get_state_dict(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Loaded checkpoint from step {self.global_step}")
    
    def log_metrics(self, loss: float, lr: float):
        """Log metrics to TensorBoard"""
        self.writer.add_scalar('train/loss', loss, self.global_step)
        self.writer.add_scalar('train/learning_rate', lr, self.global_step)
        
        # Log GPU memory
        if torch.cuda.is_available():
            self.writer.add_scalar(
                'system/gpu_memory_gb',
                torch.cuda.max_memory_allocated() / 1e9,
                self.global_step
            )
    
    def train(self):
        """Main training loop"""
        if self.dataloader is None:
            print("No data loaded. Please process your audio files first.")
            return
        
        self._prepare_accelerator()
        
        train_config = self.config['training']
        max_steps = train_config['max_train_steps']
        log_every = train_config.get('log_every', 100)
        save_every = train_config.get('save_every', 1000)
        
        print(f"\nStarting training...")
        print(f"  Max steps: {max_steps}")
        print(f"  Batch size: {train_config['batch_size']}")
        print(f"  Gradient accumulation: {train_config.get('gradient_accumulation_steps', 1)}")
        print(f"  Effective batch size: {train_config['batch_size'] * train_config.get('gradient_accumulation_steps', 1)}")
        
        self.model.train()
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        while self.global_step < max_steps:
            for batch in self.dataloader:
                if self.global_step >= max_steps:
                    break
                
                # Training step
                loss = self.train_step(batch)
                self.global_step += 1
                progress_bar.update(1)
                
                # Logging
                if self.global_step % log_every == 0:
                    lr = self.lr_scheduler.get_last_lr()[0]
                    self.log_metrics(loss, lr)
                    progress_bar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")
                
                # Checkpointing
                if self.global_step % save_every == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
                    
                    # Keep only last 3 checkpoints
                    self._cleanup_old_checkpoints(keep=3)
        
        # Final save
        self.save_checkpoint("final_model.pt")
        print("\nTraining complete!")
    
    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Remove old checkpoints to save disk space"""
        checkpoints = sorted(self.output_dir.glob("checkpoint_*.pt"))
        for old_ckpt in checkpoints[:-keep]:
            old_ckpt.unlink()
            print(f"Removed old checkpoint: {old_ckpt}")


def main():
    parser = argparse.ArgumentParser(description='Train DiT model')
    parser.add_argument('--config', '-c', default='config/train_config.yaml', help='Config file')
    parser.add_argument('--output', '-o', default='checkpoints/', help='Output directory')
    parser.add_argument('--resume', '-r', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # Create trainer
    trainer = Trainer(config_path=args.config, output_dir=args.output)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()