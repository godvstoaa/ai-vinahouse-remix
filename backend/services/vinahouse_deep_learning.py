"""
Vinahouse Deep Learning - Train from 10,000+ tracks to generate creative drops
Uses: VAE + Transformer for drop generation, Style Transfer for remix

Architecture:
1. Dataset Pipeline: Load 10,000 Vinahouse tracks
2. Feature Extraction: Mel-spectrograms, beat grids, drop sections
3. VAE: Learn latent space of Vinahouse drops
4. Transformer: Generate new drop sequences
5. Style Transfer: Apply Vinahouse style to any input
"""
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import soundfile as sf
from scipy import signal
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Check GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


class VinahouseDropDataset(Dataset):
    """
    Dataset for Vinahouse drop sections.
    Extracts drop sections from tracks and converts to mel-spectrograms.
    """
    
    def __init__(self, 
                 data_path: str,
                 sample_rate: int = 44100,
                 n_mels: int = 128,
                 drop_duration: float = 8.0,  # 8 seconds drop
                 target_bpm: int = 135):
        self.data_path = Path(data_path)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.drop_duration = drop_duration
        self.target_bpm = target_bpm
        self.drop_samples = int(drop_duration * sample_rate)
        
        # Find all audio files
        self.audio_files = self._find_audio_files()
        logger.info(f"Found {len(self.audio_files)} audio files")
        
        # Cache for processed drops
        self.cache_path = self.data_path / ".drop_cache"
        self.cache_path.mkdir(exist_ok=True)
        
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in dataset"""
        extensions = ['.mp3', '.wav', '.flac', '.m4a']
        files = []
        for ext in extensions:
            files.extend(self.data_path.rglob(f'*{ext}'))
            files.extend(self.data_path.rglob(f'*{ext.upper()}'))
        return files[:10000]  # Max 10,000 files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        cache_file = self.cache_path / f"{audio_path.stem}_drop.pt"
        
        # Check cache
        if cache_file.exists():
            return torch.load(cache_file)
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=60)
            
            # Detect BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0])
            
            # Time stretch to target BPM
            if abs(tempo - self.target_bpm) > 5:
                rate = self.target_bpm / tempo
                rate = max(0.8, min(1.2, rate))
                y = librosa.effects.time_stretch(y, rate=rate)
            
            # Find drop section (highest energy 8-second section)
            drop_audio = self._find_drop_section(y)
            
            # Convert to mel-spectrogram
            mel = self._audio_to_mel(drop_audio)
            
            # Save cache
            torch.save(mel, cache_file)
            
            return mel
            
        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")
            # Return silence
            return torch.zeros(1, self.n_mels, int(self.drop_samples / 512))
    
    def _find_drop_section(self, y: np.ndarray) -> np.ndarray:
        """Find the drop section (highest energy section)"""
        # Calculate energy in 8-second windows
        window_samples = self.drop_samples
        hop_samples = window_samples // 4
        
        if len(y) < window_samples:
            # Pad if too short
            y = np.pad(y, (0, window_samples - len(y)))
            return y
        
        num_windows = (len(y) - window_samples) // hop_samples + 1
        energies = []
        
        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            section = y[start:end]
            
            # Calculate RMS energy with emphasis on bass
            bass = librosa.effects.highpass(section, 200)
            bass_energy = np.sqrt(np.mean(bass**2))
            total_energy = np.sqrt(np.mean(section**2))
            
            # Score: bass energy + total energy
            score = bass_energy * 2 + total_energy
            energies.append(score)
        
        # Find highest energy section
        best_idx = np.argmax(energies)
        start = best_idx * hop_samples
        end = start + window_samples
        
        return y[start:end]
    
    def _audio_to_mel(self, y: np.ndarray) -> torch.Tensor:
        """Convert audio to mel-spectrogram"""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512,
            fmin=20,
            fmax=16000
        )
        
        # Log scale
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize
        mel = (mel + 80) / 80  # Scale to 0-1
        
        return torch.FloatTensor(mel).unsqueeze(0)


class DropVAE(nn.Module):
    """
    Variational Autoencoder for Vinahouse drops.
    Learns latent representation of drop patterns.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 latent_dim: int = 512,
                 hidden_dims: List[int] = [32, 64, 128, 256]):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1] * 8 * 16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 8 * 16, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 8 * 16)
        
        modules = []
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )
        
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(0.2),
                nn.Conv2d(hidden_dims[-1], 1, kernel_size=3, padding=1),
                nn.Tanh()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space"""
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        result = self.decoder_input(z)
        result = result.view(-1, 256, 8, 16)
        return self.decoder(result)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def generate(self, num_samples: int = 1) -> torch.Tensor:
        """Generate new drops from random latent vectors"""
        z = torch.randn(num_samples, self.latent_dim).to(DEVICE)
        return self.decode(z)


class DropTransformer(nn.Module):
    """
    Transformer for generating drop sequences.
    Uses attention to create musically coherent patterns.
    """
    
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_proj = nn.Linear(128, d_model)  # mel bands to d_model
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 128)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate drop sequence"""
        # x: (batch, 1, mel, time)
        x = x.squeeze(1)  # (batch, mel, time)
        x = x.permute(0, 2, 1)  # (batch, time, mel)
        
        # Project to d_model
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Project back to mel
        x = self.output_proj(x)
        
        # Back to (batch, 1, mel, time)
        x = x.permute(0, 2, 1).unsqueeze(1)
        
        return x
    
    def generate(self, seed: torch.Tensor, length: int = 688) -> torch.Tensor:
        """Generate drop from seed"""
        self.eval()
        with torch.no_grad():
            generated = seed
            for _ in range(length - seed.shape[-1]):
                output = self.forward(generated)
                # Take last frame
                next_frame = output[:, :, :, -1:].detach()
                generated = torch.cat([generated, next_frame], dim=-1)
        return generated


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VinahouseStyleTransfer(nn.Module):
    """
    Style transfer network for converting any song to Vinahouse style.
    Uses adaptive instance normalization (AdaIN).
    """
    
    def __init__(self):
        super().__init__()
        
        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 9, padding=4),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        
        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 9, padding=4),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 9, padding=4),
        )
        
        # AdaIN parameters
        self.adain_mu = nn.Linear(128, 128)
        self.adain_sigma = nn.Linear(128, 128)
        
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Encode
        content_feat = self.content_encoder(content)
        style_feat = self.style_encoder(style)
        
        # AdaIN
        style_mu = torch.mean(style_feat, dim=[2, 3])
        style_sigma = torch.std(style_feat, dim=[2, 3]) + 1e-6
        
        content_mu = torch.mean(content_feat, dim=[2, 3])
        content_sigma = torch.std(content_feat, dim=[2, 3]) + 1e-6
        
        # Normalize content, denormalize with style statistics
        normalized = (content_feat - content_mu.unsqueeze(-1).unsqueeze(-1)) / content_sigma.unsqueeze(-1).unsqueeze(-1)
        stylized = normalized * style_sigma.unsqueeze(-1).unsqueeze(-1) + style_mu.unsqueeze(-1).unsqueeze(-1)
        
        # Decode
        return self.decoder(stylized)


class MelToAudio:
    """Convert mel-spectrogram back to audio using Griffin-Lim or vocoder"""
    
    def __init__(self, sample_rate: int = 44100, n_mels: int = 128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
    def griffin_lim(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """Convert mel to audio using Griffin-Lim"""
        # Denormalize
        mel = mel * 80 - 80
        
        # Convert to power
        mel = librosa.db_to_power(mel)
        
        # Invert mel to spectrogram
        S = librosa.feature.inverse.mel_to_stft(
            mel,
            sr=self.sample_rate,
            n_fft=2048,
            fmin=20,
            fmax=16000
        )
        
        # Griffin-Lim
        y = librosa.griffinlim(
            S,
            n_iter=n_iter,
            hop_length=512,
            win_length=2048,
            n_fft=2048
        )
        
        return y


class VinahouseDeepRemixer:
    """
    Main class for training and generating Vinahouse remixes.
    """
    
    def __init__(self, 
                 data_path: str,
                 checkpoint_path: str = "checkpoints"):
        self.data_path = data_path
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(exist_ok=True)
        
        self.vae = DropVAE().to(DEVICE)
        self.transformer = DropTransformer().to(DEVICE)
        self.style_transfer = VinahouseStyleTransfer().to(DEVICE)
        self.mel_to_audio = MelToAudio()
        
        # Optimizers
        self.vae_optimizer = torch.optim.AdamW(self.vae.parameters(), lr=1e-4)
        self.transformer_optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=1e-4)
        self.style_optimizer = torch.optim.AdamW(self.style_transfer.parameters(), lr=1e-4)
        
        # Mixed precision
        self.scaler = GradScaler()
        
    def train(self, epochs: int = 100, batch_size: int = 16, num_workers: int = 4):
        """
        Train the model on Vinahouse dataset.
        """
        logger.info("Starting training...")
        
        # Dataset
        dataset = VinahouseDropDataset(self.data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(epochs):
            self.vae.train()
            total_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, batch in enumerate(pbar):
                batch = batch.to(DEVICE)
                
                # VAE forward
                with autocast():
                    recon, mu, log_var = self.vae(batch)
                    
                    # Loss
                    recon_loss = F.mse_loss(recon, batch)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_loss + 0.001 * kl_loss
                
                # Backward
                self.vae_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.vae_optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
                
                # Generate sample
                self.generate_sample(epoch + 1)
        
        logger.info("Training complete!")
        
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'transformer_state_dict': self.transformer.state_dict(),
            'style_state_dict': self.style_transfer.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
        }
        path = self.checkpoint_path / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.style_transfer.load_state_dict(checkpoint['style_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        
    def generate_sample(self, epoch: int):
        """Generate a sample drop"""
        self.vae.eval()
        
        with torch.no_grad():
            # Generate from random latent
            generated = self.vae.generate(1)
            
            # Convert to audio
            mel = generated.squeeze().cpu().numpy()
            audio = self.mel_to_audio.griffin_lim(mel)
            
            # Save
            output_path = self.checkpoint_path / f"sample_epoch_{epoch}.wav"
            sf.write(output_path, audio, 44100)
            logger.info(f"Generated sample: {output_path}")
    
    def create_remix(self,
                     input_path: str,
                     output_path: str,
                     creativity: float = 0.8,
                     preserve_melody: float = 0.3) -> Dict:
        """
        Create a Vinahouse remix using trained models.
        
        Args:
            input_path: Input song
            output_path: Output remix path
            creativity: 0-1, how much AI creativity vs original
            preserve_melody: 0-1, how much of original melody to keep
        """
        logger.info(f"Creating remix: {input_path}")
        
        # Load input
        y, sr = librosa.load(input_path, sr=44100, duration=180)
        
        # Convert to mel
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel + 80) / 80
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Generate drop
        self.vae.eval()
        with torch.no_grad():
            # Generate new drop
            generated_drop = self.vae.generate(1)
            
            # Style transfer
            vinahouse_style = self._get_vinahouse_style()
            stylized = self.style_transfer(mel_tensor, vinahouse_style)
        
        # Combine original melody with generated elements
        creativity = max(0, min(1, creativity))
        preserve_melody = max(0, min(1, preserve_melody))
        
        combined = mel_tensor * preserve_melody + stylized * (1 - preserve_melody)
        combined = combined * creativity + mel_tensor * (1 - creativity)
        
        # Convert back to audio
        combined_mel = combined.squeeze().cpu().numpy()
        remixed_audio = self.mel_to_audio.griffin_lim(combined_mel)
        
        # Apply Vinahouse processing
        remixed_audio = self._apply_vinahouse_processing(remixed_audio)
        
        # Save
        sf.write(output_path, remixed_audio, sr)
        
        return {
            "input": input_path,
            "output": output_path,
            "creativity": creativity,
            "preserve_melody": preserve_melody,
            "success": True
        }
    
    def _get_vinahouse_style(self) -> torch.Tensor:
        """Get average Vinahouse style from trained VAE"""
        with torch.no_grad():
            # Use mean latent as style reference
            style_latent = torch.zeros(1, 512).to(DEVICE)
            style_mel = self.vae.decode(style_latent)
        return style_mel
    
    def _apply_vinahouse_processing(self, y: np.ndarray) -> np.ndarray:
        """Apply final Vinahouse processing"""
        # Bass boost
        sos_bass = signal.butter(4, 100, btype='low', fs=44100, output='sos')
        bass = signal.sosfilt(sos_bass, y) * 2
        
        # High pass original
        sos_hp = signal.butter(4, 60, btype='high', fs=44100, output='sos')
        y_hp = signal.sosfilt(sos_hp, y)
        
        # Combine
        y = y_hp + bass
        
        # Soft clip
        y = np.tanh(y * 0.8)
        
        # Normalize
        y = y / (np.max(np.abs(y)) + 1e-10) * 0.9
        
        return y


# Training script
def train_vinahouse_model(data_path: str, epochs: int = 100):
    """
    Train Vinahouse model from dataset.
    
    Usage:
        python -c "from vinahouse_deep_learning import train_vinahouse_model; train_vinahouse_model('D:/VinahouseMusic', 100)"
    """
    remixer = VinahouseDeepRemixer(data_path)
    remixer.train(epochs=epochs, batch_size=8, num_workers=4)
    return remixer


def create_remix_from_trained_model(input_path: str,
                                     output_path: str,
                                     checkpoint_path: str,
                                     creativity: float = 0.8):
    """
    Create remix using trained model.
    
    Usage:
        python -c "from vinahouse_deep_learning import create_remix_from_trained_model; create_remix_from_trained_model('song.mp3', 'remix.wav', 'checkpoints/best.pt')"
    """
    remixer = VinahouseDeepRemixer("", checkpoint_path=os.path.dirname(checkpoint_path))
    remixer.load_checkpoint(checkpoint_path)
    return remixer.create_remix(input_path, output_path, creativity)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vinahouse Deep Learning Remixer")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--remix", action="store_true", help="Create remix")
    parser.add_argument("--data_path", type=str, help="Path to Vinahouse dataset")
    parser.add_argument("--input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    
    args = parser.parse_args()
    
    if args.train and args.data_path:
        train_vinahouse_model(args.data_path, args.epochs)
    elif args.remix and args.input and args.output and args.checkpoint:
        create_remix_from_trained_model(args.input, args.output, args.checkpoint)