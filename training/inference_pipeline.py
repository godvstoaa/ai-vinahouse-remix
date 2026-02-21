"""
Inference Pipeline - Sinh bản Remix Vinahouse từ bài nhạc gốc (Pop/Ballad/Hiphop)
Sử dụng DiT model đã train xong để tạo Beat + ghép với Vocal tách từ Demucs.

Usage:
    python training/inference_pipeline.py --input "path/to/song.mp3" --checkpoint "checkpoints/best.pt"
"""

import os
import sys
import yaml
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dit_model_architecture import DiT, DiTConfig, GaussianDiffusion, create_model


# Phase name to index mapping
PHASE_MAP = {"warmup": 0, "buildup": 1, "peak": 2, "cooldown": 3}

# Key name to index mapping
KEY_MAP = {k: i for i, k in enumerate([
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
    'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'
])}


class RemixInference:
    """
    Pipeline to generate a Vinahouse remix from any input song.
    
    Flow:
    1. Load trained DiT checkpoint
    2. Analyze input song (BPM, Key, Energy)
    3. Use DiT to generate Beat/Instrumental
    4. (Optional) Mix with separated Vocal from Demucs
    5. Export final remix
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "config/train_config.yaml",
        device: str = "auto"
    ):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[Inference] Device: {self.device}")
        
        # Load model
        self.model, self.diffusion = self._load_model(checkpoint_path)
        self.sample_rate = self.config.get('audio', {}).get('sample_rate', 44100)
    
    def _load_model(self, checkpoint_path: str):
        """Load trained DiT model from checkpoint"""
        print(f"[Inference] Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Build model from saved config
        model_config = checkpoint.get('config', {}).get('model', {})
        dit_config = DiTConfig(
            hidden_size=model_config.get('hidden_size', 768),
            num_layers=model_config.get('num_layers', 12),
            num_heads=model_config.get('num_heads', 12),
        )
        
        model = create_model(dit_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        diffusion = GaussianDiffusion(dit_config)
        
        print(f"[Inference] Model loaded: {model.get_param_count()['total_M']:.1f}M params")
        print(f"[Inference] Trained for {checkpoint.get('global_step', '?')} steps, epoch {checkpoint.get('epoch', '?')}")
        
        return model, diffusion
    
    def analyze_input(self, audio_path: str) -> Dict:
        """Analyze input song for BPM, Key, Energy"""
        print(f"[Inference] Analyzing: {Path(audio_path).name}")
        
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration = len(y) / sr
        
        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        
        # Key (simple chroma-based)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = int(np.argmax(np.mean(chroma, axis=1)))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_idx]
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = float(np.mean(rms))
        
        # Map to bin indices for the model
        bpm_bin = min(max(int((bpm - 100) / 2), 0), 39)  # 100-180 BPM -> 0-39
        key_class = KEY_MAP.get(key, 0)
        energy_level = min(int(avg_energy * 50), 4)  # 0-4
        
        analysis = {
            'bpm': bpm,
            'bpm_bin': bpm_bin,
            'key': key,
            'key_class': key_class,
            'energy': avg_energy,
            'energy_level': energy_level,
            'duration': duration,
        }
        
        print(f"[Inference] BPM: {bpm:.0f}, Key: {key}, Energy: {avg_energy:.3f}, Duration: {duration:.1f}s")
        return analysis
    
    @torch.no_grad()
    def generate_beat(
        self,
        analysis: Dict,
        target_duration: float = 180.0,
        temperature: float = 1.0,
        set_phase: str = "peak"
    ) -> np.ndarray:
        """
        Generate Vinahouse beat using the trained DiT model.
        
        Args:
            analysis: Output from analyze_input()
            target_duration: Target duration in seconds (default 3 minutes)  
            temperature: Sampling randomness (higher = more creative)
            set_phase: Emotional phase (warmup/buildup/peak/cooldown)
        
        Returns:
            Generated audio as numpy array
        """
        print(f"[Inference] Generating {target_duration:.0f}s beat (temperature={temperature}, phase={set_phase})...")
        
        # Prepare conditioning tensors
        bpm = torch.tensor([analysis['bpm_bin']], device=self.device)
        key = torch.tensor([analysis['key_class']], device=self.device)
        energy = torch.tensor([analysis['energy_level']], device=self.device)
        phase = torch.tensor([PHASE_MAP.get(set_phase, 2)], device=self.device)
        
        # Calculate output shape from target duration
        # Latent sequence length depends on sample_rate and hop_length
        hop_length = self.config.get('audio', {}).get('hop_length', 512)
        latent_dim = self.config.get('model', {}).get('latent_dim', 64)
        seq_len = int(target_duration * self.sample_rate / hop_length)
        
        # Clamp to model's context_length
        context_length = self.config.get('model', {}).get('context_length', 512)
        seq_len = min(seq_len, context_length)
        
        shape = (1, latent_dim, seq_len)
        
        # Generate using reverse diffusion
        generated_latent = self.diffusion.p_sample_loop(
            self.model, shape, bpm, key, energy,
            temperature=temperature, progress=True
        )
        
        # Convert latent to audio (simple mel inversion for now)
        audio = self._latent_to_audio(generated_latent, target_duration)
        
        print(f"[Inference] Generated {len(audio)/self.sample_rate:.1f}s of audio")
        return audio
    
    def _latent_to_audio(self, latent: torch.Tensor, target_duration: float) -> np.ndarray:
        """Convert model output latent back to audio waveform"""
        # Get the mel spectrogram from latent
        mel = latent.squeeze(0).cpu().numpy()
        
        # Inverse mel spectrogram to audio
        hop_length = self.config.get('audio', {}).get('hop_length', 512)
        n_fft = self.config.get('audio', {}).get('n_fft', 2048)
        
        # Use Griffin-Lim algorithm for mel -> audio conversion
        mel_linear = librosa.db_to_power(mel)
        audio = librosa.feature.inverse.mel_to_audio(
            mel_linear,
            sr=self.sample_rate,
            hop_length=hop_length,
            n_fft=n_fft
        )
        
        # Trim/pad to target duration
        target_samples = int(target_duration * self.sample_rate)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
        
        return audio
    
    def remix(
        self,
        input_path: str,
        output_path: str = "output_remix.wav",
        temperature: float = 1.0,
        set_phase: str = "peak",
        target_duration: Optional[float] = None
    ) -> str:
        """
        Full remix pipeline: Analyze -> Generate Beat -> Export
        
        Args:
            input_path: Path to input song (Pop/Ballad/Hiphop)
            output_path: Path to save the remix
            temperature: Creativity level (0.5=safe, 1.0=balanced, 1.5=wild)
            set_phase: Emotional intensity (warmup/buildup/peak/cooldown)
            target_duration: Override output duration (default: match input)
        
        Returns:
            Path to the generated remix file
        """
        print(f"\n{'='*60}")
        print(f"  AI VINAHOUSE REMIX PRO - Inference Pipeline")
        print(f"{'='*60}")
        
        # Step 1: Analyze input
        analysis = self.analyze_input(input_path)
        
        if target_duration is None:
            target_duration = min(analysis['duration'], 360)  # Cap at 6 minutes
        
        # Step 2: Generate beat
        beat = self.generate_beat(
            analysis, 
            target_duration=target_duration,
            temperature=temperature,
            set_phase=set_phase
        )
        
        # Step 3: Export
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        sf.write(output_path, beat, self.sample_rate)
        
        print(f"\n[Inference] ✅ Remix saved to: {output_path}")
        print(f"[Inference] Duration: {len(beat)/self.sample_rate:.1f}s")
        print(f"[Inference] BPM: {analysis['bpm']:.0f}, Key: {analysis['key']}")
        print(f"{'='*60}\n")
        
        return output_path


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Vinahouse Remix - Generate remix from any song')
    parser.add_argument('--input', '-i', required=True, help='Input song path (MP3/WAV/FLAC)')
    parser.add_argument('--output', '-o', default='output_remix.wav', help='Output remix path')
    parser.add_argument('--checkpoint', '-c', default='checkpoints/best.pt', help='Model checkpoint path')
    parser.add_argument('--config', default='config/train_config.yaml', help='Config file path')
    parser.add_argument('--temperature', '-t', type=float, default=1.0, help='Creativity (0.5-1.5)')
    parser.add_argument('--phase', '-p', default='peak', choices=['warmup', 'buildup', 'peak', 'cooldown'],
                       help='Emotional intensity phase')
    parser.add_argument('--duration', '-d', type=float, default=None, help='Output duration in seconds')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please train the model first using train_script.py")
        sys.exit(1)
    
    # Run inference
    pipeline = RemixInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    pipeline.remix(
        input_path=args.input,
        output_path=args.output,
        temperature=args.temperature,
        set_phase=args.phase,
        target_duration=args.duration
    )


if __name__ == '__main__':
    main()
