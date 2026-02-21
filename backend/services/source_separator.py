"""
Source Separator using spectral techniques
Separates audio into stems: vocals, drums, bass, other

Note: For higher quality separation, consider using:
- Demucs (https://github.com/facebookresearch/demucs)
- Spleeter (https://github.com/deezer/spleeter)

This implementation uses spectral filtering as a fallback.
"""
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter
from typing import Dict, Any, Optional, Tuple
import logging
import os
import uuid

logger = logging.getLogger(__name__)

class SourceSeparator:
    """
    Audio source separation using spectral techniques
    
    Separates into 4 stems:
    - vocals
    - drums  
    - bass
    - other (remaining instruments)
    """
    
    def __init__(self, output_dir: str = "uploads/stems"):
        self.output_dir = output_dir
        self.sample_rate = 44100
        self.hop_length = 1024
        self.n_fft = 4096
        os.makedirs(output_dir, exist_ok=True)
    
    async def separate(self, audio_path: str) -> Dict[str, str]:
        """
        Separate audio into stems
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Dict with paths to separated stem files
        """
        try:
            logger.info(f"Separating audio: {audio_path}")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Compute STFT
            stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Separate using spectral techniques
            stems = await self._separate_stems(magnitude, phase)
            
            # Save stems
            stem_paths = {}
            stem_id = str(uuid.uuid4())[:8]
            
            for stem_name, stem_audio in stems.items():
                filename = f"{stem_name}_{stem_id}.wav"
                filepath = os.path.join(self.output_dir, filename)
                sf.write(filepath, stem_audio, sr)
                stem_paths[stem_name] = filepath
                logger.info(f"Saved {stem_name} stem: {filepath}")
            
            return stem_paths
            
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            raise
    
    async def _separate_stems(
        self, 
        magnitude: np.ndarray, 
        phase: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Separate magnitude spectrogram into stems
        
        Uses spectral filtering based on frequency characteristics:
        - Bass: Low frequencies (20-250 Hz)
        - Drums: Transients and mid frequencies
        - Vocals: Mid-high frequencies with harmonic content
        - Other: Everything else
        """
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        # Frequency bands
        bass_mask = (freqs >= 20) & (freqs <= 250)
        low_mid_mask = (freqs > 250) & (freqs <= 500)
        mid_mask = (freqs > 500) & (freqs <= 2000)
        high_mid_mask = (freqs > 2000) & (freqs <= 4000)
        high_mask = (freqs > 4000) & (freqs <= 8000)
        
        # === BASS ===
        # Low frequency content
        bass_mag = np.zeros_like(magnitude)
        bass_mag[bass_mask] = magnitude[bass_mask]
        # Add some low-mid warmth
        bass_mag[low_mid_mask] = magnitude[low_mid_mask] * 0.6
        
        # === DRUMS ===
        # Transient detection - look for sharp attacks
        drum_mag = self._extract_drums(magnitude, freqs)
        
        # === VOCALS ===
        # Mid-range frequencies with harmonic content
        vocals_mag = self._extract_vocals(magnitude, freqs, phase)
        
        # === OTHER ===
        # Residual - what's left after removing the above
        other_mag = magnitude.copy()
        other_mag = np.maximum(other_mag - bass_mag * 0.5 - drum_mag * 0.5 - vocals_mag * 0.5, 0)
        # Enhance high frequencies (cymbals, synths, etc.)
        other_mag[high_mask] *= 1.2
        
        # Convert back to audio
        stems = {}
        
        for name, mag in [
            ('bass', bass_mag),
            ('drums', drum_mag),
            ('vocals', vocals_mag),
            ('other', other_mag)
        ]:
            # Reconstruct with original phase
            stft_separated = mag * np.exp(1j * phase)
            audio = librosa.istft(
                stft_separated, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                length=int(magnitude.shape[1] * self.hop_length)
            )
            stems[name] = audio
        
        # Normalize stems
        for name in stems:
            stems[name] = self._normalize(stems[name])
        
        return stems
    
    def _extract_drums(
        self, 
        magnitude: np.ndarray, 
        freqs: np.ndarray
    ) -> np.ndarray:
        """
        Extract drums using transient detection
        Drums have sharp attacks across many frequencies
        """
        drum_mag = np.zeros_like(magnitude)
        
        # Drums occupy broad frequency range
        # Focus on transient detection
        
        # Time-domain transient detection
        energy = np.sum(magnitude, axis=0)
        energy_diff = np.diff(energy, prepend=energy[0])
        
        # Find transients (positive energy jumps)
        transient_mask = energy_diff > np.percentile(energy_diff, 70)
        
        # Apply to drum frequencies (low-mid to high)
        drum_freq_mask = (freqs >= 60) & (freqs <= 8000)
        
        # Build drum magnitude
        for i, is_transient in enumerate(transient_mask):
            if is_transient:
                # Drums are stronger on transients
                drum_mag[drum_freq_mask, i] = magnitude[drum_freq_mask, i] * 0.8
            else:
                # Sustain is weaker
                drum_mag[drum_freq_mask, i] = magnitude[drum_freq_mask, i] * 0.3
        
        # Enhance low-end for kick
        kick_mask = (freqs >= 40) & (freqs <= 100)
        drum_mag[kick_mask] = magnitude[kick_mask] * 1.2
        
        # Enhance snare region
        snare_mask = (freqs >= 150) & (freqs <= 300)
        drum_mag[snare_mask] = magnitude[snare_mask] * 0.9
        
        return drum_mag
    
    def _extract_vocals(
        self, 
        magnitude: np.ndarray, 
        freqs: np.ndarray,
        phase: np.ndarray
    ) -> np.ndarray:
        """
        Extract vocals using harmonic content detection
        Vocals are typically in 200-3000 Hz range with harmonics
        """
        vocals_mag = np.zeros_like(magnitude)
        
        # Vocal fundamental frequency range
        vocal_range = (freqs >= 150) & (freqs <= 3000)
        
        # Extract harmonic content
        # Vocals have strong harmonic series
        for i, f in enumerate(freqs):
            if 150 <= f <= 3000:
                # Check for harmonics
                harmonics_strength = self._check_harmonics(magnitude, freqs, i)
                vocals_mag[i] = magnitude[i] * harmonics_strength
        
        # Enhance presence region (2-5 kHz for vocal clarity)
        presence_mask = (freqs >= 2000) & (freqs <= 5000)
        vocals_mag[presence_mask] = magnitude[presence_mask] * 0.7
        
        # Reduce bass frequencies (not vocals)
        bass_mask = freqs < 150
        vocals_mag[bass_mask] *= 0.2
        
        # Apply median filter to smooth
        vocals_mag = median_filter(vocals_mag, size=(1, 5))
        
        return vocals_mag
    
    def _check_harmonics(
        self, 
        magnitude: np.ndarray, 
        freqs: np.ndarray,
        base_idx: int
    ) -> float:
        """
        Check if a frequency has harmonic series (indicating pitched content like vocals)
        """
        base_freq = freqs[base_idx]
        if base_freq < 100 or base_freq > 2000:
            return 0.5
        
        base_amp = magnitude[base_idx]
        harmonic_score = 0
        count = 0
        
        for harmonic in [2, 3, 4]:
            target_freq = base_freq * harmonic
            if target_freq > freqs[-1]:
                break
            
            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - target_freq))
            
            # Check if harmonic exists
            if magnitude[idx] > base_amp * 0.1:  # Harmonic should be at least 10% of fundamental
                harmonic_score += 1
            count += 1
        
        if count == 0:
            return 0.5
        
        # Score from 0.3 to 1.0
        return 0.3 + (harmonic_score / count) * 0.7
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to -1 to 1 range"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.9
        return audio


# Singleton instance
source_separator = SourceSeparator()