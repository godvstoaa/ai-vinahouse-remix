"""
Real Remix Generator using scipy/numpy effects
Creates actual remixes with audio processing
"""
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Any, Optional, Tuple
import logging
import os
import uuid
import json

logger = logging.getLogger(__name__)

class RemixGenerator:
    """Real remix generation with audio effects"""
    
    # Default genre-specific presets (used if no trained profiles)
    DEFAULT_PRESETS = {
        'edm': {
            'target_bpm': 128,
            'kick_pattern': 'four_on_floor',
            'bass_style': 'sub',
            'effects': {'reverb': 0.3, 'delay': 0.2, 'sidechain': True}
        },
        'house': {
            'target_bpm': 124,
            'kick_pattern': 'four_on_floor',
            'bass_style': 'deep',
            'effects': {'reverb': 0.4, 'delay': 0.3, 'sidechain': True}
        },
        'hiphop': {
            'target_bpm': 90,
            'kick_pattern': 'boom_bap',
            'bass_style': '808',
            'effects': {'reverb': 0.2, 'delay': 0.1, 'sidechain': False}
        },
        'trap': {
            'target_bpm': 140,
            'kick_pattern': 'trap',
            'bass_style': '808_long',
            'effects': {'reverb': 0.5, 'delay': 0.25, 'sidechain': True}
        },
        'lofi': {
            'target_bpm': 75,
            'kick_pattern': 'swing',
            'bass_style': 'soft',
            'effects': {'reverb': 0.6, 'delay': 0.4, 'lowpass': True}
        },
        'techno': {
            'target_bpm': 132,
            'kick_pattern': 'four_on_floor',
            'bass_style': 'acid',
            'effects': {'reverb': 0.35, 'delay': 0.3, 'sidechain': True}
        },
        'dubstep': {
            'target_bpm': 140,
            'kick_pattern': 'half_time',
            'bass_style': 'wobble',
            'effects': {'reverb': 0.4, 'delay': 0.35, 'sidechain': True}
        },
        'pop': {
            'target_bpm': 120,
            'kick_pattern': 'pop',
            'bass_style': 'synth',
            'effects': {'reverb': 0.3, 'delay': 0.2, 'sidechain': True}
        },
    }
    
    def __init__(self, output_dir: str = "uploads/remixes"):
        self.output_dir = output_dir
        self.sample_rate = 44100
        os.makedirs(output_dir, exist_ok=True)
        
        # Load trained style profiles (if available)
        self.genre_presets = self.DEFAULT_PRESETS.copy()
        self.trained_profiles = {}
        self._load_trained_profiles()
    
    def _load_trained_profiles(self):
        """Load learned style profiles from training"""
        profile_path = "models/style_profiles.json"
        try:
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    self.trained_profiles = json.load(f)
                
                # Merge trained profiles with defaults
                for genre, profile in self.trained_profiles.items():
                    if genre not in self.genre_presets:
                        self.genre_presets[genre] = {}
                    
                    # Use learned BPM if available
                    if 'avg_bpm' in profile:
                        self.genre_presets[genre]['target_bpm'] = int(profile['avg_bpm'])
                    
                    # Use learned remix params
                    if 'remix_params' in profile:
                        self.genre_presets[genre]['effects'] = profile['remix_params']
                    
                    logger.info(f"Loaded trained profile for {genre}: BPM={self.genre_presets[genre].get('target_bpm')}")
                
                logger.info(f"Loaded {len(self.trained_profiles)} trained genre profiles!")
        except Exception as e:
            logger.warning(f"Could not load trained profiles: {e}")
    
    def get_available_genres(self):
        """Get list of available genres (including trained ones)"""
        return list(self.genre_presets.keys())
    
    def get_genre_info(self, genre: str) -> Dict:
        """Get genre preset info"""
        return self.genre_presets.get(genre, self.genre_presets.get('edm', {}))
    
    async def generate(
        self,
        audio_path: str,
        stems: Dict[str, str],
        settings: Dict[str, Any],
        original_bpm: float = 120.0,
        original_key: str = "C major"
    ) -> Dict[str, Any]:
        """
        Generate a remix
        
        Args:
            audio_path: Path to original audio
            stems: Dict of stem paths (vocals, drums, bass, other)
            settings: Remix settings (genre, bpm, effects, etc.)
            original_bpm: Original track BPM
            original_key: Original track key
            
        Returns:
            Dict with remix path and metadata
        """
        try:
            logger.info(f"Generating remix with settings: {settings}")
            
            genre = settings.get('genre', 'edm')
            target_bpm = settings.get('bpm', 128)
            energy_level = settings.get('energyLevel', 0.7)
            style = settings.get('style', 'club')
            
            # Get genre preset (uses trained profiles if available!)
            preset = self.genre_presets.get(genre, self.genre_presets.get('edm', {}))
            logger.info(f"Using preset for {genre}: {preset}")
            
            # Load all stems
            vocals, _ = self._load_audio(stems.get('vocals'))
            drums, _ = self._load_audio(stems.get('drums'))
            bass, _ = self._load_audio(stems.get('bass'))
            other, _ = self._load_audio(stems.get('other'))
            
            # Ensure same length
            max_len = max(len(vocals), len(drums), len(bass), len(other))
            vocals = self._pad_to_length(vocals, max_len)
            drums = self._pad_to_length(drums, max_len)
            bass = self._pad_to_length(bass, max_len)
            other = self._pad_to_length(other, max_len)
            
            # Apply tempo change if needed
            tempo_ratio = target_bpm / original_bpm if original_bpm > 0 else 1.0
            
            if abs(tempo_ratio - 1.0) > 0.05:  # More than 5% difference
                logger.info(f"Applying tempo change: {tempo_ratio:.2f}x")
                vocals = self._change_tempo(vocals, tempo_ratio)
                drums = self._change_tempo(drums, tempo_ratio)
                bass = self._change_tempo(bass, tempo_ratio)
                other = self._change_tempo(other, tempo_ratio)
            
            # Apply genre-specific processing
            vocals = self._process_vocals(vocals, genre, settings)
            drums = self._process_drums(drums, genre, settings, energy_level)
            bass = self._process_bass(bass, genre, settings, energy_level)
            other = self._process_other(other, genre, settings)
            
            # Mix stems together with new levels
            mix = self._mix_stems(vocals, drums, bass, other, genre, energy_level)
            
            # Apply master effects
            mix = self._apply_master_effects(mix, settings)
            
            # Normalize
            mix = self._normalize(mix)
            
            # Save remix
            remix_id = str(uuid.uuid4())
            remix_filename = f"remix_{remix_id}.wav"
            remix_path = os.path.join(self.output_dir, remix_filename)
            
            sf.write(remix_path, mix, self.sample_rate)
            
            logger.info(f"Remix saved: {remix_path}")
            
            return {
                'remix_id': remix_id,
                'remix_path': remix_path,
                'remix_url': f"/remixes/{remix_filename}",
                'duration': len(mix) / self.sample_rate,
                'genre': genre,
                'bpm': target_bpm,
                'style': style
            }
            
        except Exception as e:
            logger.error(f"Remix generation failed: {e}")
            raise
    
    def _load_audio(self, path: Optional[str]) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        if path and os.path.exists(path):
            y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return y, sr
        return np.zeros(self.sample_rate * 30), self.sample_rate  # 30 sec silence
    
    def _pad_to_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad audio to target length"""
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)))
        return audio[:target_length]
    
    def _change_tempo(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """Change tempo using librosa time stretch"""
        try:
            # Use librosa's time stretch (inverse of ratio for tempo)
            stretch_rate = 1.0 / ratio
            return librosa.effects.time_stretch(audio, rate=stretch_rate)
        except Exception as e:
            logger.warning(f"Tempo change failed: {e}")
            return audio
    
    def _process_vocals(self, vocals: np.ndarray, genre: str, settings: Dict) -> np.ndarray:
        """Process vocals based on genre"""
        reverb = settings.get('reverb', 0.3)
        delay = settings.get('delay', 0.2)
        
        # Apply reverb
        if reverb > 0:
            vocals = self._apply_reverb(vocals, reverb * 0.5)
        
        # Apply delay
        if delay > 0:
            vocals = self._apply_delay(vocals, delay * 0.3, delay_time=0.25)
        
        # Genre-specific processing
        if genre == 'trap':
            # Pitch shift down slightly
            vocals = self._pitch_shift(vocals, -2)
        elif genre == 'lofi':
            # Add subtle pitch wobble
            vocals = self._add_pitch_wobble(vocals, depth=0.5)
        
        return vocals
    
    def _process_drums(
        self, 
        drums: np.ndarray, 
        genre: str, 
        settings: Dict,
        energy: float
    ) -> np.ndarray:
        """Process drums based on genre"""
        # Enhance transients
        drums = self._enhance_transients(drums, amount=energy)
        
        # Genre-specific processing
        if genre in ['edm', 'house', 'techno']:
            # Compress and enhance
            drums = self._compress(drums, threshold=0.5, ratio=3.0)
            drums = self._enhance_low_end(drums, gain=1.2)
        elif genre == 'trap':
            # Heavy 808 style
            drums = self._enhance_low_end(drums, gain=1.5)
        elif genre == 'lofi':
            # Degrade quality slightly for lofi feel
            drums = self._apply_lofi_effect(drums)
        
        return drums
    
    def _process_bass(
        self, 
        bass: np.ndarray, 
        genre: str, 
        settings: Dict,
        energy: float
    ) -> np.ndarray:
        """Process bass based on genre"""
        # Enhance low end
        bass = self._enhance_low_end(bass, gain=1.3)
        
        if genre == 'dubstep':
            # Add wobble effect
            bass = self._add_wobble(bass, rate=2.0, depth=0.4)
        elif genre == 'trap':
            # Sustained 808
            bass = self._extend_bass_sustain(bass)
        
        return bass
    
    def _process_other(self, other: np.ndarray, genre: str, settings: Dict) -> np.ndarray:
        """Process other instruments"""
        reverb = settings.get('reverb', 0.3)
        filter_val = settings.get('filter', 0.5)
        
        # Apply filter
        if filter_val > 0.5:
            # Low pass
            cutoff = 2000 + (1 - filter_val) * 8000
            other = self._low_pass(other, cutoff)
        elif filter_val < 0.3:
            # High pass for clarity
            other = self._high_pass(other, 200)
        
        # Apply reverb
        if reverb > 0:
            other = self._apply_reverb(other, reverb * 0.4)
        
        return other
    
    def _mix_stems(
        self,
        vocals: np.ndarray,
        drums: np.ndarray,
        bass: np.ndarray,
        other: np.ndarray,
        genre: str,
        energy: float
    ) -> np.ndarray:
        """Mix all stems together with genre-appropriate levels"""
        
        # Genre-specific mix levels
        levels = {
            'edm': {'vocals': 0.7, 'drums': 1.0, 'bass': 0.9, 'other': 0.6},
            'house': {'vocals': 0.8, 'drums': 1.0, 'bass': 0.85, 'other': 0.65},
            'hiphop': {'vocals': 0.9, 'drums': 0.95, 'bass': 0.8, 'other': 0.5},
            'trap': {'vocals': 0.75, 'drums': 1.0, 'bass': 0.95, 'other': 0.55},
            'lofi': {'vocals': 0.6, 'drums': 0.7, 'bass': 0.65, 'other': 0.8},
            'techno': {'vocals': 0.5, 'drums': 1.0, 'bass': 0.9, 'other': 0.6},
            'dubstep': {'vocals': 0.6, 'drums': 1.0, 'bass': 1.0, 'other': 0.5},
            'pop': {'vocals': 1.0, 'drums': 0.85, 'bass': 0.7, 'other': 0.65},
        }
        
        level = levels.get(genre, levels['edm'])
        
        # Ensure same length
        max_len = max(len(vocals), len(drums), len(bass), len(other))
        vocals = self._pad_to_length(vocals, max_len)
        drums = self._pad_to_length(drums, max_len)
        bass = self._pad_to_length(bass, max_len)
        other = self._pad_to_length(other, max_len)
        
        # Mix
        mix = (
            vocals * level['vocals'] * energy +
            drums * level['drums'] +
            bass * level['bass'] * (0.5 + energy * 0.5) +
            other * level['other']
        )
        
        return mix
    
    def _apply_master_effects(self, audio: np.ndarray, settings: Dict) -> np.ndarray:
        """Apply master bus effects"""
        reverb = settings.get('reverb', 0.3)
        energy = settings.get('energyLevel', 0.7)
        
        # Light master reverb
        audio = self._apply_reverb(audio, reverb * 0.15)
        
        # Multiband compression simulation
        audio = self._compress(audio, threshold=0.7, ratio=2.0)
        
        # Final limiter
        audio = self._limit(audio, threshold=0.95)
        
        return audio
    
    # === Audio Effects ===
    
    def _apply_reverb(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Simple reverb using convolution"""
        if amount <= 0:
            return audio
        
        # Create impulse response
        delay_samples = int(self.sample_rate * 0.1)  # 100ms
        decay = np.exp(-np.linspace(0, 5, delay_samples))
        impulse = np.zeros(delay_samples)
        impulse[0] = 1
        impulse[delay_samples//4::delay_samples//8] = decay[::8][:len(impulse[delay_samples//4::delay_samples//8])]
        
        # Convolve
        reverb = np.convolve(audio, impulse, mode='same') * amount
        
        return audio + reverb
    
    def _apply_delay(
        self, 
        audio: np.ndarray, 
        amount: float, 
        delay_time: float = 0.25
    ) -> np.ndarray:
        """Simple delay effect"""
        if amount <= 0:
            return audio
        
        delay_samples = int(self.sample_rate * delay_time)
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples] * amount
        
        return audio + delayed
    
    def _compress(
        self, 
        audio: np.ndarray, 
        threshold: float = 0.5, 
        ratio: float = 4.0
    ) -> np.ndarray:
        """Simple compression"""
        abs_audio = np.abs(audio)
        over_threshold = abs_audio > threshold
        
        compressed = audio.copy()
        compressed[over_threshold] = np.sign(audio[over_threshold]) * (
            threshold + (abs_audio[over_threshold] - threshold) / ratio
        )
        
        return compressed
    
    def _limit(self, audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Hard limiter"""
        return np.clip(audio, -threshold, threshold)
    
    def _enhance_transients(self, audio: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """Enhance transient attacks"""
        # Differentiate to find transients
        diff = np.diff(audio, prepend=audio[0])
        
        # Enhance
        enhanced = audio + diff * amount * 2
        
        return enhanced
    
    def _enhance_low_end(self, audio: np.ndarray, gain: float = 1.2) -> np.ndarray:
        """Enhance bass frequencies"""
        # Simple low shelf boost
        low_passed = self._low_pass(audio, 200)
        
        return audio + low_passed * (gain - 1)
    
    def _low_pass(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """Low pass filter"""
        nyquist = self.sample_rate / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio)
    
    def _high_pass(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """High pass filter"""
        nyquist = self.sample_rate / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def _pitch_shift(self, audio: np.ndarray, semitones: int) -> np.ndarray:
        """Pitch shift using librosa"""
        try:
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=semitones)
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio
    
    def _add_pitch_wobble(self, audio: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """Add subtle pitch wobble for lofi effect"""
        # Create LFO
        t = np.arange(len(audio)) / self.sample_rate
        lfo = np.sin(2 * np.pi * 0.2 * t) * depth  # 0.2 Hz LFO
        
        # This is a simplified version - real pitch wobble requires phase vocoder
        return audio * (1 + lfo * 0.01)
    
    def _add_wobble(self, audio: np.ndarray, rate: float = 2.0, depth: float = 0.3) -> np.ndarray:
        """Add wobble effect for dubstep bass"""
        t = np.arange(len(audio)) / self.sample_rate
        lfo = (np.sin(2 * np.pi * rate * t) + 1) / 2  # 0 to 1
        
        # Modulate amplitude
        return audio * (1 - depth + lfo * depth)
    
    def _extend_bass_sustain(self, bass: np.ndarray) -> np.ndarray:
        """Extend bass sustain for trap 808 style"""
        # Simple envelope following and extension
        envelope = np.abs(bass)
        smoothed = gaussian_filter1d(envelope, sigma=int(self.sample_rate * 0.1))
        
        # Extend
        return bass * (1 + smoothed * 0.5)
    
    def _apply_lofi_effect(self, audio: np.ndarray) -> np.ndarray:
        """Apply lofi degradation"""
        # Reduce sample rate simulation
        downsample = 4
        upsampled = np.repeat(audio[::downsample], downsample)
        
        # Match length
        if len(upsampled) < len(audio):
            upsampled = np.pad(upsampled, (0, len(audio) - len(upsampled)))
        else:
            upsampled = upsampled[:len(audio)]
        
        # Mix with original
        return audio * 0.6 + upsampled * 0.4
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to -1 to 1 range"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.9  # Leave headroom
        return audio


# Singleton instance
remix_generator = RemixGenerator()