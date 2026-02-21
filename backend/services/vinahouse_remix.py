"""
Vinahouse Remix Engine - Club-Ready Vinahouse Production
Optimized for Vietnamese Club Music
BPM: 128-140 | Style: High Energy Dance
"""
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VinahouseRemixEngine:
    """
    Specialized engine for Vinahouse remixes.
    
    Vinahouse Characteristics:
    - BPM: 128-140 (typically 132-138)
    - Strong 4/4 kick pattern
    - Simple, catchy melodies
    - High energy bass
    - Vocal chops common
    - Sidechain compression
    - Bright, punchy sound
    """
    
    # Standard Vinahouse BPM range
    VINAHOUSE_BPM_RANGE = (128, 140)
    DEFAULT_BPM = 135
    
    # Vinahouse-specific EQ curves
    EQ_PRESETS = {
        "club": {
            "sub_bass": (40, 4),      # Sub bass boost
            "kick": (60, 3),          # Kick punch
            "bass": (100, 2),         # Bass body
            "low_mid": (250, -2),     # Remove mud
            "mid": (1000, 0),         # Neutral
            "presence": (3000, 2),    # Presence
            "high": (8000, 3),        # Brightness
            "air": (12000, 2),        # Air
        },
        "festival": {
            "sub_bass": (40, 6),
            "kick": (60, 4),
            "bass": (100, 3),
            "low_mid": (250, -3),
            "mid": (1000, 0),
            "presence": (3000, 3),
            "high": (8000, 4),
            "air": (12000, 3),
        },
        "radio": {
            "sub_bass": (40, 2),
            "kick": (60, 2),
            "bass": (100, 1),
            "low_mid": (250, -1),
            "mid": (1000, 1),
            "presence": (3000, 2),
            "high": (8000, 2),
            "air": (12000, 2),
        },
    }
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def create_vinahouse_remix(self,
                               input_path: str,
                               output_path: str,
                               target_bpm: int = 135,
                               preset: str = "club",
                               energy_level: float = 1.0,
                               add_buildups: bool = True,
                               sidechain_intensity: float = 0.7) -> Dict:
        """
        Create a Vinahouse club remix.
        
        Args:
            input_path: Input audio file
            output_path: Output remix file
            target_bpm: Target BPM (128-140)
            preset: "club", "festival", "radio"
            energy_level: 0.5-1.5 energy multiplier
            add_buildups: Add build-up sections
            sidechain_intensity: Sidechain compression (0-1)
            
        Returns:
            Dict with remix info
        """
        logger.info(f"Creating Vinahouse remix: {input_path}")
        
        # Validate BPM
        if target_bpm < self.VINAHOUSE_BPM_RANGE[0]:
            target_bpm = self.VINAHOUSE_BPM_RANGE[0]
        elif target_bpm > self.VINAHOUSE_BPM_RANGE[1]:
            target_bpm = self.VINAHOUSE_BPM_RANGE[1]
        
        # Load audio
        y, sr = librosa.load(input_path, sr=self.sample_rate)
        original_bpm = self._detect_bpm(y)
        
        logger.info(f"Original BPM: {original_bpm}, Target BPM: {target_bpm}")
        
        # Step 1: Time stretch to target BPM
        y = self._time_stretch_to_bpm(y, original_bpm, target_bpm)
        
        # Step 2: Apply Vinahouse EQ
        y = self._apply_vinahouse_eq(y, preset, energy_level)
        
        # Step 3: Apply sidechain compression (ducking)
        y = self._apply_sidechain(y, intensity=sidechain_intensity, bpm=target_bpm)
        
        # Step 4: Add bass enhancement
        y = self._enhance_bass(y, energy_level)
        
        # Step 5: Add brightness/presence
        y = self._add_presence(y, preset)
        
        # Step 6: Apply multiband compression
        y = self._apply_multiband_compression(y, style="vinahouse")
        
        # Step 7: Final limiting for club
        y = self._apply_club_limiting(y)
        
        # Step 8: Normalize to club standard (-8 LUFS)
        y = self._normalize_for_club(y)
        
        # Save
        sf.write(output_path, y, sr)
        
        return {
            "input": input_path,
            "output": output_path,
            "original_bpm": original_bpm,
            "target_bpm": target_bpm,
            "preset": preset,
            "energy_level": energy_level,
            "success": True
        }
    
    def _detect_bpm(self, y: np.ndarray) -> float:
        """Detect BPM"""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sample_rate)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0])
            return float(tempo)
        except:
            return self.DEFAULT_BPM
    
    def _time_stretch_to_bpm(self, y: np.ndarray, original_bpm: float, target_bpm: float) -> np.ndarray:
        """Stretch audio to target BPM"""
        if abs(original_bpm - target_bpm) < 1:
            return y
        
        rate = target_bpm / original_bpm
        
        # Don't stretch too much (max 20% change)
        if rate > 1.2:
            rate = 1.2
        elif rate < 0.8:
            rate = 0.8
        
        logger.info(f"Time stretch rate: {rate}")
        return librosa.effects.time_stretch(y, rate=rate)
    
    def _apply_vinahouse_eq(self, y: np.ndarray, preset: str, energy: float) -> np.ndarray:
        """Apply Vinahouse-style EQ"""
        eq_settings = self.EQ_PRESETS.get(preset, self.EQ_PRESETS["club"])
        
        for band_name, (freq, gain_db) in eq_settings.items():
            # Adjust gain based on energy level
            adjusted_gain = gain_db * energy
            
            if freq < 200:
                # Low shelf for bass frequencies
                y = self._apply_low_shelf(y, freq, adjusted_gain)
            elif freq > 6000:
                # High shelf for treble
                y = self._apply_high_shelf(y, freq, adjusted_gain)
            else:
                # Bell EQ for mids
                y = self._apply_bell_eq(y, freq, adjusted_gain, q=1.0)
        
        return y
    
    def _apply_low_shelf(self, y: np.ndarray, freq: float, gain_db: float) -> np.ndarray:
        """Apply low shelf EQ"""
        if abs(gain_db) < 0.1:
            return y
        
        sos = signal.butter(2, freq / (self.sample_rate / 2), btype='low', output='sos')
        filtered = signal.sosfilt(sos, y)
        
        gain = 10 ** (gain_db / 20)
        return y + (filtered * (gain - 1) * 0.6)
    
    def _apply_high_shelf(self, y: np.ndarray, freq: float, gain_db: float) -> np.ndarray:
        """Apply high shelf EQ"""
        if abs(gain_db) < 0.1:
            return y
        
        sos = signal.butter(2, freq / (self.sample_rate / 2), btype='high', output='sos')
        filtered = signal.sosfilt(sos, y)
        
        gain = 10 ** (gain_db / 20)
        return y + (filtered * (gain - 1) * 0.4)
    
    def _apply_bell_eq(self, y: np.ndarray, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
        """Apply bell EQ"""
        if abs(gain_db) < 0.1:
            return y
        
        w0 = 2 * np.pi * freq / self.sample_rate
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0) / (2 * q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        b = [b0/a0, b1/a0, b2/a0]
        a = [1, a1/a0, a2/a0]
        
        return signal.lfilter(b, a, y)
    
    def _apply_sidechain(self, y: np.ndarray, intensity: float, bpm: float) -> np.ndarray:
        """
        Apply sidechain compression (pumping effect).
        Classic Vinahouse technique - ducks on every beat.
        """
        # Calculate beat interval in samples
        beat_interval = int(60 / bpm * self.sample_rate)
        
        # Create sidechain envelope
        num_beats = len(y) // beat_interval + 1
        envelope = np.ones(len(y))
        
        for i in range(num_beats):
            start = i * beat_interval
            end = min(start + beat_interval, len(y))
            
            # Create ducking envelope for each beat
            t = np.linspace(0, 1, end - start)
            
            # Fast attack, slower release
            duck = 1 - intensity * 0.3 * np.exp(-t * 8)  # Quick duck
            duck = duck * (1 - (1 - duck[-1]) * np.exp(-np.linspace(0, 5, len(t))))  # Release
            
            envelope[start:end] = duck[:end-start]
        
        return y * envelope
    
    def _enhance_bass(self, y: np.ndarray, energy: float) -> np.ndarray:
        """Enhance bass for club sound"""
        # Extract sub bass
        sos = signal.butter(4, 100 / (self.sample_rate / 2), btype='low', output='sos')
        sub_bass = signal.sosfilt(sos, y)
        
        # Add harmonic excitement to bass
        sub_harmonic = np.tanh(sub_bass * 1.5) * 0.3
        
        # Mix back
        enhanced_bass = sub_bass + sub_harmonic
        
        # Apply gain based on energy
        gain = 1 + (energy - 1) * 0.5
        enhanced_bass = enhanced_bass * gain
        
        # High-pass original and add enhanced bass
        sos_hp = signal.butter(4, 60 / (self.sample_rate / 2), btype='high', output='sos')
        y_highpassed = signal.sosfilt(sos_hp, y)
        
        return y_highpassed + enhanced_bass
    
    def _add_presence(self, y: np.ndarray, preset: str) -> np.ndarray:
        """Add presence and brightness"""
        # Presence boost around 3-5kHz
        presence_gain = 2.0 if preset == "club" else 1.5
        y = self._apply_bell_eq(y, 4000, presence_gain, q=0.7)
        
        # Air boost around 10-12kHz
        y = self._apply_bell_eq(y, 11000, 1.5, q=0.5)
        
        return y
    
    def _apply_multiband_compression(self, y: np.ndarray, style: str = "vinahouse") -> np.ndarray:
        """Apply multiband compression for club sound"""
        # Split into 3 bands
        low = self._band_filter(y, 20, 200)
        mid = self._band_filter(y, 200, 2000)
        high = self._band_filter(y, 2000, 20000)
        
        # Compress each band differently
        low = self._compress_band(low, threshold=-10, ratio=3.0)  # Heavy bass compression
        mid = self._compress_band(mid, threshold=-12, ratio=2.5)
        high = self._compress_band(high, threshold=-15, ratio=2.0)  # Lighter high compression
        
        return low + mid + high
    
    def _band_filter(self, y: np.ndarray, low: float, high: float) -> np.ndarray:
        """Bandpass filter"""
        nyquist = self.sample_rate / 2
        low_norm = max(low / nyquist, 0.001)
        high_norm = min(high / nyquist, 0.999)
        
        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        return signal.sosfilt(sos, y)
    
    def _compress_band(self, y: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
        """Compress a frequency band"""
        envelope = np.abs(y)
        
        # Smooth envelope
        window = int(0.01 * self.sample_rate)  # 10ms window
        envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Apply compression
        threshold_linear = 10 ** (threshold / 20)
        gain = np.ones_like(y)
        
        above = envelope_smooth > threshold_linear
        gain[above] = (threshold_linear / envelope_smooth[above]) ** (1 - 1/ratio)
        
        return y * gain
    
    def _apply_club_limiting(self, y: np.ndarray) -> np.ndarray:
        """Apply hard limiting for club play"""
        # Club standard: -0.3dB ceiling
        ceiling = 10 ** (-0.3 / 20)
        
        # Soft clip
        y_soft = np.tanh(y * 1.2) * ceiling
        
        # Hard limit
        y_limited = np.clip(y_soft, -ceiling, ceiling)
        
        return y_limited
    
    def _normalize_for_club(self, y: np.ndarray) -> np.ndarray:
        """Normalize to club standard loudness"""
        # Club tracks are typically -6 to -8 LUFS
        rms = np.sqrt(np.mean(y ** 2))
        
        # Target RMS for club (-8 LUFS approx)
        target_rms = 10 ** (-8 / 20)
        
        gain = target_rms / (rms + 1e-10)
        
        # Limit gain to prevent excessive boost
        if gain > 3:
            gain = 3
        
        y = y * gain
        
        # Final safety clip
        peak = np.max(np.abs(y))
        if peak > 0.95:
            y = y * (0.95 / peak)
        
        return y
    
    def add_vinahouse_kick(self, y: np.ndarray, bpm: int, kick_volume: float = 0.3) -> np.ndarray:
        """
        Add synthesized Vinahouse kick drum.
        4/4 pattern on every beat.
        """
        beat_interval = int(60 / bpm * self.sample_rate)
        num_beats = len(y) // beat_interval
        
        # Generate kick sample
        kick_duration = int(0.2 * self.sample_rate)  # 200ms kick
        t = np.linspace(0, 0.2, kick_duration)
        
        # Pitch sweep from 150Hz to 50Hz
        freq_start = 150
        freq_end = 50
        freq = freq_start * np.exp(-t * 10) + freq_end
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        
        # Kick waveform with envelope
        envelope = np.exp(-t * 15)  # Quick decay
        kick = np.sin(phase) * envelope * kick_volume
        
        # Add to audio
        result = y.copy()
        for i in range(num_beats):
            start = i * beat_interval
            end = min(start + len(kick), len(y))
            kick_len = end - start
            result[start:end] += kick[:kick_len]
        
        return result
    
    def add_vinahouse_clap(self, y: np.ndarray, bpm: int, clap_volume: float = 0.2) -> np.ndarray:
        """
        Add Vinahouse clap on every 2 and 4 beat.
        """
        beat_interval = int(60 / bpm * self.sample_rate)
        num_beats = len(y) // beat_interval
        
        # Generate clap sample (white noise burst)
        clap_duration = int(0.1 * self.sample_rate)
        noise = np.random.randn(clap_duration)
        
        # Envelope
        envelope = np.exp(-np.linspace(0, 5, clap_duration))
        clap = noise * envelope * clap_volume
        
        # Bandpass filter (1-8kHz)
        sos = signal.butter(4, [1000 / (self.sample_rate/2), 8000 / (self.sample_rate/2)], 
                           btype='band', output='sos')
        clap = signal.sosfilt(sos, clap)
        
        # Add on beats 2 and 4
        result = y.copy()
        for i in range(2, num_beats, 4):  # Beat 3 (index 2)
            start = i * beat_interval
            end = min(start + len(clap), len(y))
            clap_len = end - start
            result[start:end] += clap[:clap_len]
        
        for i in range(0, num_beats, 4):  # Beat 1 (index 0, then 4, etc)
            start = (i + 2) * beat_interval
            if start < len(y):
                end = min(start + len(clap), len(y))
                clap_len = end - start
                result[start:end] += clap[:clap_len]
        
        return result


def create_vinahouse_remix(input_path: str, 
                           output_path: str,
                           target_bpm: int = 135,
                           preset: str = "club") -> Dict:
    """
    Quick function to create Vinahouse remix.
    
    Args:
        input_path: Input audio file
        output_path: Output file path
        target_bpm: Target BPM (128-140)
        preset: "club", "festival", "radio"
    """
    engine = VinahouseRemixEngine()
    return engine.create_vinahouse_remix(
        input_path=input_path,
        output_path=output_path,
        target_bpm=target_bpm,
        preset=preset
    )


# Default instance
vinahouse_engine = VinahouseRemixEngine()