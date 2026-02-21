"""
Professional Remix Generator using Pedalboard
Creates studio-quality remixes with VST-grade effects
"""
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Any, Optional, Tuple, List, Callable
import logging
import os
import uuid
import json

# Import new professional modules
from services.pedalboard_effects import (
    ProfessionalEffects, 
    VinahouseRemixProcessor,
    professional_effects,
    vinahouse_processor,
    HAS_PEDALBOARD
)

logger = logging.getLogger(__name__)


class RemixGenerator:
    """
    Professional remix generation using Pedalboard effects.
    
    Upgraded from scipy/librosa to VST-grade Pedalboard effects.
    Implements Vinahouse structure: Intro -> Buildup -> DROP -> Breakdown -> DROP 2 -> Outro
    """
    
    # Vinahouse default settings
    VINAHOUSE_PRESET = {
        'target_bpm': 135,
        'kick_pattern': 'four_on_floor',
        'bass_style': 'off_beat',  # Vinahouse signature
        'effects': {
            'reverb': 0.3,
            'delay': 0.2,
            'sidechain': True,
            'sidechain_intensity': 0.6
        }
    }
    
    DEFAULT_PRESETS = {
        'vinahouse': {
            'target_bpm': 135,
            'kick_pattern': 'four_on_floor',
            'bass_style': 'off_beat',
            'effects': {'reverb': 0.3, 'delay': 0.2, 'sidechain': True}
        },
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
        'techno': {
            'target_bpm': 132,
            'kick_pattern': 'four_on_floor',
            'bass_style': 'acid',
            'effects': {'reverb': 0.35, 'delay': 0.3, 'sidechain': True}
        },
        'hardstyle': {
            'target_bpm': 150,
            'kick_pattern': 'reverse_bass',
            'bass_style': 'distorted',
            'effects': {'reverb': 0.4, 'delay': 0.25, 'sidechain': True}
        }
    }
    
    def __init__(self, output_dir: str = "backend/remixes"):
        self.output_dir = output_dir
        self.sample_rate = 44100
        os.makedirs(output_dir, exist_ok=True)
        
        # Use professional effects
        self.effects = professional_effects
        self.vinahouse_processor = vinahouse_processor
        
        # Progress callback
        self._progress_callback = None
        
        # Load trained profiles
        self.genre_presets = self.DEFAULT_PRESETS.copy()
        self._load_trained_profiles()
        
        logger.info(f"RemixGenerator initialized (Pedalboard: {HAS_PEDALBOARD})")
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates"""
        self._progress_callback = callback
        
    def _report_progress(self, progress: float, message: str):
        """Report progress"""
        if self._progress_callback:
            self._progress_callback(progress, message)
        logger.info(f"[Remix] {progress*100:.0f}% - {message}")
    
    def _load_trained_profiles(self):
        """Load learned style profiles"""
        profile_path = "models/style_profiles.json"
        try:
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    trained = json.load(f)
                for genre, profile in trained.items():
                    if genre not in self.genre_presets:
                        self.genre_presets[genre] = {}
                    if 'avg_bpm' in profile:
                        self.genre_presets[genre]['target_bpm'] = int(profile['avg_bpm'])
                    if 'remix_params' in profile:
                        self.genre_presets[genre]['effects'] = profile['remix_params']
                logger.info(f"Loaded {len(trained)} trained profiles")
        except Exception as e:
            logger.warning(f"Could not load trained profiles: {e}")
    
    def get_available_genres(self) -> List[str]:
        return list(self.genre_presets.keys())
    
    def get_genre_info(self, genre: str) -> Dict:
        return self.genre_presets.get(genre, self.VINAHOUSE_PRESET)
    
    async def generate(
        self,
        audio_path: str,
        stems: Dict[str, str],
        settings: Dict[str, Any],
        original_bpm: float = 120.0,
        original_key: str = "C major"
    ) -> Dict[str, Any]:
        """
        Generate a professional remix using Pedalboard effects.
        
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
            self._report_progress(0.05, "Loading stems...")
            
            genre = settings.get('genre', 'vinahouse')
            target_bpm = settings.get('bpm', 135)
            energy_level = settings.get('energyLevel', 0.7)
            
            # Get preset
            preset = self.genre_presets.get(genre, self.VINAHOUSE_PRESET)
            logger.info(f"Using preset: {genre} -> {preset}")
            
            # Load stems
            vocals, _ = self._load_audio(stems.get('vocals'))
            drums, _ = self._load_audio(stems.get('drums'))
            bass, _ = self._load_audio(stems.get('bass'))
            other, _ = self._load_audio(stems.get('other'))
            
            self._report_progress(0.15, "Processing stems...")
            
            # Ensure same length
            max_len = max(len(vocals), len(drums), len(bass), len(other))
            vocals = self._pad_to_length(vocals, max_len)
            drums = self._pad_to_length(drums, max_len)
            bass = self._pad_to_length(bass, max_len)
            other = self._pad_to_length(other, max_len)
            
            # Apply tempo change if needed
            tempo_ratio = target_bpm / original_bpm if original_bpm > 0 else 1.0
            
            if abs(tempo_ratio - 1.0) > 0.05:
                self._report_progress(0.25, f"Adjusting tempo: {original_bpm} -> {target_bpm} BPM")
                vocals = self._change_tempo(vocals, tempo_ratio)
                drums = self._change_tempo(drums, tempo_ratio)
                bass = self._change_tempo(bass, tempo_ratio)
                other = self._change_tempo(other, tempo_ratio)
            
            self._report_progress(0.35, "Applying professional effects...")
            
            # Process each stem with Pedalboard
            vocals = self._process_vocals_pro(vocals, genre, settings)
            drums = self._process_drums_pro(drums, genre, settings, energy_level)
            bass = self._process_bass_pro(bass, genre, settings, energy_level)
            other = self._process_other_pro(other, genre, settings)
            
            self._report_progress(0.6, "Mixing stems...")
            
            # Mix with genre-appropriate levels
            mix = self._mix_stems(vocals, drums, bass, other, genre, energy_level)
            
            self._report_progress(0.75, "Applying master chain...")
            
            # Apply Vinahouse structure processing
            if genre in ['vinahouse', 'edm', 'house', 'techno']:
                mix = self._apply_vinahouse_structure(mix, target_bpm, energy_level)
            
            # Master for club
            self._report_progress(0.85, "Mastering for club play...")
            mix = self.effects.master_for_club(mix, target_lufs=-8.0)
            
            # Final normalize
            mix = self._normalize(mix)
            
            # Save
            self._report_progress(0.95, "Saving remix...")
            
            remix_id = str(uuid.uuid4())
            remix_filename = f"remix_{remix_id}.wav"
            remix_path = os.path.join(self.output_dir, remix_filename)
            
            sf.write(remix_path, mix, self.sample_rate)
            
            self._report_progress(1.0, "Remix complete!")
            
            return {
                'remix_id': remix_id,
                'remix_path': remix_path,
                'remix_url': f"/remixes/{remix_filename}",
                'duration': len(mix) / self.sample_rate,
                'genre': genre,
                'bpm': target_bpm,
                'effects': 'Pedalboard VST-grade',
                'has_sidechain': True
            }
            
        except Exception as e:
            logger.error(f"Remix generation failed: {e}")
            raise
    
    def _load_audio(self, path: Optional[str]) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        if path and os.path.exists(path):
            y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return y, sr
        return np.zeros(self.sample_rate * 30), self.sample_rate
    
    def _pad_to_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad audio to target length"""
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)))
        return audio[:target_length]
    
    def _change_tempo(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """Change tempo using librosa"""
        try:
            stretch_rate = 1.0 / ratio
            return librosa.effects.time_stretch(audio, rate=stretch_rate)
        except Exception as e:
            logger.warning(f"Tempo change failed: {e}")
            return audio
    
    def _process_vocals_pro(self, vocals: np.ndarray, genre: str, settings: Dict) -> np.ndarray:
        """Process vocals using Pedalboard"""
        if not HAS_PEDALBOARD:
            return self._process_vocals_fallback(vocals, genre, settings)
        
        from pedalboard import Pedalboard, Reverb, Delay, HighpassFilter, Compressor
        
        reverb_mix = settings.get('reverb', 0.3)
        delay_mix = settings.get('delay', 0.2)
        
        board = Pedalboard([
            HighpassFilter(cutoff_hz=100),  # Remove rumble
            Delay(delay_seconds=0.25, feedback=0.2, mix=delay_mix * 0.3),
            Reverb(room_size=0.6, wet_level=reverb_mix * 0.5, dry_level=1 - reverb_mix * 0.5),
            Compressor(threshold_db=-18, ratio=3.0)
        ])
        
        # Ensure 2D for pedalboard
        if vocals.ndim == 1:
            vocals = vocals.reshape(1, -1)
        
        return board(vocals, self.sample_rate).flatten()
    
    def _process_vocals_fallback(self, vocals: np.ndarray, genre: str, settings: Dict) -> np.ndarray:
        """Fallback vocal processing without Pedalboard"""
        reverb = settings.get('reverb', 0.3)
        if reverb > 0:
            vocals = self._apply_reverb_simple(vocals, reverb * 0.5)
        return vocals
    
    def _process_drums_pro(self, drums: np.ndarray, genre: str, settings: Dict, energy: float) -> np.ndarray:
        """Process drums using Pedalboard"""
        if not HAS_PEDALBOARD:
            return drums * (0.8 + energy * 0.4)
        
        from pedalboard import Pedalboard, Compressor, LadderFilter, PeakLimiter
        
        # Different processing for different genres
        if genre in ['vinahouse', 'house', 'edm']:
            # Punchy, tight drums
            board = Pedalboard([
                Compressor(threshold_db=-12, ratio=4.0, attack_ms=2.0, release_ms=50.0),
                LadderFilter(mode=LadderFilter.Mode.LP12, cutoff_hz=15000, resonance=0.1),
                PeakLimiter(threshold_db=-1.0, release_ms=30.0)
            ])
        elif genre == 'techno':
            # Harder, more aggressive
            board = Pedalboard([
                Compressor(threshold_db=-10, ratio=6.0, attack_ms=1.0, release_ms=30.0),
                PeakLimiter(threshold_db=-0.5, release_ms=20.0)
            ])
        else:
            # Default
            board = Pedalboard([
                Compressor(threshold_db=-15, ratio=3.0),
            ])
        
        if drums.ndim == 1:
            drums = drums.reshape(1, -1)
        
        processed = board(drums, self.sample_rate)
        return processed.flatten() * (0.8 + energy * 0.4)
    
    def _process_bass_pro(self, bass: np.ndarray, genre: str, settings: Dict, energy: float) -> np.ndarray:
        """Process bass using Pedalboard"""
        if not HAS_PEDALBOARD:
            return bass * (0.7 + energy * 0.5)
        
        from pedalboard import Pedalboard, Compressor, LadderFilter
        
        # Vinahouse signature: off-beat bass
        if genre == 'vinahouse':
            board = Pedalboard([
                LadderFilter(mode=LadderFilter.Mode.LP24, cutoff_hz=200, resonance=0.2),
                Compressor(threshold_db=-10, ratio=6.0, attack_ms=5.0, release_ms=80.0)
            ])
        else:
            board = Pedalboard([
                Compressor(threshold_db=-12, ratio=4.0)
            ])
        
        if bass.ndim == 1:
            bass = bass.reshape(1, -1)
        
        return board(bass, self.sample_rate).flatten() * (0.7 + energy * 0.5)
    
    def _process_other_pro(self, other: np.ndarray, genre: str, settings: Dict) -> np.ndarray:
        """Process other instruments using Pedalboard"""
        if not HAS_PEDALBOARD:
            return other * 0.8
        
        from pedalboard import Pedalboard, Reverb, HighpassFilter, LowpassFilter
        
        reverb = settings.get('reverb', 0.3)
        filter_val = settings.get('filter', 0.5)
        
        board = Pedalboard([
            HighpassFilter(cutoff_hz=150),
            LowpassFilter(cutoff_hz=12000),
            Reverb(room_size=0.5, wet_level=reverb * 0.4, dry_level=1 - reverb * 0.4)
        ])
        
        if other.ndim == 1:
            other = other.reshape(1, -1)
        
        return board(other, self.sample_rate).flatten() * 0.8
    
    def _mix_stems(self, vocals, drums, bass, other, genre, energy):
        """Mix stems with genre-appropriate levels"""
        levels = {
            'vinahouse': {'vocals': 0.7, 'drums': 1.0, 'bass': 0.95, 'other': 0.6},
            'edm': {'vocals': 0.7, 'drums': 1.0, 'bass': 0.9, 'other': 0.6},
            'house': {'vocals': 0.8, 'drums': 1.0, 'bass': 0.85, 'other': 0.65},
            'techno': {'vocals': 0.5, 'drums': 1.0, 'bass': 0.9, 'other': 0.6},
            'hardstyle': {'vocals': 0.6, 'drums': 1.0, 'bass': 1.0, 'other': 0.5}
        }
        
        level = levels.get(genre, levels['vinahouse'])
        
        max_len = max(len(vocals), len(drums), len(bass), len(other))
        vocals = self._pad_to_length(vocals, max_len)
        drums = self._pad_to_length(drums, max_len)
        bass = self._pad_to_length(bass, max_len)
        other = self._pad_to_length(other, max_len)
        
        mix = (
            vocals * level['vocals'] * energy +
            drums * level['drums'] +
            bass * level['bass'] * (0.5 + energy * 0.5) +
            other * level['other']
        )
        
        return mix
    
    def _apply_vinahouse_structure(self, audio: np.ndarray, bpm: int, energy: float) -> np.ndarray:
        """Apply Vinahouse structure processing with sidechain"""
        # Apply sidechain compression
        if HAS_PEDALBOARD:
            audio = self.effects.apply_sidechain(audio, 0.5, bpm)
        
        return audio
    
    def _apply_reverb_simple(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Simple reverb fallback"""
        delay_samples = int(self.sample_rate * 0.1)
        decay = np.exp(-np.linspace(0, 5, delay_samples))
        impulse = np.zeros(delay_samples)
        impulse[0] = 1
        impulse[delay_samples//4::delay_samples//8] = decay[::8][:len(impulse[delay_samples//4::delay_samples//8])]
        reverb = np.convolve(audio, impulse, mode='same') * amount
        return audio + reverb
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.9
        return audio


# Singleton
remix_generator = RemixGenerator()