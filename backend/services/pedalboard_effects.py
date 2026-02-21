"""
Professional Audio Effects using Spotify's Pedalboard
VST-grade effects for studio quality remixes
"""
import numpy as np
import soundfile as sf
from typing import Dict, Any, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Try import pedalboard
try:
    from pedalboard import (
        Pedalboard,
        Reverb,
        Delay,
        Compressor,
        Limiter,
        HighpassFilter,
        LowpassFilter,
        Chorus,
        Distortion,
        Gain,
        NoiseGate,
        PeakLimiter,
        LadderFilter,
        Phaser,
        ClippingLimiter,
    )
    from pedalboard.io import AudioFile, ReadableAudioFile
    HAS_PEDALBOARD = True
    logger.info("Pedalboard loaded successfully - Studio quality effects enabled!")
except ImportError:
    HAS_PEDALBOARD = False
    logger.warning("Pedalboard not installed - falling back to scipy effects")


class ProfessionalEffects:
    """
    Professional VST-grade audio effects using Pedalboard.
    
    Effects available:
    - Reverb: Studio reverb with room size, damping, wet/dry
    - Delay: Tempo-synced delay with feedback
    - Compressor: Professional dynamics control
    - Sidechain: Ducking effect for EDM/Vinahouse
    - Limiter: Brick wall limiting
    - Filters: High/Low pass with resonance
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.has_pedalboard = HAS_PEDALBOARD
        
    def create_vinahouse_board(self, 
                                reverb_mix: float = 0.3,
                                delay_mix: float = 0.2,
                                sidechain_intensity: float = 0.6,
                                target_bpm: int = 135) -> 'Pedalboard':
        """
        Create a Pedalboard chain optimized for Vinahouse.
        
        Args:
            reverb_mix: Reverb wet/dry (0-1)
            delay_mix: Delay wet/dry (0-1)
            sidechain_intensity: Sidechain ducking (0-1)
            target_bpm: BPM for tempo-synced effects
            
        Returns:
            Pedalboard instance with Vinahouse effects chain
        """
        if not self.has_pedalboard:
            return None
            
        # Calculate delay time in seconds based on BPM
        # 1/4 note delay
        quarter_note = 60 / target_bpm
        delay_time = quarter_note  # Quarter note
        
        board = Pedalboard([
            # Subtle chorus for width
            Chorus(
                rate_hz=1.5,
                depth=0.25,
                centre_delay_ms=7.0,
                feedback=0.0,
                mix=0.3
            ),
            
            # Tempo-synced delay
            Delay(
                delay_seconds=delay_time,
                feedback=0.3,
                mix=delay_mix
            ),
            
            # Hall reverb for space
            Reverb(
                room_size=0.7,
                damping=0.5,
                wet_level=reverb_mix,
                dry_level=1.0 - reverb_mix,
                width=0.8,
                freeze_mode=0.0
            ),
            
            # Compressor for punch
            Compressor(
                threshold_db=-18,
                ratio=4.0,
                attack_ms=5.0,
                release_ms=50.0
            ),
            
            # Final limiter
            Limiter(
                threshold_db=-1.0,
                release_ms=50.0
            )
        ])
        
        return board
    
    def create_drop_chain(self, intensity: float = 1.0) -> 'Pedalboard':
        """
        Create effects chain for DROP section - maximum impact!
        
        Features:
        - Heavy sidechain simulation via compressor
        - Bass enhancement
        - Aggressive limiting
        """
        if not self.has_pedalboard:
            return None
            
        board = Pedalboard([
            # Bass boost via gain on filtered signal
            LadderFilter(
                mode=LadderFilter.Mode.LP12,
                cutoff_hz=200,
                resonance=0.3,
                drive=1.5  # Add warmth
            ),
            
            # Heavy compressor for pumping sidechain effect
            Compressor(
                threshold_db=-12,
                ratio=8.0,  # Heavy ratio for pumping
                attack_ms=2.0,  # Fast attack for punch
                release_ms=80.0  # Medium release for groove
            ),
            
            # Peak limiter to prevent clipping
            PeakLimiter(
                threshold_db=-0.5,
                release_ms=30.0
            )
        ])
        
        return board
    
    def create_buildup_chain(self, target_bpm: int = 135) -> 'Pedalboard':
        """
        Create effects chain for BUILDUP section.
        
        Features:
        - Filter sweep (simulated)
        - Rising tension
        """
        if not self.has_pedalboard:
            return None
            
        board = Pedalboard([
            # High-pass for tension
            HighpassFilter(cutoff_hz=100),
            
            # Phaser for movement
            Phaser(
                rate_hz=0.5,
                depth=0.8,
                centre_frequency_hz=1000,
                feedback=0.5,
                mix=0.5
            ),
            
            # Light reverb
            Reverb(
                room_size=0.5,
                damping=0.3,
                wet_level=0.2,
                dry_level=0.8,
                width=0.7
            )
        ])
        
        return board
    
    def apply_effects(self, 
                      audio: np.ndarray, 
                      board: 'Pedalboard') -> np.ndarray:
        """
        Apply Pedalboard effects to audio.
        
        Args:
            audio: Audio array (samples,) or (channels, samples)
            board: Pedalboard instance
            
        Returns:
            Processed audio
        """
        if not self.has_pedalboard or board is None:
            return audio
            
        # Ensure 2D for pedalboard (channels, samples)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
            
        # Apply effects
        processed = board(audio, self.sample_rate)
        
        return processed
    
    def apply_sidechain(self,
                        audio: np.ndarray,
                        intensity: float = 0.6,
                        bpm: int = 135) -> np.ndarray:
        """
        Apply sidechain compression effect (pumping).
        
        This simulates the ducking effect when kick hits.
        
        Args:
            audio: Input audio
            intensity: Ducking intensity (0-1)
            bpm: Track BPM for timing
            
        Returns:
            Audio with sidechain effect
        """
        # Calculate beat duration in samples
        beat_samples = int(60 / bpm * self.sample_rate)
        
        # Create ducking envelope
        num_beats = len(audio) // beat_samples + 1
        envelope = np.ones(len(audio))
        
        for i in range(num_beats):
            start = i * beat_samples
            end = min(start + beat_samples, len(audio))
            
            # Create duck envelope for this beat
            t = np.arange(end - start) / self.sample_rate
            # Quick duck, slow recovery
            duck = 1 - intensity * 0.5 * np.exp(-t * 15)
            
            envelope[start:end] = duck[:end-start]
        
        return audio * envelope
    
    def master_for_club(self, 
                        audio: np.ndarray,
                        target_lufs: float = -8.0) -> np.ndarray:
        """
        Master audio for club play.
        
        Args:
            audio: Input audio
            target_lufs: Target loudness (default -8 LUFS for club)
            
        Returns:
            Mastered audio
        """
        if not self.has_pedalboard:
            return self._fallback_master(audio)
            
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
            
        # Create mastering chain
        master_board = Pedalboard([
            # Multiband-style compression via multiple compressors
            Compressor(
                threshold_db=-15,
                ratio=3.0,
                attack_ms=10.0,
                release_ms=100.0
            ),
            
            # Final limiter
            PeakLimiter(
                threshold_db=-0.3,
                release_ms=50.0
            )
        ])
        
        # Apply
        mastered = master_board(audio, self.sample_rate)
        
        # Normalize to target loudness (simplified)
        rms = np.sqrt(np.mean(mastered**2))
        target_rms = 10 ** (target_lufs / 20)
        gain = min(target_rms / (rms + 1e-10), 4.0)
        mastered = mastered * gain
        
        # Final safety clip
        mastered = np.clip(mastered, -0.95, 0.95)
        
        return mastered
    
    def _fallback_master(self, audio: np.ndarray) -> np.ndarray:
        """Fallback mastering without pedalboard"""
        # Simple soft clip
        mastered = np.tanh(audio * 1.5) * 0.8
        
        # Normalize
        peak = np.max(np.abs(mastered))
        if peak > 0.9:
            mastered = mastered * (0.9 / peak)
            
        return mastered


class VinahouseRemixProcessor:
    """
    Complete Vinahouse remix processor using Pedalboard.
    
    Implements the full Vinahouse structure:
    - Intro (16-32 bars)
    - Buildup (8 bars with snare roll)
    - DROP 1 (16-32 bars with sidechain)
    - Verse/Chorus
    - Breakdown
    - Buildup 2
    - DROP 2 (with octave shift)
    - Outro
    """
    
    # Vinahouse Structure Constants (in beats)
    INTRO_BEATS = 32       # 8 bars
    BUILDUP_BEATS = 16     # 4 bars  
    DROP_BEATS = 64        # 16 bars
    VERSE_BEATS = 32       # 8 bars
    BREAKDOWN_BEATS = 32   # 8 bars
    OUTRO_BEATS = 32       # 8 bars
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.effects = ProfessionalEffects(sample_rate)
        
    def create_vinahouse_structure(self, 
                                   total_beats: int,
                                   energy_level: float = 1.0) -> list:
        """
        Create Vinahouse song structure.
        
        Returns list of sections with beat positions and energy.
        """
        sections = []
        current_beat = 0
        
        # INTRO
        sections.append({
            "type": "intro",
            "start_beat": current_beat,
            "duration_beats": self.INTRO_BEATS,
            "energy": 0.3 * energy_level
        })
        current_beat += self.INTRO_BEATS
        
        # BUILDUP 1
        sections.append({
            "type": "buildup",
            "start_beat": current_beat,
            "duration_beats": self.BUILDUP_BEATS,
            "energy": 0.7 * energy_level
        })
        current_beat += self.BUILDUP_BEATS
        
        # DROP 1
        sections.append({
            "type": "drop",
            "start_beat": current_beat,
            "duration_beats": self.DROP_BEATS,
            "energy": 1.0 * energy_level
        })
        current_beat += self.DROP_BEATS
        
        # VERSE
        sections.append({
            "type": "verse",
            "start_beat": current_beat,
            "duration_beats": self.VERSE_BEATS,
            "energy": 0.5 * energy_level
        })
        current_beat += self.VERSE_BEATS
        
        # BREAKDOWN
        sections.append({
            "type": "breakdown",
            "start_beat": current_beat,
            "duration_beats": self.BREAKDOWN_BEATS,
            "energy": 0.2 * energy_level
        })
        current_beat += self.BREAKDOWN_BEATS
        
        # BUILDUP 2
        sections.append({
            "type": "buildup",
            "start_beat": current_beat,
            "duration_beats": self.BUILDUP_BEATS,
            "energy": 0.8 * energy_level
        })
        current_beat += self.BUILDUP_BEATS
        
        # DROP 2 (Maximum energy!)
        sections.append({
            "type": "drop",
            "start_beat": current_beat,
            "duration_beats": self.DROP_BEATS,
            "energy": 1.0 * energy_level
        })
        current_beat += self.DROP_BEATS
        
        # OUTRO
        sections.append({
            "type": "outro",
            "start_beat": current_beat,
            "duration_beats": self.OUTRO_BEATS,
            "energy": 0.3 * energy_level
        })
        
        return sections
    
    def process_section(self,
                        audio: np.ndarray,
                        section_type: str,
                        energy: float,
                        bpm: int = 135) -> np.ndarray:
        """
        Process audio for a specific section type.
        
        Args:
            audio: Section audio
            section_type: 'intro', 'buildup', 'drop', 'verse', 'breakdown', 'outro'
            energy: Energy level (0-1)
            bpm: Track BPM
            
        Returns:
            Processed audio
        """
        if section_type == "intro":
            # Intro: Filtered, atmospheric
            board = Pedalboard([
                HighpassFilter(cutoff_hz=200),
                Reverb(room_size=0.8, wet_level=0.4, dry_level=0.6)
            ])
            processed = self.effects.apply_effects(audio, board)
            return processed * energy
            
        elif section_type == "buildup":
            # Buildup: Rising, phasing, filter opening
            board = self.effects.create_buildup_chain(bpm)
            processed = self.effects.apply_effects(audio, board)
            # Add rising volume
            rise = np.linspace(0.5, 1.0, len(processed))
            if processed.ndim == 2:
                rise = rise.reshape(1, -1)
            return processed * rise * energy
            
        elif section_type == "drop":
            # Drop: Full energy, sidechain pumping!
            board = self.effects.create_drop_chain()
            processed = self.effects.apply_effects(audio, board)
            processed = self.effects.apply_sidechain(processed, 0.6, bpm)
            return processed * energy
            
        elif section_type == "breakdown":
            # Breakdown: Sparse, lots of reverb
            board = Pedalboard([
                HighpassFilter(cutoff_hz=100),
                Reverb(room_size=0.9, wet_level=0.6, dry_level=0.4, damping=0.7)
            ])
            processed = self.effects.apply_effects(audio, board)
            return processed * energy
            
        elif section_type == "outro":
            # Outro: Fading, filter closing
            board = Pedalboard([
                LowpassFilter(cutoff_hz=2000),
                Reverb(room_size=0.6, wet_level=0.3)
            ])
            processed = self.effects.apply_effects(audio, board)
            # Fade out
            fade = np.linspace(1.0, 0.0, len(processed))
            if processed.ndim == 2:
                fade = fade.reshape(1, -1)
            return processed * fade * energy
            
        else:
            # Default: just apply energy
            return audio * energy


# Singleton instances
professional_effects = ProfessionalEffects()
vinahouse_processor = VinahouseRemixProcessor()