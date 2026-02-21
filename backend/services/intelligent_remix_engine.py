"""
Intelligent Remix Engine - Auto Remix Based on Source Track Analysis
Analyzes source track → Learns from music library → Creates structured remix
"""
import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Dict, Optional, List, Tuple
import logging
import json
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class RemixSection(Enum):
    """Remix structure sections"""
    INTRO = "intro"
    BUILDUP = "buildup"
    DROP = "drop"
    VERSE = "verse"
    CHORUS = "chorus"
    BREAKDOWN = "breakdown"
    BRIDGE = "bridge"
    OUTRO = "outro"


@dataclass
class TrackAnalysis:
    """Analysis result of a track"""
    bpm: float
    key: str
    energy: float  # 0-1
    danceability: float  # 0-1
    genre: str
    sections: List[Dict]  # List of sections with start, end, type
    stem_analysis: Dict  # Analysis of each stem
    spectral_features: Dict
    duration: float
    drop_ready: bool  # Has energy for drop section


@dataclass
class RemixStructure:
    """Structure of a remix"""
    sections: List[Dict]  # [{type, start_beat, end_beat, duration_beats, energy}]
    total_beats: int
    bpm: float
    
    @classmethod
    def create_vinahouse_structure(cls, original_duration_beats: int, energy_level: float = 1.0) -> 'RemixStructure':
        """Create standard Vinahouse remix structure"""
        # Standard Vinahouse structure (32-beat phrases)
        # 8 bars = 32 beats per section typically
        
        sections = []
        current_beat = 0
        
        # INTRO (16-32 beats)
        intro_beats = 32
        sections.append({
            "type": RemixSection.INTRO.value,
            "start_beat": current_beat,
            "end_beat": current_beat + intro_beats,
            "duration_beats": intro_beats,
            "energy": 0.3 * energy_level
        })
        current_beat += intro_beats
        
        # BUILDUP 1 (16-32 beats)
        buildup1_beats = 16
        sections.append({
            "type": RemixSection.BUILDUP.value,
            "start_beat": current_beat,
            "end_beat": current_beat + buildup1_beats,
            "duration_beats": buildup1_beats,
            "energy": 0.6 * energy_level
        })
        current_beat += buildup1_beats
        
        # DROP 1 (32-64 beats) - Main energy
        drop1_beats = 64
        sections.append({
            "type": RemixSection.DROP.value,
            "start_beat": current_beat,
            "end_beat": current_beat + drop1_beats,
            "duration_beats": drop1_beats,
            "energy": 1.0 * energy_level
        })
        current_beat += drop1_beats
        
        # VERSE (from original, 32 beats)
        verse_beats = 32
        sections.append({
            "type": RemixSection.VERSE.value,
            "start_beat": current_beat,
            "end_beat": current_beat + verse_beats,
            "duration_beats": verse_beats,
            "energy": 0.5 * energy_level
        })
        current_beat += verse_beats
        
        # CHORUS (from original, 32 beats)
        chorus_beats = 32
        sections.append({
            "type": RemixSection.CORUS.value,
            "start_beat": current_beat,
            "end_beat": current_beat + chorus_beats,
            "duration_beats": chorus_beats,
            "energy": 0.7 * energy_level
        })
        current_beat += chorus_beats
        
        # BREAKDOWN (16-32 beats)
        breakdown_beats = 32
        sections.append({
            "type": RemixSection.BREAKDOWN.value,
            "start_beat": current_beat,
            "end_beat": current_beat + breakdown_beats,
            "duration_beats": breakdown_beats,
            "energy": 0.2 * energy_level
        })
        current_beat += breakdown_beats
        
        # BUILDUP 2 (16 beats)
        buildup2_beats = 16
        sections.append({
            "type": RemixSection.BUILDUP.value,
            "start_beat": current_beat,
            "end_beat": current_beat + buildup2_beats,
            "duration_beats": buildup2_beats,
            "energy": 0.7 * energy_level
        })
        current_beat += buildup2_beats
        
        # DROP 2 (64 beats) - Maximum energy
        drop2_beats = 64
        sections.append({
            "type": RemixSection.DROP.value,
            "start_beat": current_beat,
            "end_beat": current_beat + drop2_beats,
            "duration_beats": drop2_beats,
            "energy": 1.0 * energy_level
        })
        current_beat += drop2_beats
        
        # OUTRO (32 beats)
        outro_beats = 32
        sections.append({
            "type": RemixSection.OUTRO.value,
            "start_beat": current_beat,
            "end_beat": current_beat + outro_beats,
            "duration_beats": outro_beats,
            "energy": 0.3 * energy_level
        })
        current_beat += outro_beats
        
        return cls(
            sections=sections,
            total_beats=current_beat,
            bpm=135  # Default Vinahouse BPM
        )


@dataclass
class LearnedStyle:
    """Style learned from music library"""
    avg_bpm: float
    common_keys: List[str]
    eq_curve: Dict  # Frequency -> gain
    compression_ratio: float
    reverb_amount: float
    bass_enhancement: float
    stereo_width: float
    drop_characteristics: Dict
    buildup_style: Dict


class IntelligentRemixEngine:
    """
    Intelligent remix engine that:
    1. Analyzes source track
    2. Learns style from music library
    3. Creates structured remix based on both
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.learned_style: Optional[LearnedStyle] = None
        self.track_analysis: Optional[TrackAnalysis] = None
        self.remix_structure: Optional[RemixStructure] = None
        
    def analyze_source_track(self, audio_path: str) -> TrackAnalysis:
        """
        Deep analysis of source track.
        
        Analyzes:
        - BPM and key
        - Energy and danceability
        - Section detection (verse, chorus, etc.)
        - Spectral features
        - Drop potential
        """
        logger.info(f"Analyzing source track: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr
        
        # BPM detection
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        bpm = float(tempo)
        
        # Key detection
        key = self._detect_key(y)
        
        # Energy analysis
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms))
        energy_normalized = min(energy * 10, 1.0)  # Normalize to 0-1
        
        # Danceability (based on rhythm strength)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        danceability = float(np.std(onset_env) / (np.mean(onset_env) + 1e-10))
        danceability = min(danceability / 5, 1.0)
        
        # Section detection
        sections = self._detect_sections(y, sr, bpm)
        
        # Spectral features
        spectral_features = self._analyze_spectral(y, sr)
        
        # Genre classification (simplified)
        genre = self._classify_genre(y, sr, bpm, spectral_features)
        
        # Drop ready check
        drop_ready = energy_normalized > 0.3 and danceability > 0.4
        
        self.track_analysis = TrackAnalysis(
            bpm=bpm,
            key=key,
            energy=energy_normalized,
            danceability=danceability,
            genre=genre,
            sections=sections,
            stem_analysis={},  # Will be filled if stems available
            spectral_features=spectral_features,
            duration=duration,
            drop_ready=drop_ready
        )
        
        logger.info(f"Analysis complete: BPM={bpm:.1f}, Key={key}, Energy={energy_normalized:.2f}")
        
        return self.track_analysis
    
    def learn_from_library(self, library_path: str, max_tracks: int = 100) -> LearnedStyle:
        """
        Learn production style from music library.
        
        Analyzes:
        - Average BPM preference
        - Common keys
        - EQ characteristics
        - Compression style
        - Bass characteristics
        - Drop/buildup patterns
        """
        logger.info(f"Learning style from library: {library_path}")
        
        # Check for cached style
        cache_path = os.path.join(library_path, ".learned_style.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                style_data = json.load(f)
            self.learned_style = LearnedStyle(**style_data)
            logger.info("Loaded cached style profile")
            return self.learned_style
        
        # Analyze tracks
        audio_files = self._find_audio_files(library_path, max_tracks)
        
        if not audio_files:
            # Use default Vinahouse style
            logger.warning("No audio files found, using default Vinahouse style")
            self.learned_style = self._get_default_vinahouse_style()
            return self.learned_style
        
        bpms = []
        keys = []
        eq_curves = []
        
        for i, audio_path in enumerate(audio_files):
            try:
                logger.info(f"Analyzing {i+1}/{len(audio_files)}: {audio_path}")
                y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30)  # First 30s
                
                # BPM
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                if isinstance(tempo, np.ndarray):
                    tempo = float(tempo[0])
                bpms.append(float(tempo))
                
                # Key
                keys.append(self._detect_key(y))
                
                # EQ curve
                eq_curves.append(self._analyze_eq_curve(y, sr))
                
            except Exception as e:
                logger.warning(f"Failed to analyze {audio_path}: {e}")
        
        # Aggregate learned style
        self.learned_style = LearnedStyle(
            avg_bpm=float(np.mean(bpms)) if bpms else 135,
            common_keys=list(set(keys))[:5] if keys else ["C major"],
            eq_curve=self._aggregate_eq_curves(eq_curves),
            compression_ratio=3.0,  # Typical for Vinahouse
            reverb_amount=0.3,
            bass_enhancement=4.0,  # dB boost for club
            stereo_width=1.2,
            drop_characteristics={
                "energy_spike": 0.8,
                "bass_emphasis": 6.0,
                "kick_pattern": "4-on-floor"
            },
            buildup_style={
                "snare_roll": True,
                "rising_pitch": True,
                "filter_sweep": True
            }
        )
        
        # Cache learned style
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(asdict(self.learned_style), f, indent=2)
        
        logger.info(f"Learned style: avg BPM={self.learned_style.avg_bpm:.1f}")
        
        return self.learned_style
    
    def create_intelligent_remix(self,
                                  input_path: str,
                                  output_path: str,
                                  style: str = "vinahouse",
                                  target_bpm: Optional[int] = None,
                                  energy_level: float = 1.0) -> Dict:
        """
        Create intelligent remix based on source analysis and learned style.
        
        Process:
        1. Analyze source track
        2. Apply learned style from library
        3. Create remix structure
        4. Process each section appropriately
        5. Apply section-specific processing
        6. Master for target platform
        """
        logger.info(f"Creating intelligent remix: {input_path}")
        
        # Step 1: Analyze source
        if not self.track_analysis or self.track_analysis != input_path:
            self.analyze_source_track(input_path)
        
        # Step 2: Determine target BPM
        if target_bpm is None:
            if self.learned_style:
                target_bpm = int(self.learned_style.avg_bpm)
            else:
                target_bpm = 135  # Default Vinahouse
        
        # Ensure BPM is in Vinahouse range
        target_bpm = max(128, min(140, target_bpm))
        
        # Step 3: Create remix structure
        self.remix_structure = RemixStructure.create_vinahouse_structure(
            original_duration_beats=int(self.track_analysis.duration * target_bpm / 60),
            energy_level=energy_level
        )
        self.remix_structure.bpm = target_bpm
        
        # Step 4: Load and process source
        y, sr = librosa.load(input_path, sr=self.sample_rate)
        
        # Step 5: Time stretch to target BPM
        original_bpm = self.track_analysis.bpm
        if abs(original_bpm - target_bpm) > 1:
            rate = target_bpm / original_bpm
            rate = max(0.8, min(1.2, rate))  # Limit stretch
            y = librosa.effects.time_stretch(y, rate=rate)
        
        # Step 6: Create remix sections
        remixed = self._create_remix_sections(y, self.remix_structure, target_bpm)
        
        # Step 7: Apply learned style
        if self.learned_style:
            remixed = self._apply_learned_style(remixed, self.learned_style)
        else:
            remixed = self._apply_default_vinahouse_processing(remixed)
        
        # Step 8: Master for club
        remixed = self._master_for_club(remixed)
        
        # Save
        sf.write(output_path, remixed, sr)
        
        return {
            "input": input_path,
            "output": output_path,
            "original_bpm": original_bpm,
            "target_bpm": target_bpm,
            "structure": [s["type"] for s in self.remix_structure.sections],
            "analysis": asdict(self.track_analysis),
            "success": True
        }
    
    def _detect_key(self, y: np.ndarray) -> str:
        """Detect musical key"""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            major_idx = np.argmax(chroma_mean)
            
            # Check minor (relative minor is 9 semitones below)
            minor_idx = (major_idx - 3) % 12
            minor_strength = chroma_mean[minor_idx]
            
            if minor_strength > chroma_mean[major_idx] * 0.8:
                return f"{key_names[minor_idx]} minor"
            return f"{key_names[major_idx]} major"
        except:
            return "C major"
    
    def _detect_sections(self, y: np.ndarray, sr: int, bpm: float) -> List[Dict]:
        """Detect song sections (intro, verse, chorus, etc.)"""
        # Use onset strength and spectral contrast for section detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Normalize
        onset_env = onset_env / (np.max(onset_env) + 1e-10)
        
        # Find section boundaries using novelty
        novelty = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
        
        # Create sections
        sections = []
        beat_duration = 60 / bpm
        
        # Group into 8-bar sections (32 beats)
        section_duration = 32 * beat_duration
        total_duration = len(y) / sr
        
        num_sections = int(total_duration / section_duration)
        
        section_types = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
        
        for i in range(num_sections):
            start = i * section_duration
            end = min((i + 1) * section_duration, total_duration)
            
            # Determine section type based on position and energy
            if i < len(section_types):
                section_type = section_types[i]
            else:
                section_type = "verse"
            
            # Calculate energy for this section
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            section_audio = y[start_sample:end_sample]
            section_energy = float(np.sqrt(np.mean(section_audio**2)))
            
            sections.append({
                "type": section_type,
                "start": start,
                "end": end,
                "duration": end - start,
                "energy": section_energy
            })
        
        return sections
    
    def _analyze_spectral(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze spectral features"""
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            "centroid_mean": float(np.mean(centroid)),
            "rolloff_mean": float(np.mean(rolloff)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "brightness": float(np.mean(centroid) / (sr / 2)),  # Normalized brightness
            "mfcc_mean": [float(m) for m in np.mean(mfccs, axis=1)]
        }
    
    def _classify_genre(self, y: np.ndarray, sr: int, bpm: float, spectral: Dict) -> str:
        """Simple genre classification"""
        brightness = spectral.get("brightness", 0.5)
        
        if bpm > 140:
            return "hardstyle"
        elif bpm > 128 and brightness > 0.3:
            return "vinahouse"
        elif bpm > 120 and brightness > 0.4:
            return "house"
        elif bpm > 100 and brightness < 0.3:
            return "hiphop"
        elif bpm < 100:
            return "chill"
        else:
            return "pop"
    
    def _find_audio_files(self, path: str, max_files: int) -> List[str]:
        """Find audio files in directory"""
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        files = []
        
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in audio_extensions):
                    files.append(os.path.join(root, filename))
                    if len(files) >= max_files:
                        return files
        
        return files
    
    def _analyze_eq_curve(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze EQ curve of audio"""
        # Simple frequency band analysis
        bands = {
            "sub": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 4000),
            "high": (4000, 8000),
            "air": (8000, 20000)
        }
        
        eq_curve = {}
        
        for band_name, (low, high) in bands.items():
            # Bandpass filter
            low_norm = max(low / (sr / 2), 0.001)
            high_norm = min(high / (sr / 2), 0.999)
            
            sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
            filtered = signal.sosfilt(sos, y)
            
            # Calculate energy in band
            energy = np.sqrt(np.mean(filtered**2))
            eq_curve[band_name] = float(energy)
        
        return eq_curve
    
    def _aggregate_eq_curves(self, eq_curves: List[Dict]) -> Dict:
        """Aggregate multiple EQ curves"""
        if not eq_curves:
            return self._get_default_vinahouse_style().eq_curve
        
        aggregated = {}
        bands = ["sub", "bass", "low_mid", "mid", "high_mid", "high", "air"]
        
        for band in bands:
            values = [eq.get(band, 0) for eq in eq_curves]
            aggregated[band] = float(np.mean(values))
        
        return aggregated
    
    def _get_default_vinahouse_style(self) -> LearnedStyle:
        """Get default Vinahouse style"""
        return LearnedStyle(
            avg_bpm=135,
            common_keys=["C major", "A minor", "G major"],
            eq_curve={
                "sub": 0.5,
                "bass": 0.4,
                "low_mid": 0.2,
                "mid": 0.3,
                "high_mid": 0.35,
                "high": 0.4,
                "air": 0.3
            },
            compression_ratio=3.0,
            reverb_amount=0.3,
            bass_enhancement=4.0,
            stereo_width=1.2,
            drop_characteristics={
                "energy_spike": 0.8,
                "bass_emphasis": 6.0,
                "kick_pattern": "4-on-floor"
            },
            buildup_style={
                "snare_roll": True,
                "rising_pitch": True,
                "filter_sweep": True
            }
        )
    
    def _create_remix_sections(self, y: np.ndarray, structure: RemixStructure, bpm: int) -> np.ndarray:
        """Create remix with proper structure"""
        beat_samples = int(60 / bpm * self.sample_rate)
        
        sections_audio = []
        
        for section in structure.sections:
            duration_samples = section["duration_beats"] * beat_samples
            section_energy = section.get("energy", 0.5)
            section_type = section["type"]
            
            # Get source material for this section
            if len(y) < duration_samples:
                # Loop if needed
                repeats = int(np.ceil(duration_samples / len(y)))
                source = np.tile(y, repeats)[:duration_samples]
            else:
                # Select best part of source
                source = self._select_best_section(y, duration_samples, section_type, section_energy)
            
            # Process section based on type
            processed = self._process_section(source, section_type, section_energy, bpm)
            sections_audio.append(processed)
        
        # Concatenate all sections
        return np.concatenate(sections_audio)
    
    def _select_best_section(self, y: np.ndarray, duration: int, section_type: str, energy: float) -> np.ndarray:
        """Select best portion of source for section type"""
        if len(y) <= duration:
            return y
        
        # For drops, select highest energy section
        if section_type == RemixSection.DROP.value:
            # Find highest energy region
            num_sections = len(y) // duration
            max_energy = 0
            best_start = 0
            
            for i in range(num_sections):
                start = i * duration
                end = start + duration
                section = y[start:end]
                section_energy = np.sqrt(np.mean(section**2))
                
                if section_energy > max_energy:
                    max_energy = section_energy
                    best_start = start
            
            return y[best_start:best_start + duration]
        
        # For intro/outro, select low energy sections
        elif section_type in [RemixSection.INTRO.value, RemixSection.OUTRO.value]:
            # Find lowest energy region
            num_sections = len(y) // duration
            min_energy = float('inf')
            best_start = 0
            
            for i in range(num_sections):
                start = i * duration
                end = start + duration
                section = y[start:end]
                section_energy = np.sqrt(np.mean(section**2))
                
                if section_energy < min_energy:
                    min_energy = section_energy
                    best_start = start
            
            return y[best_start:best_start + duration]
        
        # For other sections, use middle portion
        else:
            start = (len(y) - duration) // 2
            return y[start:start + duration]
    
    def _process_section(self, y: np.ndarray, section_type: str, energy: float, bpm: int) -> np.ndarray:
        """Process audio for specific section type"""
        
        if section_type == RemixSection.INTRO.value:
            # Intro: Filtered, low energy, building anticipation
            y = self._apply_lowpass(y, 2000)
            y = y * energy
            
        elif section_type == RemixSection.BUILDUP.value:
            # Buildup: Rising energy, filter opening, snare roll
            y = self._apply_buildup_effects(y, bpm)
            
        elif section_type == RemixSection.DROP.value:
            # Drop: Maximum energy, full spectrum, punchy
            y = self._apply_drop_effects(y, bpm, energy)
            
        elif section_type == RemixSection.BREAKDOWN.value:
            # Breakdown: Sparse, atmospheric, reverb
            y = self._apply_lowpass(y, 3000)
            y = self._add_reverb(y, 0.5)
            y = y * energy
            
        elif section_type == RemixSection.OUTRO.value:
            # Outro: Fading, filter closing
            y = self._apply_lowpass(y, 1000)
            fade = np.linspace(1, 0, len(y))
            y = y * fade * energy
            
        return y
    
    def _apply_lowpass(self, y: np.ndarray, freq: float) -> np.ndarray:
        """Apply lowpass filter"""
        sos = signal.butter(4, freq / (self.sample_rate / 2), btype='low', output='sos')
        return signal.sosfilt(sos, y)
    
    def _apply_highpass(self, y: np.ndarray, freq: float) -> np.ndarray:
        """Apply highpass filter"""
        sos = signal.butter(4, freq / (self.sample_rate / 2), btype='high', output='sos')
        return signal.sosfilt(sos, y)
    
    def _apply_buildup_effects(self, y: np.ndarray, bpm: int) -> np.ndarray:
        """Apply buildup effects"""
        beat_samples = int(60 / bpm * self.sample_rate)
        
        # Filter sweep (opening)
        num_steps = 16
        step_length = len(y) // num_steps
        result = np.zeros_like(y)
        
        for i in range(num_steps):
            start = i * step_length
            end = min((i + 1) * step_length, len(y))
            
            # Frequency rises from 500Hz to 10000Hz
            freq = 500 + (i / num_steps) * 9500
            section = y[start:end]
            
            # Apply increasing highpass (filter opening)
            sos = signal.butter(2, freq / (self.sample_rate / 2), btype='high', output='sos')
            result[start:end] = signal.sosfilt(sos, section)
        
        # Add rising volume
        rise = np.linspace(0.3, 1.0, len(y))
        result = result * rise
        
        return result
    
    def _apply_drop_effects(self, y: np.ndarray, bpm: int, energy: float) -> np.ndarray:
        """Apply drop effects - maximum impact"""
        # Bass boost
        sos_low = signal.butter(4, 200 / (self.sample_rate / 2), btype='low', output='sos')
        bass = signal.sosfilt(sos_low, y) * 2.0
        
        # Highpass original
        sos_high = signal.butter(4, 60 / (self.sample_rate / 2), btype='high', output='sos')
        y_filtered = signal.sosfilt(sos_high, y)
        
        # Combine
        result = y_filtered + bass
        
        # Apply sidechain
        result = self._apply_sidechain(result, 0.6, bpm)
        
        # Compress
        result = self._compress(result, -10, 3)
        
        # Normalize to energy level
        result = result * energy
        
        return result
    
    def _apply_sidechain(self, y: np.ndarray, intensity: float, bpm: int) -> np.ndarray:
        """Apply sidechain compression"""
        beat_samples = int(60 / bpm * self.sample_rate)
        
        envelope = np.ones(len(y))
        num_beats = len(y) // beat_samples
        
        for i in range(num_beats):
            start = i * beat_samples
            end = min(start + beat_samples, len(y))
            t = np.linspace(0, 1, end - start)
            
            # Duck on beat
            duck = 1 - intensity * 0.4 * np.exp(-t * 10)
            envelope[start:end] = duck
        
        return y * envelope
    
    def _compress(self, y: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
        """Apply compression"""
        envelope = np.abs(y)
        window = int(0.01 * self.sample_rate)
        envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        threshold_linear = 10 ** (threshold / 20)
        gain = np.ones_like(y)
        
        above = envelope_smooth > threshold_linear
        gain[above] = (threshold_linear / envelope_smooth[above]) ** (1 - 1/ratio)
        
        return y * gain
    
    def _add_reverb(self, y: np.ndarray, amount: float) -> np.ndarray:
        """Add reverb"""
        delay_samples = int(0.05 * self.sample_rate)
        impulse = np.zeros(delay_samples * 5)
        impulse[::delay_samples] = 0.3 ** np.arange(5)
        
        reverb = np.convolve(y, impulse, mode='same')
        return y * (1 - amount) + reverb * amount
    
    def _apply_learned_style(self, y: np.ndarray, style: LearnedStyle) -> np.ndarray:
        """Apply learned style to audio"""
        
        # Apply EQ curve
        if style.eq_curve:
            y = self._apply_style_eq(y, style.eq_curve)
        
        # Apply bass enhancement
        if style.bass_enhancement > 0:
            sos = signal.butter(4, 100 / (self.sample_rate / 2), btype='low', output='sos')
            bass = signal.sosfilt(sos, y)
            bass_gain = 10 ** (style.bass_enhancement / 20)
            y = y + bass * (bass_gain - 1)
        
        # Apply reverb
        if style.reverb_amount > 0:
            y = self._add_reverb(y, style.reverb_amount)
        
        # Apply compression
        y = self._compress(y, -12, style.compression_ratio)
        
        return y
    
    def _apply_style_eq(self, y: np.ndarray, eq_curve: Dict) -> np.ndarray:
        """Apply EQ from style"""
        bands = {
            "sub": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 4000),
            "high": (4000, 8000),
            "air": (8000, 16000)
        }
        
        # Normalize EQ curve
        max_val = max(eq_curve.values()) if eq_curve else 1
        
        for band_name, (low, high) in bands.items():
            if band_name in eq_curve:
                # Normalize gain
                gain_db = (eq_curve[band_name] / max_val - 0.5) * 10
                
                if abs(gain_db) > 0.5:
                    center_freq = (low + high) / 2
                    y = self._apply_bell_eq(y, center_freq, gain_db)
        
        return y
    
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
    
    def _apply_default_vinahouse_processing(self, y: np.ndarray) -> np.ndarray:
        """Apply default Vinahouse processing"""
        style = self._get_default_vinahouse_style()
        return self._apply_learned_style(y, style)
    
    def _master_for_club(self, y: np.ndarray) -> np.ndarray:
        """Master for club play"""
        # Multiband compression
        low = self._band_filter(y, 20, 200)
        mid = self._band_filter(y, 200, 2000)
        high = self._band_filter(y, 2000, 20000)
        
        low = self._compress(low, -8, 4.0)   # Heavy bass compression
        mid = self._compress(mid, -12, 2.5)
        high = self._compress(high, -15, 2.0)
        
        y = low + mid + high
        
        # Limiting
        ceiling = 10 ** (-0.3 / 20)
        y = np.tanh(y * 1.2) * ceiling
        y = np.clip(y, -ceiling, ceiling)
        
        # Normalize to -8 LUFS (club standard)
        rms = np.sqrt(np.mean(y**2))
        target_rms = 10 ** (-8 / 20)
        gain = min(target_rms / (rms + 1e-10), 3)
        y = y * gain
        
        # Final safety
        peak = np.max(np.abs(y))
        if peak > 0.95:
            y = y * (0.95 / peak)
        
        return y
    
    def _band_filter(self, y: np.ndarray, low: float, high: float) -> np.ndarray:
        """Bandpass filter"""
        nyquist = self.sample_rate / 2
        low_norm = max(low / nyquist, 0.001)
        high_norm = min(high / nyquist, 0.999)
        
        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        return signal.sosfilt(sos, y)


# Convenience functions
def create_intelligent_remix(input_path: str,
                             output_path: str,
                             library_path: Optional[str] = None,
                             target_bpm: int = 135) -> Dict:
    """
    Create intelligent remix.
    
    Args:
        input_path: Source track
        output_path: Output file
        library_path: Path to music library for style learning
        target_bpm: Target BPM
    """
    engine = IntelligentRemixEngine()
    
    # Learn from library if provided
    if library_path and os.path.exists(library_path):
        engine.learn_from_library(library_path)
    
    return engine.create_intelligent_remix(
        input_path=input_path,
        output_path=output_path,
        target_bpm=target_bpm
    )


# Default instance
intelligent_remix_engine = IntelligentRemixEngine()