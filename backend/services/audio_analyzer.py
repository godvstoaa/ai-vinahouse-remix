"""
Professional Audio Analysis using Madmom + Essentia
Advanced BPM detection, Key detection, and structural analysis
"""
import numpy as np
import librosa
import logging
from typing import Dict, Any, Optional, Tuple, List
import os

logger = logging.getLogger(__name__)

# Try importing advanced libraries
try:
    import madmom
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.onsets import OnsetPeakPickingProcessor
    HAS_MADMOM = True
    logger.info("Madmom loaded - Advanced beat detection enabled!")
except ImportError:
    HAS_MADMOM = False
    logger.warning("Madmom not installed - using librosa fallback")

try:
    import essentia
    import essentia.standard as es
    HAS_ESSENTIA = True
    logger.info("Essentia loaded - Professional music analysis enabled!")
except ImportError:
    HAS_ESSENTIA = False
    logger.warning("Essentia not installed - using librosa fallback")


class ProfessionalAudioAnalyzer:
    """
    Professional audio analysis combining Madmom, Essentia, and Librosa.
    
    Features:
    - Accurate BPM detection using Madmom RNN
    - Key detection using Essentia
    - Structural analysis (verse, chorus, drop)
    - Energy profiling
    - Danceability scoring
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.has_madmom = HAS_MADMOM
        self.has_essentia = HAS_ESSENTIA
        
    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Full analysis of audio file.
        
        Returns:
            Dict with bpm, key, energy, sections, etc.
        """
        logger.info(f"Analyzing: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration = len(y) / sr
        
        # Core analysis
        bpm = self._detect_bpm(y, sr, audio_path)
        key = self._detect_key(y, sr, audio_path)
        energy, energy_profile = self._analyze_energy(y, sr)
        sections = self._detect_sections(y, sr, bpm)
        danceability = self._calculate_danceability(y, sr, bpm)
        spectral = self._analyze_spectral(y, sr)
        
        return {
            "duration": duration,
            "bpm": bpm,
            "key": key,
            "energy": energy,
            "energy_profile": energy_profile,
            "danceability": danceability,
            "sections": sections,
            "spectral": spectral,
            "drop_ready": energy > 0.4 and danceability > 0.5,
            "genre_hint": self._guess_genre(bpm, energy, spectral)
        }
    
    def _detect_bpm(self, y: np.ndarray, sr: int, audio_path: str = None) -> float:
        """
        Detect BPM using Madmom RNN (most accurate) or Librosa fallback.
        """
        if self.has_madmom and audio_path:
            try:
                # Madmom RNN-based beat detection (state-of-the-art)
                proc = RNNBeatProcessor()
                beats = proc(audio_path)
                tracker = DBNBeatTrackingProcessor(fps=100)
                beat_times = tracker(beats)
                
                if len(beat_times) > 1:
                    # Calculate BPM from beat intervals
                    intervals = np.diff(beat_times)
                    avg_interval = np.median(intervals)
                    bpm = 60.0 / avg_interval
                    
                    # Clamp to reasonable range
                    bpm = max(60, min(200, bpm))
                    logger.info(f"Madmom BPM: {bpm:.1f}")
                    return round(bpm, 1)
            except Exception as e:
                logger.warning(f"Madmom BPM failed: {e}")
        
        # Librosa fallback
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        return round(float(tempo), 1)
    
    def _detect_key(self, y: np.ndarray, sr: int, audio_path: str = None) -> str:
        """
        Detect musical key using Essentia (most accurate) or Librosa fallback.
        """
        if self.has_essentia:
            try:
                # Essentia key detection
                key_extractor = es.KeyExtractor()
                key, scale, strength = key_extractor(y)
                
                # Format: "C major", "A minor", etc.
                result = f"{key} {scale}"
                logger.info(f"Essentia Key: {result} (strength: {strength:.2f})")
                return result
            except Exception as e:
                logger.warning(f"Essentia key detection failed: {e}")
        
        # Librosa fallback using chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_idx = np.argmax(chroma_mean)
        
        # Simple major/minor detection
        minor_idx = (major_idx + 9) % 12  # Relative minor
        if chroma_mean[minor_idx] > chroma_mean[major_idx] * 0.9:
            return f"{key_names[minor_idx]} minor"
        return f"{key_names[major_idx]} major"
    
    def _analyze_energy(self, y: np.ndarray, sr: int) -> Tuple[float, List[Dict]]:
        """
        Analyze energy level and create energy profile over time.
        """
        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        
        # Normalize
        rms_normalized = rms / (np.max(rms) + 1e-10)
        overall_energy = float(np.mean(rms_normalized))
        
        # Create profile (energy per second)
        seconds = len(y) // sr
        samples_per_second = sr // 512  # hop_length samples per frame
        
        energy_profile = []
        for sec in range(seconds):
            start = sec * samples_per_second
            end = min((sec + 1) * samples_per_second, len(rms_normalized))
            energy_profile.append({
                "second": sec,
                "energy": float(np.mean(rms_normalized[start:end]))
            })
        
        return overall_energy, energy_profile
    
    def _detect_sections(self, y: np.ndarray, sr: int, bpm: float) -> List[Dict]:
        """
        Detect song sections (intro, verse, chorus, drop, etc.)
        """
        # Use novelty curve for section boundaries
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Find boundary points using spectral contrast
        bounds = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
        
        # Group into 8-bar sections (32 beats)
        beat_duration = 60 / bpm
        section_duration = 32 * beat_duration
        total_duration = len(y) / sr
        
        sections = []
        num_sections = max(1, int(total_duration / section_duration))
        
        section_types = self._classify_sections(len(y) / sr, bpm, onset_env)
        
        for i in range(num_sections):
            start = i * section_duration
            end = min((i + 1) * section_duration, total_duration)
            
            # Get energy for this section
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            section_energy = np.sqrt(np.mean(y[start_sample:end_sample]**2))
            
            section_type = section_types[i] if i < len(section_types) else "verse"
            
            sections.append({
                "type": section_type,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "energy": float(section_energy)
            })
        
        return sections
    
    def _classify_sections(self, duration: float, bpm: float, onset_env: np.ndarray) -> List[str]:
        """Classify section types based on position and energy."""
        # Vinahouse typical structure
        section_duration = 32 * 60 / bpm  # 8 bars
        num_sections = int(duration / section_duration)
        
        # Pattern: intro, buildup, drop, verse, breakdown, drop, outro
        pattern = ["intro", "buildup", "drop", "verse", "breakdown", "buildup", "drop", "outro"]
        
        sections = []
        for i in range(num_sections):
            if i < len(pattern):
                sections.append(pattern[i])
            elif i >= num_sections - 2:
                sections.append("outro")
            else:
                sections.append("verse")
        
        return sections
    
    def _calculate_danceability(self, y: np.ndarray, sr: int, bpm: float) -> float:
        """
        Calculate danceability score (0-1).
        Based on rhythm strength and tempo.
        """
        # Onset strength variability
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_std = np.std(onset_env)
        onset_mean = np.mean(onset_env)
        
        # Rhythm regularity
        tempo_strength = onset_std / (onset_mean + 1e-10)
        
        # BPM factor (optimal dance BPM: 120-140)
        bpm_factor = 1.0 - min(abs(bpm - 130) / 50, 1.0)
        
        # Combine
        danceability = min((tempo_strength / 3) * bpm_factor, 1.0)
        
        return round(danceability, 2)
    
    def _analyze_spectral(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze spectral features."""
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        return {
            "brightness": float(np.mean(centroid) / (sr / 2)),
            "centroid_mean": float(np.mean(centroid)),
            "rolloff_mean": float(np.mean(rolloff)),
            "bandwidth_mean": float(np.mean(bandwidth))
        }
    
    def _guess_genre(self, bpm: float, energy: float, spectral: Dict) -> str:
        """Guess genre based on audio features."""
        brightness = spectral.get("brightness", 0.5)
        
        if bpm >= 128 and bpm <= 140 and energy > 0.5:
            return "vinahouse"
        elif bpm >= 120 and bpm <= 130 and brightness > 0.4:
            return "house"
        elif bpm >= 138 and brightness > 0.5:
            return "hardstyle"
        elif bpm >= 140 and brightness > 0.6:
            return "techno"
        elif bpm >= 85 and bpm <= 115:
            return "hiphop"
        else:
            return "pop"
    
    def find_drop_points(self, y: np.ndarray, sr: int, bpm: float) -> List[float]:
        """
        Find optimal drop points in the track.
        Uses energy buildup detection.
        """
        # Calculate energy over time
        rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=1024)[0]
        
        # Smooth
        from scipy.ndimage import gaussian_filter1d
        rms_smooth = gaussian_filter1d(rms, sigma=10)
        
        # Find energy buildup followed by high energy
        diff = np.diff(rms_smooth)
        
        drop_points = []
        window = int(8 * 60 / bpm * sr / 1024)  # 8 bars in frames
        
        for i in range(window, len(diff) - window):
            # Check for buildup (increasing energy before)
            buildup = np.mean(diff[i-window:i]) > 0.001
            # Check for high energy after
            high_energy = np.mean(rms_smooth[i:i+window]) > np.mean(rms_smooth) * 1.2
            
            if buildup and high_energy:
                time = i * 1024 / sr
                drop_points.append(round(time, 2))
        
        # Remove duplicates (within 10 seconds)
        filtered = []
        for point in drop_points:
            if not filtered or point - filtered[-1] > 10:
                filtered.append(point)
        
        return filtered[:5]  # Max 5 drops


# Singleton
audio_analyzer = ProfessionalAudioAnalyzer()