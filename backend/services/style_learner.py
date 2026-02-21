"""
Producer Style Learning Engine - Learns YOUR production style
Extracts tuning, effects, mixing techniques from your music collection
"""
import numpy as np
import librosa
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from scipy import signal
from scipy.stats import mode

logger = logging.getLogger(__name__)


@dataclass
class ProducerStyle:
    """Captures a producer's unique style"""
    name: str
    
    # EQ Profile
    eq_bass_boost: float = 0.0  # Low freq adjustment
    eq_mid_shape: float = 0.0   # Mid freq scoop/boost
    eq_high_shelf: float = 0.0  # High freq brightness
    eq_low_cut: float = 20.0    # HPF frequency
    
    # Compression Settings
    comp_threshold: float = -20.0  # dB
    comp_ratio: float = 4.0
    comp_attack: float = 10.0  # ms
    comp_release: float = 100.0  # ms
    
    # Reverb Characteristics
    reverb_size: str = "medium"  # small, medium, large, hall
    reverb_decay: float = 2.0  # seconds
    reverb_predelay: float = 20.0  # ms
    reverb_mix: float = 0.3  # wet/dry
    
    # Delay Settings
    delay_time: float = 0.25  # quarter note
    delay_feedback: float = 0.3
    delay_mix: float = 0.2
    
    # Stereo Width
    stereo_width: float = 1.0  # 0 = mono, 1 = normal, >1 = wide
    
    # Dynamic Range
    dynamic_range: float = 10.0  # dB
    
    # Saturation/Distortion
    saturation_amount: float = 0.0
    saturation_type: str = "soft"  # soft, hard, tape, tube
    
    # BPM Preferences
    preferred_bpm_range: Tuple[float, float] = (120.0, 130.0)
    
    # Key Preferences
    preferred_keys: List[str] = None
    
    # Genre Signature
    genre_signatures: Dict[str, float] = None
    
    # Mixing Levels (relative)
    vocal_level: float = 0.0  # dB
    drums_level: float = 0.0
    bass_level: float = 0.0
    synths_level: float = 0.0
    
    # Frequency Distribution
    freq_balance: Dict[str, float] = None
    
    # Punch/Transient
    transient_shape: float = 0.5  # 0 = soft, 1 = punchy
    
    # Low End Character
    bass_character: str = "tight"  # tight, deep, boomy, punchy
    
    # High End Character  
    high_character: str = "smooth"  # smooth, bright, harsh, airy
    
    def to_dict(self):
        return asdict(self)


class StyleLearner:
    """
    Learns production style from a collection of tracks.
    Extracts detailed audio processing parameters.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.tracks_analyzed = []
        self.style_profiles = {}
        
    def analyze_track_style(self, audio_path: str) -> Dict[str, Any]:
        """
        Deep analysis of a single track's production style.
        Extracts all processing parameters.
        """
        logger.info(f"Analyzing production style: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
        
        # Convert to mono for analysis if stereo
        if y.ndim > 1:
            y_mono = np.mean(y, axis=0)
            y_stereo = y
        else:
            y_mono = y
            y_stereo = np.array([y, y])
        
        analysis = {
            'file': audio_path,
            'duration': len(y_mono) / sr,
            
            # EQ Analysis
            'eq_profile': self._analyze_eq_profile(y_mono, sr),
            
            # Compression Analysis
            'compression': self._analyze_compression(y_mono, sr),
            
            # Reverb Analysis
            'reverb': self._analyze_reverb(y_mono, sr),
            
            # Delay Analysis
            'delay': self._analyze_delay(y_mono, sr),
            
            # Stereo Analysis
            'stereo': self._analyze_stereo(y_stereo, sr),
            
            # Dynamic Range
            'dynamics': self._analyze_dynamics(y_mono, sr),
            
            # Saturation Analysis
            'saturation': self._analyze_saturation(y_mono, sr),
            
            # Frequency Balance
            'freq_balance': self._analyze_freq_balance(y_mono, sr),
            
            # Transient Analysis
            'transients': self._analyze_transients(y_mono, sr),
            
            # BPM
            'bpm': self._detect_bpm(y_mono, sr),
            
            # Key
            'key': self._detect_key(y_mono, sr),
            
            # Energy Profile
            'energy_profile': self._analyze_energy_profile(y_mono, sr),
            
            # Spectral Characteristics
            'spectral': self._analyze_spectral(y_mono, sr),
            
            # Bass Character
            'bass_character': self._analyze_bass_character(y_mono, sr),
            
            # High End Character
            'high_character': self._analyze_high_character(y_mono, sr),
        }
        
        return analysis
    
    def _analyze_eq_profile(self, y: np.ndarray, sr: int) -> Dict:
        """Extract EQ settings from frequency response"""
        # Compute frequency spectrum
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Average spectrum
        avg_spectrum = np.mean(D, axis=1)
        
        # Normalize
        avg_spectrum = avg_spectrum / (np.max(avg_spectrum) + 1e-10)
        
        # Analyze bass (20-200 Hz)
        bass_mask = freqs < 200
        bass_level = np.mean(avg_spectrum[bass_mask])
        
        # Analyze low-mids (200-500 Hz)
        low_mid_mask = (freqs >= 200) & (freqs < 500)
        low_mid_level = np.mean(avg_spectrum[low_mid_mask])
        
        # Analyze mids (500-2000 Hz)
        mid_mask = (freqs >= 500) & (freqs < 2000)
        mid_level = np.mean(avg_spectrum[mid_mask])
        
        # Analyze high-mids (2000-8000 Hz)
        high_mid_mask = (freqs >= 2000) & (freqs < 8000)
        high_mid_level = np.mean(avg_spectrum[high_mid_mask])
        
        # Analyze highs (8000+ Hz)
        high_mask = freqs >= 8000
        high_level = np.mean(avg_spectrum[high_mask])
        
        # Find low cut (where bass starts)
        bass_start_idx = np.where(bass_mask)[0]
        if len(bass_start_idx) > 0:
            bass_region = avg_spectrum[bass_mask]
            # Find where bass becomes significant
            threshold = np.max(bass_region) * 0.1
            significant_bass = np.where(bass_region > threshold)[0]
            if len(significant_bass) > 0:
                low_cut = freqs[bass_start_idx[significant_bass[0]]]
            else:
                low_cut = 20.0
        else:
            low_cut = 20.0
        
        return {
            'bass_level': float(bass_level),
            'low_mid_level': float(low_mid_level),
            'mid_level': float(mid_level),
            'high_mid_level': float(high_mid_level),
            'high_level': float(high_level),
            'bass_boost': float(bass_level - mid_level),  # Relative to mids
            'mid_scoop': float(mid_level - (low_mid_level + high_mid_level) / 2),
            'high_shelf': float(high_level - high_mid_level),
            'low_cut_hz': float(low_cut),
            'spectrum': avg_spectrum.tolist()[:100],  # First 100 bins for visualization
        }
    
    def _analyze_compression(self, y: np.ndarray, sr: int) -> Dict:
        """Infer compression settings from dynamic analysis"""
        # RMS in windows
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Analyze dynamic range
        dynamic_range = np.percentile(rms_db, 95) - np.percentile(rms_db, 5)
        
        # Infer compression from distribution shape
        # Compressed audio has more uniform RMS distribution
        rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-10)
        rms_variance = np.var(rms_normalized)
        
        # Low variance = heavy compression
        compression_ratio_estimate = 1.0 + (1.0 - rms_variance) * 10
        
        # Estimate threshold from level distribution
        # Heavily compressed audio has peaks limited
        peak_level = np.max(np.abs(y))
        rms_level = np.sqrt(np.mean(y**2))
        crest_factor = peak_level / (rms_level + 1e-10)
        
        # Low crest factor = heavy limiting/compression
        if crest_factor < 3:
            threshold_estimate = -10
        elif crest_factor < 5:
            threshold_estimate = -15
        elif crest_factor < 8:
            threshold_estimate = -20
        else:
            threshold_estimate = -30
        
        # Attack estimation from transient analysis
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        if len(onset_peaks) > 10:
            # Analyze transient decay
            avg_transient_length = np.mean(np.diff(onset_peaks)) * 512 / sr * 1000  # ms
            attack_estimate = min(50, avg_transient_length * 0.1)
        else:
            attack_estimate = 10.0
        
        return {
            'dynamic_range_db': float(dynamic_range),
            'estimated_threshold_db': float(threshold_estimate),
            'estimated_ratio': float(min(compression_ratio_estimate, 20)),
            'estimated_attack_ms': float(attack_estimate),
            'estimated_release_ms': float(attack_estimate * 10),  # Rule of thumb
            'crest_factor': float(crest_factor),
            'rms_variance': float(rms_variance),
        }
    
    def _analyze_reverb(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze reverb characteristics"""
        # Use autocorrelation to find reverb tail
        autocorr = np.correlate(y[-sr*5:], y[-sr*5:], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        # Find decay rate
        decay_threshold = 0.1
        try:
            decay_point = np.where(autocorr < decay_threshold)[0][0]
            reverb_decay_estimate = decay_point / sr
        except:
            reverb_decay_estimate = 1.0
        
        # Estimate reverb size
        if reverb_decay_estimate < 0.5:
            size = "small"
        elif reverb_decay_estimate < 1.5:
            size = "medium"
        elif reverb_decay_estimate < 3.0:
            size = "large"
        else:
            size = "hall"
        
        # Analyze early reflections (predelay)
        # Look for second peak in onset correlation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        return {
            'estimated_size': size,
            'estimated_decay_s': float(min(reverb_decay_estimate, 5.0)),
            'estimated_predelay_ms': float(20.0),  # Default estimate
            'estimated_wet_dry': float(0.2 + reverb_decay_estimate * 0.1),
        }
    
    def _analyze_delay(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze delay/echo characteristics"""
        # Look for repeated patterns in onset structure
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Autocorrelate onset envelope
        autocorr = np.correlate(onset_env, onset_env, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        # Find delay taps (peaks after first)
        # Skip first 10 samples to avoid self-correlation
        skip = 10
        if len(autocorr) > skip + 50:
            search_region = autocorr[skip:skip+200]  # Search first ~1 second
            if len(search_region) > 0 and np.max(search_region) > 0.3:
                delay_peak = np.argmax(search_region) + skip
                delay_time = delay_peak * 512 / sr  # hop_length = 512
                
                # Snap to musical values
                if 0.1 < delay_time < 2.0:
                    delay_estimate = delay_time
                else:
                    delay_estimate = 0.25  # Default quarter note
            else:
                delay_estimate = 0.0
        else:
            delay_estimate = 0.0
        
        return {
            'detected': delay_estimate > 0,
            'estimated_time_s': float(delay_estimate),
            'estimated_feedback': float(0.3),  # Default estimate
            'estimated_mix': float(0.2 if delay_estimate > 0 else 0.0),
        }
    
    def _analyze_stereo(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze stereo width and imaging"""
        if y.ndim < 2:
            return {
                'width': 0.0,
                'is_mono': True,
                'phase_correlation': 1.0,
            }
        
        left = y[0]
        right = y[1]
        
        # Stereo width (difference vs sum)
        mid = (left + right) / 2
        side = (left - right) / 2
        
        mid_power = np.mean(mid**2)
        side_power = np.mean(side**2)
        
        width = np.sqrt(side_power / (mid_power + 1e-10))
        
        # Phase correlation
        correlation = np.corrcoef(left, right)[0, 1]
        
        return {
            'width': float(min(width, 2.0)),
            'is_mono': width < 0.1,
            'phase_correlation': float(correlation),
        }
    
    def _analyze_dynamics(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze overall dynamic characteristics"""
        # Peak and RMS
        peak = np.max(np.abs(y))
        rms = np.sqrt(np.mean(y**2))
        
        # Dynamic range in dB
        # Use percentiles to avoid outliers
        rms_windows = librosa.feature.rms(y=y)[0]
        rms_db = 20 * np.log10(rms_windows + 1e-10)
        
        dr = np.percentile(rms_db, 95) - np.percentile(rms_db, 20)
        
        # LUFS approximation
        # Weighted RMS (simplified)
        lufs_estimate = 20 * np.log10(rms + 1e-10) - 10  # Rough approximation
        
        return {
            'peak': float(peak),
            'rms': float(rms),
            'dynamic_range_db': float(dr),
            'estimated_lufs': float(lufs_estimate),
            'is_loud': rms > 0.1,
        }
    
    def _analyze_saturation(self, y: np.ndarray, sr: int) -> Dict:
        """Detect saturation/distortion characteristics"""
        # Check for harmonic distortion
        D = np.abs(librosa.stft(y))
        
        # Analyze harmonic content
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Look for even harmonics (indicates saturation)
        # and odd harmonics (indicates clipping)
        
        # Simplified: check crest factor and THD estimate
        peak = np.max(np.abs(y))
        rms = np.sqrt(np.mean(y**2))
        crest = peak / (rms + 1e-10)
        
        # Low crest factor with high RMS = saturation
        saturation_indicator = (1.0 / crest) * rms * 10
        
        # Analyze high frequency content (saturation adds harmonics)
        high_freq_mask = freqs > 5000
        high_freq_energy = np.mean(D[high_freq_mask, :])
        mid_freq_mask = (freqs > 500) & (freqs < 2000)
        mid_freq_energy = np.mean(D[mid_freq_mask, :])
        
        brightness_ratio = high_freq_energy / (mid_freq_energy + 1e-10)
        
        return {
            'amount': float(min(saturation_indicator, 1.0)),
            'type': "tape" if brightness_ratio > 0.1 else "soft",
            'has_harmonic_distortion': brightness_ratio > 0.05,
        }
    
    def _analyze_freq_balance(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze frequency band balance"""
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        bands = {
            'sub': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'air': (6000, 20000),
        }
        
        balance = {}
        for name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                balance[name] = float(np.mean(D[mask, :]))
            else:
                balance[name] = 0.0
        
        # Normalize
        total = sum(balance.values()) + 1e-10
        balance = {k: v/total for k, v in balance.items()}
        
        return balance
    
    def _analyze_transients(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze transient characteristics"""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Transient sharpness
        onset_diff = np.diff(onset_env)
        sharpness = np.std(onset_diff) / (np.mean(onset_env) + 1e-10)
        
        # Normalize to 0-1
        transient_score = min(1.0, sharpness / 5)
        
        return {
            'sharpness': float(transient_score),
            'density': float(len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y)/sr)),
            'shape': "punchy" if transient_score > 0.5 else "soft",
        }
    
    def _detect_bpm(self, y: np.ndarray, sr: int) -> float:
        """Detect BPM"""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0])
            return float(tempo)
        except:
            return 120.0
    
    def _detect_key(self, y: np.ndarray, sr: int) -> str:
        """Detect musical key"""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Simple key detection
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            major_idx = np.argmax(chroma_mean)
            
            return f"{key_names[major_idx]} major"
        except:
            return "C major"
    
    def _analyze_energy_profile(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze energy distribution over time"""
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr)
        
        # Find energy peaks (chorus/drop sections)
        threshold = np.mean(rms) + np.std(rms)
        peaks = rms > threshold
        
        # Calculate energy statistics
        return {
            'mean': float(np.mean(rms)),
            'std': float(np.std(rms)),
            'max': float(np.max(rms)),
            'min': float(np.min(rms)),
            'variance': float(np.var(rms)),
        }
    
    def _analyze_spectral(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze spectral characteristics"""
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        
        return {
            'centroid_mean': float(np.mean(centroid)),
            'bandwidth_mean': float(np.mean(bandwidth)),
            'rolloff_mean': float(np.mean(rolloff)),
            'flatness_mean': float(np.mean(flatness)),
            'brightness': float(np.mean(centroid) / 5000),  # Normalized
        }
    
    def _analyze_bass_character(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze low frequency character"""
        # Isolate bass
        bass = librosa.effects.lowpass(y, freq=200)
        
        # Analyze bass transients
        bass_rms = librosa.feature.rms(y=bass)[0]
        bass_dynamic = np.std(bass_rms) / (np.mean(bass_rms) + 1e-10)
        
        # Character classification
        if bass_dynamic > 0.5:
            character = "punchy"
        elif np.mean(bass_rms) > 0.1:
            character = "deep"
        else:
            character = "tight"
        
        return {
            'character': character,
            'dynamic_ratio': float(bass_dynamic),
            'sustain': float(1.0 - bass_dynamic),
        }
    
    def _analyze_high_character(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze high frequency character"""
        # Isolate highs
        highs = librosa.effects.highpass(y, freq=5000)
        
        high_rms = np.sqrt(np.mean(highs**2))
        
        # Check for harsh frequencies
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        harsh_range = (freqs >= 3000) & (freqs < 6000)
        harsh_energy = np.mean(D[harsh_range, :]) if np.any(harsh_range) else 0
        
        air_range = freqs >= 10000
        air_energy = np.mean(D[air_range, :]) if np.any(air_range) else 0
        
        # Character classification
        if air_energy > harsh_energy * 0.5:
            character = "airy"
        elif harsh_energy > 0.1:
            character = "bright"
        else:
            character = "smooth"
        
        return {
            'character': character,
            'brightness': float(high_rms),
            'air_content': float(air_energy),
        }
    
    def learn_from_collection(self, music_dir: str, producer_name: str = "default") -> ProducerStyle:
        """
        Learn production style from a collection of tracks.
        Averages all parameters across tracks.
        """
        logger.info(f"Learning style from collection: {music_dir}")
        
        # Find all audio files
        audio_files = []
        for root, dirs, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.aiff', '.m4a')):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            logger.warning(f"No audio files found in {music_dir}")
            return ProducerStyle(name=producer_name)
        
        logger.info(f"Found {len(audio_files)} tracks to analyze")
        
        # Analyze all tracks
        all_analyses = []
        for i, audio_file in enumerate(audio_files):
            try:
                logger.info(f"Analyzing track {i+1}/{len(audio_files)}: {audio_file}")
                analysis = self.analyze_track_style(audio_file)
                all_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze {audio_file}: {e}")
        
        if not all_analyses:
            return ProducerStyle(name=producer_name)
        
        # Aggregate results
        style = self._aggregate_styles(all_analyses, producer_name)
        
        # Save style profile
        self.style_profiles[producer_name] = style
        self._save_style_profile(style, producer_name)
        
        logger.info(f"Style profile '{producer_name}' created from {len(all_analyses)} tracks")
        
        return style
    
    def _aggregate_styles(self, analyses: List[Dict], name: str) -> ProducerStyle:
        """Aggregate multiple track analyses into a single style profile"""
        
        # EQ
        eq_bass = np.mean([a['eq_profile']['bass_boost'] for a in analyses])
        eq_mid = np.mean([a['eq_profile']['mid_scoop'] for a in analyses])
        eq_high = np.mean([a['eq_profile']['high_shelf'] for a in analyses])
        eq_lowcut = np.mean([a['eq_profile']['low_cut_hz'] for a in analyses])
        
        # Compression
        comp_thresh = np.mean([a['compression']['estimated_threshold_db'] for a in analyses])
        comp_ratio = np.mean([a['compression']['estimated_ratio'] for a in analyses])
        comp_attack = np.mean([a['compression']['estimated_attack_ms'] for a in analyses])
        comp_release = np.mean([a['compression']['estimated_release_ms'] for a in analyses])
        
        # Reverb
        reverb_sizes = [a['reverb']['estimated_size'] for a in analyses]
        size_map = {"small": 1, "medium": 2, "large": 3, "hall": 4}
        avg_size_num = np.mean([size_map.get(s, 2) for s in reverb_sizes])
        if avg_size_num < 1.5:
            reverb_size = "small"
        elif avg_size_num < 2.5:
            reverb_size = "medium"
        elif avg_size_num < 3.5:
            reverb_size = "large"
        else:
            reverb_size = "hall"
        
        reverb_decay = np.mean([a['reverb']['estimated_decay_s'] for a in analyses])
        reverb_mix = np.mean([a['reverb']['estimated_wet_dry'] for a in analyses])
        
        # Delay
        delay_times = [a['delay']['estimated_time_s'] for a in analyses if a['delay']['detected']]
        delay_time = np.mean(delay_times) if delay_times else 0.25
        
        # Stereo
        stereo_width = np.mean([a['stereo']['width'] for a in analyses])
        
        # Dynamics
        dynamic_range = np.mean([a['dynamics']['dynamic_range_db'] for a in analyses])
        
        # Saturation
        sat_amount = np.mean([a['saturation']['amount'] for a in analyses])
        sat_types = [a['saturation']['type'] for a in analyses]
        sat_type = max(set(sat_types), key=sat_types.count) if sat_types else "soft"
        
        # BPM
        bpms = [a['bpm'] for a in analyses]
        avg_bpm = np.mean(bpms)
        bpm_std = np.std(bpms)
        bpm_range = (avg_bpm - bpm_std, avg_bpm + bpm_std)
        
        # Keys
        keys = [a['key'] for a in analyses]
        preferred_keys = list(set(keys))[:5]  # Top 5 unique keys
        
        # Frequency balance
        freq_balance = {}
        for band in ['sub', 'bass', 'low_mid', 'mid', 'high_mid', 'presence', 'air']:
            freq_balance[band] = np.mean([a['freq_balance'][band] for a in analyses])
        
        # Transients
        transient_scores = [a['transients']['sharpness'] for a in analyses]
        avg_transient = np.mean(transient_scores)
        transient_shape = "punchy" if avg_transient > 0.5 else "soft"
        
        # Bass character
        bass_chars = [a['bass_character']['character'] for a in analyses]
        bass_character = max(set(bass_chars), key=bass_chars.count) if bass_chars else "tight"
        
        # High character
        high_chars = [a['high_character']['character'] for a in analyses]
        high_character = max(set(high_chars), key=high_chars.count) if high_chars else "smooth"
        
        return ProducerStyle(
            name=name,
            eq_bass_boost=eq_bass,
            eq_mid_shape=eq_mid,
            eq_high_shelf=eq_high,
            eq_low_cut=eq_lowcut,
            comp_threshold=comp_thresh,
            comp_ratio=comp_ratio,
            comp_attack=comp_attack,
            comp_release=comp_release,
            reverb_size=reverb_size,
            reverb_decay=reverb_decay,
            reverb_mix=reverb_mix,
            delay_time=delay_time,
            delay_mix=0.2 if delay_time > 0 else 0.0,
            stereo_width=stereo_width,
            dynamic_range=dynamic_range,
            saturation_amount=sat_amount,
            saturation_type=sat_type,
            preferred_bpm_range=bpm_range,
            preferred_keys=preferred_keys,
            freq_balance=freq_balance,
            transient_shape=avg_transient,
            bass_character=bass_character,
            high_character=high_character,
        )
    
    def _save_style_profile(self, style: ProducerStyle, name: str):
        """Save style profile to disk"""
        os.makedirs("models/producer_styles", exist_ok=True)
        path = f"models/producer_styles/{name}.json"
        
        with open(path, 'w') as f:
            json.dump(style.to_dict(), f, indent=2)
        
        logger.info(f"Style profile saved: {path}")
    
    def load_style_profile(self, name: str) -> Optional[ProducerStyle]:
        """Load a saved style profile"""
        path = f"models/producer_styles/{name}.json"
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return ProducerStyle(**data)
    
    def list_styles(self) -> List[str]:
        """List available style profiles"""
        style_dir = "models/producer_styles"
        if not os.path.exists(style_dir):
            return []
        
        return [f.replace('.json', '') for f in os.listdir(style_dir) if f.endswith('.json')]


# Singleton
style_learner = StyleLearner()