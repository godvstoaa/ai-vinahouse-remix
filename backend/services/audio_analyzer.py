"""
Real Audio Analyzer using Librosa
Analyzes BPM, Key, Energy, and other features
"""
import librosa
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Real audio analysis using librosa"""
    
    # Musical key mapping
    KEY_MAP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    MODE_MAP = {0: 'minor', 1: 'major'}
    
    # Genre mapping based on tempo and features
    GENRE_FEATURES = {
        'edm': {'tempo_range': (120, 150), 'energy': 'high', 'beat_strength': 'strong'},
        'house': {'tempo_range': (115, 135), 'energy': 'high', 'beat_strength': 'strong'},
        'hiphop': {'tempo_range': (80, 115), 'energy': 'medium', 'beat_strength': 'strong'},
        'pop': {'tempo_range': (100, 130), 'energy': 'medium', 'beat_strength': 'medium'},
        'trap': {'tempo_range': (130, 180), 'energy': 'high', 'beat_strength': 'strong'},
        'lofi': {'tempo_range': (60, 90), 'energy': 'low', 'beat_strength': 'soft'},
        'techno': {'tempo_range': (120, 150), 'energy': 'high', 'beat_strength': 'strong'},
        'dubstep': {'tempo_range': (135, 145), 'energy': 'high', 'beat_strength': 'strong'},
        'r&b': {'tempo_range': (70, 110), 'energy': 'low', 'beat_strength': 'medium'},
        'rock': {'tempo_range': (90, 140), 'energy': 'high', 'beat_strength': 'strong'},
    }
    
    def __init__(self):
        self.sample_rate = 22050
    
    async def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Full audio analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Analyzing audio: {audio_path}")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get duration in 30-second segments for faster analysis
            segment_duration = 30  # seconds
            if duration > segment_duration:
                # Analyze first 30 seconds for quick results
                segment_samples = int(segment_duration * sr)
                y_analysis = y[:segment_samples]
            else:
                y_analysis = y
            
            # Run all analyses
            results = {
                'duration': float(duration),
                'sample_rate': int(sr),
                'bpm': await self._analyze_bpm(y_analysis, sr),
                'key': await self._analyze_key(y_analysis, sr),
                'energy': await self._analyze_energy(y_analysis),
                'danceability': await self._analyze_danceability(y_analysis, sr),
                'sections': await self._analyze_sections(y, sr),
                'spectral_features': await self._analyze_spectral(y_analysis, sr),
            }
            
            # Predict genre based on features
            results['predicted_genre'] = self._predict_genre(results)
            
            logger.info(f"Analysis complete: BPM={results['bpm']}, Key={results['key']}")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    async def _analyze_bpm(self, y: np.ndarray, sr: int) -> float:
        """Analyze tempo/BPM using librosa"""
        try:
            # Use librosa's beat tracking
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            
            # Handle array output
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)
            
            # Round to common BPM values
            bpm = round(tempo, 1)
            
            # Validate range
            if bpm < 60:
                bpm *= 2  # Double if too slow
            elif bpm > 200:
                bpm /= 2  # Half if too fast
                
            return round(bpm, 1)
            
        except Exception as e:
            logger.warning(f"BPM analysis failed: {e}")
            return 120.0
    
    async def _analyze_key(self, y: np.ndarray, sr: int) -> str:
        """Analyze musical key using Krumhansl-Schmuckler algorithm"""
        try:
            # Get chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Average chroma
            chroma_mean = np.mean(chroma, axis=1)
            
            # Krumhansl-Schmuckler major key profile
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            
            # Krumhansl-Schmuckler minor key profile  
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            # Calculate correlations for all keys
            major_correlations = []
            minor_correlations = []
            
            for i in range(12):
                # Rotate profile
                major_rotated = np.roll(major_profile, i)
                minor_rotated = np.roll(minor_profile, i)
                
                # Correlation
                major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
                
                major_correlations.append(major_corr)
                minor_correlations.append(minor_corr)
            
            # Find best match
            major_best_idx = np.argmax(major_correlations)
            minor_best_idx = np.argmax(minor_correlations)
            
            if major_correlations[major_best_idx] > minor_correlations[minor_best_idx]:
                return f"{self.KEY_MAP[major_best_idx]} major"
            else:
                return f"{self.KEY_MAP[minor_best_idx]} minor"
                
        except Exception as e:
            logger.warning(f"Key analysis failed: {e}")
            return "C major"
    
    async def _analyze_energy(self, y: np.ndarray) -> Dict[str, float]:
        """Analyze energy/intensity of audio"""
        try:
            # RMS energy
            rms = librosa.feature.rms(y=y)
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
            # Spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(y=y)
            centroid_mean = float(np.mean(centroid))
            
            # Dynamic range
            dynamic_range = float(np.max(y) - np.min(y))
            
            # Normalize energy to 0-1 scale
            # RMS typically 0-0.5 for most music
            energy_score = min(1.0, rms_mean * 3)
            
            return {
                'score': round(energy_score, 2),
                'rms': round(rms_mean, 4),
                'rms_variance': round(rms_std, 4),
                'brightness': round(centroid_mean / 10000, 2),  # Normalized
                'dynamic_range': round(dynamic_range, 2)
            }
            
        except Exception as e:
            logger.warning(f"Energy analysis failed: {e}")
            return {'score': 0.5, 'rms': 0.1, 'brightness': 0.5}
    
    async def _analyze_danceability(self, y: np.ndarray, sr: int) -> float:
        """Analyze danceability based on rhythm features"""
        try:
            # Get tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0])
            
            # Get onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_variance = np.std(onset_env)
            
            # Get beat pluse
            beat_frames = librosa.beat.beat_track(y=y, sr=sr)[1]
            beat_variance = np.var(beat_frames) if len(beat_frames) > 1 else 0
            
            # Danceability heuristics
            # Optimal dance tempo: 100-130 BPM
            tempo_score = 1.0 - min(1.0, abs(tempo - 115) / 50)
            
            # Strong beat = more danceable
            beat_score = min(1.0, onset_variance * 10)
            
            # Regular rhythm = more danceable
            regularity_score = max(0, 1.0 - beat_variance / 1000)
            
            danceability = (tempo_score * 0.4 + beat_score * 0.4 + regularity_score * 0.2)
            
            return round(danceability, 2)
            
        except Exception as e:
            logger.warning(f"Danceability analysis failed: {e}")
            return 0.5
    
    async def _analyze_sections(self, y: np.ndarray, sr: int) -> list:
        """Analyze song structure/sections"""
        try:
            # Get segment boundaries using spectral contrast
            boundaries = librosa.segment.agglomerative(y, k=10)
            
            # Convert to time
            times = librosa.frames_to_time(boundaries, sr=sr)
            
            sections = []
            for i, time in enumerate(times):
                sections.append({
                    'start': float(time),
                    'section': i + 1
                })
            
            return sections
            
        except Exception as e:
            logger.warning(f"Section analysis failed: {e}")
            return []
    
    async def _analyze_spectral(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze spectral features"""
        try:
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)
            
            return {
                'bandwidth_mean': round(float(np.mean(bandwidth)), 2),
                'rolloff_mean': round(float(np.mean(rolloff)), 2),
                'zcr_mean': round(float(np.mean(zcr)), 4),
                'flatness_mean': round(float(np.mean(flatness)), 4)
            }
            
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            return {}
    
    def _predict_genre(self, features: Dict) -> str:
        """Predict genre based on audio features"""
        bpm = features.get('bpm', 120)
        energy = features.get('energy', {}).get('score', 0.5)
        
        best_genre = 'pop'
        best_score = 0
        
        for genre, genre_features in self.GENRE_FEATURES.items():
            score = 0
            
            # Tempo matching
            tempo_range = genre_features['tempo_range']
            if tempo_range[0] <= bpm <= tempo_range[1]:
                score += 0.5
            else:
                # Distance from range
                distance = min(abs(bpm - tempo_range[0]), abs(bpm - tempo_range[1]))
                score += max(0, 0.5 - distance / 50)
            
            # Energy matching
            energy_level = genre_features['energy']
            if energy_level == 'high' and energy > 0.6:
                score += 0.3
            elif energy_level == 'low' and energy < 0.4:
                score += 0.3
            elif energy_level == 'medium' and 0.4 <= energy <= 0.6:
                score += 0.3
            
            if score > best_score:
                best_score = score
                best_genre = genre
        
        return best_genre


# Singleton instance
audio_analyzer = AudioAnalyzer()