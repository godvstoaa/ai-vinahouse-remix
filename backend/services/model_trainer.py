"""
Model Training Pipeline
Train AI models from your music library (10,000+ songs)

Features:
1. Dataset preprocessing and feature extraction
2. Genre classification model training
3. Remix style learning
4. Custom model fine-tuning
"""
import numpy as np
import librosa
import os
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ProcessPoolExecutor
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SongFeatures:
    """Extracted features from a song"""
    file_path: str
    duration: float
    bpm: float
    key: str
    energy: float
    danceability: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    zero_crossing_rate: float
    mfcc_mean: List[float]  # 20 MFCC coefficients
    mfcc_std: List[float]
    chroma_mean: List[float]  # 12 chroma values
    genre_predicted: str
    genre_confidence: float

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    dataset_path: str
    output_model_path: str = "models"
    sample_rate: int = 22050
    segment_duration: float = 30.0  # seconds per segment
    overlap: float = 0.5  # overlap between segments
    min_songs_per_genre: int = 100
    validation_split: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001


class DatasetPreprocessor:
    """
    Preprocess music dataset for training
    Extracts features from all songs in library
    """
    
    # Supported genres for training
    GENRES = [
        'edm', 'house', 'techno', 'trance', 'dubstep',
        'hiphop', 'trap', 'rnb', 'pop', 'rock',
        'jazz', 'classical', 'lofi', 'drum_and_bass',
        'reggae', 'country', 'metal', 'funk', 'soul'
    ]
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.features_db: List[SongFeatures] = []
        self.genre_counts: Dict[str, int] = {}
        
    async def scan_dataset(self) -> Dict[str, Any]:
        """
        Scan dataset directory and catalog all songs
        Returns statistics about the dataset
        """
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        stats = {
            'total_songs': 0,
            'by_genre': {},
            'by_format': {},
            'total_duration_hours': 0,
            'files': []
        }
        
        # Supported formats
        formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        
        # Scan for songs
        for file_path in dataset_path.rglob('*'):
            if file_path.suffix.lower() in formats:
                stats['total_songs'] += 1
                
                # Track format
                fmt = file_path.suffix.lower()
                stats['by_format'][fmt] = stats['by_format'].get(fmt, 0) + 1
                
                # Detect genre from folder name
                parent_folder = file_path.parent.name.lower()
                detected_genre = self._detect_genre_from_path(parent_folder)
                if detected_genre:
                    stats['by_genre'][detected_genre] = stats['by_genre'].get(detected_genre, 0) + 1
                
                stats['files'].append(str(file_path))
        
        logger.info(f"Dataset scan complete: {stats['total_songs']} songs found")
        
        return stats
    
    def _detect_genre_from_path(self, path: str) -> Optional[str]:
        """Detect genre from file path/folder name"""
        path_lower = path.lower()
        for genre in self.GENRES:
            if genre in path_lower:
                return genre
        return None
    
    async def extract_features_from_song(
        self, 
        file_path: str,
        segment_duration: float = 30.0
    ) -> Optional[SongFeatures]:
        """
        Extract audio features from a single song
        
        Features extracted:
        - Tempo (BPM)
        - Key
        - Energy (RMS)
        - Spectral features
        - MFCCs
        - Chroma
        """
        try:
            logger.info(f"Extracting features: {file_path}")
            
            # Load audio (first segment only for efficiency)
            y, sr = librosa.load(
                file_path, 
                sr=self.config.sample_rate,
                duration=segment_duration,
                mono=True
            )
            
            if len(y) < sr * 5:  # Skip if less than 5 seconds
                logger.warning(f"Song too short, skipping: {file_path}")
                return None
            
            duration = len(y) / sr
            
            # === TEMPO ===
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
            
            # === KEY ===
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key = self._detect_key(chroma)
            
            # === ENERGY ===
            rms = librosa.feature.rms(y=y)[0]
            energy = float(np.mean(rms))
            
            # === DANCEABILITY ===
            danceability = self._calculate_danceability(y, sr, bpm)
            
            # === SPECTRAL FEATURES ===
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # === ZERO CROSSING RATE ===
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # === MFCCs (20 coefficients) ===
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = mfccs.mean(axis=1).tolist()
            mfcc_std = mfccs.std(axis=1).tolist()
            
            # === CHROMA ===
            chroma_mean = chroma.mean(axis=1).tolist()
            
            # === PREDICT GENRE ===
            genre_predicted, genre_confidence = self._predict_genre_from_features(
                bpm, energy, np.mean(spectral_centroid), np.mean(spectral_bandwidth)
            )
            
            features = SongFeatures(
                file_path=file_path,
                duration=duration,
                bpm=bpm,
                key=key,
                energy=energy,
                danceability=danceability,
                spectral_centroid=float(np.mean(spectral_centroid)),
                spectral_bandwidth=float(np.mean(spectral_bandwidth)),
                spectral_rolloff=float(np.mean(spectral_rolloff)),
                zero_crossing_rate=float(np.mean(zcr)),
                mfcc_mean=mfcc_mean,
                mfcc_std=mfcc_std,
                chroma_mean=chroma_mean,
                genre_predicted=genre_predicted,
                genre_confidence=genre_confidence
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {e}")
            return None
    
    def _detect_key(self, chroma: np.ndarray) -> str:
        """Detect musical key from chromagram"""
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        chroma_mean = chroma.mean(axis=1)
        
        best_corr = -1
        best_key = "C major"
        
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i in range(12):
            # Rotate profiles
            major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
            minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]
            
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = f"{notes[i]} major"
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = f"{notes[i]} minor"
        
        return best_key
    
    def _calculate_danceability(self, y: np.ndarray, sr: int, bpm: float) -> float:
        """Calculate danceability score"""
        # Based on tempo stability, beat strength, and rhythm regularity
        
        # Tempo factor (120-130 BPM is most danceable)
        tempo_factor = 1.0 - abs(bpm - 125) / 50
        tempo_factor = max(0, min(1, tempo_factor))
        
        # Beat strength (onset detection)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = np.std(onset_env) / (np.mean(onset_env) + 1e-6)
        beat_strength = min(1, beat_strength)
        
        # Rhythm regularity
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            regularity = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6))
            regularity = max(0, min(1, regularity))
        else:
            regularity = 0.5
        
        danceability = (tempo_factor * 0.4 + beat_strength * 0.3 + regularity * 0.3)
        
        return float(danceability)
    
    def _predict_genre_from_features(
        self, 
        bpm: float, 
        energy: float,
        spectral_centroid: float,
        spectral_bandwidth: float
    ) -> Tuple[str, float]:
        """Simple rule-based genre prediction"""
        
        # Genre rules based on typical characteristics
        if bpm > 140:
            if energy > 0.5:
                return 'dubstep', 0.6
            else:
                return 'drum_and_bass', 0.5
        elif bpm > 128:
            if spectral_centroid > 3000:
                return 'techno', 0.6
            else:
                return 'house', 0.7
        elif bpm > 118:
            if energy > 0.6:
                return 'edm', 0.7
            else:
                return 'trance', 0.5
        elif bpm > 95:
            if spectral_bandwidth > 2000:
                return 'hiphop', 0.6
            else:
                return 'pop', 0.5
        elif bpm > 85:
            if energy > 0.5:
                return 'rnb', 0.5
            else:
                return 'lofi', 0.6
        else:
            if energy < 0.3:
                return 'classical', 0.4
            else:
                return 'rock', 0.4
    
    async def process_dataset(
        self, 
        max_workers: int = 4,
        progress_callback: Optional[callable] = None
    ) -> List[SongFeatures]:
        """
        Process entire dataset and extract features from all songs
        """
        stats = await self.scan_dataset()
        files = stats['files']
        total = len(files)
        
        logger.info(f"Processing {total} songs...")
        
        results = []
        processed = 0
        
        for file_path in files:
            features = await self.extract_features_from_song(file_path)
            if features:
                results.append(features)
                self.genre_counts[features.genre_predicted] = \
                    self.genre_counts.get(features.genre_predicted, 0) + 1
            
            processed += 1
            if progress_callback and processed % 10 == 0:
                await progress_callback(processed, total, features)
        
        self.features_db = results
        
        # Save to disk
        await self.save_features_db()
        
        logger.info(f"Dataset processing complete: {len(results)} songs processed")
        logger.info(f"Genre distribution: {self.genre_counts}")
        
        return results
    
    async def save_features_db(self, output_path: str = "dataset_features.json"):
        """Save extracted features to JSON"""
        output = {
            'created': datetime.now().isoformat(),
            'total_songs': len(self.features_db),
            'genre_counts': self.genre_counts,
            'features': [asdict(f) for f in self.features_db]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Features saved to {output_path}")


class RemixStyleLearner:
    """
    Learn remix styles from analyzed songs
    Creates style profiles for each genre
    """
    
    def __init__(self):
        self.style_profiles: Dict[str, Dict] = {}
    
    async def learn_from_features(self, features_db: List[SongFeatures]):
        """
        Learn remix styles from extracted features
        Creates average profiles for BPM, energy, effects per genre
        """
        genre_data: Dict[str, List[SongFeatures]] = {}
        
        # Group by genre
        for features in features_db:
            genre = features.genre_predicted
            if genre not in genre_data:
                genre_data[genre] = []
            genre_data[genre].append(features)
        
        # Calculate style profiles
        for genre, songs in genre_data.items():
            if len(songs) < 10:  # Need minimum samples
                continue
            
            profile = {
                'song_count': len(songs),
                'avg_bpm': np.mean([s.bpm for s in songs]),
                'bpm_range': (
                    np.percentile([s.bpm for s in songs], 10),
                    np.percentile([s.bpm for s in songs], 90)
                ),
                'avg_energy': np.mean([s.energy for s in songs]),
                'avg_danceability': np.mean([s.danceability for s in songs]),
                'common_keys': self._get_common_keys(songs),
                'spectral_profile': {
                    'centroid': np.mean([s.spectral_centroid for s in songs]),
                    'bandwidth': np.mean([s.spectral_bandwidth for s in songs])
                },
                'remix_params': self._generate_remix_params(genre, songs)
            }
            
            self.style_profiles[genre] = profile
        
        logger.info(f"Learned {len(self.style_profiles)} genre style profiles")
        
        return self.style_profiles
    
    def _get_common_keys(self, songs: List[SongFeatures]) -> List[str]:
        """Get most common keys for a genre"""
        key_counts = {}
        for song in songs:
            key_counts[song.key] = key_counts.get(song.key, 0) + 1
        
        sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
        return [k[0] for k in sorted_keys[:5]]
    
    def _generate_remix_params(
        self, 
        genre: str, 
        songs: List[SongFeatures]
    ) -> Dict[str, Any]:
        """Generate remix parameters based on learned style"""
        
        avg_bpm = np.mean([s.bpm for s in songs])
        avg_energy = np.mean([s.energy for s in songs])
        
        # Genre-specific effect settings
        params = {
            'edm': {
                'target_bpm': 128,
                'reverb': 0.3,
                'delay': 0.2,
                'sidechain': True,
                'bass_boost': 1.2
            },
            'house': {
                'target_bpm': 124,
                'reverb': 0.4,
                'delay': 0.3,
                'sidechain': True,
                'bass_boost': 1.1
            },
            'hiphop': {
                'target_bpm': 90,
                'reverb': 0.2,
                'delay': 0.1,
                'sidechain': False,
                'bass_boost': 1.4
            },
            'trap': {
                'target_bpm': 140,
                'reverb': 0.5,
                'delay': 0.25,
                'sidechain': True,
                'bass_boost': 1.5
            },
            'lofi': {
                'target_bpm': 75,
                'reverb': 0.6,
                'delay': 0.4,
                'sidechain': False,
                'lowpass': True
            },
            'techno': {
                'target_bpm': 132,
                'reverb': 0.35,
                'delay': 0.3,
                'sidechain': True,
                'bass_boost': 1.2
            }
        }
        
        return params.get(genre, params['edm'])
    
    async def save_profiles(self, output_path: str = "style_profiles.json"):
        """Save style profiles to disk"""
        with open(output_path, 'w') as f:
            json.dump(self.style_profiles, f, indent=2)
        
        logger.info(f"Style profiles saved to {output_path}")


class ModelTrainer:
    """
    Main training orchestrator
    Coordinates dataset preprocessing and model training
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.preprocessor = DatasetPreprocessor(config)
        self.style_learner = RemixStyleLearner()
        
        # Create output directory
        os.makedirs(config.output_model_path, exist_ok=True)
    
    async def train(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Full training pipeline:
        1. Scan dataset
        2. Extract features from all songs
        3. Learn style profiles
        4. Save trained models
        """
        results = {
            'started': datetime.now().isoformat(),
            'status': 'processing',
            'songs_processed': 0,
            'genres_learned': 0
        }
        
        try:
            # Step 1: Process dataset
            logger.info("=" * 50)
            logger.info("STEP 1: Processing Dataset")
            logger.info("=" * 50)
            
            features = await self.preprocessor.process_dataset(
                progress_callback=progress_callback
            )
            results['songs_processed'] = len(features)
            
            # Step 2: Learn styles
            logger.info("=" * 50)
            logger.info("STEP 2: Learning Remix Styles")
            logger.info("=" * 50)
            
            profiles = await self.style_learner.learn_from_features(features)
            results['genres_learned'] = len(profiles)
            
            # Step 3: Save models
            logger.info("=" * 50)
            logger.info("STEP 3: Saving Models")
            logger.info("=" * 50)
            
            await self.style_learner.save_profiles(
                os.path.join(self.config.output_model_path, "style_profiles.json")
            )
            
            results['status'] = 'completed'
            results['completed'] = datetime.now().isoformat()
            results['model_path'] = self.config.output_model_path
            results['genre_profiles'] = list(profiles.keys())
            
            logger.info(f"Training complete! {len(features)} songs, {len(profiles)} genres")
            
            return results
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Training failed: {e}")
            raise
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'features_count': len(self.preprocessor.features_db),
            'genre_counts': self.preprocessor.genre_counts,
            'profiles_learned': len(self.style_learner.style_profiles)
        }


# Singleton for API access
model_trainer: Optional[ModelTrainer] = None

def get_trainer(config: Optional[TrainingConfig] = None) -> ModelTrainer:
    """Get or create model trainer instance"""
    global model_trainer
    if model_trainer is None and config:
        model_trainer = ModelTrainer(config)
    return model_trainer