"""
Data Preparation Pipeline - Cắt và chuẩn bị dữ liệu training
Biến 10,000 bài Vinahouse + segments từ Long Audio thành hàng triệu segments 30 giây
"""

import numpy as np
import librosa
import soundfile as sf
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass, asdict
from tqdm import tqdm
import yaml
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AudioSegment:
    """Metadata for a single audio segment"""
    segment_id: str
    source_file: str
    start_time: float  # seconds
    end_time: float
    duration: float
    bpm: Optional[float]
    key: Optional[str]
    energy_label: str  # 'low', 'medium', 'high', 'build', 'drop'
    mel_spectrogram_path: Optional[str]
    audio_path: str
    
    def to_dict(self):
        return asdict(self)


class DataPrepPipeline:
    """
    Prepare audio data for DiT training.
    - Cuts tracks into 30-second segments
    - Extracts BPM/Key/Energy labels
    - Generates Mel spectrograms
    - Saves processed data for streaming
    """
    
    ENERGY_LABELS = ['low', 'medium', 'high', 'build', 'drop']
    
    def __init__(self, config_path: str = "config/train_config.yaml"):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.n_mels = self.config['audio']['n_mels']
        self.n_fft = self.config['audio']['n_fft']
        self.hop_length = self.config['audio']['hop_length']
        
        self.segment_duration = self.config['data']['segment_duration']
        self.segment_overlap = self.config['data']['segment_overlap']
        
        # Output directories
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.metadata_dir = Path(self.config['data']['metadata_dir'])
        self.cache_dir = Path(self.config['data']['cache_dir'])
        
        # Create directories
        for d in [self.processed_dir, self.metadata_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def generate_segment_id(self, file_path: str, start_time: float) -> str:
        """Generate unique ID for a segment"""
        unique_str = f"{file_path}_{start_time}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract log-mel spectrogram"""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel
    
    def estimate_bpm(self, y: np.ndarray) -> Optional[float]:
        """Estimate tempo"""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sample_rate)
            return float(tempo)
        except:
            return None
    
    def estimate_key(self, y: np.ndarray) -> Optional[str]:
        """Estimate musical key using Krumhansl-Schmuckler algorithm"""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Major and minor profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                        'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Find best correlation
            best_corr = -1
            best_key = None
            
            for shift in range(12):
                major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, shift))[0, 1]
                minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, shift))[0, 1]
                
                if major_corr > best_corr:
                    best_corr = major_corr
                    best_key = f"{key_names[shift]} major"
                if minor_corr > best_corr:
                    best_corr = minor_corr
                    best_key = f"{key_names[shift]} minor"
            
            return best_key
        except:
            return None
    
    def classify_energy(self, y: np.ndarray) -> str:
        """Classify energy level of segment"""
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]
        centroid_mean = np.mean(spectral_centroid)
        
        # Onset rate (events per second)
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate)
        onset_rate = np.sum(onset_env > np.mean(onset_env)) / (len(y) / self.sample_rate)
        
        # Classification logic
        if rms_std > 0.1 and onset_rate > 3:
            return 'build'
        elif rms_mean > 0.15 and centroid_mean > 4000:
            return 'drop'
        elif rms_mean > 0.1:
            return 'high'
        elif rms_mean > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def process_file(
        self,
        file_path: str,
        save_audio: bool = True,
        save_spectrogram: bool = True
    ) -> List[AudioSegment]:
        """
        Process a single audio file into segments.
        """
        segments = []
        
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            duration = len(y) / sr
            
            # Skip very short files
            if duration < self.segment_duration:
                return segments
            
            # Calculate segment parameters
            segment_samples = int(self.segment_duration * sr)
            overlap_samples = int(self.segment_overlap * sr)
            step = segment_samples - overlap_samples
            
            # Process each segment
            for start_sample in range(0, len(y) - segment_samples + 1, step):
                start_time = start_sample / sr
                end_time = start_time + self.segment_duration
                
                # Extract segment
                segment_audio = y[start_sample:start_sample + segment_samples]
                
                # Generate ID
                segment_id = self.generate_segment_id(file_path, start_time)
                
                # Extract features
                bpm = self.estimate_bpm(segment_audio)
                key = self.estimate_key(segment_audio)
                energy_label = self.classify_energy(segment_audio)
                
                # Paths
                audio_path = None
                mel_path = None
                
                if save_audio:
                    audio_path = self.processed_dir / f"{segment_id}.wav"
                    sf.write(str(audio_path), segment_audio, sr)
                
                if save_spectrogram:
                    mel = self.extract_mel_spectrogram(segment_audio)
                    mel_path = self.cache_dir / f"{segment_id}_mel.npy"
                    np.save(str(mel_path), mel)
                
                # Create segment metadata
                segment = AudioSegment(
                    segment_id=segment_id,
                    source_file=file_path,
                    start_time=start_time,
                    end_time=end_time,
                    duration=self.segment_duration,
                    bpm=bpm,
                    key=key,
                    energy_label=energy_label,
                    mel_spectrogram_path=str(mel_path) if mel_path else None,
                    audio_path=str(audio_path) if audio_path else None
                )
                
                segments.append(segment)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return segments
    
    def process_folder(
        self,
        folder_path: str,
        extensions: List[str] = ['.mp3', '.wav', '.flac'],
        num_workers: int = 4,
        max_files: Optional[int] = None
    ) -> Dict:
        """
        Process all audio files in a folder using parallel processing.
        """
        folder = Path(folder_path)
        
        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(folder.glob(f"**/*{ext}"))
            audio_files.extend(folder.glob(f"**/*{ext.upper()}"))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        print(f"Found {len(audio_files)} files to process")
        
        all_segments = []
        failed_files = []
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_file, str(f)): f 
                for f in audio_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    segments = future.result()
                    all_segments.extend(segments)
                except Exception as e:
                    failed_files.append(str(futures[future]))
        
        # Save metadata
        manifest = {
            'total_segments': len(all_segments),
            'total_files': len(audio_files),
            'failed_files': len(failed_files),
            'segments': [s.to_dict() for s in all_segments]
        }
        
        manifest_path = self.metadata_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        # Create index files for fast loading
        self._create_indices(all_segments)
        
        # Print statistics
        self._print_statistics(all_segments)
        
        return manifest
    
    def process_all_sources(self, num_workers: int = 4, max_files: Optional[int] = None) -> Dict:
        """
        Process ALL data sources as defined in train_config.yaml:
        1. vinahouse_dir (10k original tracks)
        2. long_audio_cache_dir (segments from Smart DJ Set Processor)
        
        Merges everything into a single manifest.json for training.
        """
        all_segments = []
        
        # Source 1: Original Vinahouse tracks
        vinahouse_dir = self.config['data'].get('vinahouse_dir')
        if vinahouse_dir and Path(vinahouse_dir).exists():
            print(f"\n[Pipeline] Processing Source 1: {vinahouse_dir}")
            result = self.process_folder(vinahouse_dir, num_workers=num_workers, max_files=max_files)
            print(f"[Pipeline] Source 1: {result.get('total_segments', 0)} segments")
        
        # Source 2: Long Audio segments (from Smart DJ Set Processor)
        long_audio_dir = self.config['data'].get('long_audio_cache_dir')
        long_audio_manifest = None
        
        if long_audio_dir:
            # Check for the training manifest from smart_dj_set_processor.py
            manifest_candidates = [
                Path(long_audio_dir) / 'long_audio_manifest.json',
                Path('datasets/long_audio_segments') / 'long_audio_manifest.json',
            ]
            
            for candidate in manifest_candidates:
                if candidate.exists():
                    long_audio_manifest = candidate
                    break
        
        if long_audio_manifest:
            print(f"\n[Pipeline] Merging Source 2: Long Audio segments from {long_audio_manifest}")
            with open(long_audio_manifest, 'r') as f:
                la_data = json.load(f)
            
            la_segments = la_data.get('segments', [])
            
            # Load existing manifest and merge
            main_manifest_path = self.metadata_dir / 'manifest.json'
            if main_manifest_path.exists():
                with open(main_manifest_path, 'r') as f:
                    main_manifest = json.load(f)
            else:
                main_manifest = {'total_segments': 0, 'total_files': 0, 'segments': []}
            
            # Convert Long Audio chunks to our segment format
            for chunk in la_segments:
                spec_path = chunk.get('spectrogram_path')
                main_manifest['segments'].append({
                    'segment_id': chunk.get('chunk_id', ''),
                    'source_file': chunk.get('source_file', ''),
                    'start_time': chunk.get('start_time', 0),
                    'end_time': chunk.get('end_time', 0),
                    'duration': chunk.get('duration', 30),
                    'bpm': chunk.get('bpm', 128),
                    'key': chunk.get('key', None),
                    'energy_label': 'high' if chunk.get('energy', 0.5) > 0.6 else 'medium',
                    'mel_spectrogram_path': spec_path if spec_path and os.path.exists(spec_path) else None,
                    'audio_path': None,
                    'set_position': chunk.get('set_position', 0.0),
                    'set_phase': chunk.get('set_phase', 'main'),
                })
            
            main_manifest['total_segments'] = len(main_manifest['segments'])
            
            # Save merged manifest
            with open(main_manifest_path, 'w') as f:
                json.dump(main_manifest, f, indent=2)
            
            print(f"[Pipeline] Merged: +{len(la_segments)} Long Audio segments")
            print(f"[Pipeline] Total training segments: {main_manifest['total_segments']}")
            return main_manifest
        else:
            print(f"\n[Pipeline] No Long Audio manifest found. Using vinahouse data only.")
            return {}
    
    def _create_indices(self, segments: List[AudioSegment]):
        """Create index files for efficient data loading"""
        
        # By energy label
        energy_index = {label: [] for label in self.ENERGY_LABELS}
        for seg in segments:
            if seg.energy_label in energy_index:
                energy_index[seg.energy_label].append(seg.segment_id)
        
        with open(self.metadata_dir / 'energy_index.json', 'w') as f:
            json.dump(energy_index, f, indent=2)
        
        # By BPM range
        bpm_index = {}
        for seg in segments:
            if seg.bpm:
                bpm_range = int(seg.bpm // 10) * 10  # Round to nearest 10
                key = f"{bpm_range}-{bpm_range + 10}"
                if key not in bpm_index:
                    bpm_index[key] = []
                bpm_index[key].append(seg.segment_id)
        
        with open(self.metadata_dir / 'bpm_index.json', 'w') as f:
            json.dump(bpm_index, f, indent=2)
    
    def _print_statistics(self, segments: List[AudioSegment]):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Total segments created: {len(segments)}")
        
        # Energy distribution
        energy_counts = {}
        for seg in segments:
            energy_counts[seg.energy_label] = energy_counts.get(seg.energy_label, 0) + 1
        
        print("\nEnergy Distribution:")
        for label, count in sorted(energy_counts.items()):
            pct = count / len(segments) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # BPM distribution
        bpms = [seg.bpm for seg in segments if seg.bpm]
        if bpms:
            print(f"\nBPM Stats:")
            print(f"  Min: {min(bpms):.1f}")
            print(f"  Max: {max(bpms):.1f}")
            print(f"  Mean: {np.mean(bpms):.1f}")
        
        print(f"\nManifest saved to: {self.metadata_dir / 'manifest.json'}")
        print("="*50)


class StreamingDataset:
    """
    Streaming dataset for training.
    Loads segments on-demand to avoid memory overflow.
    """
    
    def __init__(
        self,
        manifest_path: str,
        cache_dir: str,
        sample_rate: int = 44100,
        segment_duration: float = 30.0
    ):
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.segments = self.manifest['segments']
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict:
        segment = self.segments[idx]
        
        # Load mel spectrogram
        mel_path = segment['mel_spectrogram_path']
        if mel_path and os.path.exists(mel_path):
            mel = np.load(mel_path)
        else:
            # Generate on-the-fly
            audio_path = segment['audio_path']
            y, _ = librosa.load(audio_path, sr=self.sample_rate)
            mel = librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, n_mels=80
            )
            mel = librosa.power_to_db(mel, ref=np.max)
        
        return {
            'mel': mel,
            'bpm': segment.get('bpm', 128),
            'energy_label': segment.get('energy_label', 'medium'),
            'segment_id': segment['segment_id'],
            'set_phase': segment.get('set_phase', 'peak'),
        }
    
    def get_batch(self, indices: List[int]) -> Dict:
        """Get a batch of segments"""
        batch = [self[i] for i in indices]
        
        # Pad to same length
        mels = [item['mel'] for item in batch]
        max_len = max(m.shape[1] for m in mels)
        
        padded_mels = []
        for mel in mels:
            if mel.shape[1] < max_len:
                pad = np.zeros((mel.shape[0], max_len - mel.shape[1]))
                mel = np.concatenate([mel, pad], axis=1)
            padded_mels.append(mel)
        
        return {
            'mel': np.stack(padded_mels),
            'bpm': np.array([item['bpm'] for item in batch]),
            'energy_label': [item['energy_label'] for item in batch],
            'segment_id': [item['segment_id'] for item in batch]
        }


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Preparation Pipeline')
    parser.add_argument('input', nargs='?', default=None, help='Input folder with audio files')
    parser.add_argument('--config', '-c', default='config/train_config.yaml', help='Config file')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of workers')
    parser.add_argument('--max-files', type=int, default=None, help='Max files to process')
    parser.add_argument('--all', action='store_true', help='Process ALL sources from config (vinahouse + long audio)')
    
    args = parser.parse_args()
    
    pipeline = DataPrepPipeline(config_path=args.config)
    
    if args.output:
        pipeline.processed_dir = Path(args.output)
        pipeline.processed_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        # Process all sources defined in train_config.yaml
        pipeline.process_all_sources(num_workers=args.workers, max_files=args.max_files)
    elif args.input:
        pipeline.process_folder(
            folder_path=args.input,
            num_workers=args.workers,
            max_files=args.max_files
        )
    else:
        print("Error: Specify --all or provide an input folder path")
        parser.print_help()


if __name__ == '__main__':
    main()