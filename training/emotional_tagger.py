"""
Emotional Tagger - Phân tích năng lượng và cảm xúc từ Nonstop tracks
Trích xuất Build/Drop/Breakdown events để dạy AI cách lên xuống nhịp tim
"""

import numpy as np
import librosa
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


@dataclass
class EnergyEvent:
    """Represents an energy event in the track"""
    timestamp: float  # seconds
    duration: float   # seconds
    event_type: str   # 'build', 'drop', 'breakdown', 'intro', 'outro'
    intensity: float  # 0.0 - 1.0
    rms_energy: float
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class TrackMetadata:
    """Metadata for a processed track"""
    file_path: str
    duration: float
    sample_rate: int
    bpm: Optional[float]
    key: Optional[str]
    average_energy: float
    peak_energy: float
    energy_events: List[Dict]
    energy_curve: List[float]  # Downsampled energy over time
    
    def to_dict(self):
        return {
            'file_path': self.file_path,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'bpm': self.bpm,
            'key': self.key,
            'average_energy': self.average_energy,
            'peak_energy': self.peak_energy,
            'energy_events': self.energy_events,
            'energy_curve': self.energy_curve
        }


class EmotionalTagger:
    """
    Analyze emotional energy flow in music tracks.
    Optimized for long Nonstop tracks (1-4 hours) using chunked processing.
    """
    
    # Event type labels
    LABELS = ['intro', 'build', 'drop', 'breakdown', 'outro', 'verse', 'chorus']
    
    def __init__(
        self,
        sample_rate: int = 44100,
        rms_window: int = 2048,
        spectral_flux_window: int = 2048,
        build_threshold: float = 0.7,
        drop_threshold: float = 0.9,
        breakdown_threshold: float = 0.3,
        energy_curve_fps: int = 10  # Energy samples per second
    ):
        self.sample_rate = sample_rate
        self.rms_window = rms_window
        self.spectral_flux_window = spectral_flux_window
        self.build_threshold = build_threshold
        self.drop_threshold = drop_threshold
        self.breakdown_threshold = breakdown_threshold
        self.energy_curve_fps = energy_curve_fps
        
    def load_audio_chunked(
        self, 
        file_path: str, 
        chunk_duration: float = 60.0
    ) -> Tuple[np.ndarray, float]:
        """
        Load audio file in chunks to avoid memory overflow.
        Yields (chunk_audio, chunk_start_time)
        """
        try:
            # Get total duration first
            duration = librosa.get_duration(path=file_path)
            
            # Load full file if short enough
            if duration < chunk_duration * 2:
                y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                return y, duration
            
            # For long files, return None and use streaming
            return None, duration
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, 0
    
    def compute_rms_energy(self, y: np.ndarray) -> np.ndarray:
        """Compute RMS energy over time"""
        hop_length = self.rms_window // 4
        rms = librosa.feature.rms(
            y=y, 
            frame_length=self.rms_window,
            hop_length=hop_length
        )[0]
        return rms
    
    def compute_spectral_flux(self, y: np.ndarray) -> np.ndarray:
        """Compute spectral flux (onset strength) for build-up detection"""
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=self.sample_rate,
            n_fft=self.spectral_flux_window
        )
        return onset_env
    
    def detect_energy_events(
        self,
        rms_energy: np.ndarray,
        spectral_flux: np.ndarray,
        duration: float
    ) -> List[EnergyEvent]:
        """
        Detect energy events: builds, drops, breakdowns
        """
        events = []
        
        # Normalize energy
        rms_norm = (rms_energy - rms_energy.min()) / (rms_energy.max() - rms_energy.min() + 1e-8)
        flux_norm = (spectral_flux - spectral_flux.min()) / (spectral_flux.max() - spectral_flux.min() + 1e-8)
        
        # Resample to same length if needed
        if len(flux_norm) != len(rms_norm):
            flux_norm = np.interp(
                np.linspace(0, 1, len(rms_norm)),
                np.linspace(0, 1, len(flux_norm)),
                flux_norm
            )
        
        # Time per frame
        hop_length = self.rms_window // 4
        time_per_frame = hop_length / self.sample_rate
        
        # Detect events using sliding window
        window_size = int(5.0 / time_per_frame)  # 5 second window
        
        for i in range(window_size, len(rms_norm) - window_size):
            # Local statistics
            local_rms = rms_norm[i-window_size:i+window_size]
            current_rms = rms_norm[i]
            
            # Before/after energy comparison
            before_avg = np.mean(rms_norm[max(0, i-window_size*2):i])
            after_avg = np.mean(rms_norm[i:min(len(rms_norm), i+window_size*2)])
            
            # High flux = building up
            local_flux = np.mean(flux_norm[max(0, i-window_size//2):i+window_size//2])
            
            timestamp = i * time_per_frame
            intensity = current_rms
            
            # DROP detection: sudden high energy after build-up
            if (after_avg > before_avg * 1.5 and 
                current_rms > self.drop_threshold and
                local_flux > 0.5):
                events.append(EnergyEvent(
                    timestamp=timestamp,
                    duration=3.0,
                    event_type='drop',
                    intensity=intensity,
                    rms_energy=current_rms
                ))
            
            # BUILD detection: rising energy with high flux
            elif (local_flux > 0.6 and 
                  after_avg > before_avg * 1.2 and
                  current_rms > self.build_threshold):
                events.append(EnergyEvent(
                    timestamp=timestamp,
                    duration=8.0,  # Builds typically 8 seconds
                    event_type='build',
                    intensity=intensity,
                    rms_energy=current_rms
                ))
            
            # BREAKDOWN detection: sudden low energy
            elif (after_avg < before_avg * 0.7 and
                  current_rms < self.breakdown_threshold):
                events.append(EnergyEvent(
                    timestamp=timestamp,
                    duration=16.0,  # Breakdowns typically 16 seconds
                    event_type='breakdown',
                    intensity=intensity,
                    rms_energy=current_rms
                ))
        
        # Merge overlapping events of same type
        events = self._merge_events(events)
        
        # Add intro/outro
        if duration > 30:
            events.insert(0, EnergyEvent(
                timestamp=0.0,
                duration=30.0,
                event_type='intro',
                intensity=0.3,
                rms_energy=rms_norm[0]
            ))
            events.append(EnergyEvent(
                timestamp=duration - 30.0,
                duration=30.0,
                event_type='outro',
                intensity=0.3,
                rms_energy=rms_norm[-1]
            ))
        
        return events
    
    def _merge_events(self, events: List[EnergyEvent]) -> List[EnergyEvent]:
        """Merge overlapping events of the same type"""
        if not events:
            return events
            
        # Sort by timestamp
        events = sorted(events, key=lambda e: e.timestamp)
        merged = [events[0]]
        
        for event in events[1:]:
            last = merged[-1]
            # Merge if same type and overlapping
            if (event.event_type == last.event_type and
                event.timestamp < last.timestamp + last.duration):
                # Extend duration
                merged[-1] = EnergyEvent(
                    timestamp=last.timestamp,
                    duration=max(last.duration, event.timestamp + event.duration - last.timestamp),
                    event_type=last.event_type,
                    intensity=max(last.intensity, event.intensity),
                    rms_energy=max(last.rms_energy, event.rms_energy)
                )
            else:
                merged.append(event)
        
        return merged
    
    def generate_energy_curve(
        self, 
        rms_energy: np.ndarray,
        target_duration: float
    ) -> List[float]:
        """Generate downsampled energy curve for the entire track"""
        target_samples = int(target_duration * self.energy_curve_fps)
        
        # Resample RMS to target rate
        if len(rms_energy) != target_samples:
            curve = np.interp(
                np.linspace(0, 1, target_samples),
                np.linspace(0, 1, len(rms_energy)),
                rms_energy
            )
        else:
            curve = rms_energy
            
        # Normalize to 0-1
        curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)
        
        return curve.tolist()
    
    def estimate_bpm(self, y: np.ndarray) -> Optional[float]:
        """Estimate tempo using librosa"""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sample_rate)
            return float(tempo)
        except:
            return None
    
    def analyze_track(self, file_path: str) -> Optional[TrackMetadata]:
        """
        Analyze a single track and extract energy metadata.
        Handles long tracks with streaming.
        """
        print(f"Analyzing: {file_path}")
        
        try:
            # Load audio
            y, duration = self.load_audio_chunked(file_path)
            
            if y is None:
                # Long file - use streaming approach
                return self._analyze_long_track(file_path, duration)
            
            # Short file - process directly
            rms_energy = self.compute_rms_energy(y)
            spectral_flux = self.compute_spectral_flux(y)
            
            events = self.detect_energy_events(rms_energy, spectral_flux, duration)
            energy_curve = self.generate_energy_curve(rms_energy, duration)
            
            bpm = self.estimate_bpm(y)
            
            return TrackMetadata(
                file_path=file_path,
                duration=duration,
                sample_rate=self.sample_rate,
                bpm=bpm,
                key=None,  # Key detection is expensive
                average_energy=float(np.mean(rms_energy)),
                peak_energy=float(np.max(rms_energy)),
                energy_events=[e.to_dict() for e in events],
                energy_curve=energy_curve
            )
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _analyze_long_track(
        self, 
        file_path: str, 
        duration: float
    ) -> Optional[TrackMetadata]:
        """
        Analyze a long track (1-4 hours) using CHUNKED STREAMING.
        Processes 60-second chunks to avoid memory overflow.
        Never loads more than ~10MB of audio into RAM at once.
        """
        chunk_duration = 60  # 60 second chunks - safe for any RAM size
        all_rms = []
        all_flux = []
        bpm = None
        
        # Cap analysis at 4 hours max
        analysis_duration = min(duration, 14400)
        num_chunks = int(np.ceil(analysis_duration / chunk_duration))
        
        print(f"  Streaming {num_chunks} chunks ({analysis_duration/3600:.1f}h)...")
        
        for i in range(num_chunks):
            offset = i * chunk_duration
            try:
                y_chunk, sr = librosa.load(
                    file_path,
                    sr=self.sample_rate,
                    mono=True,
                    offset=offset,
                    duration=chunk_duration
                )
            except Exception:
                break
            
            if len(y_chunk) == 0:
                break
            
            # Compute features for this chunk
            all_rms.append(self.compute_rms_energy(y_chunk))
            all_flux.append(self.compute_spectral_flux(y_chunk))
            
            # Estimate BPM from first chunk only
            if bpm is None:
                bpm = self.estimate_bpm(y_chunk)
            
            # Free memory immediately
            del y_chunk
        
        if not all_rms:
            return None
        
        # Concatenate all chunk results
        rms_energy = np.concatenate(all_rms)
        spectral_flux = np.concatenate(all_flux)
        
        # Free chunk arrays
        del all_rms, all_flux
        
        events = self.detect_energy_events(rms_energy, spectral_flux, analysis_duration)
        energy_curve = self.generate_energy_curve(rms_energy, analysis_duration)
        
        return TrackMetadata(
            file_path=file_path,
            duration=duration,
            sample_rate=self.sample_rate,
            bpm=bpm,
            key=None,
            average_energy=float(np.mean(rms_energy)),
            peak_energy=float(np.max(rms_energy)),
            energy_events=[e.to_dict() for e in events],
            energy_curve=energy_curve
        )
    
    def process_folder(
        self,
        folder_path: str,
        output_dir: str,
        extensions: List[str] = ['.mp3', '.wav', '.flac'],
        num_workers: int = 4
    ) -> List[str]:
        """
        Process all audio files in a folder.
        Saves metadata JSON files to output directory.
        """
        folder = Path(folder_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(folder.glob(f"**/*{ext}"))
            audio_files.extend(folder.glob(f"**/*{ext.upper()}"))
        
        print(f"Found {len(audio_files)} audio files in {folder_path}")
        
        # Process files
        processed = []
        for file_path in tqdm(audio_files, desc="Processing tracks"):
            metadata = self.analyze_track(str(file_path))
            
            if metadata:
                # Save to JSON
                output_file = output / f"{file_path.stem}_metadata.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
                processed.append(str(output_file))
        
        # Generate summary
        summary = {
            'total_files': len(audio_files),
            'processed': len(processed),
            'failed': len(audio_files) - len(processed),
            'output_dir': str(output)
        }
        
        with open(output / '_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nProcessing complete: {len(processed)}/{len(audio_files)} files")
        return processed


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotional Tagger for Nonstop tracks')
    parser.add_argument('input', help='Input folder with audio files')
    parser.add_argument('--output', '-o', default='datasets/metadata/', help='Output directory')
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--build-threshold', type=float, default=0.7)
    parser.add_argument('--drop-threshold', type=float, default=0.9)
    
    args = parser.parse_args()
    
    tagger = EmotionalTagger(
        sample_rate=args.sample_rate,
        build_threshold=args.build_threshold,
        drop_threshold=args.drop_threshold
    )
    
    tagger.process_folder(args.input, args.output)


if __name__ == '__main__':
    main()