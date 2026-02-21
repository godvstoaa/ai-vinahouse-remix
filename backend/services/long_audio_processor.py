"""
Smart DJ Set Processor - Content-Aware Segmentation
Detect track boundaries thay vì cắt cứng theo thời gian
"""

import numpy as np
import librosa
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator, Callable
from dataclasses import dataclass, asdict
import threading
import time
import gc
from datetime import datetime

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except:
    HAS_SCIPY = False


@dataclass
class ChunkInfo:
    chunk_id: str
    source_file: str
    start_time: float
    end_time: float
    duration: float
    bpm: float
    key: str
    energy: float
    track_index: int = 0
    spectrogram_path: Optional[str] = None
    # Macro-Conditioning: position in the full DJ set (0.0 = start, 1.0 = end)
    set_position: float = 0.0
    # Macro-Conditioning: phase label derived from position in the DJ set
    set_phase: str = "main"  # warmup / buildup / peak / cooldown


@dataclass
class FileProgress:
    file_path: str
    file_name: str
    status: str
    progress: float
    message: str
    chunks_created: int = 0
    error: Optional[str] = None


@dataclass
class FolderProgress:
    total_files: int
    completed_files: int
    current_file: Optional[str]
    files: List[FileProgress]
    total_chunks: int
    total_duration_hours: float
    started_at: float
    elapsed_seconds: float = 0.0
    
    def to_dict(self): return asdict(self)


class LongAudioProcessor:
    """Smart DJ Set Processor với Content-Aware Track Detection"""
    
    CHUNK_DURATION = 30
    SAMPLE_RATE = 11025
    HOP_LENGTH = 512
    N_FFT = 1024
    N_MELS = 64
    MIN_TRACK_LENGTH = 90  # Minimum 90 seconds per track
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma']
    
    def __init__(self, cache_dir: str = "backend/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.spectrogram_dir = self.cache_dir / "spectrograms"
        self.metadata_dir = self.cache_dir / "metadata"
        self.dataset_dir = self.cache_dir / "datasets"
        
        for d in [self.spectrogram_dir, self.metadata_dir, self.dataset_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self._progress_callback = None
        self._cancel_flag = False
        self._folder_progress = None
        self._lock = threading.Lock()
    
    def set_progress_callback(self, cb): self._progress_callback = cb
    
    def _report_progress(self, stage, current, total, msg):
        if self._progress_callback:
            self._progress_callback({
                "stage": stage, "current": current, "total": total,
                "message": msg, "percentage": (current/total*100) if total > 0 else 0
            })
    
    def cancel(self): self._cancel_flag = True
    
    def get_file_hash(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
        return h.hexdigest()
    
    def is_cached(self, path: str) -> bool:
        try: return (self.metadata_dir / f"{self.get_file_hash(path)}.json").exists()
        except: return False
    
    def load_cached_metadata(self, path: str) -> Optional[Dict]:
        try:
            p = self.metadata_dir / f"{self.get_file_hash(path)}.json"
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f: return json.load(f)
        except: pass
        return None
    
    def scan_folder(self, folder_path: str, recursive: bool = True) -> List[Dict]:
        """Scan folder for audio files with accurate duration"""
        folder = Path(folder_path)
        if not folder.exists(): raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        files = []
        patterns = ['**/*'] if recursive else ['*']
        print(f"[LongAudio] Scanning: {folder_path}")
        
        for pattern in patterns:
            for ext in self.SUPPORTED_FORMATS:
                for fp in folder.glob(f"{pattern}{ext}"):
                    if fp.is_file():
                        try:
                            stat = fp.stat()
                            size_mb = stat.st_size / (1024*1024)
                            try:
                                dur = librosa.get_duration(path=str(fp))
                                dur_min = dur / 60
                            except:
                                dur_min = size_mb / 1.5
                            
                            print(f"[LongAudio] Found: {fp.name} ({size_mb:.0f}MB, {dur_min:.0f}min)")
                            files.append({
                                "path": str(fp), "name": fp.name,
                                "size_mb": round(size_mb, 2),
                                "estimated_duration_min": round(dur_min, 1),
                                "cached": self.is_cached(str(fp))
                            })
                        except Exception as e: print(f"[LongAudio] Error: {e}")
        
        files.sort(key=lambda x: x['name'])
        print(f"[LongAudio] Total: {len(files)} files")
        return files
    
    def _detect_key(self, y: np.ndarray, sr: int) -> str:
        """Detect musical key using Krumhansl-Schmuckler algorithm"""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.HOP_LENGTH)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Camelot wheel keys
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_idx = np.argmax(chroma_mean)
            return keys[key_idx]
        except:
            return "Unknown"
    
    def _classify_segment(self, rms_segment: np.ndarray) -> str:
        """Classify segment type based on energy profile"""
        if len(rms_segment) < 3:
            return "unknown"
        
        # Normalize
        rms_norm = (rms_segment - np.min(rms_segment)) / (np.max(rms_segment) - np.min(rms_segment) + 1e-8)
        
        # Analyze shape
        start_energy = np.mean(rms_norm[:len(rms_norm)//4])
        mid_energy = np.mean(rms_norm[len(rms_norm)//4:3*len(rms_norm)//4])
        end_energy = np.mean(rms_norm[3*len(rms_norm)//4:])
        
        if mid_energy > 0.7 and start_energy < mid_energy:
            return "buildup"
        elif mid_energy > 0.7 and end_energy < mid_energy:
            return "breakdown"
        elif np.mean(rms_norm) > 0.6:
            return "drop"
        elif np.mean(rms_norm) < 0.3:
            return "intro"
        else:
            return "main"
    
    def _detect_track_boundaries(self, y: np.ndarray, duration: float) -> List[Dict]:
        """
        SMART TRACK DETECTION with Key Analysis
        Detect chỗ chuyển bài trong DJ set + phân tích key
        """
        print(f"[SmartDetect] Analyzing {duration/60:.1f}min set for track boundaries...")
        
        hop = self.HOP_LENGTH
        sr = self.SAMPLE_RATE
        frame_time = hop / sr
        
        # 1. Energy curve
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        
        # 2. Spectral flux (timbre changes)
        spec = np.abs(librosa.stft(y, hop_length=hop, n_fft=2048))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        flux = np.pad(flux, (1, 0))
        flux_norm = (flux - flux.min()) / (flux.max() - flux.min() + 1e-8)
        
        # 3. Key change detection (harmonic analysis)
        print("[SmartDetect] Analyzing harmonic content...")
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
            # Compute key change = chroma difference
            key_change = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)
            key_change = np.pad(key_change, (1, 0))
            key_change_norm = (key_change - key_change.min()) / (key_change.max() - key_change.min() + 1e-8)
        except:
            key_change_norm = np.zeros(len(rms))
        
        # 4. Novelty = energy change + timbre change + key change
        energy_change = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
        novelty = 0.35 * flux_norm + 0.25 * energy_change + 0.25 * rms_norm + 0.15 * key_change_norm
        
        # Smooth
        if HAS_SCIPY:
            novelty = gaussian_filter1d(novelty, sigma=80)
        else:
            k = 80
            novelty = np.convolve(novelty, np.ones(k)/k, mode='same')
        
        # 5. Find peaks = boundaries
        threshold = np.percentile(novelty, 65)
        min_frames = int(self.MIN_TRACK_LENGTH / frame_time)
        
        peaks = []
        for i in range(1, len(novelty)-1):
            if novelty[i] > novelty[i-1] and novelty[i] > novelty[i+1] and novelty[i] > threshold:
                peaks.append(i)
        
        # Filter minimum distance
        selected = [0]
        for p in peaks:
            if p - selected[-1] >= min_frames:
                selected.append(p)
        
        # Convert to time with metadata
        boundaries = []
        for i in range(len(selected)):
            start = selected[i] * frame_time
            end = (selected[i+1] * frame_time) if i+1 < len(selected) else duration
            end = min(end, duration)
            
            # Analyze this segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = y[start_sample:end_sample]
            
            # Get key
            key = self._detect_key(segment, sr)
            
            # Get segment type
            start_frame = int(start / frame_time)
            end_frame = int(end / frame_time)
            segment_rms = rms_norm[start_frame:end_frame]
            segment_type = self._classify_segment(segment_rms)
            
            # Confidence based on novelty
            confidence = float(novelty[selected[i]]) if selected[i] < len(novelty) else 0.5
            
            boundaries.append({
                "start": start,
                "end": end,
                "duration": end - start,
                "key": key,
                "segment_type": segment_type,
                "confidence": confidence,
                "frame_idx": selected[i]
            })
        
        print(f"[SmartDetect] Found {len(boundaries)} tracks")
        for i, b in enumerate(boundaries):
            print(f"  Track {i+1}: {b['start']/60:.1f}-{b['end']/60:.1f}min | Key: {b['key']} | Type: {b['segment_type']}")
        
        return boundaries
    
    def process_file(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        """Process single DJ set file with smart segmentation"""
        self._cancel_flag = False
        start_time = time.time()
        file_hash = self.get_file_hash(file_path)
        
        # Check cache
        if not force:
            cached = self.load_cached_metadata(file_path)
            if cached:
                self._report_progress("cache", 1, 1, "Loaded from cache")
                return cached
        
        self._report_progress("loading", 0, 100, "Loading audio...")
        duration = librosa.get_duration(path=file_path)
        print(f"[LongAudio] Processing: {Path(file_path).name} ({duration/60:.1f}min)")
        
        if self._cancel_flag: raise InterruptedError("Cancelled")
        
        # Load audio
        self._report_progress("loading", 30, 100, "Decoding...")
        y, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, mono=True)
        
        if self._cancel_flag: raise InterruptedError("Cancelled")
        
        # SMART: Detect track boundaries
        self._report_progress("detect", 0, 100, "Detecting tracks...")
        boundaries = self._detect_track_boundaries(y, duration)
        
        if self._cancel_flag: raise InterruptedError("Cancelled")
        
        # Process each detected track
        all_chunks = []
        self._report_progress("chunks", 0, len(boundaries), "Creating chunks...")
        
        for track_idx, track_info in enumerate(boundaries):
            if self._cancel_flag: raise InterruptedError("Cancelled")
            
            t_start = track_info['start']
            t_end = track_info['end']
            track_key = track_info.get('key', 'Unknown')
            track_type = track_info.get('segment_type', 'main')
            
            self._report_progress("chunks", track_idx, len(boundaries), 
                                 f"Track {track_idx+1}/{len(boundaries)} ({t_start/60:.1f}-{t_end/60:.1f}min) [{track_key}]")
            
            # Analyze BPM for this track
            s_start = int(t_start * sr)
            s_end = int(t_end * sr)
            segment = y[s_start:s_end]
            
            try:
                tempo, _ = librosa.beat.beat_track(y=segment, sr=sr, hop_length=self.HOP_LENGTH)
                bpm = float(tempo) if tempo > 0 else 128.0
            except:
                bpm = 128.0
            
            # Create 30s chunks within this track
            chunk_start = t_start
            chunk_idx = 0
            
            while chunk_start < t_end - 5:  # At least 5s remaining
                chunk_end = min(chunk_start + self.CHUNK_DURATION, t_end)
                
                cs = int(chunk_start * sr)
                ce = int(chunk_end * sr)
                chunk_audio = y[cs:ce]
                
                if len(chunk_audio) < sr * 5: break
                
                chunk_id = f"{file_hash[:8]}_t{track_idx:02d}_c{chunk_idx:03d}"
                
                # Energy
                rms_chunk = np.sqrt(np.mean(chunk_audio**2))
                energy = min(1.0, rms_chunk * 10)
                
                chunk_info = ChunkInfo(
                    chunk_id=chunk_id,
                    source_file=Path(file_path).name,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    duration=chunk_end - chunk_start,
                    bpm=bpm,
                    key=track_key,
                    energy=energy,
                    track_index=track_idx,
                    # Macro-Conditioning: where in the 4-hour set is this chunk?
                    set_position=round(chunk_start / max(duration, 1.0), 4),
                    set_phase=(
                        "warmup" if chunk_start / max(duration, 1.0) < 0.15 else
                        "buildup" if chunk_start / max(duration, 1.0) < 0.35 else
                        "peak" if chunk_start / max(duration, 1.0) < 0.75 else
                        "cooldown"
                    )
                )
                
                # Save spectrogram
                spec_path = self.spectrogram_dir / f"{chunk_id}.npy"
                try:
                    mel = librosa.feature.melspectrogram(y=chunk_audio, sr=sr,
                                                         n_fft=self.N_FFT, 
                                                         hop_length=self.HOP_LENGTH,
                                                         n_mels=self.N_MELS)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    np.save(spec_path, mel_db.astype(np.float32))
                    chunk_info.spectrogram_path = str(spec_path)
                except: pass
                
                all_chunks.append(chunk_info)
                chunk_start = chunk_end
                chunk_idx += 1
        
        print(f"[LongAudio] Created {len(all_chunks)} chunks from {len(boundaries)} tracks")
        
        # Metadata with full track info
        metadata = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_hash": file_hash,
            "duration": duration,
            "sample_rate": sr,
            "num_tracks": len(boundaries),
            "num_chunks": len(all_chunks),
            "tracks": boundaries,  # Full track info with key, type, confidence
            "chunks": [asdict(c) for c in all_chunks],
            "processing_time": time.time() - start_time,
            "processed_at": datetime.now().isoformat()
        }
        
        # Save
        meta_path = self.metadata_dir / f"{file_hash}.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Cleanup
        del y
        gc.collect()
        
        self._report_progress("complete", 1, 1, "Complete!")
        return metadata
    
    def process_folder(self, folder_path: str, recursive: bool = True, 
                       force: bool = False, max_workers: int = 1,
                       skip_cached: bool = True) -> Dict[str, Any]:
        """Process all files in folder"""
        self._cancel_flag = False
        start = time.time()
        
        files = self.scan_folder(folder_path, recursive)
        if not files:
            return {"success": False, "error": "No files", "folder_path": folder_path}
        
        self._folder_progress = FolderProgress(
            total_files=len(files), completed_files=0, current_file=None,
            files=[FileProgress(file_path=f['path'], file_name=f['name'],
                                status='cached' if f['cached'] and skip_cached else 'pending',
                                progress=0, message='') for f in files],
            total_chunks=0, total_duration_hours=0, started_at=start
        )
        
        all_meta = []
        
        for f in files:
            if self._cancel_flag: break
            
            if f['cached'] and skip_cached and not force:
                cached = self.load_cached_metadata(f['path'])
                if cached:
                    all_meta.append(cached)
                    self._folder_progress.completed_files += 1
                    continue
            
            self._folder_progress.current_file = f['path']
            self._update_file_progress(f['path'], 'processing', 0, 'Starting...')
            
            try:
                meta = self.process_file(f['path'], force)
                all_meta.append(meta)
                self._update_file_progress(f['path'], 'completed', 100, 
                                          f"Done - {meta['num_chunks']} chunks", 
                                          chunks=meta['num_chunks'])
                self._folder_progress.total_chunks += meta['num_chunks']
                self._folder_progress.total_duration_hours += meta['duration'] / 3600
            except Exception as e:
                self._update_file_progress(f['path'], 'error', 0, str(e)[:50], error=str(e)[:100])
            
            self._folder_progress.completed_files += 1
        
        self._folder_progress.elapsed_seconds = time.time() - start
        
        # Merge
        dataset = self._merge_metadata(all_meta, folder_path)
        
        ds_path = self.dataset_dir / f"dataset_{int(time.time())}.json"
        with open(ds_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        dataset['dataset_path'] = str(ds_path)
        
        print(f"[LongAudio] Dataset: {dataset['total_chunks']} chunks, {dataset['total_files']} files")
        return dataset
    
    def _update_file_progress(self, path, status, prog, msg, chunks=0, error=None):
        with self._lock:
            if self._folder_progress:
                for fp in self._folder_progress.files:
                    if fp.file_path == path:
                        fp.status, fp.progress, fp.message = status, prog, msg
                        if chunks: fp.chunks_created = chunks
                        if error: fp.error = error
                        break
    
    def get_folder_progress(self) -> Optional[Dict]:
        if self._folder_progress:
            self._folder_progress.elapsed_seconds = time.time() - self._folder_progress.started_at
            return self._folder_progress.to_dict()
        return None
    
    def _merge_metadata(self, all_meta: List[Dict], folder: str) -> Dict:
        chunks = []
        dur = 0
        for m in all_meta:
            chunks.extend(m.get("chunks", []))
            dur += m.get("duration", 0)
        
        return {
            "success": True,
            "folder_path": folder,
            "total_files": len(all_meta),
            "total_chunks": len(chunks),
            "total_duration_hours": dur / 3600,
            "chunks": chunks[:100],
            "total_chunks_full": len(chunks),
            "created_at": datetime.now().isoformat(),
            "cache_size_mb": self._get_cache_size()
        }
    
    def _get_cache_size(self) -> float:
        total = 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
                try: total += p.stat().st_size
                except: pass
        return round(total / (1024*1024), 2)
    
    def get_chunk_generator(self, meta: Dict, batch: int = 8, shuffle: bool = True) -> Generator:
        chunks = meta.get("chunks", [])
        if shuffle:
            import random
            random.shuffle(chunks)
        
        specs, metas = [], []
        for c in chunks:
            sp = c.get("spectrogram_path")
            if sp and os.path.exists(sp):
                try:
                    specs.append(np.load(sp))
                    metas.append(c)
                    if len(specs) >= batch:
                        yield np.stack(specs), metas
                        specs, metas = [], []
                except: pass
        if specs: yield np.stack(specs), metas
    
    def get_dataset_stats(self, meta: Dict) -> Dict:
        chunks = meta.get("chunks", [])
        return {
            "total_chunks": len(chunks),
            "total_duration_hours": meta.get("duration", 0) / 3600,
            "cache_size_mb": self._get_cache_size()
        }
    
    def clear_cache(self, path: Optional[str] = None):
        import shutil
        if path:
            h = self.get_file_hash(path)
            p = self.metadata_dir / f"{h}.json"
            if p.exists(): p.unlink()
        else:
            for d in [self.spectrogram_dir, self.metadata_dir, self.dataset_dir]:
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
    
    def get_cached_datasets(self) -> List[Dict]:
        datasets = []
        for p in sorted(self.dataset_dir.glob("dataset_*.json"), reverse=True):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                datasets.append({
                    "path": str(p), "name": p.name,
                    "total_files": data.get("total_files", 0),
                    "total_chunks": data.get("total_chunks_full", 0),
                    "created_at": data.get("created_at", "")
                })
            except: pass
        return datasets


_processor = None

def get_long_audio_processor() -> LongAudioProcessor:
    global _processor
    if _processor is None:
        _processor = LongAudioProcessor()
    return _processor