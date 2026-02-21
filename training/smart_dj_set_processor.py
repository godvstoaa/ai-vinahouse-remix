"""
Smart DJ Set Processor - Training-side wrapper
Tái sử dụng engine từ backend/services/long_audio_processor.py
Tích hợp vào training pipeline để:
1. Băm 2000 bài Nonstop 4 tiếng -> hàng chục nghìn bài Remix lẻ
2. Dán nhãn Macro-Conditioning (set_position + set_phase) lên từng bài

Usage:
    python training/smart_dj_set_processor.py --input "D:/Music/Nonstop/" --output "datasets/long_audio_segments/"
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.long_audio_processor import LongAudioProcessor, get_long_audio_processor


class TrainingDJSetProcessor:
    """
    Wrapper around LongAudioProcessor for training pipeline integration.
    
    Reads from nonstop_dir, runs Content-Aware Track Detection,
    outputs segmented tracks with Macro-Conditioning labels
    into a format that data_prep_pipeline.py can consume.
    """
    
    def __init__(self, cache_dir: str = "datasets/long_audio_segments"):
        self.processor = LongAudioProcessor(cache_dir=cache_dir)
        self.output_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_nonstop_folder(
        self,
        nonstop_dir: str,
        force: bool = False,
        skip_cached: bool = True
    ) -> Dict:
        """
        Process all Nonstop tracks in folder.
        
        Returns dict with:
        - total_files: Number of nonstop files processed
        - total_chunks: Number of short segments created
        - total_duration_hours: Total audio hours processed
        - segments_manifest_path: Path to manifest JSON for data_prep_pipeline
        """
        print(f"\n{'='*60}")
        print(f"  SMART DJ SET PROCESSOR - Training Data Extraction")
        print(f"  Input: {nonstop_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        result = self.processor.process_folder(
            folder_path=nonstop_dir,
            recursive=True,
            force=force,
            skip_cached=skip_cached
        )
        
        if not result.get('success', False):
            print(f"[ERROR] Processing failed: {result.get('error', 'Unknown')}")
            return result
        
        # Create a manifest for data_prep_pipeline.py consumption
        manifest = self._create_training_manifest(result)
        
        print(f"\n{'='*60}")
        print(f"  RESULTS:")
        print(f"  Files processed: {result.get('total_files', 0)}")
        print(f"  Segments created: {result.get('total_chunks', 0)}")
        print(f"  Total hours: {result.get('total_duration_hours', 0):.1f}h")
        print(f"  Manifest: {manifest['manifest_path']}")
        print(f"{'='*60}\n")
        
        return manifest
    
    def _create_training_manifest(self, result: Dict) -> Dict:
        """
        Create a manifest file that data_prep_pipeline.py can read
        to include Long Audio segments in training.
        """
        chunks = result.get('chunks', [])
        
        # Enhance chunks with set_phase info for training
        training_segments = []
        for chunk in chunks:
            training_segments.append({
                'chunk_id': chunk.get('chunk_id', ''),
                'source_file': chunk.get('source_file', ''),
                'start_time': chunk.get('start_time', 0),
                'end_time': chunk.get('end_time', 0),
                'duration': chunk.get('duration', 0),
                'bpm': chunk.get('bpm', 128.0),
                'key': chunk.get('key', 'Unknown'),
                'energy': chunk.get('energy', 0.5),
                'set_position': chunk.get('set_position', 0.0),
                'set_phase': chunk.get('set_phase', 'main'),
                'spectrogram_path': chunk.get('spectrogram_path', None),
            })
        
        manifest = {
            'source': 'smart_dj_set_processor',
            'total_segments': len(training_segments),
            'total_files': result.get('total_files', 0),
            'segments': training_segments
        }
        
        manifest_path = self.output_dir / 'long_audio_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        manifest['manifest_path'] = str(manifest_path)
        return manifest
    
    def get_segment_count(self) -> int:
        """Get number of cached segments"""
        manifest_path = self.output_dir / 'long_audio_manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            return data.get('total_segments', 0)
        return 0


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart DJ Set Processor for Training')
    parser.add_argument('--input', '-i', required=True, help='Nonstop tracks folder')
    parser.add_argument('--output', '-o', default='datasets/long_audio_segments/', help='Output directory')
    parser.add_argument('--force', action='store_true', help='Re-process cached files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input folder not found: {args.input}")
        sys.exit(1)
    
    processor = TrainingDJSetProcessor(cache_dir=args.output)
    processor.process_nonstop_folder(args.input, force=args.force)


if __name__ == '__main__':
    main()
