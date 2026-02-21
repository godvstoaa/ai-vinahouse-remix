"""
AI Remix Backend - Main FastAPI Application
Handles audio upload, analysis, source separation, and remix generation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
import asyncio
import json
from datetime import datetime

# Import ML modules - Professional Version
from services.source_separator import SourceSeparator
from services.genre_detector import GenreDetector
from services.audio_analyzer import ProfessionalAudioAnalyzer, audio_analyzer
from services.remix_generator import RemixGenerator, remix_generator
from services.demucs_separator import DemucsSeparator, demucs_separator
from services.pedalboard_effects import HAS_PEDALBOARD, professional_effects
from services.model_trainer import ModelTrainer, TrainingConfig, get_trainer
from watermark import embed_watermark, detect_watermark, analyze_spectrum
from services.long_audio_processor import get_long_audio_processor, LongAudioProcessor

# Training router
from routers.training import router as training_router

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Remix Studio API",
    description="Backend for AI-powered music remix application",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training_router)

# Directories
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
STEMS_DIR = "stems"
REMIX_DIR = "remixes"

for dir_path in [UPLOAD_DIR, PROCESSED_DIR, STEMS_DIR, REMIX_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Initialize ML models (lazy loading)
source_separator = None
genre_detector = None
audio_analyzer = None
remix_generator = None

# Job tracking
jobs: Dict[str, Dict[str, Any]] = {}


class RemixSettings(BaseModel):
    genre: str = "edm"
    style: str = "club"
    bpm: int = 128
    original_bpm: int = 120
    energy_level: float = 0.7
    reverb: float = 0.3
    delay: float = 0.2
    filter: float = 0.0


def get_models():
    """Lazy load ML models"""
    global source_separator, genre_detector, audio_analyzer, remix_generator
    
    if source_separator is None:
        print("Loading ML models...")
        source_separator = SourceSeparator()
        genre_detector = GenreDetector()
        audio_analyzer = AudioAnalyzer()
        remix_generator = RemixGenerator()
        print("ML models loaded!")
    
    return source_separator, genre_detector, audio_analyzer, remix_generator


@app.get("/")
async def root():
    return {"message": "AI Remix Studio API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Enhanced health check with system info"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "pedalboard": HAS_PEDALBOARD,
            "cuda": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else 0
        },
        "jobs_count": len(jobs)
    }


@app.get("/api/system/info")
async def get_system_info():
    """Get detailed system information"""
    return {
        "audio_engine": {
            "separator": "Demucs v4 (htdemucs)",
            "effects": "Pedalboard VST-grade" if HAS_PEDALBOARD else "scipy fallback",
            "analyzer": "Madmom + Essentia + Librosa"
        },
        "genres_available": list(remix_generator.genre_presets.keys()) if remix_generator else ["vinahouse", "edm", "house"],
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for processing"""
    
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "audio/x-flac"]
    if file.content_type not in allowed_types:
        # Check by extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.mp3', '.wav', '.flac']:
            raise HTTPException(status_code=400, detail="Only MP3, WAV, and FLAC files are supported")
    
    # Generate unique ID
    job_id = str(uuid.uuid4())
    
    # Save file
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_ext}")
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Initialize job
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "created_at": datetime.now().isoformat(),
        "file_path": file_path,
        "analysis": None,
        "stems": None,
        "remix": None
    }
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "message": "File uploaded successfully"
    }


@app.post("/api/analyze/{job_id}")
async def analyze_audio(job_id: str, background_tasks: BackgroundTasks):
    """Analyze uploaded audio - BPM, key, genre, energy"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Start analysis in background
    background_tasks.add_task(process_analysis, job_id)
    
    jobs[job_id]["status"] = "analyzing"
    
    return {"job_id": job_id, "status": "analyzing", "message": "Analysis started"}


async def process_analysis(job_id: str):
    """Background task for audio analysis"""
    try:
        _, genre_detector, audio_analyzer, _ = get_models()
        
        job = jobs[job_id]
        file_path = job["file_path"]
        
        # Run analysis
        analysis = audio_analyzer.analyze(file_path)
        genre = genre_detector.detect(file_path)
        
        analysis["genre"] = genre
        
        jobs[job_id]["analysis"] = analysis
        jobs[job_id]["status"] = "analyzed"
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"Analysis error: {e}")


@app.post("/api/separate/{job_id}")
async def separate_stems(job_id: str, background_tasks: BackgroundTasks):
    """Separate audio into stems (vocals, drums, bass, other)"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Start separation in background
    background_tasks.add_task(process_separation, job_id)
    
    jobs[job_id]["status"] = "separating"
    
    return {"job_id": job_id, "status": "separating", "message": "Source separation started"}


async def process_separation(job_id: str):
    """Background task for source separation"""
    try:
        source_separator, _, _, _ = get_models()
        
        job = jobs[job_id]
        file_path = job["file_path"]
        output_dir = os.path.join(STEMS_DIR, job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run source separation
        stems = source_separator.separate(file_path, output_dir)
        
        jobs[job_id]["stems"] = stems
        jobs[job_id]["status"] = "separated"
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"Separation error: {e}")


@app.post("/api/remix/{job_id}")
async def generate_remix(
    job_id: str, 
    settings: RemixSettings, 
    background_tasks: BackgroundTasks
):
    """Generate remix based on settings"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.get("stems") is None:
        raise HTTPException(status_code=400, detail="Source separation must be done first")
    
    # Start remix generation in background
    background_tasks.add_task(process_remix, job_id, settings)
    
    jobs[job_id]["status"] = "remixing"
    jobs[job_id]["remix_settings"] = settings.dict()
    
    return {"job_id": job_id, "status": "remixing", "message": "Remix generation started"}


async def process_remix(job_id: str, settings: RemixSettings):
    """Background task for remix generation"""
    try:
        _, _, _, remix_generator = get_models()
        
        job = jobs[job_id]
        stems = job["stems"]
        analysis = job.get("analysis", {})
        
        output_path = os.path.join(REMIX_DIR, f"{job_id}_remix.wav")
        
        # Generate remix
        remix_path = remix_generator.generate(
            stems=stems,
            analysis=analysis,
            settings=settings.dict(),
            output_path=output_path
        )
        
        jobs[job_id]["remix"] = {
            "path": remix_path,
            "url": f"/api/download/remix/{job_id}"
        }
        jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"Remix error: {e}")


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get current status of a job"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "analysis": job.get("analysis"),
        "stems": job.get("stems"),
        "remix": job.get("remix"),
        "error": job.get("error")
    }


@app.get("/api/download/stem/{job_id}/{stem_name}")
async def download_stem(job_id: str, stem_name: str):
    """Download individual stem file"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    stems = job.get("stems", {})
    
    if stem_name not in stems:
        raise HTTPException(status_code=404, detail="Stem not found")
    
    stem_path = stems[stem_name]
    
    if not os.path.exists(stem_path):
        raise HTTPException(status_code=404, detail="Stem file not found")
    
    return FileResponse(
        stem_path,
        media_type="audio/wav",
        filename=f"{stem_name}.wav"
    )


@app.get("/api/download/remix/{job_id}")
async def download_remix(job_id: str):
    """Download generated remix"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    remix = job.get("remix")
    
    if not remix:
        raise HTTPException(status_code=404, detail="Remix not found")
    
    remix_path = remix["path"]
    
    if not os.path.exists(remix_path):
        raise HTTPException(status_code=404, detail="Remix file not found")
    
    return FileResponse(
        remix_path,
        media_type="audio/wav",
        filename=f"remix_{job['filename']}"
    )


# Training state
training_status: Dict[str, Any] = {
    "is_training": False,
    "progress": 0,
    "total_songs": 0,
    "processed_songs": 0,
    "current_song": None,
    "results": None
}
trainer: Optional[ModelTrainer] = None


class TrainingRequest(BaseModel):
    dataset_path: str
    output_model_path: str = "models"
    segment_duration: float = 30.0


@app.post("/api/train/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training from music library"""
    global trainer, training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset path not found: {request.dataset_path}")
    
    # Initialize trainer
    config = TrainingConfig(
        dataset_path=request.dataset_path,
        output_model_path=request.output_model_path,
        segment_duration=request.segment_duration
    )
    
    trainer = ModelTrainer(config)
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["results"] = None
    
    # Start training in background
    background_tasks.add_task(run_training)
    
    return {
        "message": "Training started",
        "dataset_path": request.dataset_path
    }


async def run_training():
    """Background task for model training"""
    global training_status
    
    try:
        async def progress_callback(processed, total, current_song):
            training_status["processed_songs"] = processed
            training_status["total_songs"] = total
            training_status["progress"] = (processed / total * 100) if total > 0 else 0
            training_status["current_song"] = current_song.file_path if current_song else None
        
        results = await trainer.train(progress_callback=progress_callback)
        
        training_status["is_training"] = False
        training_status["results"] = results
        training_status["progress"] = 100
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["results"] = {"status": "failed", "error": str(e)}
        print(f"Training error: {e}")


@app.get("/api/train/status")
async def get_training_status():
    """Get current training status"""
    return training_status


@app.post("/api/train/stop")
async def stop_training():
    """Stop current training"""
    global training_status
    training_status["is_training"] = False
    return {"message": "Training stopped"}


@app.get("/api/train/results")
async def get_training_results():
    """Get training results"""
    if not training_status["results"]:
        raise HTTPException(status_code=404, detail="No training results available")
    return training_status["results"]


# =====================
# FAST TRAINING API
# =====================

from services.fast_training import create_fast_trainer, get_available_models, TrainingConfig as FastTrainingConfig

fast_training_status: Dict[str, Any] = {
    "is_training": False,
    "progress": 0,
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "results": None
}
fast_trainer_instance = None


class FastTrainingRequest(BaseModel):
    method: str = "lora"
    pretrained_model: str = "demucs_ht"
    dataset_path: str = "./datasets/vinahouse"
    output_path: str = "./checkpoints"
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: float = 16.0


@app.get("/api/fast-train/models")
async def api_get_fast_train_models():
    """Lấy danh sách models hỗ trợ Fast Training"""
    return await get_available_models()


@app.post("/api/fast-train/start")
async def api_start_fast_training(request: FastTrainingRequest, background_tasks: BackgroundTasks):
    global fast_trainer_instance, fast_training_status
    
    if fast_training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Fast Training already in progress")
        
    config_dict = request.dict()
    fast_trainer_instance = create_fast_trainer(config_dict)
    
    fast_training_status.update({
        "is_training": True,
        "progress": 0,
        "epoch": 0,
        "total_epochs": request.epochs,
        "loss": 0.0,
        "results": None
    })
    
    background_tasks.add_task(run_fast_training)
    return {"message": "Fast Training started", "config": config_dict}


async def run_fast_training():
    global fast_trainer_instance, fast_training_status
    try:
        async def progress_callback(info):
            fast_training_status["epoch"] = info["epoch"]
            fast_training_status["total_epochs"] = info["total_epochs"]
            fast_training_status["progress"] = info["progress"]
            fast_training_status["loss"] = info["loss"]
            
        results = await fast_trainer_instance.train(progress_callback=progress_callback)
        
        fast_training_status["is_training"] = False
        fast_training_status["progress"] = 100
        fast_training_status["results"] = results
    except Exception as e:
        fast_training_status["is_training"] = False
        fast_training_status["results"] = {"status": "failed", "error": str(e)}
        print(f"Fast Training error: {e}")


@app.get("/api/fast-train/status")
async def api_get_fast_training_status():
    return fast_training_status


@app.post("/api/fast-train/stop")
async def api_stop_fast_training():
    global fast_trainer_instance, fast_training_status
    if fast_trainer_instance:
        fast_trainer_instance.stop_training()
    fast_training_status["is_training"] = False
    return {"message": "Fast Training stopped"}


# =====================
# WATERMARK API
# =====================

class WatermarkRequest(BaseModel):
    audio_path: str
    producer_id: str
    track_id: str = ""
    output_path: Optional[str] = None


class WatermarkDetectRequest(BaseModel):
    audio_path: str


@app.post("/api/watermark/embed")
async def api_embed_watermark(request: WatermarkRequest):
    """
    Embed ultrasonic watermark into audio file
    
    - **audio_path**: Path to audio file
    - **producer_id**: Your producer identifier (max 32 chars)
    - **track_id**: Track identifier (max 32 chars)
    - **output_path**: Optional output path (default: <input>_watermarked.<ext>)
    
    Watermark is embedded at 19kHz (ultrasonic, inaudible)
    """
    if not os.path.exists(request.audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")
    
    result = embed_watermark(
        audio_path=request.audio_path,
        producer_id=request.producer_id,
        track_id=request.track_id,
        output_path=request.output_path
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Watermark embedding failed"))
    
    return result


@app.post("/api/watermark/detect")
async def api_detect_watermark(request: WatermarkDetectRequest):
    """
    Detect ultrasonic watermark in audio file
    
    Returns:
    - **has_watermark**: Boolean indicating watermark presence
    - **producer_id**: Extracted producer ID (if found)
    - **track_id**: Extracted track ID (if found)
    - **timestamp**: Original embed timestamp (if found)
    - **confidence**: Detection confidence (0-1)
    - **ultrasonic_energy**: Energy level in ultrasonic range
    """
    if not os.path.exists(request.audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")
    
    result = detect_watermark(request.audio_path)
    return result


@app.post("/api/watermark/analyze")
async def api_analyze_spectrum(request: WatermarkDetectRequest):
    """
    Analyze audio spectrum - useful for visualization
    
    Returns frequency spectrum data focused on ultrasonic region
    """
    if not os.path.exists(request.audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")
    
    result = analyze_spectrum(request.audio_path)
    return result


@app.post("/api/watermark/embed-job/{job_id}")
async def embed_watermark_to_job(job_id: str, producer_id: str, track_id: str = ""):
    """
    Embed watermark to an existing job's remix output
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    remix = job.get("remix")
    
    if not remix:
        raise HTTPException(status_code=400, detail="No remix available for this job")
    
    remix_path = remix["path"]
    
    result = embed_watermark(
        audio_path=remix_path,
        producer_id=producer_id,
        track_id=track_id,
        output_path=remix_path  # Overwrite
    )
    
    if result.get("success"):
        job["watermark"] = {
            "producer_id": producer_id,
            "track_id": track_id,
            "embedded": True
        }
    
    return result


# =====================
# LONG AUDIO PROCESSING API (DJ Sets 2-5h)
# =====================

class LongAudioRequest(BaseModel):
    file_path: str
    force_reprocess: bool = False


class FolderProcessRequest(BaseModel):
    folder_path: str
    recursive: bool = True
    force_reprocess: bool = False
    max_workers: int = 2
    skip_cached: bool = True


long_audio_status: Dict[str, Any] = {
    "is_processing": False,
    "current_file": None,
    "progress": None,
    "metadata": None,
    "folder_progress": None,
    "dataset": None
}


@app.post("/api/long-audio/process")
async def process_long_audio(request: LongAudioRequest, background_tasks: BackgroundTasks):
    """
    Process a long audio file (DJ set, 2-5 hours)
    
    - Creates hierarchical chunks (30s segments)
    - Caches spectrograms for efficient training
    - Returns metadata with chunk information
    """
    global long_audio_status
    
    if long_audio_status["is_processing"]:
        raise HTTPException(status_code=400, detail="Already processing a file")
    
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    
    long_audio_status["is_processing"] = True
    long_audio_status["current_file"] = request.file_path
    long_audio_status["progress"] = None
    
    background_tasks.add_task(run_long_audio_processing, request.file_path, request.force_reprocess)
    
    return {"message": "Processing started", "file_path": request.file_path}


async def run_long_audio_processing(file_path: str, force_reprocess: bool):
    """Background task for long audio processing"""
    global long_audio_status
    
    try:
        processor = get_long_audio_processor()
        
        def progress_callback(progress):
            long_audio_status["progress"] = {
                "stage": progress.stage,
                "current": progress.current,
                "total": progress.total,
                "message": progress.message,
                "percentage": progress.percentage
            }
        
        processor.set_progress_callback(progress_callback)
        metadata = processor.process_file(file_path, force_reprocess)
        
        long_audio_status["is_processing"] = False
        long_audio_status["current_file"] = None
        long_audio_status["progress"] = None
        long_audio_status["metadata"] = metadata
        
    except Exception as e:
        long_audio_status["is_processing"] = False
        long_audio_status["progress"] = {"error": str(e)}
        print(f"Long audio processing error: {e}")


@app.get("/api/long-audio/status")
async def get_long_audio_status():
    """Get current long audio processing status"""
    return long_audio_status


@app.get("/api/long-audio/stats")
async def get_long_audio_stats():
    """Get statistics about processed long audio files"""
    if not long_audio_status.get("metadata"):
        return {"message": "No files processed yet"}
    
    processor = get_long_audio_processor()
    stats = processor.get_dataset_stats(long_audio_status["metadata"])
    return stats


@app.post("/api/long-audio/cancel")
async def cancel_long_audio():
    """Cancel current long audio processing"""
    processor = get_long_audio_processor()
    processor.cancel()
    return {"message": "Cancellation requested"}


@app.delete("/api/long-audio/cache")
async def clear_long_audio_cache(file_path: Optional[str] = None):
    """Clear cache for long audio processing"""
    processor = get_long_audio_processor()
    processor.clear_cache(file_path)
    return {"message": "Cache cleared"}


# =====================
# FOLDER PROCESSING API
# =====================

@app.get("/api/long-audio/metadata")
async def get_file_metadata(file_path: str):
    """Get cached metadata for a specific file"""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    processor = get_long_audio_processor()
    metadata = processor.load_cached_metadata(file_path)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found in cache")
    
    return metadata


@app.get("/api/long-audio/scan-folder")
async def scan_folder_for_audio(folder_path: str, recursive: bool = True):
    """
    Scan a folder for audio files
    
    Returns list of found audio files with:
    - path, name, size_mb, estimated_duration_min, cached status
    """
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    
    processor = get_long_audio_processor()
    
    try:
        files = processor.scan_folder(folder_path, recursive)
        return {
            "folder_path": folder_path,
            "total_files": len(files),
            "total_size_mb": sum(f['size_mb'] for f in files),
            "total_duration_hours": sum(f['estimated_duration_min'] for f in files) / 60,
            "cached_files": sum(1 for f in files if f['cached']),
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/long-audio/process-folder")
async def process_folder(request: FolderProcessRequest, background_tasks: BackgroundTasks):
    """
    Process all audio files in a folder
    
    - Scans folder for audio files (MP3, WAV, FLAC, etc.)
    - Processes each file in parallel
    - Creates combined dataset for training
    - Returns dataset metadata
    """
    global long_audio_status
    
    if long_audio_status["is_processing"]:
        raise HTTPException(status_code=400, detail="Already processing")
    
    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")
    
    long_audio_status["is_processing"] = True
    long_audio_status["current_file"] = request.folder_path
    long_audio_status["folder_progress"] = None
    long_audio_status["progress"] = None
    
    background_tasks.add_task(
        run_folder_processing,
        request.folder_path,
        request.recursive,
        request.force_reprocess,
        request.max_workers,
        request.skip_cached
    )
    
    return {
        "message": "Folder processing started",
        "folder_path": request.folder_path
    }


async def run_folder_processing(
    folder_path: str,
    recursive: bool,
    force_reprocess: bool,
    max_workers: int,
    skip_cached: bool
):
    """Background task for folder processing"""
    global long_audio_status
    
    try:
        processor = get_long_audio_processor()
        
        def progress_callback(progress):
            long_audio_status["progress"] = progress
            long_audio_status["folder_progress"] = processor.get_folder_progress()
        
        processor.set_progress_callback(progress_callback)
        
        dataset = processor.process_folder(
            folder_path=folder_path,
            recursive=recursive,
            force_reprocess=force_reprocess,
            max_workers=max_workers,
            skip_cached=skip_cached
        )
        
        long_audio_status["is_processing"] = False
        long_audio_status["current_file"] = None
        long_audio_status["progress"] = None
        long_audio_status["folder_progress"] = processor.get_folder_progress()
        long_audio_status["dataset"] = dataset
        
    except Exception as e:
        long_audio_status["is_processing"] = False
        long_audio_status["progress"] = {"error": str(e)}
        print(f"Folder processing error: {e}")


@app.get("/api/long-audio/folder-progress")
async def get_folder_progress():
    """Get detailed progress for folder processing"""
    processor = get_long_audio_processor()
    progress = processor.get_folder_progress()
    
    return {
        "is_processing": long_audio_status["is_processing"],
        "progress": long_audio_status.get("progress"),
        "folder_progress": progress
    }


@app.get("/api/long-audio/datasets")
async def list_cached_datasets():
    """List all cached datasets"""
    processor = get_long_audio_processor()
    datasets = processor.get_cached_datasets()
    return {"datasets": datasets}


@app.get("/api/long-audio/dataset")
async def get_current_dataset():
    """Get current processed dataset"""
    if not long_audio_status.get("dataset"):
        return {"message": "No dataset processed yet"}
    
    return long_audio_status["dataset"]


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Clean up files
    import shutil
    
    try:
        if os.path.exists(job.get("file_path", "")):
            os.remove(job["file_path"])
        
        stem_dir = os.path.join(STEMS_DIR, job_id)
        if os.path.exists(stem_dir):
            shutil.rmtree(stem_dir)
        
        if job.get("remix") and os.path.exists(job["remix"].get("path", "")):
            os.remove(job["remix"]["path"])
    except Exception as e:
        print(f"Cleanup error: {e}")
    
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
