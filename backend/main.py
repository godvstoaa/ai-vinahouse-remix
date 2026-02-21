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

# Import ML modules
from services.source_separator import SourceSeparator
from services.genre_detector import GenreDetector
from services.audio_analyzer import AudioAnalyzer
from services.remix_generator import RemixGenerator
from services.model_trainer import ModelTrainer, TrainingConfig, get_trainer

app = FastAPI(
    title="AI Remix Studio API",
    description="Backend for AI-powered music remix application",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    models = get_models()
    return {
        "status": "healthy",
        "models_loaded": source_separator is not None,
        "gpu_available": torch.cuda.is_available() if 'torch' in globals() else False
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
    uvicorn.run(app, host="0.0.0.0", port=8000)