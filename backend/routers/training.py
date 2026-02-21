"""
Training API Router - Endpoints cho việc train model
"""

import os
import sys
import json
import threading
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter(prefix="/training", tags=["training"])

# Global state
training_state = {
    "is_running": False,
    "current_step": 0,
    "total_steps": 100000,
    "loss": 0.0,
    "learning_rate": 0.0001,
    "gpu_memory_gb": 0.0,
    "eta_hours": 0.0,
    "status_message": "Idle",
    "process": None
}

data_state = {
    "total_files": 0,
    "processed_files": 0,
    "total_segments": 0,
    "is_processing": False
}


class DataPrepRequest(BaseModel):
    input_path: str
    workers: int = 4


class EmotionalTaggerRequest(BaseModel):
    input_path: str


class TrainingConfig(BaseModel):
    batch_size: int = 2
    learning_rate: float = 0.0001
    gradient_accumulation_steps: int = 16
    max_steps: int = 100000
    mixed_precision: str = "fp16"


@router.get("/status")
async def get_training_status():
    """Get current training status"""
    return training_state


@router.get("/data-status")
async def get_data_status():
    """Get data preparation status"""
    # Check manifest file
    manifest_path = Path("datasets/metadata/manifest.json")
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)
                data_state["total_segments"] = data.get("total_segments", 0)
                data_state["total_files"] = data.get("total_files", 0)
                data_state["processed_files"] = data.get("processed", 0)
        except:
            pass
    
    return data_state


@router.post("/data-prep")
async def start_data_prep(request: DataPrepRequest, background_tasks: BackgroundTasks):
    """Start data preparation pipeline"""
    if data_state["is_processing"]:
        raise HTTPException(status_code=400, detail="Data preparation already running")
    
    input_path = Path(request.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {request.input_path}")
    
    def run_data_prep():
        data_state["is_processing"] = True
        data_state["status_message"] = "Processing..."
        
        try:
            # Run data prep script
            cmd = [
                sys.executable, "-m", "training.data_prep_pipeline",
                str(input_path),
                "--workers", str(request.workers)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent.parent.parent)
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                data_state["status_message"] = "Completed"
            else:
                data_state["status_message"] = f"Error: {stderr.decode()[:200]}"
                
        except Exception as e:
            data_state["status_message"] = f"Error: {str(e)}"
        finally:
            data_state["is_processing"] = False
    
    background_tasks.add_task(run_data_prep)
    
    return {"message": "Data preparation started", "input_path": str(input_path)}


@router.post("/emotional-tagger")
async def start_emotional_tagger(request: EmotionalTaggerRequest, background_tasks: BackgroundTasks):
    """Start emotional tagging for Nonstop tracks"""
    input_path = Path(request.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {request.input_path}")
    
    def run_tagger():
        try:
            cmd = [
                sys.executable, "-m", "training.emotional_tagger",
                str(input_path),
                "--output", "datasets/metadata/"
            ]
            
            subprocess.run(cmd, cwd=str(Path(__file__).parent.parent.parent))
        except Exception as e:
            print(f"Emotional tagger error: {e}")
    
    background_tasks.add_task(run_tagger)
    
    return {"message": "Emotional tagging started", "input_path": str(input_path)}


@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start training"""
    if training_state["is_running"]:
        raise HTTPException(status_code=400, detail="Training already running")
    
    def run_training():
        training_state["is_running"] = True
        training_state["total_steps"] = config.max_steps
        training_state["status_message"] = "Starting training..."
        
        try:
            cmd = [
                sys.executable, "-m", "training.train_script",
                "--config", "config/train_config.yaml"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent.parent.parent)
            )
            
            training_state["process"] = process
            
            # Read output line by line
            for line in process.stdout:
                line = line.decode().strip()
                if "step" in line.lower():
                    try:
                        # Parse step number
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "step" and i + 1 < len(parts):
                                training_state["current_step"] = int(parts[i + 1])
                    except:
                        pass
                        
                if "loss" in line.lower():
                    try:
                        # Parse loss
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "loss:" and i + 1 < len(parts):
                                training_state["loss"] = float(parts[i + 1])
                    except:
                        pass
                
                training_state["status_message"] = line[:100]
            
            process.wait()
            training_state["status_message"] = "Training completed" if process.returncode == 0 else "Training failed"
            
        except Exception as e:
            training_state["status_message"] = f"Error: {str(e)}"
        finally:
            training_state["is_running"] = False
            training_state["process"] = None
    
    background_tasks.add_task(run_training)
    
    return {"message": "Training started", "config": config.dict()}


@router.post("/stop")
async def stop_training():
    """Stop training"""
    if not training_state["is_running"]:
        raise HTTPException(status_code=400, detail="No training running")
    
    if training_state["process"]:
        training_state["process"].terminate()
        training_state["status_message"] = "Training stopped by user"
    
    training_state["is_running"] = False
    
    return {"message": "Training stopped"}


@router.get("/checkpoints")
async def list_checkpoints():
    """List available checkpoints"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return {"checkpoints": []}
    
    checkpoints = []
    for f in checkpoints_dir.glob("*.pt"):
        checkpoints.append({
            "name": f.name,
            "size_mb": f.stat().st_size / (1024 * 1024),
            "modified": f.stat().st_mtime
        })
    
    return {"checkpoints": sorted(checkpoints, key=lambda x: x["modified"], reverse=True)}