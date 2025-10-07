#!/usr/bin/env python3
"""
FastAPI Server with Celery Integration

This module provides a REST API that uses Celery for background video processing.
Jobs are queued and processed asynchronously, returning immediately with job IDs.
"""

import os
import sys
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import torchreid compatibility module first
import torchreid_compat

# Import Celery app and tasks
from celery_app import app as celery_app
from celery_tasks import process_video_task, process_batch_task, health_check_task

# Initialize FastAPI app
app = FastAPI(
    title="Video Processing API with Celery",
    description="API for person re-identification and video processing with background job queue",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class VideoJobRequest(BaseModel):
    video_path: str = Field(..., description="Path to the input video file")
    output_dir: str = Field(default="celery_results", description="Output directory for results")
    max_players: int = Field(default=2, description="Maximum number of players to track")
    detection_conf: float = Field(default=0.6, description="Person detection confidence threshold")
    reid_threshold: float = Field(default=0.35, description="Re-ID similarity threshold")
    use_gpu: bool = Field(default=True, description="Use GPU acceleration")
    sample_rate: int = Field(default=30, description="Video sampling rate")
    min_segment_duration: float = Field(default=2.0, description="Minimum segment duration in seconds")
    max_empty_duration: float = Field(default=1.0, description="Maximum empty duration in seconds")
    create_visualization: bool = Field(default=True, description="Create visualization videos")

class BatchJobRequest(BaseModel):
    video_paths: List[str] = Field(..., description="List of video file paths")
    output_dir: str = Field(default="celery_results", description="Output directory for results")
    max_players: int = Field(default=2, description="Maximum number of players to track")
    detection_conf: float = Field(default=0.6, description="Person detection confidence threshold")
    reid_threshold: float = Field(default=0.35, description="Re-ID similarity threshold")
    use_gpu: bool = Field(default=True, description="Use GPU acceleration")

class JobStatus(BaseModel):
    task_id: str
    status: str  # PENDING, PROGRESS, SUCCESS, FAILURE
    progress: float
    message: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Video Processing API with Celery",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "celery_monitoring": "/flower"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Celery worker health
        celery_health = health_check_task.delay().get(timeout=5)
        celery_status = "healthy"
    except Exception as e:
        celery_health = {"error": str(e)}
        celery_status = "unhealthy"
    
    return {
        "api_status": "healthy",
        "celery_status": celery_status,
        "timestamp": datetime.now().isoformat(),
        "celery_health": celery_health
    }

@app.post("/jobs", response_model=Dict[str, str])
async def submit_job(job_request: VideoJobRequest):
    """Submit a video processing job to Celery queue"""
    
    # Validate video file exists
    if not os.path.exists(job_request.video_path):
        raise HTTPException(status_code=400, detail=f"Video file not found: {job_request.video_path}")
    
    # Prepare parameters
    clip_extraction_params = {
        'sample_rate': job_request.sample_rate,
        'min_segment_duration': job_request.min_segment_duration,
        'max_empty_duration': job_request.max_empty_duration,
        'create_visualization': job_request.create_visualization
    }
    
    reid_params = {
        'detection_conf': job_request.detection_conf,
        'reid_threshold': job_request.reid_threshold,
        'use_gpu': job_request.use_gpu,
        'max_players': job_request.max_players
    }
    
    # Submit job to Celery
    try:
        task = process_video_task.delay(
            video_path=job_request.video_path,
            clip_extraction_params=clip_extraction_params,
            reid_params=reid_params,
            output_dir=job_request.output_dir,
            job_id=str(uuid.uuid4())
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Job submitted to Celery queue successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@app.post("/jobs/batch", response_model=Dict[str, Any])
async def submit_batch_job(batch_request: BatchJobRequest):
    """Submit multiple video processing jobs to Celery queue"""
    
    # Validate all video files exist
    missing_files = []
    for video_path in batch_request.video_paths:
        if not os.path.exists(video_path):
            missing_files.append(video_path)
    
    if missing_files:
        raise HTTPException(
            status_code=400, 
            detail=f"Video files not found: {missing_files}"
        )
    
    # Prepare parameters
    clip_extraction_params = {
        'sample_rate': 30,
        'min_segment_duration': 2.0,
        'max_empty_duration': 1.0,
        'create_visualization': True
    }
    
    reid_params = {
        'detection_conf': batch_request.detection_conf,
        'reid_threshold': batch_request.reid_threshold,
        'use_gpu': batch_request.use_gpu,
        'max_players': batch_request.max_players
    }
    
    # Submit batch job to Celery
    try:
        batch_id = str(uuid.uuid4())
        task = process_batch_task.delay(
            video_paths=batch_request.video_paths,
            clip_extraction_params=clip_extraction_params,
            reid_params=reid_params,
            output_dir=batch_request.output_dir,
            batch_id=batch_id
        )
        
        return {
            "task_id": task.id,
            "batch_id": batch_id,
            "status": "queued",
            "total_videos": len(batch_request.video_paths),
            "message": "Batch job submitted to Celery queue successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit batch job: {str(e)}")

@app.get("/jobs/{task_id}", response_model=JobStatus)
async def get_job_status(task_id: str):
    """Get the status of a specific Celery task"""
    try:
        # Get task result from Celery
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            status = "pending"
            progress = 0.0
            message = "Job is waiting to be processed"
            result_data = None
            error = None
            
        elif result.state == 'PROGRESS':
            status = "processing"
            meta = result.info or {}
            progress = meta.get('progress', 0.0)
            message = meta.get('status', 'Processing...')
            result_data = meta
            error = None
            
        elif result.state == 'SUCCESS':
            status = "completed"
            progress = 100.0
            message = "Job completed successfully"
            result_data = result.result
            error = None
            
        elif result.state == 'FAILURE':
            status = "failed"
            progress = 0.0
            message = "Job failed"
            result_data = None
            error = str(result.info) if result.info else "Unknown error"
            
        else:
            status = "unknown"
            progress = 0.0
            message = f"Unknown state: {result.state}"
            result_data = None
            error = None
        
        return JobStatus(
            task_id=task_id,
            status=status,
            progress=progress,
            message=message,
            created_at=datetime.now(),  # Celery doesn't provide creation time
            result=result_data,
            error=error
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/jobs/{task_id}/result")
async def get_job_result(task_id: str):
    """Get the result of a completed Celery task"""
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'SUCCESS':
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result.result,
                "completed_at": datetime.now().isoformat()
            }
        elif result.state == 'FAILURE':
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(result.info) if result.info else "Unknown error",
                "failed_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Task is not completed. Current state: {result.state}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job result: {str(e)}")

@app.delete("/jobs/{task_id}")
async def cancel_job(task_id: str):
    """Cancel a pending Celery task"""
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state in ['SUCCESS', 'FAILURE']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel task with state: {result.state}"
            )
        
        # Revoke the task
        result.revoke(terminate=True)
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancelled successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@app.get("/jobs")
async def list_jobs(limit: int = 50):
    """List recent Celery tasks (Note: This is a simplified implementation)"""
    # Note: In production, you'd want to store task metadata in a database
    # This is a basic implementation that returns active tasks
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        
        # This is a simplified response - in production you'd want more details
        return {
            "message": "Active tasks retrieved",
            "active_tasks": active_tasks,
            "note": "For detailed task history, use a database or Celery monitoring tool"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...), output_dir: str = Form(default="celery_results")):
    """Upload a video file for processing"""
    
    # Create upload directory
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Submit job for uploaded video
    try:
        task = process_video_task.delay(
            video_path=str(file_path),
            output_dir=output_dir,
            job_id=str(uuid.uuid4())
        )
        
        return {
            "task_id": task.id,
            "filename": file.filename,
            "file_path": str(file_path),
            "status": "uploaded_and_queued"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded video: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Get Celery worker stats
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active_tasks = inspect.active()
        
        return {
            "celery_workers": len(stats) if stats else 0,
            "active_tasks": len(active_tasks) if active_tasks else 0,
            "worker_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Failed to get stats: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Main function to run the server
def main():
    """Main function to run the FastAPI server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Processing FastAPI Server with Celery")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting Video Processing API with Celery...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"Celery Monitoring: http://{args.host}:5555 (if Flower is running)")
    
    uvicorn.run(
        "celery_fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
