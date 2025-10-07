#!/usr/bin/env python3
"""
Celery Tasks for Video Processing

This module contains Celery tasks that run the integrated video processing workflow.
Tasks are designed to work on both Windows and Linux environments.
"""

import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, List
from celery import current_task
from celery.exceptions import Retry, SoftTimeLimitExceeded

# Import torchreid compatibility module first
import torchreid_compat

# Import our modules
from integrated_workflow import IntegratedPersonReIDWorkflow
from celery_app import app

@app.task(bind=True)
def process_video_task(self, video_path: str, 
                      clip_extraction_params: Dict = None,
                      reid_params: Dict = None,
                      output_dir: str = 'celery_results',
                      job_id: str = None) -> Dict[str, Any]:
    """
    Process a single video using the integrated workflow
    
    Args:
        video_path: Path to the input video file
        clip_extraction_params: Parameters for clip extraction
        reid_params: Parameters for person re-identification
        output_dir: Output directory for results
        job_id: Optional job ID for tracking
    
    Returns:
        Dict containing processing results
    """
    try:
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Starting video processing',
                'progress': 0,
                'video_path': video_path,
                'job_id': job_id or self.request.id
            }
        )
        
        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Set default parameters
        if clip_extraction_params is None:
            clip_extraction_params = {
                'sample_rate': 30,
                'min_segment_duration': 2.0,
                'max_empty_duration': 1.0,
                'create_visualization': True
            }
        
        if reid_params is None:
            reid_params = {
                'detection_conf': 0.6,
                'reid_threshold': 0.35,
                'use_gpu': True,
                'max_players': 2
            }
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Initializing workflow',
                'progress': 10,
                'video_path': video_path,
                'job_id': job_id or self.request.id
            }
        )
        
        # Initialize workflow
        workflow = IntegratedPersonReIDWorkflow(
            clip_extraction_params=clip_extraction_params,
            reid_params=reid_params,
            results_dir=output_dir
        )
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Processing video',
                'progress': 20,
                'video_path': video_path,
                'job_id': job_id or self.request.id
            }
        )
        
        # Process video
        results = workflow.process_video(
            video_path=video_path,
            output_dir=output_dir,
            run_name=f"celery_job_{job_id or self.request.id}"
        )
        
        # Update final progress
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Finalizing results',
                'progress': 90,
                'video_path': video_path,
                'job_id': job_id or self.request.id
            }
        )
        
        # Add task metadata to results
        results['task_id'] = self.request.id
        results['job_id'] = job_id or self.request.id
        results['completed_at'] = datetime.now().isoformat()
        results['video_path'] = video_path
        results['output_dir'] = output_dir
        
        # Update final state
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'Video processing completed successfully',
                'progress': 100,
                'video_path': video_path,
                'job_id': job_id or self.request.id,
                'results': results
            }
        )
        
        return results
        
    except SoftTimeLimitExceeded:
        # Handle soft time limit
        error_msg = "Video processing timed out (soft limit)"
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Failed',
                'progress': 0,
                'error': error_msg,
                'video_path': video_path,
                'job_id': job_id or self.request.id
            }
        )
        raise SoftTimeLimitExceeded(error_msg)
        
    except Exception as e:
        # Handle any other errors
        error_msg = f"Video processing failed: {str(e)}"
        error_traceback = traceback.format_exc()
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Failed',
                'progress': 0,
                'error': error_msg,
                'traceback': error_traceback,
                'video_path': video_path,
                'job_id': job_id or self.request.id
            }
        )
        
        # Log the error
        print(f"Task {self.request.id} failed: {error_msg}")
        print(f"Traceback: {error_traceback}")
        
        raise e

@app.task(bind=True)
def process_batch_task(self, video_paths: List[str],
                      clip_extraction_params: Dict = None,
                      reid_params: Dict = None,
                      output_dir: str = 'celery_results',
                      batch_id: str = None) -> Dict[str, Any]:
    """
    Process multiple videos in batch
    
    Args:
        video_paths: List of video file paths
        clip_extraction_params: Parameters for clip extraction
        reid_params: Parameters for person re-identification
        output_dir: Output directory for results
        batch_id: Optional batch ID for tracking
    
    Returns:
        Dict containing batch processing results
    """
    try:
        batch_id = batch_id or self.request.id
        results = []
        failed_videos = []
        
        # Update initial state
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Starting batch processing',
                'progress': 0,
                'total_videos': len(video_paths),
                'processed': 0,
                'failed': 0,
                'batch_id': batch_id
            }
        )
        
        # Process each video
        for i, video_path in enumerate(video_paths):
            try:
                # Update progress
                progress = int((i / len(video_paths)) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'status': f'Processing video {i+1}/{len(video_paths)}',
                        'progress': progress,
                        'current_video': video_path,
                        'total_videos': len(video_paths),
                        'processed': i,
                        'failed': len(failed_videos),
                        'batch_id': batch_id
                    }
                )
                
                # Process single video
                video_result = process_video_task.apply_async(
                    args=[video_path],
                    kwargs={
                        'clip_extraction_params': clip_extraction_params,
                        'reid_params': reid_params,
                        'output_dir': output_dir,
                        'job_id': f"{batch_id}_video_{i+1}"
                    }
                ).get()  # Wait for completion
                
                results.append({
                    'video_path': video_path,
                    'success': True,
                    'result': video_result
                })
                
            except Exception as e:
                # Handle individual video failure
                failed_videos.append({
                    'video_path': video_path,
                    'error': str(e)
                })
                
                results.append({
                    'video_path': video_path,
                    'success': False,
                    'error': str(e)
                })
        
        # Final results
        batch_results = {
            'batch_id': batch_id,
            'total_videos': len(video_paths),
            'successful': len(results) - len(failed_videos),
            'failed': len(failed_videos),
            'results': results,
            'failed_videos': failed_videos,
            'completed_at': datetime.now().isoformat(),
            'output_dir': output_dir
        }
        
        # Update final state
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'Batch processing completed',
                'progress': 100,
                'total_videos': len(video_paths),
                'successful': len(results) - len(failed_videos),
                'failed': len(failed_videos),
                'batch_id': batch_id,
                'results': batch_results
            }
        )
        
        return batch_results
        
    except Exception as e:
        # Handle batch processing failure
        error_msg = f"Batch processing failed: {str(e)}"
        error_traceback = traceback.format_exc()
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Batch processing failed',
                'progress': 0,
                'error': error_msg,
                'traceback': error_traceback,
                'batch_id': batch_id or self.request.id
            }
        )
        
        print(f"Batch task {self.request.id} failed: {error_msg}")
        print(f"Traceback: {error_traceback}")
        
        raise e

@app.task
def health_check_task() -> Dict[str, Any]:
    """
    Health check task to verify Celery is working
    
    Returns:
        Dict containing health status
    """
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'task_id': current_task.request.id if current_task else None,
        'worker_info': {
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'pid': os.getpid()
        }
    }
