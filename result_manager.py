"""
Result Management Utilities

This module provides utilities for managing results, creating directory structures,
and saving run information for the person re-identification workflow.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path


def create_result_directory(base_dir="results", run_name=None):
    """
    Create a structured result directory for a run
    
    Args:
        base_dir: Base directory for results
        run_name: Custom name for the run (optional, will use timestamp if not provided)
        
    Returns:
        dict: Dictionary containing paths to different result subdirectories
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate run name if not provided
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    # Create run-specific directory
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories - only create necessary folders
    subdirs = {
        'root': run_dir,
        'reid_results': os.path.join(run_dir, 'reid_results'),
        'player_metadata': os.path.join(run_dir, 'player_metadata')
    }
    
    # Create all subdirectories
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    return subdirs


def get_result_path(result_dirs, input_path, subdir="videos", suffix=""):
    """
    Generate a result path for a file
    
    Args:
        result_dirs: Dictionary of result directories
        input_path: Original input file path
        subdir: Subdirectory to place the file in
        suffix: Suffix to add to the filename
        
    Returns:
        str: Full path for the result file
    """
    # Get filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Add suffix if provided
    if suffix:
        base_name = f"{base_name}_{suffix}"
    
    # Get file extension
    ext = os.path.splitext(input_path)[1]
    
    # Generate result filename
    result_filename = f"{base_name}{ext}"
    
    # Return full path
    return os.path.join(result_dirs[subdir], result_filename)


def copy_input_to_results(input_path, result_dirs):
    """
    Copy input file to results directory for reference
    
    Args:
        input_path: Path to input file
        result_dirs: Dictionary of result directories
        
    Returns:
        str: Path to copied file
    """
    if not os.path.exists(input_path):
        print(f"Warning: Input file not found: {input_path}")
        return None
    
    # Generate output path - save directly in root directory
    filename = os.path.basename(input_path)
    output_path = os.path.join(result_dirs['root'], f"input_{filename}")
    
    try:
        # Copy file
        shutil.copy2(input_path, output_path)
        print(f"Input file copied to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error copying input file: {e}")
        return None


def save_run_info(result_dirs, params, description=""):
    """
    Save run information and parameters to a JSON file
    
    Args:
        result_dirs: Dictionary of result directories
        params: Dictionary of parameters
        description: Description of the run
        
    Returns:
        str: Path to saved info file
    """
    # Prepare run information
    run_info = {
        'timestamp': datetime.now().isoformat(),
        'description': description,
        'parameters': params,
        'directories': result_dirs
    }
    
    # Save to JSON file
    info_path = os.path.join(result_dirs['root'], 'run_info.json')
    
    try:
        with open(info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        print(f"Run information saved to: {info_path}")
        return info_path
    except Exception as e:
        print(f"Error saving run info: {e}")
        return None


def save_processing_log(result_dirs, log_data):
    """
    Save processing log data
    
    Args:
        result_dirs: Dictionary of result directories
        log_data: Dictionary or list of log data
        
    Returns:
        str: Path to saved log file
    """
    log_path = os.path.join(result_dirs['root'], 'processing_log.json')
    
    try:
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Processing log saved to: {log_path}")
        return log_path
    except Exception as e:
        print(f"Error saving processing log: {e}")
        return None


def cleanup_old_runs(base_dir="results", keep_last_n=5):
    """
    Clean up old result directories, keeping only the last N runs
    
    Args:
        base_dir: Base results directory
        keep_last_n: Number of recent runs to keep
    """
    if not os.path.exists(base_dir):
        return
    
    # Get all run directories
    run_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Get modification time
            mtime = os.path.getmtime(item_path)
            run_dirs.append((item_path, mtime))
    
    # Sort by modification time (newest first)
    run_dirs.sort(key=lambda x: x[1], reverse=True)
    
    # Remove old directories
    if len(run_dirs) > keep_last_n:
        for run_dir, _ in run_dirs[keep_last_n:]:
            try:
                shutil.rmtree(run_dir)
                print(f"Removed old run directory: {run_dir}")
            except Exception as e:
                print(f"Error removing directory {run_dir}: {e}")


def get_latest_run_dir(base_dir="results"):
    """
    Get the path to the most recent run directory
    
    Args:
        base_dir: Base results directory
        
    Returns:
        str: Path to latest run directory, or None if no runs exist
    """
    if not os.path.exists(base_dir):
        return None
    
    # Get all run directories
    run_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            mtime = os.path.getmtime(item_path)
            run_dirs.append((item_path, mtime))
    
    if not run_dirs:
        return None
    
    # Return the most recent one
    run_dirs.sort(key=lambda x: x[1], reverse=True)
    return run_dirs[0][0]
