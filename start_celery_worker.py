#!/usr/bin/env python3
"""
Start Celery Worker Script

This script starts a Celery worker with platform-specific configuration.
- Windows: Uses solo pool with single worker
- Linux: Uses prefork pool with multiple workers
"""

import os
import sys
import platform
from celery import Celery

# Import our Celery app
from celery_app import app

def main():
    """Start Celery worker with appropriate configuration"""
    
    # Detect platform
    is_windows = platform.system().lower() == 'windows'
    
    print("🚀 Starting Celery Worker...")
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version}")
    
    if is_windows:
        print("🔧 Windows detected: Using solo pool")
        print("Command: celery -A celery_app worker --pool=solo --loglevel=info")
        
        # Start worker with Windows-specific settings
        app.worker_main([
            'worker',
            '--pool=solo',
            '--loglevel=info',
            '--concurrency=1'
        ])
    else:
        print("🐧 Linux detected: Using prefork pool")
        concurrency = os.getenv('CELERY_WORKER_CONCURRENCY', '4')
        print(f"Command: celery -A celery_app worker --loglevel=info --concurrency={concurrency}")
        
        # Start worker with Linux-specific settings
        app.worker_main([
            'worker',
            '--loglevel=info',
            f'--concurrency={concurrency}'
        ])

if __name__ == "__main__":
    main()
