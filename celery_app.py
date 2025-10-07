#!/usr/bin/env python3
"""
Cross-platform Celery Configuration

This module configures Celery for both Windows and Linux environments.
- Windows: Uses solo pool for compatibility
- Linux: Uses prefork pool for better performance
"""

import os
import platform
from celery import Celery

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
REDIS_RESULT_URL = os.getenv('REDIS_RESULT_URL', 'redis://localhost:6379/1')

# Detect operating system
IS_WINDOWS = platform.system().lower() == 'windows'

# Celery configuration
app = Celery('video_processing')

# Basic configuration
app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_RESULT_URL,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_ignore_result=False,
    task_store_eager_result=True,
    result_expires=3600,  # 1 hour
    result_persistent=True,
    task_compression='gzip',
    result_compression='gzip',
)

# Windows-specific configuration
if IS_WINDOWS:
    app.conf.update(
        worker_pool='solo',  # Required for Windows
        worker_concurrency=1,  # Single worker on Windows
        worker_prefetch_multiplier=1,
        worker_disable_rate_limits=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
    )
    print("🔧 Windows detected: Using solo pool with single worker")
else:
    # Linux/Ubuntu configuration
    app.conf.update(
        worker_pool='prefork',  # Better performance on Linux
        worker_concurrency=int(os.getenv('CELERY_WORKER_CONCURRENCY', '4')),
        worker_prefetch_multiplier=1,
        worker_disable_rate_limits=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
    )
    print("🐧 Linux detected: Using prefork pool with multiple workers")

# Task routing
app.conf.task_routes = {
    'celery_tasks.process_video_task': {'queue': 'video_processing'},
    'celery_tasks.process_batch_task': {'queue': 'video_processing'},
}

# Queue configuration
app.conf.task_default_queue = 'video_processing'
app.conf.task_queues = {
    'video_processing': {
        'routing_key': 'video_processing',
    },
}

# Task execution settings
app.conf.task_annotations = {
    'celery_tasks.process_video_task': {
        'rate_limit': '10/m',  # 10 tasks per minute
        'time_limit': 3600,    # 1 hour hard limit
        'soft_time_limit': 3300,  # 55 minutes soft limit
    },
    'celery_tasks.process_batch_task': {
        'rate_limit': '5/m',   # 5 batch tasks per minute
        'time_limit': 7200,    # 2 hours hard limit
        'soft_time_limit': 6900,  # 1 hour 55 minutes soft limit
    }
}

# Import tasks directly instead of autodiscovery
from celery_tasks import process_video_task, process_batch_task, health_check_task

if __name__ == '__main__':
    app.start()
