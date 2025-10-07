#!/usr/bin/env python3
"""
Start Flower Monitoring Script

This script starts Flower for monitoring Celery tasks and workers.
"""

import os
import sys
import platform
from celery import Celery

# Import our Celery app
from celery_app import app

def main():
    """Start Flower monitoring dashboard"""
    
    print("🌸 Starting Flower Monitoring Dashboard...")
    print(f"Platform: {platform.system()}")
    
    # Start Flower
    app.control.broadcast('shutdown')  # Clean shutdown of any existing workers
    
    print("Command: celery -A celery_app flower --port=5555")
    print("Access Flower at: http://localhost:5555")
    
    # Start Flower
    app.control.broadcast('shutdown')  # Clean shutdown
    app.control.broadcast('shutdown')  # Clean shutdown
    
    # Start Flower with our app
    from flower.command import FlowerCommand
    flower = FlowerCommand()
    flower.run_from_argv(['flower', '--port=5555', '--broker=redis://localhost:6379/0'])

if __name__ == "__main__":
    main()
