# Video Processing API with Celery

A high-performance video processing system for person re-identification and tracking, built with FastAPI and Celery for background job processing.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Redis
```bash
# Using Docker (Recommended)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or using Docker Compose
docker-compose up -d redis
```

### 3. Start the System
```bash
# Terminal 1: Start Celery Worker
python start_celery_worker.py

# Terminal 2: Start FastAPI Server
python celery_fastapi_server.py

# Terminal 3: Start Flower Monitoring (Optional)
celery -A celery_app flower --port=5555
```

### 4. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Flower Monitoring**: http://localhost:5555
- **Health Check**: http://localhost:8000/health

## 📋 API Usage

### Submit a Video Processing Job
```bash
curl -X POST "http://localhost:8000/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "path/to/video.mp4",
    "max_players": 2,
    "use_gpu": true,
    "output_dir": "results"
  }'
```

### Check Job Status
```bash
curl "http://localhost:8000/jobs/{task_id}"
```

### Get Results
```bash
curl "http://localhost:8000/jobs/{task_id}/result"
```

## 🏗️ System Architecture

```
FastAPI Server → Redis → Celery Workers
     ↓              ↓        ↓
  Returns job    Stores    Process
     ID         job data   videos
```

## 📁 Project Structure

```
├── celery_app.py              # Celery configuration
├── celery_tasks.py            # Background tasks
├── celery_fastapi_server.py   # FastAPI server
├── integrated_workflow.py     # Core video processing
├── squash_player_tracker/     # Player tracking module
├── start_celery_worker.py     # Worker startup script
├── start_flower.py           # Monitoring startup script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Job Parameters
- `video_path`: Path to input video file
- `max_players`: Maximum number of players to track (default: 2)
- `detection_conf`: Person detection confidence (default: 0.6)
- `reid_threshold`: Re-ID similarity threshold (default: 0.35)
- `use_gpu`: Enable GPU acceleration (default: true)
- `output_dir`: Output directory for results (default: "celery_results")

### Environment Variables
- `REDIS_URL`: Redis broker URL (default: redis://localhost:6379/0)
- `REDIS_RESULT_URL`: Redis result backend URL (default: redis://localhost:6379/1)
- `CELERY_WORKER_CONCURRENCY`: Worker concurrency (Linux only, default: 4)

## 📊 Output Structure

Each completed job creates results in the specified output directory:

```
celery_results/
├── celery_job_{task_id}/
│   ├── player_metadata/
│   │   ├── player_1.json
│   │   └── player_2.json
│   ├── reid_results/
│   │   └── reid_output.mp4
│   ├── run_info.json
│   └── workflow_summary.json
```

## 🐍 Python Client Example

```python
import requests

# Submit job
response = requests.post("http://localhost:8000/jobs", json={
    "video_path": "path/to/video.mp4",
    "max_players": 2,
    "use_gpu": True
})

task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/jobs/{task_id}")
print(status.json())

# Get results when completed
result = requests.get(f"http://localhost:8000/jobs/{task_id}/result")
print(result.json())
```

## 🚨 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check if Redis is running
   docker ps | grep redis
   
   # Test connection
   docker exec -it redis redis-cli ping
   ```

2. **Worker Not Starting**
   ```bash
   # Check Redis connection
   python -c "import redis; r = redis.Redis(); print(r.ping())"
   
   # Start worker manually
   celery -A celery_app worker --pool=solo --loglevel=info
   ```

3. **API Not Responding**
   ```bash
   # Check if API is running
   curl http://localhost:8000/health
   
   # Check worker status
   curl http://localhost:8000/stats
   ```

## 🎯 Features

- ✅ **Immediate API responses** - No waiting for video processing
- ✅ **Background processing** - Videos processed asynchronously
- ✅ **Real-time monitoring** - Flower dashboard for task tracking
- ✅ **Cross-platform** - Works on Windows and Linux
- ✅ **Scalable** - Add more workers for better performance
- ✅ **GPU acceleration** - CUDA support for faster processing
- ✅ **RESTful API** - Easy integration with any client

## 📝 License

This project is for educational and research purposes.