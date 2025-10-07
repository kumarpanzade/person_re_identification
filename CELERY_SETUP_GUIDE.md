# Celery Setup Guide for Video Processing API

This guide shows how to set up and run the Celery-based video processing system on both Windows and Ubuntu.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Redis (Required)
```bash
# Windows: Download Redis from GitHub releases
# Or use Docker: docker run -d -p 6379:6379 redis:alpine

# Ubuntu: 
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
```

### 3. Start the System

#### **Windows:**
```bash
# Terminal 1: Start Celery Worker
python start_celery_worker.py

# Terminal 2: Start FastAPI Server
python celery_fastapi_server.py

# Terminal 3: Start Flower Monitoring (Optional)
celery -A celery_app flower --port=5555
```

#### **Ubuntu:**
```bash
# Terminal 1: Start Celery Worker
python start_celery_worker.py

# Terminal 2: Start FastAPI Server
python celery_fastapi_server.py

# Terminal 3: Start Flower Monitoring (Optional)
celery -A celery_app flower --port=5555
```

## 📋 System Components

### **1. Celery App (`celery_app.py`)**
- Cross-platform configuration
- Windows: Uses `solo` pool (single worker)
- Linux: Uses `prefork` pool (multiple workers)
- Redis as message broker and result backend

### **2. Celery Tasks (`celery_tasks.py`)**
- `process_video_task`: Process single video
- `process_batch_task`: Process multiple videos
- `health_check_task`: System health check

### **3. FastAPI Server (`celery_fastapi_server.py`)**
- REST API endpoints
- Job submission and monitoring
- File upload support
- Real-time job status

### **4. Worker Script (`start_celery_worker.py`)**
- Platform-specific worker startup
- Automatic pool selection
- Queue configuration

## 🔧 Configuration

### **Environment Variables**
```bash
# Redis Configuration
export REDIS_URL="redis://localhost:6379/0"
export REDIS_RESULT_URL="redis://localhost:6379/1"

# Worker Configuration (Linux only)
export CELERY_WORKER_CONCURRENCY="4"
```

### **Windows-Specific Settings**
- **Pool**: `solo` (required for Windows)
- **Concurrency**: 1 (single worker)
- **Prefetch**: 1 (one task at a time)

### **Linux-Specific Settings**
- **Pool**: `prefork` (better performance)
- **Concurrency**: 4 (configurable)
- **Prefetch**: 1 (one task per worker)

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check (includes Celery status) |
| POST | `/jobs` | Submit video processing job |
| POST | `/jobs/batch` | Submit batch processing jobs |
| GET | `/jobs/{task_id}` | Get job status |
| GET | `/jobs/{task_id}/result` | Get job results |
| DELETE | `/jobs/{task_id}` | Cancel job |
| GET | `/jobs` | List active tasks |
| POST | `/upload` | Upload video file |
| GET | `/stats` | System statistics |

## 🧪 Testing the System

### **1. Health Check**
```bash
curl http://localhost:8000/health
```

### **2. Submit a Job**
```bash
curl -X POST "http://localhost:8000/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "path/to/video.mp4",
    "max_players": 2,
    "use_gpu": true
  }'
```

### **3. Check Job Status**
```bash
curl "http://localhost:8000/jobs/{task_id}"
```

### **4. Get Results**
```bash
curl "http://localhost:8000/jobs/{task_id}/result"
```

## 🔍 Monitoring

### **Flower Dashboard**
- **URL**: http://localhost:5555
- **Features**: Task monitoring, worker status, real-time updates
- **Start**: `celery -A celery_app flower --port=5555`

### **API Statistics**
- **Endpoint**: `GET /stats`
- **Shows**: Active workers, queued tasks, worker statistics

## 🚨 Troubleshooting

### **Common Issues**

#### **1. Redis Connection Error**
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG
```

#### **2. Worker Not Starting (Windows)**
```bash
# Make sure to use solo pool
celery -A celery_app worker --pool=solo --loglevel=info
```

#### **3. Task Stuck in PENDING**
- Check if worker is running
- Check Redis connection
- Check task queue configuration

#### **4. Import Errors**
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Check Python path
python -c "import celery_app; import celery_tasks"
```

### **Debug Commands**

#### **Check Celery Status**
```bash
celery -A celery_app inspect active
celery -A celery_app inspect stats
```

#### **Check Redis**
```bash
redis-cli monitor
```

#### **Check Worker Logs**
```bash
# Look for error messages in worker output
python start_celery_worker.py
```

## 📈 Performance Tuning

### **Windows Optimization**
- Use single worker (`--pool=solo`)
- Process one video at a time
- Monitor memory usage
- Use SSD for better I/O

### **Linux Optimization**
- Increase worker concurrency: `export CELERY_WORKER_CONCURRENCY="8"`
- Use multiple worker processes
- Monitor CPU and memory usage
- Use Redis persistence

### **General Optimization**
- Use GPU acceleration when available
- Optimize video processing parameters
- Use appropriate Redis configuration
- Monitor disk space for results

## 🚀 Production Deployment

### **Ubuntu Server Setup**
```bash
# Install system dependencies
sudo apt update
sudo apt install redis-server nginx supervisor

# Install Python dependencies
pip install -r requirements.txt

# Configure supervisor for auto-start
# Create /etc/supervisor/conf.d/celery.conf
# Create /etc/supervisor/conf.d/fastapi.conf

# Start services
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start all
```

### **Docker Deployment**
```dockerfile
# Use multi-stage build
# Install dependencies
# Copy application code
# Expose ports
# Start services
```

## 📝 Notes

- **Windows**: Limited to single worker due to multiprocessing limitations
- **Linux**: Full multiprocessing support with multiple workers
- **Redis**: Required for message broker and result storage
- **GPU**: Optional but recommended for better performance
- **Monitoring**: Use Flower for production monitoring
- **Scaling**: Add more workers on Linux for better throughput

## 🎯 Next Steps

1. **Test the system** with small videos
2. **Monitor performance** using Flower dashboard
3. **Scale workers** based on workload
4. **Set up monitoring** and alerting
5. **Deploy to production** server
