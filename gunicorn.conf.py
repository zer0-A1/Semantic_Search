# Gunicorn configuration file for FastAPI application
import multiprocessing
import os

# Server socket
port = int(os.getenv('PORT', 8000))
bind = f"0.0.0.0:{port}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WEB_CONCURRENCY',
                        multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "semantic_search_api"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

# Worker timeout for graceful shutdown
graceful_timeout = 30

# Preload app for better performance
preload_app = True

# Environment variables
raw_env = [
    'PYTHONPATH=/app',
]

# Worker class for ASGI applications
worker_class = "uvicorn.workers.UvicornWorker"
