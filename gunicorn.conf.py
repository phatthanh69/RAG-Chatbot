# Gunicorn configuration file
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WEB_CONCURRENCY', multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 5

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
logconfig = None

# Process naming
proc_name = 'rag_chatbot_api'

# Server mechanics
daemon = False
pidfile = 'logs/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (if configured)
keyfile = None
certfile = None

# Performance tuning
preload_app = True
worker_tmp_dir = '/dev/shm'
