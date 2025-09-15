#!/bin/bash
# Startup script for production deployment

# Set environment variables
export PYTHONPATH=/app
export ENVIRONMENT=production

# Run database migrations/initialization
echo "Starting application..."

# Start Gunicorn with the configuration
exec gunicorn app:app -c gunicorn.conf.py
