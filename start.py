#!/usr/bin/env python3
"""
Startup script for Render.com deployment
"""
import os
import subprocess
import sys


def main():
    # Get port from environment variable
    port = os.getenv('PORT', '8000')

    # Build gunicorn command
    cmd = [
        'gunicorn', 'app:app', '--bind', f'0.0.0.0:{port}', '--workers', '4',
        '--worker-class', 'uvicorn.workers.UvicornWorker', '--timeout', '30',
        '--keep-alive', '2', '--max-requests', '1000', '--max-requests-jitter',
        '50', '--access-logfile', '-', '--error-logfile', '-', '--log-level',
        'info'
    ]

    print(f"Starting server on port {port}")
    print(f"Command: {' '.join(cmd)}")

    # Start the server
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Server stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
