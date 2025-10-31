#!/usr/bin/env python3
"""
Vision Assistant Server Startup Script
"""

import sys
import os


server_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, server_dir)

if __name__ == "__main__":
    from server import app
    import uvicorn
    
    print("Starting Vision Assistant Server...")
    print("Server will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 