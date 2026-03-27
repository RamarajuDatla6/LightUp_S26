#!/usr/bin/env python3
"""
Vision Assistant Client Startup Script
"""

import sys
import os
import subprocess


client_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, client_dir)

if __name__ == "__main__":
    print("Starting Vision Assistant Client...")
    print("Make sure the server is running first!")
    print("Press Ctrl+C to stop the client")
    
    # Run the Streamlit client
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        os.path.join(client_dir, "client.py"),
        "--server.port", "8501",
        "--server.address", "localhost"
    ]) 