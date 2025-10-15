"""
start_api.py
Script to start the FastAPI server
"""

import subprocess
import sys
import time
import requests

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI Server...")
    
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, "api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("⏳ Waiting for server to start...")
        time.sleep(5)  # Give server time to start
        
        # Test if server is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                print("✅ Server started successfully!")
                print("🌐 API is running at: http://localhost:8000")
                print("📚 API docs available at: http://localhost:8000/docs")
                return process
            else:
                print("❌ Server started but health check failed")
        except:
            print("❌ Server failed to start properly")
            
        return None
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

if __name__ == "__main__":
    start_server()