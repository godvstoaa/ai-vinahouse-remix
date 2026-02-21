"""Start the AI Remix Backend Server"""
import uvicorn
import os

# Change to backend directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )