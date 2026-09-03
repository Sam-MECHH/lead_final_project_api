import uvicorn
# Import your FastAPI instance from src/demoday_dashboard_app.py
from src.demoday_dashboard_app import app

if __name__ == "__main__":
    # Hugging Face Spaces require port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)