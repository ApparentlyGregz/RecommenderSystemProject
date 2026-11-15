"""Launch the Révise Malin API (the Streamlit interface is added in Week 3).

Usage:
    python run.py      then open http://localhost:8000/docs
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)
