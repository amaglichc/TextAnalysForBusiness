from fastapi import FastAPI
from pydantic import BaseModel
from schemas import TextRequest
import joblib
import uvicorn
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.summarizer_model = joblib.load("../models/summarizer_model.pkl")
    app.state.sentiment_model = joblib.load("../models/sentimental_model.pkl")
    app.state.topic_model = joblib.load("../models/topic_model.pkl")
    yield
    del app.state.summarizer_model
    del app.state.sentiment_model
    del app.state.topic_model
    
app = FastAPI(
    title="NLP Models API",
    description="API for text summarization, sentiment analysis, and topic modeling",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["health"])
async def health():
    try:
        if not hasattr(app.state, 'summarizer_model') or app.state.summarizer_model is None:
            return {
                "status": "error",
                "message": "Summarizer model not loaded"
            }
        if not hasattr(app.state, 'sentiment_model') or app.state.sentiment_model is None:
            return {
                "status": "error",
                "message": "Sentiment model not loaded"
            }
        if not hasattr(app.state, 'topic_model') or app.state.topic_model is None:
            return {
                "status": "error",
                "message": "Topic model not loaded"
            }
        return {
            "status": "ok",
            "model_loaded": True
        }
    except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
        
@app.post("/sentiment", tags=["sentiment"])
async def sentiment(request: TextRequest):
    try:
        sentiment = app.state.sentiment_model.predict([request.text])[0]
        return {
            "sentiment": sentiment
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
@app.post("/topic", tags=["topic"])
async def topic(request: TextRequest):
    try:
        topic = app.state.topic_model.predict([request.text])[0]
        return {
            "topic": topic
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=True)