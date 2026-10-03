import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analysis import textblob_sentiment, analyze_text, DEFAULT_MODEL

app = FastAPI(title="Trade Sentiment API")

# Bubble calls server-side so CORS is not strictly needed, but this makes
# browser-side testing painless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_SECRET = os.environ.get("API_SECRET")


def check_key(provided):
    if API_SECRET and provided != API_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")


class TextIn(BaseModel):
    text: str
    model: str | None = None


@app.get("/")
def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL}


@app.post("/sentiment")
def sentiment(payload: TextIn, x_api_key: str = Header(None)):
    """TextBlob only. Fast, free, no LLM call."""
    check_key(x_api_key)
    return textblob_sentiment(payload.text)


@app.post("/analyze")
def analyze(payload: TextIn, x_api_key: str = Header(None)):
    """Full financial analysis from the notebook's system prompt."""
    check_key(x_api_key)
    try:
        return analyze_text(payload.text, payload.model or DEFAULT_MODEL)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
