# Trade Sentiment API

Your notebook's analysis logic, exposed as a web API that Bubble can call.

## Files

- `analysis.py` — the notebook logic, stripped of Colab/pandas/plotting
- `main.py` — the FastAPI wrapper (the only new code)
- `requirements.txt`
- `render.yaml` — optional, lets Render configure itself

## Endpoints

| Method | Path | What it does | Cost |
|---|---|---|---|
| GET | `/` | Health check | free |
| POST | `/sentiment` | TextBlob polarity + label | free, instant |
| POST | `/analyze` | Full 8-field financial analysis via OpenRouter | ~1 LLM call |

Both POST routes take `{"text": "..."}`. `/analyze` also accepts an optional
`"model"` to override the default.

### `/analyze` response

```json
{
  "action_required": "Yes",
  "market_sentiment": "Negative",
  "mentions_macro_risks": "Yes",
  "mentions_corporate_strengths": "No",
  "bullish_intensity": 2,
  "bearish_intensity": 6,
  "predicted_impact_score": -3,
  "relevance_for_trading": 6,
  "model": "meta-llama/llama-3.1-8b-instruct",
  "parsed_ok": true,
  "raw": ""
}
```

Flat on purpose. Bubble's API Connector maps flat JSON cleanly; nested objects
turn into a mess of sub-types.

## Run locally

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
export API_SECRET=pick-any-random-string
uvicorn main:app --reload
```

No NLTK corpora download is needed. TextBlob's `.sentiment.polarity` uses the
bundled pattern analyzer, not an NLTK corpus. (Verified by deleting `nltk_data`
and re-running.) You would only need `python -m textblob.download_corpora lite`
if you later use noun phrase extraction or POS tagging.

Then open http://127.0.0.1:8000/docs for an interactive test page.

## Deploy to Render

Render → New → Web Service → connect this repo, then:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Instance type: Free
- Environment variables: `OPENROUTER_API_KEY` and `API_SECRET`

Render auto-detects the build command but usually does NOT get the start
command right, so type that one in by hand.

`render.yaml` only configures the service automatically through Render's
**Blueprint** flow (New → Blueprint). The ordinary New → Web Service flow
ignores it.

Free tier sleeps after ~15 min idle, so the first call after a quiet spell
takes 30–50s. Fine for a demo, bad during a live walkthrough. Ping it on a
schedule or pay for the $7 tier if that matters.

## Wire into Bubble

API Connector → new API → new call:

- Method: `POST`
- URL: `https://your-app.onrender.com/analyze`
- Header: `x-api-key` = your `API_SECRET`
- Body type: JSON
- Body: `{ "text": "<text>" }`

Initialize with a sample headline. Bubble will map every field above, so
`predicted_impact_score` and `market_sentiment` become usable in workflows and
conditionals right next to your Alpaca buy/sell actions.

## Two notebook bugs this fixes

**The 404 on `analyze_review`.** OpenRouter dropped every provider for
`llama-4-scout` because none supported `response_format={"type":"json_object"}`,
leaving zero candidates. Here the JSON-mode request is wrapped in a try/except
that retries without it, and `parse_json_analysis` handles a plain-text reply.

**Bare model names.** `llama-4-scout` and `gemma-4-31b-it` only resolve through
the course proxy. Against OpenRouter directly you need the full slug, e.g.
`meta-llama/llama-3.1-8b-instruct`.
