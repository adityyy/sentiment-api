"""
Core logic lifted from Trade_f.ipynb.
No Colab, no pandas, no plotting - just the two functions the API needs.
"""

import os
import re
import json
from textblob import TextBlob
from openai import OpenAI

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Full OpenRouter slugs. Bare names like "llama-4-scout" return 404 outside
# the course proxy, which is what broke the notebook run.
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"

SYSTEM_PROMPT = """
You are an AI financial analyst and trader designed to extract actionable intelligence from news, blogs, and market commentaries. Your task is to extract the following information from the provided text and return it as a JSON object:

1.  **Action_Required**: Determine if immediate trading attention/action is needed. Return "Yes" if there is a severe warning, regulatory threat, or major trading signal. Otherwise, return "No".
2.  **Market_Sentiment**: Classify the overall market sentiment of the text as "Positive", "Negative", or "Neutral".
3.  **Mentions_Macro_Risks**: Indicate if the text explicitly mentions macroeconomic issues (inflation, interest rates, etc.). Return "Yes" or "No".
4.  **Mentions_Corporate_Strengths**: Indicate if the text explicitly highlights corporate strengths or strong earnings. Return "Yes" or "No".
5.  **Bullish_Intensity**: Measure the level of bullishness on a scale from 1 (very low) to 7 (extremely high).
6.  **Bearish_Intensity**: Measure the level of bearishness on a scale from 1 (no bearish signals) to 7 (extremely bearish).
7.  **Predicted_Impact_Score**: Rate the likely short-term price impact on a scale from -5 (strongly negative) to +5 (strongly positive). Return an integer.
8.  **Relevance_For_Trading**: Rate on a scale from 1 (not relevant) to 7 (extremely relevant) how useful this information is for making an immediate trading execution.
9.  **Primary_Ticker**: The single US-listed stock ticker this text is most directly about, in uppercase (e.g. "TSLA", "AAPL"). If the text is about the broad market, a sector, or no identifiable listed company, return an empty string "". Never guess a ticker you are not confident about, and never return an index like SPX or an ETF unless the text names it directly.
10. **Ticker_Confidence**: How confident you are in Primary_Ticker, on a scale from 1 (guessing) to 7 (the company is named explicitly). Return 1 if Primary_Ticker is empty.

Ensure your response is a valid JSON object containing only these keys and their corresponding extracted values. Do not include any other text or formatting outside the JSON object.
"""


def parse_json_analysis(json_string):
    """Strip markdown fences and <think> tags, then parse. From the notebook."""
    if not json_string:
        return {}

    json_string = str(json_string)
    json_string = re.sub(r"<think>.*?</think>", "", json_string, flags=re.DOTALL)

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_string, re.DOTALL)
    if match:
        json_string = match.group(1)

    json_string = json_string.strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block
        brace = re.search(r"\{.*\}", json_string, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def textblob_sentiment(text):
    """Free, instant, no API call. Good for a fast path in the UI."""
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        label = "Positive"
    elif polarity < 0:
        label = "Negative"
    else:
        label = "Neutral"
    return {"polarity": round(polarity, 4), "label": label}


def analyze_text(text, model_name=DEFAULT_MODEL):
    """The notebook's analyze_review, returning a flat dict for Bubble."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(text)},
    ]

    # Ask for JSON mode, but fall back if no provider for this model supports it.
    # That mismatch is what produced the 404 "Filter by Parameters removed..." error.
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(model=model_name, messages=messages)

    raw = resp.choices[0].message.content
    parsed = parse_json_analysis(raw)

    def num(key, default=None):
        try:
            return int(parsed.get(key))
        except (TypeError, ValueError):
            return default

    # Ticker: uppercase, letters only, 1-5 chars. Anything else becomes "".
    ticker = str(parsed.get("Primary_Ticker") or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{1,5}", ticker):
        ticker = ""

    # Flat keys. Bubble's API Connector handles flat JSON far better than nested.
    return {
        "ticker": ticker,
        "ticker_confidence": num("Ticker_Confidence", 1),
        "action_required": parsed.get("Action_Required", "No"),
        "market_sentiment": parsed.get("Market_Sentiment", "Neutral"),
        "mentions_macro_risks": parsed.get("Mentions_Macro_Risks", "No"),
        "mentions_corporate_strengths": parsed.get("Mentions_Corporate_Strengths", "No"),
        "bullish_intensity": num("Bullish_Intensity", 0),
        "bearish_intensity": num("Bearish_Intensity", 0),
        "predicted_impact_score": num("Predicted_Impact_Score", 0),
        "relevance_for_trading": num("Relevance_For_Trading", 0),
        "model": model_name,
        "parsed_ok": bool(parsed),
        "raw": raw if not parsed else "",
    }
