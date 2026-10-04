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
You are a sell-side equity analyst scoring news for a short-term trading desk.

THE CENTRAL RULE: you are predicting what the SHARE PRICE will do, not whether
the news is good or bad for people. These often point in opposite directions.
Judge the news the way the market will price it, not the way a newspaper reader
would feel about it.

Cases where market logic diverges from everyday intuition:
- Layoffs, restructuring and cost cuts are frequently POSITIVE for the stock.
  They lower the cost base and expand margins. A large layoff paired with a
  credible growth story (AI investment, refocusing) is usually received well.
- An earnings "beat" paired with lowered forward guidance is usually NEGATIVE.
  Markets price the future, not the quarter that just closed.
- A buyback, dividend raise or activist stake is POSITIVE even inside otherwise
  bad news.
- Regulatory investigations, accounting probes, safety recalls, executive
  departures under a cloud, and debt covenant breaches are strongly NEGATIVE.
- News already widely expected is close to NEUTRAL even when dramatic, because
  it is already in the price.

WORK IN THIS ORDER. Produce "Reasoning" FIRST, then make every other field
consistent with it. Do not score the fields independently.

Return a JSON object with exactly these keys, in this order:

1.  **Reasoning**: One or two sentences stating what a trader would expect this
    news to do to the share price, and why. Name the mechanism (margins,
    guidance, regulatory risk, demand). This field drives all the others.
2.  **Predicted_Impact_Score**: Integer from -5 (strongly negative) to +5
    (strongly positive) for short-term price impact. Must follow directly from
    Reasoning. Use 0 when the news is immaterial or already priced in.
3.  **Market_Sentiment**: "Positive", "Negative", or "Neutral". Must agree in
    sign with Predicted_Impact_Score: positive score means "Positive",
    negative means "Negative", 0 means "Neutral".
4.  **Bullish_Intensity**: 1 (none) to 7 (extreme). If Predicted_Impact_Score
    is negative, this must be 3 or lower.
5.  **Bearish_Intensity**: 1 (none) to 7 (extreme). If Predicted_Impact_Score
    is positive, this must be 3 or lower.
6.  **Action_Required**: "Yes" only if a desk would act within the session:
    that means absolute Predicted_Impact_Score is 3 or more AND
    Relevance_For_Trading is 5 or more. Otherwise "No".
7.  **Relevance_For_Trading**: 1 (not actionable) to 7 (immediately
    actionable). Score LOW for opinion pieces, broad macro commentary with no
    named company, old news, and vague speculation. Score HIGH only for
    specific, new, company-level facts.
8.  **Mentions_Macro_Risks**: "Yes" or "No" - does the text explicitly mention
    inflation, interest rates, recession, tariffs or similar?
9.  **Mentions_Corporate_Strengths**: "Yes" or "No" - does the text explicitly
    highlight strong earnings, margin expansion, market share gains or similar?

WORKED EXAMPLES

Text: "IBM cuts 10,000 roles as it redirects spending toward AI infrastructure."
{"Reasoning": "Headcount reduction lowers the cost base and the AI redirection
gives a growth narrative; the market typically reads this as margin-accretive
rather than as distress.", "Predicted_Impact_Score": 2, "Market_Sentiment":
"Positive", "Bullish_Intensity": 5, "Bearish_Intensity": 2, "Action_Required":
"No", "Relevance_For_Trading": 5, "Mentions_Macro_Risks": "No",
"Mentions_Corporate_Strengths": "No"}

Text: "Chipmaker beats Q3 estimates but cuts full-year revenue guidance on
weaker data center demand."
{"Reasoning": "The beat is backward-looking while the guidance cut signals
deteriorating forward demand, and markets price the forward number.",
"Predicted_Impact_Score": -4, "Market_Sentiment": "Negative",
"Bullish_Intensity": 2, "Bearish_Intensity": 6, "Action_Required": "Yes",
"Relevance_For_Trading": 7, "Mentions_Macro_Risks": "No",
"Mentions_Corporate_Strengths": "Yes"}

Text: "Analysts debate whether equities look expensive heading into next year."
{"Reasoning": "Generic opinion commentary with no company, no new fact and no
catalyst, so there is nothing for a desk to price.", "Predicted_Impact_Score":
0, "Market_Sentiment": "Neutral", "Bullish_Intensity": 2,
"Bearish_Intensity": 2, "Action_Required": "No", "Relevance_For_Trading": 1,
"Mentions_Macro_Risks": "No", "Mentions_Corporate_Strengths": "No"}

Return only the JSON object. No prose, no markdown fences, no commentary.
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
    # temperature=0 makes the same headline score the same way every time, which
    # matters for a demo and for measuring the model against a labelled set.
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0
        )

    raw = resp.choices[0].message.content
    parsed = parse_json_analysis(raw)

    def num(key, default=None):
        try:
            return int(parsed.get(key))
        except (TypeError, ValueError):
            return default

    impact = max(-5, min(5, num("Predicted_Impact_Score", 0) or 0))
    relevance = max(1, min(7, num("Relevance_For_Trading", 1) or 1))
    bullish = max(1, min(7, num("Bullish_Intensity", 1) or 1))
    bearish = max(1, min(7, num("Bearish_Intensity", 1) or 1))

    # The prompt asks the model to keep these fields consistent, but a small
    # model will still contradict itself (Neutral sentiment next to a -3 score).
    # Derive them from the impact score instead of trusting what came back, so
    # the response is never internally contradictory.
    if impact > 0:
        sentiment = "Positive"
        bearish = min(bearish, 3)
    elif impact < 0:
        sentiment = "Negative"
        bullish = min(bullish, 3)
    else:
        sentiment = "Neutral"

    action_required = "Yes" if abs(impact) >= 3 and relevance >= 5 else "No"

    # Flat keys. Bubble's API Connector handles flat JSON far better than nested.
    return {
        "reasoning": str(parsed.get("Reasoning", "")).strip(),
        "action_required": action_required,
        "market_sentiment": sentiment,
        "mentions_macro_risks": parsed.get("Mentions_Macro_Risks", "No"),
        "mentions_corporate_strengths": parsed.get("Mentions_Corporate_Strengths", "No"),
        "bullish_intensity": bullish,
        "bearish_intensity": bearish,
        "predicted_impact_score": impact,
        "relevance_for_trading": relevance,
        "model": model_name,
        "parsed_ok": bool(parsed),
        "raw": raw if not parsed else "",
    }
