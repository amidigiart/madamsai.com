# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import os
import time

import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters import OpenAICompatAdapter
from crisis import detect_crisis, crisis_response
from dual_engine import DualEngine

GROK_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.5")
GROK_KEY = os.getenv("GROK_API_KEY", "")

DEEPSEEK_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MOCK_MODE = os.getenv("MADAMSAI_MOCK", "false").lower() == "true"

STRIPE_SECRET = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
SITE_URL = os.getenv("SITE_URL", "https://madamsai.com")

stripe.api_key = STRIPE_SECRET

ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "https://madamsai.com,http://madamsai.com,http://localhost:8000"
).split(",")

SYSTEM_PROMPT = (
    "You are MadamsAI — a sharp, supportive, and empowering AI companion for women "
    "entrepreneurs, founders, and career professionals. You are an AI system (not a "
    "human) and say so if asked. You speak in the user's language (detect from their "
    "message).\n\n"
    "YOUR EXPERTISE:\n"
    "- Business launch: ideation, validation, lean startup methodology, MVP design, "
    "business model canvas, value proposition design, market-founder fit\n"
    "- Funding & finance: bootstrapping strategies, grant programs for women founders "
    "(EU, national), angel investing, pitch deck crafting, revenue models, cash flow "
    "management, pricing strategies\n"
    "- Leadership & management: team building, hiring frameworks, delegation, "
    "conflict resolution, imposter syndrome management, executive presence, "
    "board communication\n"
    "- Career growth: negotiation tactics (salary, contracts, equity), personal "
    "branding, networking strategies, career pivots, mentorship frameworks, "
    "visibility without burnout\n"
    "- Work-life integration: boundary setting, energy management, sustainable "
    "productivity, parental leave planning, return-to-work strategies, "
    "managing guilt narratives\n"
    "- Marketing & sales: personal brand building, content strategy, social media "
    "for founders, community building, B2B/B2C sales, storytelling, PR basics\n"
    "- Legal & admin: business registration (EU focus), contracts basics, IP "
    "protection, employment law essentials, GDPR for small businesses\n\n"
    "OUTPUT FORMAT:\n"
    "- Use 💎/🚀/💡/📊/✅/⚡ icons for visual clarity\n"
    "- Be direct and actionable — no fluff\n"
    "- Structure advice as concrete next steps\n"
    "- When relevant, mention specific programs, tools, or resources\n\n"
    "CRITICAL RULES:\n"
    "- Be empowering without being patronizing. Treat users as capable professionals.\n"
    "- You are NOT a lawyer, financial advisor, or therapist.\n"
    "- Never fabricate statistics, funding amounts, or program details.\n"
    "- For mental health topics (burnout, anxiety), provide frameworks but always "
    "recommend professional support.\n"
    "- Be inclusive — 'women entrepreneurs' includes all who identify as women.\n"
    "- Acknowledge systemic barriers honestly without being discouraging.\n"
    "- For legal/financial specifics, always recommend consulting a qualified professional."
)

RATE_LIMIT: dict[str, list[float]] = {}


def _rate_ok(ip: str, max_req: int = 20, window: float = 60.0) -> bool:
    now = time.time()
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT.get(ip, []) if now - t < window] + [now]
    return len(RATE_LIMIT[ip]) <= max_req


def _sign(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _build_engine() -> DualEngine | None:
    if MOCK_MODE or not GROK_KEY or not DEEPSEEK_KEY:
        return None
    grok = OpenAICompatAdapter(
        base_url=GROK_URL, model=GROK_MODEL, api_key=GROK_KEY,
        temperature=0.4, max_tokens=600, timeout=30, name="grok",
    )
    deepseek = OpenAICompatAdapter(
        base_url=DEEPSEEK_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_KEY,
        temperature=0.4, max_tokens=600, timeout=30, name="deepseek",
    )
    return DualEngine(grok, deepseek, system_prompt=SYSTEM_PROMPT, threshold=0.45)


ENGINE = _build_engine()

app = FastAPI(title="MadamsAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    locale: str = "EN"


class ChatResponse(BaseModel):
    response: str
    engine: str
    decision: str
    certified: bool
    signature: str
    is_crisis_response: bool
    concordance: float | None = None
    latency_s: float = 0.0


@app.get("/")
def root():
    return {"service": "MadamsAI API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "dual (Grok + DeepSeek)" if ENGINE else "mock",
        "grok_configured": bool(GROK_KEY),
        "deepseek_configured": bool(DEEPSEEK_KEY),
        "stripe_configured": bool(STRIPE_SECRET and STRIPE_PRICE_ID),
        "mock_mode": MOCK_MODE or not ENGINE,
    }


@app.post("/companion/madams/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        raise HTTPException(429, "Too many requests — please wait a moment")

    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    if len(msg) > 2000:
        raise HTTPException(400, "Message too long (max 2000 chars)")

    locale = req.locale.upper()[:2] if req.locale else "EN"

    crisis = detect_crisis(msg)
    if crisis.is_crisis:
        resp = crisis_response(locale.lower())
        return ChatResponse(
            response=resp, engine="safety-layer", decision="crisis-intercept",
            certified=True, signature=_sign(resp), is_crisis_response=True,
        )

    if ENGINE:
        result = ENGINE.ask(msg)
        return ChatResponse(
            response=result.reply, engine=result.engine, decision=result.decision,
            certified=True, signature=_sign(result.reply), is_crisis_response=False,
            concordance=result.concordance, latency_s=result.latency_s,
        )

    mock = "Great question! In production, I'd cross-verify this with two AI engines. Check back soon for verified insights."
    return ChatResponse(
        response=mock, engine="mock", decision="mock",
        certified=False, signature=_sign(mock), is_crisis_response=False,
    )


@app.post("/create-checkout-session")
def create_checkout():
    if not STRIPE_SECRET or not STRIPE_PRICE_ID:
        raise HTTPException(503, "Payment system not configured yet")
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=SITE_URL + "?payment=success",
            cancel_url=SITE_URL + "?payment=cancelled",
        )
    except stripe.error.StripeError as e:
        raise HTTPException(502, f"Payment error: {e.user_message or str(e)}")
    return {"url": session.url}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(400, "Invalid webhook signature")
    if event["type"] == "checkout.session.completed":
        print(f"[STRIPE] New sub: {event['data']['object'].get('customer_email', '?')}")
    elif event["type"] == "customer.subscription.deleted":
        print(f"[STRIPE] Sub cancelled: {event['data']['object'].get('id')}")
    return {"received": True}


@app.post("/gdpr/data-request")
def gdpr_data():
    return {
        "message": "MadamsAI does not store chat messages or queries. Processing is stateless. "
                   "For Stripe subscription data, email privacy@madamsai.com.",
        "data_stored": "none",
    }


@app.delete("/gdpr/delete")
def gdpr_delete():
    return {
        "message": "No personal data held server-side. "
                   "For Stripe data deletion, email privacy@madamsai.com.",
        "status": "no_data_held",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
