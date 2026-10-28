"""Optional natural-language layer for trade negotiation.

This module is INERT unless a local LLM is already running. Nothing here is
required for the bot to play or trade: trade decisions are made by the
trained critic in trading.py, and this only ever translates them into words
(or translates a human's words into a structured offer).

Design rule, deliberate: the LLM never decides. It phrases and it parses.
The critic prices. That keeps decisions grounded in a value function trained
over tens of millions of steps, keeps training reproducible, and keeps the
per-game cost at zero.

Detection order (first hit wins):
  1. Luminal Cloud           $LUMINAL_API_KEY + $LUMINAL_BASE_URL
  2. $CATAN_LLM_URL          any OpenAI-compatible /v1 endpoint
                             (bearer key from $CATAN_LLM_API_KEY, optional)
  3. Ollama                  http://127.0.0.1:11434
  4. LM Studio               http://127.0.0.1:1234/v1
  5. llama.cpp server        http://127.0.0.1:8080/v1

Use Luminal's hosted API (OpenAI-compatible chat completions, bearer auth).
Copy .env.example to .env at the repo root (gitignored) and fill it in, or:
  export LUMINAL_API_KEY=...                       # never commit this
  export LUMINAL_BASE_URL=https://<your endpoint>/v1
  export LUMINAL_MODEL=<model id>                  # optional if /models lists one
  python -m catan_rl.llm_negotiator                # prints provider + test call
Remote calls get $CATAN_LLM_TIMEOUT seconds (default 15) instead of the 2s
local budget; on timeout or any error the layer still falls back silently.

Enable a local model on a machine that has one:
  ollama serve & ; ollama pull qwen2.5:7b-instruct
Then anything importing this module picks it up automatically. If nothing is
listening, available() is False and every method returns None, so callers
fall back to the structured text that trading.py already produces.

    neg = get_negotiator()
    text = neg.describe_offer(offer, context) or str(offer)   # graceful

TODO (stage 3), in rough order of value:
  * describe_offer: phrase the critic's chosen offer as table talk, with a
    reason drawn from real state (what the bot is short of, what it is
    building). Prompt must receive only public information plus the bot's
    own hand, never another player's hand.
  * parse_human_offer: turn "2 wood for your ore" into a structured Offer.
    Must validate against the legal vocabulary in trading.py and reject
    anything unaffordable, since an LLM will happily invent cards.
  * respond_to_human: given the critic's accept/reject and its gain, write
    the reply. The verdict is fixed before the LLM sees it; the LLM only
    writes the sentence.
  * explain_move: post-hoc explanation of any action for demos and teaching.
  * Wire into lobby.py: a chat box per seat, human free text -> parse ->
    critic verdict -> phrased reply. Keep a structured "offer card" in the
    UI as the source of truth so a hallucinated sentence can never move
    cards; the card is what executes.
  * Guard rails to add with the first real call: hard timeout (2s) with
    silent fallback to structured text, output length cap, and a check that
    any parsed offer round-trips to a legal Offer before it is shown.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def _load_dotenv(path: Path = Path(__file__).resolve().parents[1] / ".env"):
    """Minimal KEY=VALUE loader for the repo-root .env; real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, val.strip().strip('"').strip("'"))


_load_dotenv()

_TIMEOUT = 1.0         # detection only; keep startup snappy
_CALL_TIMEOUT = 2.0     # per generation call, fall back on timeout
_REMOTE_TIMEOUT = float(os.environ.get("CATAN_LLM_TIMEOUT", "15"))
_cached = None


def _auth_headers(api_key: Optional[str]) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _probe(url: str, timeout=_TIMEOUT, api_key: Optional[str] = None) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers=_auth_headers(api_key))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception:
        return None


def _remote(kind: str, base: str, api_key: Optional[str],
            model: Optional[str]) -> Optional[dict]:
    """OpenAI-compatible remote endpoint; model from env or the first /models entry."""
    base = base.rstrip("/")
    if not model:
        info = _probe(f"{base}/models", timeout=_REMOTE_TIMEOUT, api_key=api_key)
        data = (info or {}).get("data") or []
        model = data[0].get("id") if data else None
    if model:
        return dict(kind=kind, base_url=base, model=model, api_key=api_key,
                    timeout=_REMOTE_TIMEOUT)
    return None


def detect_local_llm() -> Optional[dict]:
    """Return {kind, base_url, model, ...} for the first LLM found, else None."""
    lum_key = os.environ.get("LUMINAL_API_KEY")
    lum_url = os.environ.get("LUMINAL_BASE_URL")
    if lum_key and lum_url:
        cfg = _remote("luminal", lum_url, lum_key, os.environ.get("LUMINAL_MODEL"))
        if cfg:
            return cfg

    env_url = os.environ.get("CATAN_LLM_URL")
    if env_url:
        cfg = _remote("openai-compatible", env_url,
                      os.environ.get("CATAN_LLM_API_KEY"),
                      os.environ.get("CATAN_LLM_MODEL"))
        if cfg:
            return cfg

    tags = _probe("http://127.0.0.1:11434/api/tags")
    if tags:
        models = tags.get("models") or []
        if models:
            preferred = os.environ.get("CATAN_LLM_MODEL")
            names = [m.get("name") for m in models if m.get("name")]
            model = preferred if preferred in names else (names[0] if names else None)
            if model:
                return dict(kind="ollama", base_url="http://127.0.0.1:11434",
                            model=model)

    for base in ("http://127.0.0.1:1234/v1", "http://127.0.0.1:8080/v1"):
        info = _probe(f"{base}/models")
        if info:
            data = info.get("data") or []
            if data:
                return dict(kind="openai-compatible", base_url=base,
                            model=data[0].get("id"))
    return None


class LLMNegotiator:
    """Inert when no local LLM is present: every method returns None."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config

    def available(self) -> bool:
        return self.config is not None

    def describe(self) -> str:
        if not self.available():
            return "no LLM detected (natural-language layer off)"
        return f"{self.config['kind']} / {self.config['model']}"

    # ---- generation primitive (used by the TODO methods above) ----
    def _generate(self, system: str, user: str, max_tokens=120) -> Optional[str]:
        if not self.available():
            return None
        cfg = self.config
        try:
            if cfg["kind"] == "ollama":
                url = f"{cfg['base_url']}/api/chat"
                payload = dict(model=cfg["model"], stream=False,
                               messages=[dict(role="system", content=system),
                                         dict(role="user", content=user)],
                               options=dict(num_predict=max_tokens))
            else:
                url = f"{cfg['base_url']}/chat/completions"
                payload = dict(model=cfg["model"], max_tokens=max_tokens,
                               messages=[dict(role="system", content=system),
                                         dict(role="user", content=user)])
            headers = {"Content-Type": "application/json",
                       **_auth_headers(cfg.get("api_key"))}
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers=headers, method="POST")
            timeout = cfg.get("timeout", _CALL_TIMEOUT)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            if cfg["kind"] == "ollama":
                return (data.get("message") or {}).get("content")
            return data["choices"][0]["message"]["content"]
        except Exception:
            return None  # any failure: caller falls back to structured text

    # ---- stage 3 surface: implemented as no-ops on purpose ----
    def describe_offer(self, offer, context: Optional[dict] = None) -> Optional[str]:
        """TODO(stage 3): phrase `offer` as table talk. Returns None for now."""
        return None

    def parse_human_offer(self, text: str, hand: dict) -> Optional[object]:
        """TODO(stage 3): parse free text into a validated trading.Offer."""
        return None

    def respond_to_human(self, offer, verdict: bool,
                         gain: Optional[float] = None) -> Optional[str]:
        """TODO(stage 3): write the reply for an already-decided verdict."""
        return None

    def explain_move(self, action_label: str,
                     context: Optional[dict] = None) -> Optional[str]:
        """TODO(stage 3): post-hoc explanation for demos."""
        return None


def get_negotiator(force_redetect: bool = False) -> LLMNegotiator:
    """Cached singleton: detection runs at most once per process."""
    global _cached
    if _cached is None or force_redetect:
        _cached = LLMNegotiator(detect_local_llm())
    return _cached


if __name__ == "__main__":
    neg = get_negotiator()
    print("LLM:", neg.describe())
    if neg.available():
        out = neg._generate("You are terse.", "Say OK.")
        print("test generation:", out if out else "(call failed; layer stays off)")
