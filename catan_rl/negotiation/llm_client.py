"""Thin chat client for the negotiation layer.

Reuses the provider detection in catan_rl.llm_negotiator (Luminal, then any
OpenAI-compatible URL, then local servers), so the .env you already have
works unchanged. Differences from LLMNegotiator._generate, all needed for
parsing:

  * temperature 0, so the same utterance parses the same way
  * asks for a JSON object response when the provider supports it, and
    retries without that field if the provider rejects it
  * strips <think>...</think> blocks that reasoning models emit
  * returns token usage, for the spec's cost accounting
  * raises nothing: failures come back as LLMReply(text=None, error=...)
"""

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from ..llm_negotiator import _auth_headers, get_negotiator

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMReply:
    text: Optional[str]
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None


class LLMClient:
    def __init__(self, config: Optional[dict] = None):
        self.config = config if config is not None else get_negotiator().config
        self._json_mode = True       # turned off after the first 400

    def available(self) -> bool:
        return self.config is not None

    @property
    def model(self) -> Optional[str]:
        return (self.config or {}).get("model")

    def chat(self, system: str, user: str, max_tokens: int = 300,
             json_mode: bool = False) -> LLMReply:
        if not self.available():
            return LLMReply(None, error="no LLM configured")
        cfg = self.config
        if cfg["kind"] == "ollama":
            return self._ollama(system, user, max_tokens, json_mode)

        payload = dict(model=cfg["model"], max_tokens=max_tokens, temperature=0,
                       messages=[dict(role="system", content=system),
                                 dict(role="user", content=user)])
        if json_mode and self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        reply = self._post(f"{cfg['base_url']}/chat/completions", payload)
        if reply.error and "HTTP 400" in reply.error and "response_format" in payload:
            # provider does not support JSON mode; the prompt still asks for JSON
            self._json_mode = False
            payload.pop("response_format")
            reply = self._post(f"{cfg['base_url']}/chat/completions", payload)
        return reply

    def _ollama(self, system, user, max_tokens, json_mode) -> LLMReply:
        cfg = self.config
        payload = dict(model=cfg["model"], stream=False,
                       messages=[dict(role="system", content=system),
                                 dict(role="user", content=user)],
                       options=dict(num_predict=max_tokens, temperature=0))
        if json_mode:
            payload["format"] = "json"
        return self._post(f"{cfg['base_url']}/api/chat", payload, ollama=True)

    def _post(self, url: str, payload: dict, ollama: bool = False) -> LLMReply:
        cfg = self.config
        headers = {"Content-Type": "application/json",
                   **_auth_headers(cfg.get("api_key"))}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=cfg.get("timeout", 15)) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            return LLMReply(None, cfg["model"], latency_s=time.time() - t0,
                            error=f"HTTP {e.code}: {body}")
        except Exception as e:  # timeout, DNS, bad JSON
            return LLMReply(None, cfg["model"], latency_s=time.time() - t0,
                            error=f"{type(e).__name__}: {e}")
        dt = time.time() - t0

        if ollama:
            text = (data.get("message") or {}).get("content")
            pt, ct = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        else:
            text = data["choices"][0]["message"].get("content")
            usage = data.get("usage") or {}
            pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        if text:
            text = _THINK.sub("", text).strip()
        return LLMReply(text, cfg["model"], pt, ct, dt)
