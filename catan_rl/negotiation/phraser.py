"""TradeOffer -> one or two sentences of table talk, with a faithfulness gate.

The bot's terms are fixed before the phraser runs. The LLM only writes
words, and the words are checked: the phrase is parsed back with the same
Parser, and it is used only if the parse equals the offer exactly and adds
no promises. Otherwise the deterministic template is used. So the bot can
never say something its decider did not decide.

    from catan_rl.negotiation.phraser import Phraser
    ph = Phraser()
    out = ph.phrase(offer, reasons={"need": "ORE", "building": "city", "vp_to_win": 1},
                    hand=bot_hand)
    out.text            # what to show
    out.used_fallback   # True if the gate rejected the LLM's phrasing

Known limitation, worth stating in the report: the gate uses the same model
family to parse as to phrase, so a misreading both directions share would
pass. E2 therefore also logs the phrases themselves for spot checks.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .llm_client import LLMClient
from .parser import Parser
from .schema import SEATS, Clarify, TradeOffer

PHRASER_VERSION = "ph1"
BUILDINGS = ("road", "settlement", "city", "development card")

SYSTEM_PROMPT = """You are a Settlers of Catan player making a trade offer at the table. Write the offer as one or two short, natural sentences in the first person.

Hard rules:
- State exactly the cards in the offer: the same resources and the same counts, nothing added or left out. Always write counts as numbers or "a"/"one".
- If the offer is addressed to a player, address them by that name. If it is open, address the table.
- You may give at most one short reason, and only from the facts you are given. Do not mention any other player's cards.
- No promises, threats, favours, or future deals. No questions other than asking whether they accept.
- Output only the sentences, no quotes or labels."""


def first_person_template(offer: TradeOffer) -> str:
    """The fallback. Deterministic, and parses back to the same terms."""
    def fmt(items):
        return " and ".join(f"{n} {r.lower()}" for r, n in items)
    who = f"{offer.to_seat}, " if offer.to_seat else ""
    return f"{who}I'll give {fmt(offer.give)} for {fmt(offer.get)}."


def describe_offer(offer: TradeOffer) -> str:
    def fmt(items):
        return ", ".join(f"{n} {r.lower()}" for r, n in items)
    to = offer.to_seat or "the whole table (open offer)"
    return f"You give: {fmt(offer.give)}. You receive: {fmt(offer.get)}. Addressed to: {to}."


def describe_reasons(reasons: Optional[Dict]) -> str:
    """Only whitelisted, bot-side facts reach the prompt."""
    if not reasons:
        return "No reason facts available; do not give a reason."
    facts = []
    need = reasons.get("need")
    if need:
        facts.append(f"you are short of {str(need).lower()}")
    building = reasons.get("building")
    if building in BUILDINGS:
        facts.append(f"you are saving for a {building}")
    vp = reasons.get("vp_to_win")
    if isinstance(vp, int) and vp > 0:
        facts.append(f"you are {vp} victory point{'s' if vp != 1 else ''} from winning")
    return ("Facts you may use: " + "; ".join(facts) + ".") if facts else \
        "No reason facts available; do not give a reason."


@dataclass
class PhraseResult:
    text: str
    used_fallback: bool
    gate: str                      # "pass" or why it failed
    llm_text: Optional[str] = None
    parsed: Optional[str] = None
    tokens: int = 0                # phrase call + gate parse
    latency_s: float = 0.0


class Phraser:
    def __init__(self, client: Optional[LLMClient] = None,
                 parser: Optional[Parser] = None):
        self.client = client or LLMClient()
        self.parser = parser or Parser(client=self.client)

    def check(self, text: str, offer: TradeOffer,
              hand: Optional[Dict[str, int]] = None,
              seats: Tuple[str, ...] = SEATS) -> Tuple[str, Optional[str], int, float]:
        """Round-trip gate. Returns (verdict, parsed_repr, tokens, latency)."""
        t = self.parser.parse_traced(text, offer.from_seat, hand=hand, seats=seats)
        toks = t.prompt_tokens + t.completion_tokens
        r = t.result
        if not isinstance(r, TradeOffer):
            kind = "clarify" if isinstance(r, Clarify) else "rejected"
            return f"parse_{kind}", str(r), toks, t.latency_s
        if r.terms() != offer.terms():
            return "terms_mismatch", str(r), toks, t.latency_s
        if r.dropped_clauses:
            return "added_clause", str(r), toks, t.latency_s
        return "pass", str(r), toks, t.latency_s

    def phrase(self, offer: TradeOffer, reasons: Optional[Dict] = None,
               hand: Optional[Dict[str, int]] = None,
               seats: Tuple[str, ...] = SEATS) -> PhraseResult:
        fallback = first_person_template(offer)
        if not self.client.available():
            return PhraseResult(fallback, True, "no_llm")

        user = f"{describe_offer(offer)}\n{describe_reasons(reasons)}"
        reply = self.client.chat(SYSTEM_PROMPT, user, max_tokens=1024)
        tokens = reply.prompt_tokens + reply.completion_tokens
        if reply.text is None:
            return PhraseResult(fallback, True, "transport", tokens=tokens,
                                latency_s=reply.latency_s)
        text = reply.text.strip().strip('"').strip()
        if len(text) > 300:
            return PhraseResult(fallback, True, "too_long", llm_text=text,
                                tokens=tokens, latency_s=reply.latency_s)

        verdict, parsed, ptoks, plat = self.check(text, offer, hand, seats)
        ok = verdict == "pass"
        return PhraseResult(text if ok else fallback, not ok, verdict,
                            llm_text=text, parsed=parsed, tokens=tokens + ptoks,
                            latency_s=reply.latency_s + plat)
