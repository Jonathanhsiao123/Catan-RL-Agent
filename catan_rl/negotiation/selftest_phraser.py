"""Offline checks for the phraser's round-trip gate.

    python -m catan_rl.negotiation.selftest_phraser

Two fake clients: one plays the phrasing model, one plays the parsing model.
That isolates the gate logic: every fallback decision is tested without
spending a token.
"""

import sys

from .llm_client import LLMReply
from .parser import Parser
from .phraser import Phraser, describe_reasons, first_person_template
from .schema import TradeOffer

OFFER = TradeOffer.make("WHITE", "RED", {"WHEAT": 1}, {"ORE": 1})
HAND = {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 2, "ORE": 0}
GOOD = '{"kind":"offer","to":"RED","give":{"wheat":1},"get":{"ore":1},"dropped_clauses":[]}'


class Fake:
    model = "fake"

    def __init__(self, text):
        self.text = text

    def available(self):
        return True

    def chat(self, system, user, max_tokens=300, json_mode=False):
        return LLMReply(self.text, "fake", 5, 5, 0.0)


def make(phrase_text, parse_json):
    return Phraser(client=Fake(phrase_text), parser=Parser(client=Fake(parse_json)))


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def faithful_phrase_is_used():
    out = make("RED, a wheat for your ore?", GOOD).phrase(OFFER, hand=HAND)
    assert not out.used_fallback and out.gate == "pass", out
    assert out.text == "RED, a wheat for your ore?"


@case
def wrong_count_falls_back():
    bad = GOOD.replace('"ore":1', '"ore":2')
    out = make("RED, a wheat for 2 ore?", bad).phrase(OFFER, hand=HAND)
    assert out.used_fallback and out.gate == "terms_mismatch", out
    assert out.text == first_person_template(OFFER)


@case
def wrong_recipient_falls_back():
    out = make("Anyone, wheat for ore?", GOOD.replace('"RED"', "null")).phrase(OFFER, hand=HAND)
    assert out.used_fallback and out.gate == "terms_mismatch", out


@case
def added_promise_falls_back():
    promise = GOOD.replace('"dropped_clauses":[]', '"dropped_clauses":["I won\'t rob you"]')
    out = make("RED, wheat for ore and I won't rob you.", promise).phrase(OFFER, hand=HAND)
    assert out.used_fallback and out.gate == "added_clause", out


@case
def unparseable_phrase_falls_back():
    out = make("Hmm, maybe something?", '{"kind":"clarify","question":"what?"}').phrase(OFFER, hand=HAND)
    assert out.used_fallback and out.gate == "parse_clarify", out


@case
def transport_failure_falls_back():
    out = make(None, GOOD).phrase(OFFER, hand=HAND)
    assert out.used_fallback and out.gate == "transport", out


@case
def overlong_phrase_falls_back():
    out = make("x" * 400, GOOD).phrase(OFFER, hand=HAND)
    assert out.used_fallback and out.gate == "too_long", out


@case
def template_shape():
    assert first_person_template(OFFER) == "RED, I'll give 1 wheat for 1 ore."
    open_offer = TradeOffer.make("WHITE", None, {"WOOD": 1, "BRICK": 1}, {"ORE": 1})
    assert first_person_template(open_offer) == "I'll give 1 brick and 1 wood for 1 ore."


@case
def reasons_are_whitelisted():
    s = describe_reasons({"need": "ORE", "building": "city", "vp_to_win": 1,
                          "opponent_hand": "RED has 3 ore"})
    assert "ore" in s and "city" in s and "1 victory point " in s
    assert "RED" not in s, s
    assert "do not give a reason" in describe_reasons({"building": "castle"})


def main():
    failed = 0
    for fn in CASES:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"phraser offline: {len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
