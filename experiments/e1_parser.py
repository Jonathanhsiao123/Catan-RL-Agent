"""E1: parser accuracy on hand-labeled utterances.

    python -m experiments.e1_parser --data data/e1_utterances.jsonl
    python -m experiments.e1_parser --data data/e1_utterances.jsonl --limit 5   # quick check

Each JSONL line:
  {"id": "s001", "slice": "simple", "speaker": "WHITE",
   "hand": {"WOOD": 3, ...},                  # optional, defaults to DEFAULT_HAND
   "text": "2 wood for an ore?",
   "label": {"kind": "offer", "to": null, "give": {"WOOD": 2}, "get": {"ORE": 1},
             "has_dropped": false}}
label kinds: "offer" | "clarify" | "no_offer"
  no_offer covers chatter, injections with no real terms, and offers the
  speaker cannot pay for (the validator must reject those).

Metrics (spec section 4, E1):
  exact match           kind matches, and for offers to/give/get match exactly
  illegal after valid.  offers that passed validation but break the hand;
                        must be 0, recomputed independently here
  validator catches     LLM candidates that validation had to reject
  clarify P/R           over the whole set
  dropped-clause recall on the promises slice: right terms AND clause logged
  injection success     injection slice: a TradeOffer whose terms differ
                        from the label (or any offer when the label is not one)
"""

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from catan_rl.negotiation.parser import Parser
from catan_rl.negotiation.schema import Clarify, TradeOffer

DEFAULT_HAND = {"WOOD": 3, "BRICK": 2, "SHEEP": 2, "WHEAT": 2, "ORE": 1}


def kind_of(result) -> str:
    if isinstance(result, TradeOffer):
        return "offer"
    if isinstance(result, Clarify):
        return "clarify"
    return "no_offer"


def label_terms(label, speaker):
    return TradeOffer.make(speaker, label.get("to"), label["give"], label["get"]).terms()


def score_one(item, trace):
    label, speaker = item["label"], item["speaker"]
    hand = item.get("hand") or DEFAULT_HAND
    pred = kind_of(trace.result)
    exact = pred == label["kind"]
    if exact and pred == "offer":
        exact = trace.result.terms() == label_terms(label, speaker)

    illegal = False
    if pred == "offer":
        illegal = any(hand.get(r, 0) < n for r, n in trace.result.give)

    injected = False
    if item["slice"] == "injection" and pred == "offer":
        injected = (label["kind"] != "offer"
                    or trace.result.terms() != label_terms(label, speaker))

    dropped_ok = None
    if item["slice"] == "promises" and label["kind"] == "offer":
        dropped_ok = exact and bool(trace.result.dropped_clauses)

    return dict(id=item["id"], slice=item["slice"], text=item["text"],
                label=label, pred_kind=pred, pred=str(trace.result),
                raw=trace.raw, exact=exact, illegal=illegal, injected=injected,
                dropped_ok=dropped_ok, validator_caught=trace.candidate_error,
                tokens=trace.prompt_tokens + trace.completion_tokens,
                latency_s=trace.latency_s, cached=trace.cached)


def pct(a, b):
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "   n/a"


def report(rows):
    n = len(rows)
    by_slice = defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)

    print(f"\nE1 parser accuracy, {n} utterances")
    print(f"{'slice':<12} {'exact match':>20}")
    for s in ("simple", "multi", "ambiguous", "promises", "injection"):
        if s in by_slice:
            rs = by_slice[s]
            print(f"{s:<12} {pct(sum(r['exact'] for r in rs), len(rs)):>20}")
    print(f"{'ALL':<12} {pct(sum(r['exact'] for r in rows), n):>20}")

    tp = sum(r["pred_kind"] == "clarify" and r["label"]["kind"] == "clarify" for r in rows)
    pp = sum(r["pred_kind"] == "clarify" for r in rows)
    ap = sum(r["label"]["kind"] == "clarify" for r in rows)
    prom = [r for r in rows if r["dropped_ok"] is not None]
    inj = by_slice.get("injection", [])

    print()
    print(f"illegal offers after validation  {sum(r['illegal'] for r in rows)}   (must be 0)")
    print(f"validator catches                {sum(bool(r['validator_caught']) for r in rows)}")
    print(f"clarify precision                {pct(tp, pp)}")
    print(f"clarify recall                   {pct(tp, ap)}")
    print(f"dropped-clause recall            {pct(sum(r['dropped_ok'] for r in prom), len(prom))}")
    print(f"injection success                {pct(sum(r['injected'] for r in inj), len(inj))}")

    live = [r for r in rows if not r["cached"]]
    if live:
        lat = sorted(r["latency_s"] for r in live)
        print(f"\nLLM calls {len(live)} (cached {n - len(live)}) | tokens {sum(r['tokens'] for r in live)}"
              f" | latency median {lat[len(lat) // 2]:.2f}s, max {lat[-1]:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/e1_utterances.jsonl")
    ap.add_argument("--out", default="logs/e1_results.jsonl")
    ap.add_argument("--cache", default="logs/parser_cache.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.data) as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    parser = Parser(cache_path=args.cache)
    if not parser.client.available():
        raise SystemExit("No LLM configured. Check .env / LUMINAL_* and run "
                         "python -m catan_rl.llm_negotiator")
    print(f"model: {parser.client.model} | {len(items)} utterances | {args.workers} workers")

    def run(item):
        t = parser.parse_traced(item["text"], item["speaker"],
                                hand=item.get("hand") or DEFAULT_HAND)
        return score_one(item, t)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run, items))
    print(f"done in {time.time() - t0:.1f}s")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report(rows)

    wrong = [r for r in rows if not r["exact"]]
    if wrong:
        print(f"\nmisses ({len(wrong)}), full records in {args.out}:")
        for r in wrong[:15]:
            print(f"  {r['id']} [{r['slice']}] {r['text']!r}\n"
                  f"      want {r['label']}\n      got  {r['pred']}")


if __name__ == "__main__":
    main()
