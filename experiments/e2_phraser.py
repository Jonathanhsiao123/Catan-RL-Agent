"""E2: phraser faithfulness.

    python -m experiments.e2_phraser --n 50            # start here
    python -m experiments.e2_phraser --n 500           # the spec's full run

Generates random legal counteroffers (the shapes the decider will produce:
1:1, 2:1, 1:2, and two-for-one), phrases each one, and records whether the
phrase survived the round-trip gate.

Metrics:
  round-trip pass    phrase parsed back to exactly the offer, no added clause
  fallback rate      1 - pass (the template was shown instead)
  failure breakdown  terms_mismatch / added_clause / parse_clarify /
                     parse_rejected / transport / too_long
  template pass      the fallback text itself round-trips (sanity check:
                     if this is not ~100%, the fallback is not safe either)

Cost: each offer is one phrase call plus one or two parse calls.
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from catan_rl.negotiation.phraser import BUILDINGS, Phraser, first_person_template
from catan_rl.negotiation.schema import RESOURCES, SEATS, TradeOffer


def random_case(rng: random.Random):
    speaker = rng.choice(SEATS)
    res = list(RESOURCES)
    rng.shuffle(res)
    shape = rng.choices(["1:1", "2:1", "1:2", "2-for-1"], [4, 3, 1, 2])[0]
    if shape == "1:1":
        give, get = {res[0]: 1}, {res[1]: 1}
    elif shape == "2:1":
        give, get = {res[0]: 2}, {res[1]: 1}
    elif shape == "1:2":
        give, get = {res[0]: 1}, {res[1]: 2}
    else:
        give, get = {res[0]: 1, res[1]: 1}, {res[2]: 1}
    to = None if rng.random() < 0.5 else rng.choice([s for s in SEATS if s != speaker])
    offer = TradeOffer.make(speaker, to, give, get)

    hand = {r: rng.randint(0, 2) for r in RESOURCES}
    for r, n in give.items():
        hand[r] += n                      # the bot can always pay its own offer

    reasons = {}
    if rng.random() < 0.7:
        reasons["need"] = next(iter(get))
    if rng.random() < 0.5:
        reasons["building"] = rng.choice(BUILDINGS)
    if rng.random() < 0.4:
        reasons["vp_to_win"] = rng.randint(1, 5)
    return dict(shape=shape, offer=offer, hand=hand, reasons=reasons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-template-check", action="store_true")
    ap.add_argument("--out", default="logs/e2_results.jsonl")
    args = ap.parse_args()

    ph = Phraser()
    if not ph.client.available():
        raise SystemExit("No LLM configured. Run python -m catan_rl.llm_negotiator")
    rng = random.Random(args.seed)
    cases = [random_case(rng) for _ in range(args.n)]
    print(f"model: {ph.client.model} | {args.n} offers | {args.workers} workers")

    def run(c):
        out = ph.phrase(c["offer"], c["reasons"], hand=c["hand"])
        tmpl = None
        if not args.no_template_check:
            tmpl = ph.check(first_person_template(c["offer"]), c["offer"], hand=c["hand"])[0]
        return dict(shape=c["shape"], offer=c["offer"].template_text(),
                    reasons=c["reasons"], gate=out.gate, used_fallback=out.used_fallback,
                    llm_text=out.llm_text, parsed=out.parsed, shown=out.text,
                    template_gate=tmpl, tokens=out.tokens, latency_s=out.latency_s)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run, cases))
    print(f"done in {time.time() - t0:.1f}s\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    passed = sum(r["gate"] == "pass" for r in rows)
    print(f"round-trip pass   {100 * passed / n:5.1f}% ({passed}/{n})")
    print(f"fallback rate     {100 * (n - passed) / n:5.1f}%")
    for gate, k in Counter(r["gate"] for r in rows if r["gate"] != "pass").most_common():
        print(f"    {gate:<16} {k}")
    by_shape = Counter(r["shape"] for r in rows)
    ok_shape = Counter(r["shape"] for r in rows if r["gate"] == "pass")
    print("by shape          " + "  ".join(f"{s} {ok_shape[s]}/{by_shape[s]}" for s in by_shape))
    if not args.no_template_check:
        tp = sum(r["template_gate"] == "pass" for r in rows)
        print(f"template pass     {100 * tp / n:5.1f}% ({tp}/{n})   (should be ~100%)")
    lat = sorted(r["latency_s"] for r in rows)
    print(f"tokens/offer      {sum(r['tokens'] for r in rows) / n:.0f} (phrase + gate parse)"
          f" | latency median {lat[n // 2]:.2f}s")

    print("\nsample phrases that passed:")
    for r in [r for r in rows if r["gate"] == "pass"][:5]:
        print(f"  {r['offer']}\n      -> {r['llm_text']}")
    fails = [r for r in rows if r["gate"] != "pass"]
    if fails:
        print(f"\nfailures (first 8 of {len(fails)}), all in {args.out}:")
        for r in fails[:8]:
            print(f"  [{r['gate']}] {r['offer']}\n      said:   {r['llm_text']}"
                  f"\n      parsed: {r['parsed']}")


if __name__ == "__main__":
    main()
