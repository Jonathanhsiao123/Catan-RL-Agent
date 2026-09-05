"""Arena: round-robin tournament over checkpoints, reported as Elo.

Win rate against a fixed opponent saturates (yours is ~70% against heuristic
bots), and a single head-to-head only orders two agents. Elo from a
round-robin gives one comparable number per agent on a common scale, which
is the standard way strength is tracked in game RL.

Each game seats 4 entrants sampled from the pool, so a 4-player placement
becomes 6 pairwise results (winner beats the other three; non-winners draw
with each other, which is the honest encoding since Catan has no 2nd place
in the win condition). Ratings are fitted by repeated Elo passes over the
shuffled result list, which converges to a stable ordering.

Usage:
  python -m catan_rl.arena --ckpts checkpoints/gen3/ckpt_08000.pt \
      checkpoints/flat0v2/ckpt_04000.pt checkpoints/seed0/ckpt_02000.pt \
      --baselines weighted random --games 60 --trade table

--trade modes:
  off    (default) no player-to-player trading.
  table  every network entrant can propose and accept. Symmetric: this
         measures whether trading changes the RANKING among entrants who all
         have it, not whether trading itself helps, since a change applied
         equally to every seat cannot move relative standing by construction.
  agent  only ONE designated entrant (--trade-agent) proposes; everyone else
         may still accept if they have a critic. This is the mode that shows
         a trading advantage, matching evaluate.py's --trade agent.
"""

import argparse
import itertools
import os
import random
from collections import defaultdict

import numpy as np
import torch

torch.set_num_threads(1)  # bitwise-stable argmax across runs
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from .opponents import NNPlayer
from .trading import TradeConfig, TradeLog
from .watch import load_agent

SEAT_COLORS = [Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE]


def elo(results, k=24, iters=40, base=1500.0):
    """Fit Elo from pairwise outcomes.

    results: list of (name_a, name_b, score_a) where score_a is 1.0 for a win,
    0.5 for a draw, 0.0 for a loss. Draws matter: in a 4-player game only one
    seat wins, so the three non-winners must be recorded as drawing with each
    other. Recording only winner-over-loser pairs leaves entrants that never
    win with no head-to-head evidence against each other, and their relative
    ratings become noise.
    """
    names = {n for r in results for n in r[:2]}
    rating = {n: base for n in names}
    for _ in range(iters):
        random.shuffle(results)
        for a, b, sa in results:
            ea = 1.0 / (1.0 + 10 ** ((rating[b] - rating[a]) / 400.0))
            rating[a] += k * (sa - ea)
            rating[b] -= k * (sa - ea)
    return rating


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[])
    p.add_argument("--baselines", nargs="*", default=["weighted"],
                   choices=["weighted", "random"])
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--trade", choices=["off", "table", "agent"], default="off")
    p.add_argument("--random-offers", action="store_true",
                   help="CONTROL: the proposer picks a random legal offer "
                        "instead of the critic's best. Proposing at all is an "
                        "advantage if opponents accept anything mildly good "
                        "for them; this isolates how much of the gain comes "
                        "from pricing rather than from having the initiative.")
    p.add_argument("--trade-agent", default=None,
                   help="entrant name (checkpoint filename) that proposes, "
                        "for --trade agent. Defaults to the first --ckpts entry.")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    entrants = {}   # name -> factory(color) -> Player
    tlog = TradeLog() if args.trade != "off" else None
    designated = args.trade_agent or (
        os.path.basename(args.ckpts[0]) if args.ckpts else None)
    for path in args.ckpts:
        model, norm = load_agent(path)
        name = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        is_designated = os.path.basename(path) == designated

        if args.trade == "table":
            can_propose = True
        elif args.trade == "agent":
            can_propose = is_designated
        else:
            can_propose = False

        cfg = TradeConfig(random_offers=args.random_offers)

        def make(color, model=model, norm=norm, can_propose=can_propose, cfg=cfg):
            return NNPlayer(color, model, norm,
                            deterministic=not args.stochastic,
                            derived=getattr(model, "uses_derived", False),
                            trade=can_propose, trade_config=cfg, trade_log=tlog)

        entrants[name] = make
    if args.trade == "agent":
        ctrl = " [RANDOM-OFFER CONTROL]" if args.random_offers else ""
        print(f"trade mode: agent-only, proposer = {designated}{ctrl}\n")
    for b in args.baselines:
        cls = WeightedRandomPlayer if b == "weighted" else RandomPlayer
        entrants[f"baseline:{b}"] = (lambda color, cls=cls: cls(color))

    names = list(entrants)
    if len(names) < 2:
        raise SystemExit("need at least 2 entrants (checkpoints + baselines)")
    print(f"arena: {len(names)} entrants, {args.games} games\n  " +
          "\n  ".join(names) + "\n")

    results, wins, played = [], defaultdict(int), defaultdict(int)
    for g in range(args.games):
        random.seed(args.seed + g)
        np.random.seed(args.seed + g)
        table = (random.sample(names, 4) if len(names) >= 4
                 else [random.choice(names) for _ in range(4)])
        players = [entrants[n](c) for n, c in zip(table, SEAT_COLORS)]
        game = Game(players, seed=args.seed + g)
        game.play()
        winner_color = game.winning_color()
        for n in table:
            played[n] += 1
        if winner_color is None:
            # turn limit: every seat drew with every other seat
            for i in range(4):
                for j in range(i + 1, 4):
                    if table[i] != table[j]:
                        results.append((table[i], table[j], 0.5))
            continue
        widx = SEAT_COLORS.index(winner_color)
        wname = table[widx]
        wins[wname] += 1
        losers = [n for i, n in enumerate(table) if i != widx]
        for n in losers:
            if n != wname:
                results.append((wname, n, 1.0))
        # the three non-winners drew with each other: this is what separates
        # a weak entrant from a very weak one when neither ever wins
        for i in range(len(losers)):
            for j in range(i + 1, len(losers)):
                if losers[i] != losers[j]:
                    results.append((losers[i], losers[j], 0.5))
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.games} games")

    rating = elo(results)
    print("\n%-42s %7s %8s %7s" % ("entrant", "elo", "win%", "games"))
    for n in sorted(rating, key=lambda x: -rating[x]):
        wr = wins[n] / played[n] if played[n] else float("nan")
        print("%-42s %7.0f %7.1f%% %7d" % (n, rating[n], 100 * wr, played[n]))
    print("\nfair share in a 4-player game is 25%; elo is relative within this pool")
    if tlog is not None:
        print("trading:", tlog.summary())


if __name__ == "__main__":
    main()
