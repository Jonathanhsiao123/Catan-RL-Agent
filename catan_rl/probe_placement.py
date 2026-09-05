"""Measure initial-placement quality for a checkpoint.

For each game, intercepts the agent's two initial settlement choices and
scores them against the alternatives it had:

  * pips(chosen): production dots of the picked node (sum over adjacent
    tiles of 6 - |7 - number|)
  * pips(best):  the best node it could have picked at that moment
  * percentile:  where the choice ranked among legal options
                 (1.0 = best available, 0.5 = what random picking averages)

Usage:
  python -m catan_rl.probe_placement --ckpt checkpoints/rel0/ckpt_04000.pt --games 50
  python -m catan_rl.probe_placement --ckpt ... --random-baseline
"""

import argparse
import random

import numpy as np
import torch

torch.set_num_threads(1)  # bitwise-stable argmax across runs
from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from .opponents import NNPlayer
from .watch import load_agent

SEAT_COLORS = [Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE]


def node_pips(game):
    m = game.state.board.map
    return {n: sum(c.values()) * 36 for n, c in m.node_production.items()}


class ProbePlayer(NNPlayer):
    """NNPlayer that records the quality of its initial settlement picks."""

    records = []  # (chosen_pips, best_pips, percentile)

    def decide(self, game, playable_actions):
        action = super().decide(game, playable_actions)
        settle = [a for a in playable_actions
                  if a.action_type == ActionType.BUILD_SETTLEMENT]
        # initial phase: many legal spots (mid-game buildable spots are few)
        if (action.action_type == ActionType.BUILD_SETTLEMENT
                and len(settle) >= 8):
            pips = node_pips(game)
            options = sorted(pips[a.value] for a in settle)
            chosen = pips[action.value]
            pct = np.searchsorted(options, chosen, side="right") / len(options)
            ProbePlayer.records.append((chosen, options[-1], pct))
        return action


class RandomProbe(RandomPlayer):
    def decide(self, game, playable_actions):
        action = super().decide(game, playable_actions)
        settle = [a for a in playable_actions
                  if a.action_type == ActionType.BUILD_SETTLEMENT]
        if (action.action_type == ActionType.BUILD_SETTLEMENT
                and len(settle) >= 8):
            pips = node_pips(game)
            options = sorted(pips[a.value] for a in settle)
            chosen = pips[action.value]
            pct = np.searchsorted(options, chosen, side="right") / len(options)
            ProbePlayer.records.append((chosen, options[-1], pct))
        return action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--random-baseline", action="store_true",
                   help="probe a random picker instead of the checkpoint")
    p.add_argument("--seed", type=int, default=777)
    args = p.parse_args()

    ProbePlayer.records = []
    if args.random_baseline:
        agent = RandomProbe(Color.BLUE)
        name = "random baseline"
    else:
        model, norm = load_agent(args.ckpt)
        agent = ProbePlayer(Color.BLUE, model, norm, deterministic=True,
                            derived=getattr(model, "uses_derived", False))
        name = args.ckpt

    for g in range(args.games):
        random.seed(args.seed + g)
        np.random.seed(args.seed + g)
        players = [agent] + [WeightedRandomPlayer(c) for c in SEAT_COLORS[1:]]
        Game(players, seed=args.seed + g).play()

    rec = np.array(ProbePlayer.records)
    chosen, best, pct = rec[:, 0], rec[:, 1], rec[:, 2]
    print(f"{name} | {args.games} games, {len(rec)} initial placements")
    print(f"  chosen production : {chosen.mean():5.2f} pips (best available "
          f"averaged {best.mean():5.2f})")
    print(f"  capture ratio     : {chosen.mean() / best.mean():5.1%} of the "
          f"best spot's pips")
    print(f"  choice percentile : {pct.mean():5.1%} "
          f"(random picking = ~50%, always-best = 100%)")


if __name__ == "__main__":
    main()
