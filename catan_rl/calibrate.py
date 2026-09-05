"""Calibrate a checkpoint's value head into win-probability units.

The critic outputs shaped-return estimates, not probabilities. That matters
for trading: win probabilities across four seats sum to one, so the four
players' gains from a trade must sum to zero. If V were calibrated, "accept
iff my gain > 0" would be exactly right and the effect on the two players
not in the trade would already be priced in. Uncalibrated, both sides of a
trade can report a gain, which is impossible, and each side accepts exactly
when its own estimate is too high (adverse selection).

This script plays self-play games, records (V(s), did this player win) at
sampled decision points, and fits a monotone map from V to empirical win
rate by quantile binning with a cumulative-max monotonicity constraint. No
sklearn needed.

    python -m catan_rl.calibrate --ckpt checkpoints/gen3/ckpt_08000.pt --games 60

Writes <ckpt>.calib.json next to the checkpoint. load_agent() picks it up
automatically, and trading.py then reports gains in probability units.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from catanatron.game import Game
from catanatron.models.player import Color

torch.set_num_threads(1)

from .opponents import NNPlayer
from .watch import load_agent

SEAT_COLORS = [Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE]


class Recorder(NNPlayer):
    """Plays normally, recording its own V(s) at sampled decision points."""

    def __init__(self, *a, sample_every=4, **kw):
        super().__init__(*a, **kw)
        self.sample_every = sample_every
        self.values = []
        self._n = 0

    def decide(self, game, playable_actions):
        self._n += 1
        if self._n % self.sample_every == 0 and self.model is not None:
            try:
                obs = self._raw_obs(game)
                self.values.append(float(self._values(np.stack([obs]))[0]))
            except Exception:
                pass
        return super().decide(game, playable_actions)


def fit_monotone(values, labels, n_bins=20):
    """Quantile-bin V, take empirical win rate per bin, enforce monotonicity."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(values)
    values, labels = values[order], labels[order]
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    centers, rates = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values <= hi)
        if m.sum() < 5:
            continue
        centers.append(float(values[m].mean()))
        rates.append(float(labels[m].mean()))
    if len(centers) < 2:
        raise SystemExit("not enough data to calibrate; run more games")
    rates = np.maximum.accumulate(np.asarray(rates))  # monotone in V
    return [float(c) for c in centers], [float(r) for r in rates]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--sample-every", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, norm = load_agent(args.ckpt)
    all_v, all_y = [], []
    for g in range(args.games):
        random.seed(args.seed + g)
        np.random.seed(args.seed + g)
        players = [Recorder(c, model, norm, deterministic=False,
                            derived=getattr(model, "uses_derived", False),
                            sample_every=args.sample_every)
                   for c in SEAT_COLORS]
        game = Game(players, seed=args.seed + g)
        game.play()
        winner = game.winning_color()
        for pl in players:
            won = 1.0 if (winner is not None and pl.color == winner) else 0.0
            all_v.extend(pl.values)
            all_y.extend([won] * len(pl.values))
        if (g + 1) % 15 == 0:
            print(f"  {g+1}/{args.games} games, {len(all_v)} samples")

    centers, rates = fit_monotone(all_v, all_y)
    out = args.ckpt + ".calib.json"
    with open(out, "w") as f:
        json.dump(dict(centers=centers, rates=rates,
                       n_samples=len(all_v), games=args.games), f)
    print(f"\nfitted on {len(all_v)} samples from {args.games} games")
    print("  V range      :", f"{min(all_v):+.3f} .. {max(all_v):+.3f}")
    print("  P(win) range :", f"{rates[0]:.3f} .. {rates[-1]:.3f}")
    print("  base rate    :", f"{np.mean(all_y):.3f} (fair share 0.25)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
