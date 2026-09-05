"""Evaluate a checkpoint's win rate against heuristic lineups or another checkpoint.

Usage:
  # vs heuristic bots
  python -m catan_rl.evaluate --ckpt checkpoints/seed0/ckpt_00900.pt --games 100 --enemies weighted

  # head-to-head: --ckpt takes the BLUE seat, --vs-ckpt fills the other three
  python -m catan_rl.evaluate --ckpt checkpoints/seed0/ckpt_00900.pt \
      --vs-ckpt checkpoints/seed0/ckpt_00600.pt --games 100
"""

import argparse
import os
import random

import numpy as np
import torch

torch.set_num_threads(1)  # bitwise-stable argmax across runs
from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import get_actual_victory_points

from .env import DERIVED_SIZE, CatanEnv, RunningNorm
from .model import ActorCritic
from .opponents import NNPlayer
from .streams import build_stream_index
from .watch import load_agent

NUM_ACTIONS = 290


def evaluate(ckpt_path, games=50, enemies="weighted", vs_ckpt=None,
             deterministic=True, seed=123, trade="off", random_offers=False):
    """trade: "off" | "agent" (only the evaluated agent proposes; opponents
    may still accept) | "table" (everyone proposes). "agent" is the mode that
    can actually show an advantage, since a symmetric change to all four
    seats cannot move relative standing."""
    from .trading import TradeConfig, TradeLog

    model, norm = load_agent(ckpt_path)
    tcfg = TradeConfig(random_offers=random_offers)
    tlog = TradeLog() if trade != "off" else None

    if vs_ckpt is not None:
        enemy_model, enemy_norm = load_agent(vs_ckpt)

        def enemy_factory(color):
            return NNPlayer(color, enemy_model, enemy_norm,
                            deterministic=deterministic,
                            derived=getattr(enemy_model, "uses_derived", False),
                            trade=(trade == "table"), trade_config=tcfg,
                            trade_log=tlog)

        enemy_desc = os.path.basename(vs_ckpt)
        factories = [enemy_factory] * 3
    else:
        enemy_cls = WeightedRandomPlayer if enemies == "weighted" else RandomPlayer
        enemy_desc = enemy_cls.__name__
        factories = [enemy_cls] * 3

    env = CatanEnv(enemy_factories=factories, normalizer=norm, shaping_coef=0.0,
                   derived=getattr(model, "uses_derived", False))
    # the evaluated agent trades through the same critic-priced broker
    agent_trader = None
    if trade != "off":
        agent_trader = NNPlayer(env.env.p0.color, model, norm,
                                deterministic=deterministic,
                                derived=getattr(model, "uses_derived", False),
                                trade=True, trade_config=tcfg, trade_log=tlog)

    wins, vps = 0, []
    for g in range(games):
        # seed the global RNG too: heuristic opponents sample from it, so
        # without this the same command gives different games run to run
        random.seed(seed + g)
        np.random.seed(seed + g)
        obs, mask = env.reset(seed=seed + g)
        done = False
        while not done:
            if agent_trader is not None:
                from .trading import is_trade_window, try_trade

                agent_trader.color = env.env.p0.color
                tries = 0
                while (tries < agent_trader.trade_config.max_per_turn
                       and is_trade_window(env.env.game, agent_trader.color)):
                    if try_trade(env.env.game, agent_trader,
                                 agent_trader.trade_config, tlog) is None:
                        break
                    tries += 1
                if tries:
                    obs, mask = env.observe()
            action, _, _ = model.act(obs[None], mask[None], deterministic=deterministic)
            obs, mask, _, done, info = env.step(int(action[0]))
        if info.get("env_reward", 0) > 0:
            wins += 1
        vps.append(get_actual_victory_points(env.env.game.state, env.env.p0.color))

    se = np.sqrt(max(wins / games * (1 - wins / games), 1e-9) / games)
    print(
        f"{os.path.basename(ckpt_path)} vs 3x {enemy_desc} over {games} games: "
        f"win rate {wins / games:.2%} (+/- {1.96 * se:.1%}) | "
        f"mean final VP {np.mean(vps):.2f} | fair-share baseline 25%"
    )
    if tlog is not None:
        print(f"  trading{' [RANDOM CONTROL]' if random_offers else ''}: "
              f"{tlog.summary()} "
              f"({len(tlog.entries)/max(games,1):.1f} per game)")
    return wins / games


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--enemies", choices=["random", "weighted"], default="weighted")
    p.add_argument("--vs-ckpt", default=None,
                   help="checkpoint to fill the other 3 seats (overrides --enemies)")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--random-offers", action="store_true",
                   help="CONTROL: proposer picks random legal offers instead "
                        "of the critic's best, isolating pricing from "
                        "the advantage of proposing at all")
    p.add_argument("--trade", choices=["off", "agent", "table"], default="off",
                   help="agent: only the evaluated agent proposes (measurable); "
                        "table: everyone proposes (symmetric, shows nothing)")
    a = p.parse_args()
    evaluate(a.ckpt, a.games, a.enemies, vs_ckpt=a.vs_ckpt,
             deterministic=not a.stochastic, trade=a.trade,
             random_offers=a.random_offers)
