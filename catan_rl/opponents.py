"""Self-play opponent pool (Section 6).

NNPlayer adapts a frozen policy checkpoint into a catanatron Player so it can
sit inside the engine as an enemy. It builds the ego-centric feature vector
from its own color's perspective, masks to the currently playable actions,
and samples from the frozen policy.

OpponentPool holds heuristic bots plus frozen checkpoints and samples a
3-enemy lineup per environment, mixing styles to avoid latest-vs-latest
cycling in a 4-player game with kingmaker dynamics.
"""

import copy
import random
from typing import List

import numpy as np
import torch
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import (
    ACTION_SPACE_SIZE,
    from_action_space,
    to_action_space,
)
from catanatron_gym.features import create_sample_vector

from .trading import CriticTrader, TradeConfig, TradeLog, is_trade_window, try_trade


class NNPlayer(CriticTrader, Player):
    """A frozen policy checkpoint playing as an in-engine enemy.

    With trade=True it also proposes and accepts player-to-player trades,
    priced by its own value head (see trading.py). Trading is a wrapper on
    the engine, not an engine action, so this works with existing
    checkpoints and needs no retraining.
    """

    def __init__(self, color: Color, model, normalizer=None, deterministic=False,
                 derived=False, trade=False, trade_config=None, trade_log=None):
        super().__init__(color)
        self.model = model
        self.normalizer = normalizer
        self.deterministic = deterministic
        self.derived = derived
        self.trade = trade
        self.trade_config = trade_config or TradeConfig()
        self.trade_log = trade_log
        self._traded_this_turn = 0
        self._last_turn_seen = -1

    def decide(self, game, playable_actions):
        if self.trade and self.model is not None:
            turn = game.state.num_turns
            if turn != self._last_turn_seen:
                self._last_turn_seen = turn
                self._traded_this_turn = 0
            while (self._traded_this_turn < self.trade_config.max_per_turn
                   and is_trade_window(game, self.color)
                   and random.random() < self.trade_config.propose_chance):
                done = try_trade(game, self, self.trade_config, self.trade_log)
                if done is None:
                    break
                self._traded_this_turn += 1
            # hand may have changed; re-read the legal set from the engine
            playable_actions = game.state.playable_actions

        if len(playable_actions) == 1:
            return playable_actions[0]
        obs = np.asarray(
            create_sample_vector(game, self.color), dtype=np.float64
        )
        if self.derived:
            from .env import derived_features

            obs = np.concatenate([obs, derived_features(game.state, self.color)])
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[to_action_space(a) for a in playable_actions]] = True
        with torch.no_grad():
            action, _, _ = self.model.act(
                obs[None].astype(np.float32), mask[None], deterministic=self.deterministic
            )
        return from_action_space(int(action[0]), playable_actions)


class OpponentPool:
    """Heuristic bots + frozen checkpoints; samples 3-enemy lineups."""

    def __init__(self, model_factory, max_checkpoints: int = 8, selfplay_prob: float = 0.7,
                 derived: bool = False):
        self.model_factory = model_factory  # () -> fresh ActorCritic
        self.checkpoints: List[dict] = []   # list of (state_dict, norm_state)
        self.max_checkpoints = max_checkpoints
        self.selfplay_prob = selfplay_prob
        self.derived = derived

    def push(self, model, normalizer):
        from .model import clean_state_dict

        sd = copy.deepcopy(clean_state_dict(model))
        ns = copy.deepcopy(normalizer.state_dict()) if normalizer else None
        self.checkpoints.append((sd, ns))
        if len(self.checkpoints) > self.max_checkpoints:
            # keep the oldest anchor and a spread of recent ones
            self.checkpoints.pop(1)

    def _make_nn_enemy(self):
        sd, ns = random.choice(self.checkpoints)
        model = self.model_factory()
        model.load_state_dict(sd)
        model.eval()
        norm = None
        if ns is not None:
            from .env import RunningNorm

            norm = RunningNorm(len(ns["mean"]))
            norm.load_state_dict(ns)
            norm.frozen = True
        return lambda color: NNPlayer(color, model, norm, derived=self.derived)

    def sample_enemy_factories(self):
        factories = []
        for _ in range(3):
            if self.checkpoints and random.random() < self.selfplay_prob:
                factories.append(self._make_nn_enemy())
            else:
                factories.append(
                    random.choice([RandomPlayer, WeightedRandomPlayer])
                )
        return factories
