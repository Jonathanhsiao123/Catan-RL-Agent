"""Environment layer.

Wraps catanatron_gym's Discrete(290) flat-action env with:
  * 4-player configuration (agent is BLUE, three configurable enemies)
  * a boolean legal-action mask surfaced every step
  * potential-based reward shaping F = gamma * Phi(s') - Phi(s),
    Phi = my_actual_VP / vps_to_win (policy-invariant; terminal win stays dominant)
  * running observation normalization (feature scales range from binary to ~95)
  * a minimal synchronous vector env that auto-resets and returns masks
"""

from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
from catanatron.models.player import Color, RandomPlayer
from catanatron.state_functions import get_actual_victory_points

SETTLEMENT, CITY = "SETTLEMENT", "CITY"
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DERIVED_SIZE = 9  # 5 own per-resource pips, own total, 3 opponent totals (sorted)


def derived_features(state, color):
    """Engineered production features: what the network would otherwise have
    to learn as arithmetic over 54 node-tile relations. Per-resource and total
    expected production pips for `color` (cities x2), plus each opponent's
    total, sorted descending for permutation invariance."""
    prod_map = state.board.map.node_production

    def pips_for(c):
        per = dict.fromkeys(RESOURCES, 0.0)
        for node, (col, btype) in state.board.buildings.items():
            if col == c:
                mult = 2.0 if btype == CITY else 1.0
                for r, p in prod_map[node].items():
                    per[r] += mult * p * 36.0
        return per

    mine = pips_for(color)
    feats = [min(mine[r], 13.0) / 13.0 for r in RESOURCES]
    feats.append(min(sum(mine.values()), 26.0) / 26.0)
    opp = sorted((sum(pips_for(c).values()) for c in state.colors if c != color),
                 reverse=True)
    feats += [min(v, 26.0) / 26.0 for v in opp]
    return np.asarray(feats, dtype=np.float64)
from catanatron_gym.envs.catanatron_env import ACTION_SPACE_SIZE, CatanatronEnv

ENEMY_COLORS = [Color.RED, Color.ORANGE, Color.WHITE]


def simple_terminal_reward(game, p0_color):
    winner = game.winning_color()
    if winner is None:
        return 0.0
    return 1.0 if winner == p0_color else -1.0


class RunningNorm:
    """Per-feature running mean/std normalizer (Welford)."""

    def __init__(self, size, clip=10.0, eps=1e-8):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = eps
        self.clip = clip
        self.frozen = False

    def update(self, x: np.ndarray):
        if self.frozen:
            return
        x = np.atleast_2d(x)
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m_a = self.var * self.count
        m_b = b_var * b_n
        self.var = (m_a + m_b + delta**2 * self.count * b_n / tot) / tot
        self.count = tot

    def __call__(self, x):
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class CatanEnv:
    """Single-environment wrapper. step() returns (obs, mask, reward, done, info)."""

    def __init__(
        self,
        enemy_factories: Optional[List[Callable[[Color], object]]] = None,
        gamma: float = 0.995,
        shaping_coef: float = 0.1,
        prod_potential: float = 0.0,
        derived: bool = False,
        vps_to_win: int = 10,
        normalizer: Optional[RunningNorm] = None,
    ):
        if enemy_factories is None:
            enemy_factories = [RandomPlayer] * 3
        assert len(enemy_factories) == 3, "4-player game: exactly 3 enemies"
        enemies = [f(c) for f, c in zip(enemy_factories, ENEMY_COLORS)]
        self.env = CatanatronEnv(
            config=dict(
                enemies=enemies,
                reward_function=simple_terminal_reward,
                vps_to_win=vps_to_win,
            )
        )
        self.gamma = gamma
        self.shaping_coef = shaping_coef
        self.prod_potential = prod_potential
        self.derived = derived
        self.vps_to_win = vps_to_win
        self.norm = normalizer
        self._phi = 0.0

    @property
    def obs_size(self):
        base = self.env.observation_space.shape[0]
        return base + (DERIVED_SIZE if self.derived else 0)

    def observe(self):
        """Current observation and mask without stepping.

        Needed after an out-of-band hand change (a wrapper-level trade):
        the agent must see its new hand before choosing an action.
        """
        obs = self._augment(self.env._get_observation())
        if self.norm is not None:
            obs = self.norm(obs)
        return obs.astype(np.float32), self._mask()

    def _augment(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        if self.derived:
            d = derived_features(self.env.game.state, self.env.p0.color)
            obs = np.concatenate([obs, d])
        return obs

    def _mask(self):
        m = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        m[self.env.get_valid_actions()] = True
        return m

    def _potential(self):
        state = self.env.game.state
        color = self.env.p0.color
        vp = get_actual_victory_points(state, color)
        phi = min(vp, self.vps_to_win) / self.vps_to_win
        if self.prod_potential > 0:
            # expected production pips of our buildings (city counts double);
            # part of the potential, so shaping stays policy-invariant
            prod_map = state.board.map.node_production
            pips = 0.0
            for node, (c, btype) in state.board.buildings.items():
                if c == color:
                    mult = 2.0 if btype == CITY else 1.0
                    pips += mult * sum(prod_map[node].values()) * 36
            phi += self.prod_potential * min(pips, 26.0) / 26.0
        return phi

    def reset(self, seed=None):
        obs, _info = self.env.reset(seed=seed)
        obs = self._augment(obs)
        if self.norm is not None:
            self.norm.update(obs)
            obs = self.norm(obs)
        self._phi = self._potential()
        return obs.astype(np.float32), self._mask()

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        done = bool(terminated or truncated)
        phi_next = 0.0 if done else self._potential()  # Phi(terminal) = 0 keeps invariance
        shaped = reward + self.shaping_coef * (self.gamma * phi_next - self._phi)
        self._phi = phi_next
        obs = self._augment(obs)
        if self.norm is not None:
            self.norm.update(obs)
            obs = self.norm(obs)
        info = dict(info or {})
        info["env_reward"] = reward
        return obs.astype(np.float32), self._mask(), float(shaped), done, info


class SyncVecCatan:
    """Minimal synchronous vector env with auto-reset and shared normalizer."""

    def __init__(self, num_envs: int, env_fn: Callable[[], CatanEnv], base_seed: int = 0):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.base_seed = base_seed
        self._ep = np.zeros(num_envs, dtype=np.int64)

    def reset(self):
        obs, masks = [], []
        for i, e in enumerate(self.envs):
            o, m = e.reset(seed=self.base_seed + i)
            obs.append(o)
            masks.append(m)
        return np.stack(obs), np.stack(masks)

    def step(self, actions):
        obs, masks, rews, dones, wins = [], [], [], [], []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, m, r, d, info = e.step(a)
            if d:
                wins.append(1.0 if info.get("env_reward", 0) > 0 else 0.0)
                self._ep[i] += 1
                o, m = e.reset(seed=self.base_seed + i + 10_000 * int(self._ep[i]))
            obs.append(o)
            masks.append(m)
            rews.append(r)
            dones.append(d)
        return (
            np.stack(obs),
            np.stack(masks),
            np.asarray(rews, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            wins,
        )
