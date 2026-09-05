"""PPO with GAE, value clipping, and an entropy bonus (Section 6).

The buffer stores the legal-action mask alongside each transition so the
update evaluates log-probs and entropy under the same masked distribution
that generated the data. Entropy of a masked Categorical is computed only
over legal actions, which is exactly what we want: the bonus rewards spread
over the legal subset, not the full 290.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    rollout_steps: int = 512      # per env
    minibatch_size: int = 512
    epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.02    # deliberately high; masked space collapses easily
    value_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


class RolloutBuffer:
    def __init__(self, steps, num_envs, obs_size, num_actions):
        self.obs = np.zeros((steps, num_envs, obs_size), dtype=np.float32)
        self.masks = np.zeros((steps, num_envs, num_actions), dtype=np.bool_)
        self.actions = np.zeros((steps, num_envs), dtype=np.int64)
        self.logprobs = np.zeros((steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((steps, num_envs), dtype=np.bool_)
        self.values = np.zeros((steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, mask, action, logprob, reward, done, value):
        t = self.ptr
        self.obs[t], self.masks[t] = obs, mask
        self.actions[t], self.logprobs[t] = action, logprob
        self.rewards[t], self.dones[t], self.values[t] = reward, done, value
        self.ptr += 1

    def compute_gae(self, last_values, gamma, lam):
        steps, num_envs = self.rewards.shape
        adv = np.zeros_like(self.rewards)
        lastgae = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(steps)):
            next_values = last_values if t == steps - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t].astype(np.float32)
            delta = self.rewards[t] + gamma * next_values * nonterminal - self.values[t]
            lastgae = delta + gamma * lam * nonterminal * lastgae
            adv[t] = lastgae
        returns = adv + self.values
        return adv, returns


def ppo_update(model, optimizer, buffer: RolloutBuffer, last_values, cfg: PPOConfig):
    adv, returns = buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)

    def flat(x):
        return torch.as_tensor(x.reshape(-1, *x.shape[2:]))

    obs = flat(buffer.obs)
    masks = flat(buffer.masks)
    actions = flat(buffer.actions)
    old_logprobs = flat(buffer.logprobs)
    old_values = flat(buffer.values)
    adv_t = flat(adv)
    ret_t = flat(returns)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = obs.shape[0]
    stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, clip_frac=0.0)
    updates = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch_size):
            mb = perm[start : start + cfg.minibatch_size]
            logprobs, entropy, values = model.evaluate_actions(
                obs[mb], masks[mb], actions[mb]
            )
            ratio = torch.exp(logprobs - old_logprobs[mb])
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            v_clipped = old_values[mb] + torch.clamp(
                values - old_values[mb], -cfg.value_clip_eps, cfg.value_clip_eps
            )
            v_loss = 0.5 * torch.max(
                (values - ret_t[mb]) ** 2, (v_clipped - ret_t[mb]) ** 2
            ).mean()

            loss = policy_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
            updates += 1

    return {k: v / max(updates, 1) for k, v in stats.items()}
