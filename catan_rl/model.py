"""Network from Section 2/4 of the architecture doc.

board stream ----+
private stream ---+--> concat --> MLP trunk --+--> policy head (masked logits)
opponent set  ---+                            +--> value head (scalar)

Opponent set encoder: one shared MLP applied to each opponent's 14 public
features, mean-pooled, so the agent is invariant to seating order.
Masking: illegal logits set to a large negative value before the softmax.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .streams import StreamIndex

NEG_INF = -1e9


def mlp(sizes, act=nn.Tanh, out_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if out_act or i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        stream_index: StreamIndex,
        num_actions: int,
        board_hidden=(256, 128),
        opp_hidden=(32, 32),
        trunk_hidden=(256, 256),
        encoder: str = "flat",
    ):
        super().__init__()
        self.encoder_type = encoder
        s = stream_index.sizes
        self.register_buffer("ix_board", torch.as_tensor(stream_index.board))
        self.register_buffer("ix_private", torch.as_tensor(stream_index.private))
        self.register_buffer(
            "ix_opp", torch.stack([torch.as_tensor(o) for o in stream_index.opponents])
        )  # (num_opp, opp_dim)
        self.register_buffer("ix_global", torch.as_tensor(stream_index.global_))

        if encoder == "relational":
            from .relational import RelationalBoardEncoder

            # operates on the full observation (does its own entity slicing)
            self.board_enc = RelationalBoardEncoder(
                stream_index.feature_names, out_dim=board_hidden[-1]
            )
        else:
            self.board_enc = mlp([s["board"], *board_hidden])
        self.opp_enc = mlp([s["opponent"], *opp_hidden])  # shared across opponents
        trunk_in = board_hidden[-1] + s["private"] + opp_hidden[-1] + s["global_"]
        self.trunk = mlp([trunk_in, *trunk_hidden])
        self.policy = nn.Linear(trunk_hidden[-1], num_actions)
        self.value = nn.Linear(trunk_hidden[-1], 1)

        # smaller init on heads stabilizes early PPO
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "relational":
            board = self.board_enc(obs)
        else:
            board = self.board_enc(obs[:, self.ix_board])
        private = obs[:, self.ix_private]
        # (B, num_opp, opp_dim) -> shared MLP -> mean pool
        opp = obs[:, self.ix_opp.reshape(-1)].view(obs.shape[0], self.ix_opp.shape[0], -1)
        opp = self.opp_enc(opp).mean(dim=1)
        glob = obs[:, self.ix_global]
        return self.trunk(torch.cat([board, private, opp, glob], dim=-1))

    def dist_value(self, obs: torch.Tensor, mask: torch.Tensor):
        h = self._features(obs)
        logits = self.policy(h)
        logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
        return Categorical(logits=logits), self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        dist, value = self.dist_value(obs_t, mask_t)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return (
            action.cpu().numpy(),
            dist.log_prob(action).cpu().numpy(),
            value.cpu().numpy(),
        )

    def evaluate_actions(self, obs, mask, actions):
        dist, value = self.dist_value(obs, mask)
        return dist.log_prob(actions), dist.entropy(), value


def clean_state_dict(model):
    """state_dict with torch.compile's _orig_mod. prefixes stripped."""
    return {k.replace("._orig_mod", ""): v for k, v in model.state_dict().items()}
