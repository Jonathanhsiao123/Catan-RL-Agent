"""Entity-addressed ("pointer") actor-critic.

The plain relational encoder computed good per-entity representations, then
pooled them away before the policy head had to score entity-specific actions
("build on node 23"). This head keeps the address book:

  BUILD_SETTLEMENT[54], BUILD_CITY[54]  <- linear score of that node's embedding
  BUILD_ROAD[72]                        <- that edge's embedding
  MOVE_ROBBER[19]                       <- that tile's embedding
  remaining 91 global actions           <- trunk MLP (roll, trades, dev cards...)

Entity scores are conditioned on the global context by concatenating the trunk
state to each entity embedding before scoring. Value head reads the trunk.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY

from .model import NEG_INF, mlp
from .relational import RelationalBoardEncoder, build_board_structure
from .streams import StreamIndex


def build_action_maps(feature_names):
    """Map each of the 290 primitive actions to (head, entity_row)."""
    s = build_board_structure(feature_names)
    edge_row = {tuple(e): i for i, e in enumerate(s["edge_ends"].tolist())}
    g = Game([RandomPlayer(c) for c in
              (Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE)])
    coord_row = {coord: t.id for coord, t in g.state.board.map.land_tiles.items()}

    node_set, node_row = [], []   # (action_idx, node, col) col 0=settlement 1=city
    edge_act, tile_act, global_act = [], [], []
    for i, (atype, value) in enumerate(ACTIONS_ARRAY):
        t = atype.value
        if t == "BUILD_SETTLEMENT":
            node_set.append(i); node_row.append((value, 0))
        elif t == "BUILD_CITY":
            node_set.append(i); node_row.append((value, 1))
        elif t == "BUILD_ROAD":
            edge_act.append((i, edge_row[tuple(sorted(value))]))
        elif t == "MOVE_ROBBER":
            tile_act.append((i, coord_row[value]))
        else:
            global_act.append(i)
    return dict(
        node_actions=np.array(node_set),
        node_rows=np.array(node_row),          # (108, 2): node id, column
        edge_actions=np.array([i for i, _ in edge_act]),
        edge_rows=np.array([r for _, r in edge_act]),
        tile_actions=np.array([i for i, _ in tile_act]),
        tile_rows=np.array([r for _, r in tile_act]),
        global_actions=np.array(global_act),
    )


class PointerActorCritic(nn.Module):
    def __init__(self, stream_index: StreamIndex, num_actions: int,
                 d=48, opp_hidden=(32, 32), trunk_hidden=(256, 256)):
        super().__init__()
        self.encoder_type = "pointer"
        s = stream_index.sizes
        names = stream_index.feature_names
        self.enc = RelationalBoardEncoder(names, d=d, out_dim=128)

        m = build_action_maps(names)
        for k, v in m.items():
            self.register_buffer("am_" + k, torch.as_tensor(v), persistent=False)
        self.num_actions = num_actions

        self.register_buffer("ix_private", torch.as_tensor(stream_index.private))
        self.register_buffer(
            "ix_opp", torch.stack([torch.as_tensor(o) for o in stream_index.opponents]))
        self.register_buffer("ix_global", torch.as_tensor(stream_index.global_))
        self.opp_enc = mlp([s["opponent"], *opp_hidden])
        trunk_in = 128 + s["private"] + opp_hidden[-1] + s["global_"]
        self.trunk = mlp([trunk_in, *trunk_hidden])

        ctx = trunk_hidden[-1]
        self.node_head = nn.Linear(d + ctx, 2)   # settlement / city columns
        self.edge_head = nn.Linear(d + ctx, 1)
        self.tile_head = nn.Linear(d + ctx, 1)
        self.global_head = nn.Linear(ctx, len(m["global_actions"]))
        self.value = nn.Linear(ctx, 1)
        for head in (self.node_head, self.edge_head, self.tile_head, self.global_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def _logits_value(self, obs):
        B = obs.shape[0]
        tiles, nodes, edges, ports = self.enc.entities(obs)
        pooled = self.enc.readout(torch.cat([
            nodes.mean(1), nodes.max(1).values, tiles.mean(1), edges.mean(1), ports,
        ], -1))
        opp = obs[:, self.ix_opp.reshape(-1)].view(B, self.ix_opp.shape[0], -1)
        opp = self.opp_enc(opp).mean(dim=1)
        h = self.trunk(torch.cat(
            [pooled, obs[:, self.ix_private], opp, obs[:, self.ix_global]], -1))

        def with_ctx(ent):  # (B, N, d) -> (B, N, d+ctx)
            return torch.cat([ent, h.unsqueeze(1).expand(-1, ent.shape[1], -1)], -1)

        logits = obs.new_full((B, self.num_actions), 0.0)
        node_sc = self.node_head(with_ctx(nodes))            # (B, 54, 2)
        rows, cols = self.am_node_rows[:, 0], self.am_node_rows[:, 1]
        logits[:, self.am_node_actions] = node_sc[:, rows, cols]
        logits[:, self.am_edge_actions] = self.edge_head(
            with_ctx(edges)).squeeze(-1)[:, self.am_edge_rows]
        logits[:, self.am_tile_actions] = self.tile_head(
            with_ctx(tiles)).squeeze(-1)[:, self.am_tile_rows]
        logits[:, self.am_global_actions] = self.global_head(h)
        return logits, self.value(h).squeeze(-1)

    def dist_value(self, obs, mask):
        logits, value = self._logits_value(obs)
        logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        dist, value = self.dist_value(obs_t, mask_t)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return (action.cpu().numpy(), dist.log_prob(action).cpu().numpy(),
                value.cpu().numpy())

    def evaluate_actions(self, obs, mask, actions):
        dist, value = self.dist_value(obs, mask)
        return dist.log_prob(actions), dist.entropy(), value
