"""Relational board encoder (the doc's Section 2/8 upgrade path).

The flat MLP treats the board as an unstructured 926-dim vector and must learn
adjacency from data. This encoder embeds each tile (19), node (54), and edge
(72) separately and runs message passing over the real incidence structure,
so spatial facts (road chains, settlement spacing, tile access) are computable
rather than memorized.

Structure sources:
  * entity feature slices: parsed from catanatron's feature names
    (TILE{i}_*, NODE{i}_*, EDGE(a, b)_*)
  * edge endpoints: parsed directly from the EDGE(a, b) names
  * tile->node incidence: read from one instantiated BASE board (fixed topology)
  * ports: kept as a flat side-input to the readout in v1

Message rounds (residual):
  edge  <- MLP([edge, its two endpoint nodes])
  node  <- MLP([node, mean adj tiles, mean incident edges, mean neighbor nodes])
  tile  <- MLP([tile, mean of its six nodes])
Readout: mean+max over nodes, mean over tiles/edges, port projection -> out_dim.
"""

import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


def _pad(adj_lists, width):
    """Ragged adjacency -> (N, width) index array + float mask."""
    n = len(adj_lists)
    idx = np.zeros((n, width), dtype=np.int64)
    mask = np.zeros((n, width), dtype=np.float32)
    for i, lst in enumerate(adj_lists):
        for j, v in enumerate(lst[:width]):
            idx[i, j] = v
            mask[i, j] = 1.0
    return idx, mask


def build_board_structure(feature_names):
    """Parse entity feature slices and adjacency from names + static board."""
    tile_feats, node_feats, edge_feats, port_feats = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        [],
    )
    edge_pairs = {}
    for i, name in enumerate(feature_names):
        m = re.match(r"TILE(\d+)_", name)
        if m:
            tile_feats[int(m.group(1))].append(i)
            continue
        m = re.match(r"NODE(\d+)_", name)
        if m:
            node_feats[int(m.group(1))].append(i)
            continue
        m = re.match(r"EDGE\((\d+), (\d+)\)_", name)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            edge_feats[key].append(i)
            edge_pairs.setdefault(key, len(edge_pairs))
            continue
        if name.startswith("PORT"):
            port_feats.append(i)

    num_tiles, num_nodes, num_edges = len(tile_feats), len(node_feats), len(edge_feats)
    tile_ix = np.array([tile_feats[i] for i in range(num_tiles)])
    node_ix = np.array([node_feats[i] for i in range(num_nodes)])
    edge_keys = sorted(edge_pairs, key=edge_pairs.get)
    edge_ix = np.array([edge_feats[k] for k in edge_keys])
    edge_ends = np.array(edge_keys, dtype=np.int64)  # (E, 2) node ids

    # tile -> node incidence from one instantiated board (topology is fixed)
    from catanatron.game import Game
    from catanatron.models.player import Color, RandomPlayer

    game = Game([RandomPlayer(c) for c in
                 (Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE)])
    tile_nodes = np.zeros((num_tiles, 6), dtype=np.int64)
    for tile in game.state.board.map.land_tiles.values():
        tile_nodes[tile.id] = sorted(tile.nodes.values())

    # node-centric adjacency
    node_tiles = [[] for _ in range(num_nodes)]
    for t in range(num_tiles):
        for n in tile_nodes[t]:
            node_tiles[n].append(t)
    node_edges = [[] for _ in range(num_nodes)]
    node_nodes = [[] for _ in range(num_nodes)]
    for e, (a, b) in enumerate(edge_ends):
        node_edges[a].append(e)
        node_edges[b].append(e)
        node_nodes[a].append(b)
        node_nodes[b].append(a)

    nt_i, nt_m = _pad(node_tiles, 3)
    ne_i, ne_m = _pad(node_edges, 3)
    nn_i, nn_m = _pad(node_nodes, 3)

    assert all(1 <= len(x) <= 3 for x in node_tiles)
    assert all(2 <= len(x) <= 3 for x in node_edges)
    return dict(
        tile_ix=tile_ix, node_ix=node_ix, edge_ix=edge_ix, port_ix=np.array(port_feats),
        edge_ends=edge_ends, tile_nodes=tile_nodes,
        node_tiles=(nt_i, nt_m), node_edges=(ne_i, ne_m), node_nodes=(nn_i, nn_m),
    )


def _gather_mean(x, idx, mask):
    """x: (B, N, D); idx: (M, K); mask: (M, K) -> (B, M, D) masked mean."""
    g = x[:, idx.reshape(-1)].view(x.shape[0], *idx.shape, x.shape[-1])
    m = mask.unsqueeze(0).unsqueeze(-1)
    return (g * m).sum(2) / m.sum(2).clamp(min=1.0)


class RelationalBoardEncoder(nn.Module):
    def __init__(self, feature_names, d=48, rounds=2, out_dim=128):
        super().__init__()
        s = build_board_structure(feature_names)
        for k in ("tile_ix", "node_ix", "edge_ix", "port_ix", "edge_ends", "tile_nodes"):
            self.register_buffer(k, torch.as_tensor(s[k]), persistent=False)
        for k in ("node_tiles", "node_edges", "node_nodes"):
            self.register_buffer(k + "_i", torch.as_tensor(s[k][0]), persistent=False)
            self.register_buffer(k + "_m", torch.as_tensor(s[k][1]), persistent=False)

        self.tile_emb = nn.Linear(self.tile_ix.shape[1], d)
        self.node_emb = nn.Linear(self.node_ix.shape[1], d)
        self.edge_emb = nn.Linear(self.edge_ix.shape[1], d)
        self.rounds = rounds
        self.edge_upd = nn.ModuleList(
            nn.Sequential(nn.Linear(3 * d, d), nn.Tanh()) for _ in range(rounds))
        self.node_upd = nn.ModuleList(
            nn.Sequential(nn.Linear(4 * d, d), nn.Tanh()) for _ in range(rounds))
        self.tile_upd = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * d, d), nn.Tanh()) for _ in range(rounds))
        self.port_proj = nn.Linear(self.port_ix.shape[0], d)
        self.readout = nn.Sequential(nn.Linear(5 * d, out_dim), nn.Tanh())

    def entities(self, obs):
        """Per-entity embeddings after message passing: (tiles, nodes, edges, ports)."""
        B = obs.shape[0]

        def slice_(ix):
            return obs[:, ix.reshape(-1)].view(B, *ix.shape)

        tiles = torch.tanh(self.tile_emb(slice_(self.tile_ix)))
        nodes = torch.tanh(self.node_emb(slice_(self.node_ix)))
        edges = torch.tanh(self.edge_emb(slice_(self.edge_ix)))

        for r in range(self.rounds):
            end_nodes = nodes[:, self.edge_ends.reshape(-1)].view(B, -1, 2 * nodes.shape[-1])
            edges = edges + self.edge_upd[r](torch.cat([edges, end_nodes], -1))
            nodes = nodes + self.node_upd[r](torch.cat([
                nodes,
                _gather_mean(tiles, self.node_tiles_i, self.node_tiles_m),
                _gather_mean(edges, self.node_edges_i, self.node_edges_m),
                _gather_mean(nodes, self.node_nodes_i, self.node_nodes_m),
            ], -1))
            tile_node_mask = torch.ones(*self.tile_nodes.shape, device=obs.device)
            tiles = tiles + self.tile_upd[r](torch.cat(
                [tiles, _gather_mean(nodes, self.tile_nodes, tile_node_mask)], -1))

        ports = torch.tanh(self.port_proj(obs[:, self.port_ix]))
        return tiles, nodes, edges, ports

    def forward(self, obs):
        tiles, nodes, edges, ports = self.entities(obs)
        return self.readout(torch.cat([
            nodes.mean(1), nodes.max(1).values, tiles.mean(1), edges.mean(1), ports,
        ], -1))
