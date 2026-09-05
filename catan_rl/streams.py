"""Split catanatron's flat feature vector into the three streams from the
architecture doc: board, private (self), opponent set, plus globals.

Catanatron's get_feature_ordering(num_players) returns stable feature names.
We group by prefix once at startup and index into the observation with those
groups forever after. Opponent features (P1_, P2_, P3_) are identical per
opponent, which is what makes the shared-MLP permutation-invariant set
encoder valid.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from catanatron_gym.features import get_feature_ordering


@dataclass
class StreamIndex:
    board: np.ndarray          # TILE*, NODE*, EDGE*, PORT*
    private: np.ndarray        # P0_*
    opponents: List[np.ndarray]  # one index array per opponent, same length
    global_: np.ndarray        # BANK_*, IS_*, anything else
    feature_names: List[str] = field(default_factory=list)

    @property
    def sizes(self):
        return dict(
            board=len(self.board),
            private=len(self.private),
            opponent=len(self.opponents[0]) if self.opponents else 0,
            num_opponents=len(self.opponents),
            global_=len(self.global_),
        )


def build_stream_index(num_players: int, map_type: str = "BASE",
                       num_derived: int = 0) -> StreamIndex:
    names = get_feature_ordering(num_players, map_type)

    board, private, global_ = [], [], []
    opp = {i: [] for i in range(1, num_players)}

    for idx, name in enumerate(names):
        if name.startswith(("TILE", "NODE", "EDGE", "PORT")):
            board.append(idx)
        elif name.startswith("P0_"):
            private.append(idx)
        else:
            matched = False
            for i in range(1, num_players):
                if name.startswith(f"P{i}_"):
                    opp[i].append(idx)
                    matched = True
                    break
            if not matched:
                global_.append(idx)  # BANK_*, IS_*, etc.

    # Sanity: every opponent must expose the same features in the same order,
    # otherwise the shared encoder sees inconsistent semantics per slot.
    def suffixes(ixs, p):
        return [names[j][len(f"P{p}_"):] for j in ixs]

    ref = suffixes(opp[1], 1)
    for i in range(2, num_players):
        assert suffixes(opp[i], i) == ref, "opponent feature layout mismatch"

    # engineered features appended after the base vector go to the global stream
    global_ += list(range(len(names), len(names) + num_derived))

    return StreamIndex(
        board=np.asarray(board, dtype=np.int64),
        private=np.asarray(private, dtype=np.int64),
        opponents=[np.asarray(opp[i], dtype=np.int64) for i in range(1, num_players)],
        global_=np.asarray(global_, dtype=np.int64),
        feature_names=names,
    )
