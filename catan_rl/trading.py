"""Stage 1 trading: player-to-player trades priced by the trained critic.

The pinned engine has no player-trade actions, so trades happen as a wrapper
layer around it rather than as engine actions:

  1. A proposer enumerates a small offer vocabulary (1:1 and 2:1 single-
     resource swaps) filtered to what it can actually pay.
  2. All candidates are priced in ONE critic pass: build the proposer's
     observation, copy it once per candidate with the hand-count features
     edited to the post-trade hand, and read V for the whole batch. Best
     V(s') - V(s) wins if the gain clears a threshold.
  3. Every other trade-capable player prices the mirrored offer with its own
     critic and accepts if its own gain clears the threshold. First accepter
     takes the deal.
  4. The accepted trade executes with the engine's own resource primitives,
     so card totals stay conserved, then legality is regenerated.

No retraining needed: the critic already prices resources in context, having
learned V(s) over tens of millions of steps. It was trained in a world
without trading, so its estimates are slightly off-distribution, but resource
counts are part of its input and "more ore in this position" is exactly what
it knows.

Counterparties are found by duck typing: any player in the game exposing
trade_gain() can accept. Humans do not (that is stage 3), so mixed
human/bot tables work with the bots trading among themselves.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from catanatron.models.enums import ActionType
from catanatron.state_functions import (
    player_freqdeck_add,
    player_deck_subtract,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
# (give_count, ask_count). 1:1 is the common human trade; 2:1 sweetens it
# when the proposer is resource-rich.
TRADE_SHAPES = ((1, 1), (2, 1))


def freqdeck(counts: dict) -> List[int]:
    return [int(counts.get(r, 0)) for r in RESOURCES]


@dataclass
class Offer:
    """give/ask are {resource: count} from the proposer's perspective."""

    give: dict
    ask: dict

    def mirrored(self) -> "Offer":
        return Offer(give=dict(self.ask), ask=dict(self.give))

    def __str__(self):
        g = " + ".join(f"{n} {r.lower()}" for r, n in self.give.items() if n)
        a = " + ".join(f"{n} {r.lower()}" for r, n in self.ask.items() if n)
        return f"{g} for {a}"


@dataclass
class TradeConfig:
    """min_gain is in whatever units the critic reports: raw value units by
    default, or win-probability units once the checkpoint is calibrated (see
    calibrate.py), which is what makes the two sides' gains comparable.

    criterion:
      "relative" (default) accept iff my_gain - their_gain > min_gain. Robust
        to an uncalibrated critic: in a race, a trade that helps you more than
        me is a trade I should decline even if my own estimate is positive.
      "absolute" accept iff both sides gain > min_gain. Correct ONLY if the
        critic is calibrated to win probability, since the four seats' gains
        then sum to zero and my own gain already prices the whole table.
    """

    min_gain: float = 0.015       # was 0.01: fewer, more clearly-worthwhile offers
    criterion: str = "relative"
    random_offers: bool = False   # control: propose randomly, ignore pricing
    propose_chance: float = 0.4   # extra throttle on top of max_per_turn: a
                                  # bot only ATTEMPTS a proposal this often per
                                  # eligible turn (human explicit offers are
                                  # never throttled by this)
    max_per_turn: int = 1         # cap trades per proposer turn (was 2: felt spammy)
    max_candidates: int = 24      # offers scored per attempt (single batch)


@dataclass
class TradeLog:
    entries: List[dict] = field(default_factory=list)
    on_record: Optional[object] = None   # callable(text) for live UIs

    def record(self, proposer, accepter, offer, gains):
        entry = dict(
            proposer=proposer.value, accepter=accepter.value,
            offer=str(offer), proposer_gain=round(gains[0], 5),
            accepter_gain=round(gains[1], 5))
        self.entries.append(entry)
        if self.on_record is not None:
            try:
                self.on_record(f"{proposer.value} \u2192 {accepter.value}: {offer}")
            except Exception:
                pass

    def summary(self):
        if not self.entries:
            return "no trades"
        pg = np.mean([e["proposer_gain"] for e in self.entries])
        ag = np.mean([e["accepter_gain"] for e in self.entries])
        return (f"{len(self.entries)} trades | mean gain: "
                f"proposer {pg:+.4f}, accepter {ag:+.4f}")


def can_pay(state, color, give: dict) -> bool:
    return player_resource_freqdeck_contains(state, color, freqdeck(give))


def enumerate_offers(state, color, cfg: TradeConfig) -> List[Offer]:
    """Affordable single-resource swaps, deepest give-stacks first."""
    hand = {r: player_num_resource_cards(state, color, r) for r in RESOURCES}
    offers = []
    for give_n, ask_n in TRADE_SHAPES:
        for gr in RESOURCES:
            if hand[gr] < give_n:
                continue
            for ar in RESOURCES:
                if ar != gr:
                    offers.append(Offer({gr: give_n}, {ar: ask_n}))
    offers.sort(key=lambda o: -hand[next(iter(o.give))])
    return offers[: cfg.max_candidates]


def execute(state, proposer, accepter, offer: Offer):
    """Move the cards, then refresh legality.

    Engine primitives keep card totals conserved. The engine caches
    playable_actions per tick, so changing hands mid-tick invalidates it: an
    affordable BUY_DEVELOPMENT_CARD can become unaffordable. Regenerate or
    the engine raises on the next action.
    """
    from catanatron.state import generate_playable_actions

    give, ask = freqdeck(offer.give), freqdeck(offer.ask)
    player_deck_subtract(state, proposer, give)
    player_freqdeck_add(state, accepter, give)
    player_deck_subtract(state, accepter, ask)
    player_freqdeck_add(state, proposer, ask)
    state.playable_actions = generate_playable_actions(state)


def is_trade_window(game, color) -> bool:
    """Post-roll main phase: rolled already, and can still act."""
    from catanatron.state_functions import player_has_rolled

    if not player_has_rolled(game.state, color):
        return False
    return any(a.action_type == ActionType.END_TURN
               for a in game.state.playable_actions)


def try_trade(game, proposer_player, cfg: TradeConfig,
              log: Optional[TradeLog] = None) -> Optional[dict]:
    """One proposal round. Returns the executed trade dict, or None."""
    state = game.state
    proposer = proposer_player.color
    candidates = enumerate_offers(state, proposer, cfg)
    if not candidates:
        return None

    if cfg.random_offers:
        # CONTROL ARM: keep the proposing initiative, discard the pricing.
        # Any advantage that survives here comes from proposing at all, not
        # from the critic choosing well.
        import random as _random

        shuffled = candidates[:]
        _random.shuffle(shuffled)
        scored = [(0.0, o) for o in shuffled[:5]]
    else:
        gains = proposer_player.trade_gains(game, candidates)
        floor = 0.0 if cfg.criterion == "relative" else cfg.min_gain
        scored = [(g, o) for g, o in zip(gains, candidates)
                  if g is not None and g > floor]
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])

    # Humans answer through the UI (ask_trade); networks answer via their
    # critic (trade_gain). Humans are asked FIRST so a player at the table
    # always gets right of first refusal instead of only seeing the offers
    # that every bot already declined.
    others = [p for p in state.players
              if p.color != proposer
              and (hasattr(p, "ask_trade") or hasattr(p, "trade_gain"))]
    others.sort(key=lambda p: 0 if hasattr(p, "ask_trade") else 1)
    for my_gain, offer in scored[:5]:
        mirror = offer.mirrored()
        for other in others:
            if not can_pay(state, other.color, mirror.give):
                continue
            if hasattr(other, "ask_trade"):
                # blocks until the browser answers; offer is passed in the
                # proposer's perspective (give = what the human receives)
                if other.ask_trade(game, proposer, offer):
                    execute(state, proposer, other.color, offer)
                    if log is not None:
                        log.record(proposer, other.color, offer, (my_gain, 0.0))
                    return dict(proposer=proposer.value,
                                accepter=other.color.value,
                                offer=str(offer), gains=(my_gain, 0.0))
                continue
            their_gain = other.trade_gain(game, mirror)
            if their_gain is None:
                continue
            if cfg.random_offers:
                # responder still protects itself; only the proposer is random
                ok = their_gain > cfg.min_gain
            elif cfg.criterion == "relative":
                # the counterparty must still want it, and I must come out
                # ahead of them: gains across the table sum to zero, so a
                # mutually "positive" trade means someone is miscalibrated
                ok = (their_gain > 0) and (my_gain - their_gain > cfg.min_gain)
            else:
                ok = their_gain > cfg.min_gain
            if ok:
                execute(state, proposer, other.color, offer)
                if log is not None:
                    log.record(proposer, other.color, offer, (my_gain, their_gain))
                return dict(proposer=proposer.value, accepter=other.color.value,
                            offer=str(offer), gains=(my_gain, their_gain))
    return None


class CriticTrader:
    """Mixin giving a network-backed player a value-based trade opinion.

    Host class must provide: self.color, self.model, self.normalizer,
    self.derived.
    """

    _hand_ix = None
    _total_ix = None

    @classmethod
    def _hand_indices(cls):
        if CriticTrader._hand_ix is None:
            from catanatron_gym.features import get_feature_ordering

            names = get_feature_ordering(4)
            CriticTrader._hand_ix = {
                r: names.index(f"P0_{r}_IN_HAND") for r in RESOURCES}
            CriticTrader._total_ix = names.index("P0_NUM_RESOURCES_IN_HAND")
        return CriticTrader._hand_ix

    def _raw_obs(self, game):
        from catanatron_gym.features import create_sample_vector

        obs = np.asarray(create_sample_vector(game, self.color), dtype=np.float64)
        if getattr(self, "derived", False):
            from .env import derived_features

            obs = np.concatenate([obs, derived_features(game.state, self.color)])
        return obs

    def _values(self, obs_batch: np.ndarray) -> np.ndarray:
        """Critic values for a batch of raw observations: one forward pass."""
        import torch

        if self.normalizer is not None:
            obs_batch = np.stack([self.normalizer(o) for o in obs_batch])
        obs_t = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32))
        mask = torch.ones((obs_t.shape[0], 290), dtype=torch.bool)  # value ignores mask
        with torch.no_grad():
            _, value = self.model.dist_value(obs_t, mask)
        return value.cpu().numpy().reshape(-1)

    def _post_trade_obs(self, base, offer: Offer):
        ix = self._hand_indices()
        after = base.copy()
        delta = 0
        for r, n in offer.give.items():
            after[ix[r]] -= n
            delta -= n
        for r, n in offer.ask.items():
            after[ix[r]] += n
            delta += n
        if any(after[ix[r]] < 0 for r in RESOURCES):
            return None
        after[CriticTrader._total_ix] += delta
        return after

    def _to_prob(self, values: np.ndarray) -> np.ndarray:
        """Map raw critic values to win probability when a calibration exists.

        Without this, the two sides of a trade report gains in incomparable
        units and both can look positive, which is impossible in a race.
        """
        calib = getattr(self.model, "calibration", None)
        if not calib:
            return values
        return np.interp(values, calib["centers"], calib["rates"])

    def trade_gains(self, game, offers: List[Offer]) -> List[Optional[float]]:
        """Batched gain for many offers, in one critic pass.

        Units are win probability if the checkpoint is calibrated, raw value
        units otherwise.
        """
        if self.model is None or not offers:
            return [None] * len(offers)
        base = self._raw_obs(game)
        rows, slots = [base], []
        for o in offers:
            after = self._post_trade_obs(base, o)
            if after is None:
                slots.append(None)
            else:
                slots.append(len(rows))
                rows.append(after)
        vals = self._to_prob(self._values(np.stack(rows)))
        v_now = vals[0]
        return [None if s is None else float(vals[s] - v_now) for s in slots]

    def trade_gain(self, game, offer: Offer) -> Optional[float]:
        """Single-offer wrapper, used when responding to an offer."""
        return self.trade_gains(game, [offer])[0]


def human_propose(game, proposer_color, give: dict, ask: dict,
                  cfg: TradeConfig, log: Optional[TradeLog] = None) -> dict:
    """A human proposes an explicit offer; network players evaluate it.

    Bots judge a human offer on the ABSOLUTE criterion (their own gain over
    the threshold), not the relative one: the relative rule needs both sides'
    gains, and there is no critic estimate of what a human's hand is worth to
    the human. So a bot accepts when the trade is good for the bot, which is
    the honest reading of an offer from an unmodelled player.
    """
    state = game.state
    offer = Offer({k: v for k, v in give.items() if v},
                  {k: v for k, v in ask.items() if v})
    if not offer.give or not offer.ask:
        return dict(ok=False, reason="pick something to give and something to ask")
    if set(offer.give) & set(offer.ask):
        return dict(ok=False, reason="cannot trade a resource for itself")
    if not can_pay(state, proposer_color, offer.give):
        return dict(ok=False, reason="you do not have those cards")

    others = [p for p in state.players
              if p.color != proposer_color and hasattr(p, "trade_gain")]
    mirror = offer.mirrored()
    declined = []
    for other in others:
        if not can_pay(state, other.color, mirror.give):
            declined.append(f"{other.color.value} (cannot pay)")
            continue
        their_gain = other.trade_gain(game, mirror)
        if their_gain is not None and their_gain > cfg.min_gain:
            execute(state, proposer_color, other.color, offer)
            if log is not None:
                log.record(proposer_color, other.color, offer, (0.0, their_gain))
            return dict(ok=True, accepter=other.color.value,
                        offer=str(offer), their_gain=round(their_gain, 4))
        declined.append(other.color.value)
    return dict(ok=False, reason="declined by " + ", ".join(declined))
