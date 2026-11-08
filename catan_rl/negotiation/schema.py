"""TradeOffer data model and hard validation.

Validation is the safety boundary of the whole negotiation layer. The LLM
proposes a candidate; this module decides whether it is a real, legal offer.
An invalid candidate is rejected with a reason and is never auto-fixed,
because "fixing" an LLM's output means guessing what it meant.

Seats are engine color names ("RED", "BLUE", "ORANGE", "WHITE") rather than
the integers in the spec, so a TradeOffer maps onto engine state with no
lookup table.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

RESOURCES: Tuple[str, ...] = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
SEATS: Tuple[str, ...] = ("RED", "BLUE", "ORANGE", "WHITE")
MIN_COUNT, MAX_COUNT = 1, 4

# Table-talk names -> engine names. Plurals are handled by stripping a
# trailing "s" before lookup.
_SYNONYMS = {
    "wood": "WOOD", "lumber": "WOOD", "log": "WOOD", "tree": "WOOD",
    "brick": "BRICK", "clay": "BRICK",
    "sheep": "SHEEP", "wool": "SHEEP",
    "wheat": "WHEAT", "grain": "WHEAT",
    "ore": "ORE", "stone": "ORE", "rock": "ORE",
}


def normalize_resource(name) -> Optional[str]:
    """Map 'Lumber', 'woods', 'WOOD' etc. to an engine name, else None."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    if key.upper() in RESOURCES:
        return key.upper()
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    if key.endswith("s") and key[:-1] in _SYNONYMS:
        return _SYNONYMS[key[:-1]]
    return None


@dataclass(frozen=True)
class TradeOffer:
    """A structured offer. give/get are from the proposer's perspective."""

    from_seat: str
    to_seat: Optional[str]          # None = open offer to the table
    give: Tuple[Tuple[str, int], ...]
    get: Tuple[Tuple[str, int], ...]
    dropped_clauses: Tuple[str, ...] = ()   # logged, never acted on

    @staticmethod
    def make(from_seat, to_seat, give: Dict[str, int], get: Dict[str, int],
             dropped_clauses: List[str] = ()) -> "TradeOffer":
        """Build with canonical ordering so equal offers compare equal."""
        return TradeOffer(
            from_seat=from_seat,
            to_seat=to_seat,
            give=tuple(sorted(give.items())),
            get=tuple(sorted(get.items())),
            dropped_clauses=tuple(dropped_clauses),
        )

    @property
    def give_dict(self) -> Dict[str, int]:
        return dict(self.give)

    @property
    def get_dict(self) -> Dict[str, int]:
        return dict(self.get)

    def terms(self) -> tuple:
        """What the trade actually does. Used for exact-match comparisons:
        dropped clauses are commentary, not terms."""
        return (self.to_seat, self.give, self.get)

    def to_engine_offer(self):
        """Convert to trading.Offer, the only type trading.execute accepts."""
        from ..trading import Offer
        return Offer(give=self.give_dict, ask=self.get_dict)

    def template_text(self) -> str:
        """Deterministic phrasing; the phraser's fallback."""
        def fmt(items):
            return " and ".join(f"{n} {r.lower()}" for r, n in items)
        to = f" to {self.to_seat}" if self.to_seat else ""
        return f"{self.from_seat} offers {fmt(self.give)} for {fmt(self.get)}{to}."


@dataclass(frozen=True)
class Clarify:
    """The utterance names a want but not terms. Ask, don't guess."""

    question: str


@dataclass(frozen=True)
class Rejected:
    """The candidate failed validation, or the LLM output was unusable."""

    reason: str
    raw: Optional[str] = None       # the LLM output, kept for the E1 error log


ParseResult = Union[TradeOffer, Clarify, Rejected]


def _check_side(side: Dict, label: str) -> Optional[str]:
    if not isinstance(side, dict) or not side:
        return f"{label} is empty"
    for r, n in side.items():
        if r not in RESOURCES:
            return f"{label}: unknown resource {r!r}"
        if isinstance(n, bool) or not isinstance(n, int):
            return f"{label}: count for {r} is not an integer"
        if not MIN_COUNT <= n <= MAX_COUNT:
            return f"{label}: count {n} for {r} outside {MIN_COUNT}..{MAX_COUNT}"
    return None


def validate(offer: TradeOffer, hand: Optional[Dict[str, int]] = None,
             seats: Optional[Tuple[str, ...]] = None) -> Union[TradeOffer, Rejected]:
    """Hard checks. Returns the offer unchanged, or Rejected with a reason.

    hand:  the proposer's resource counts; when given, the proposer must
           hold everything it offers.
    seats: the colors actually at this table; defaults to all four.
    """
    seats = seats or SEATS
    if offer.from_seat not in seats:
        return Rejected(f"unknown proposer {offer.from_seat!r}")
    if offer.to_seat is not None:
        if offer.to_seat not in seats:
            return Rejected(f"unknown recipient {offer.to_seat!r}")
        if offer.to_seat == offer.from_seat:
            return Rejected("cannot trade with yourself")

    give, get = offer.give_dict, offer.get_dict
    for side, label in ((give, "give"), (get, "get")):
        err = _check_side(side, label)
        if err:
            return Rejected(err)
    overlap = set(give) & set(get)
    if overlap:
        return Rejected(f"same resource on both sides: {sorted(overlap)}")

    if hand is not None:
        for r, n in give.items():
            if hand.get(r, 0) < n:
                return Rejected(f"proposer holds {hand.get(r, 0)} {r}, offered {n}")
    return offer


def hand_of(state, color_name: str) -> Dict[str, int]:
    """Resource counts for one seat, read from engine state."""
    from catanatron.models.player import Color
    from catanatron.state_functions import player_num_resource_cards

    color = Color[color_name]
    return {r: player_num_resource_cards(state, color, r) for r in RESOURCES}


def validate_in_game(offer: TradeOffer, game) -> Union[TradeOffer, Rejected]:
    """validate() against live engine state: real seats, real hand."""
    seats = tuple(p.color.value for p in game.state.players)
    if offer.from_seat not in seats:
        return Rejected(f"unknown proposer {offer.from_seat!r}")
    return validate(offer, hand=hand_of(game.state, offer.from_seat), seats=seats)
