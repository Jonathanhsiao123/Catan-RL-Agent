"""Natural-language negotiation layer.

The LLM parses and phrases. The critic in catan_rl.trading decides.
Nothing in this package can move cards: only a TradeOffer that passed
validation can be turned into a trading.Offer, and only trading.execute
moves cards.
"""

from .schema import (
    RESOURCES,
    SEATS,
    Clarify,
    ParseResult,
    Rejected,
    TradeOffer,
    validate,
    validate_in_game,
)

__all__ = [
    "RESOURCES",
    "SEATS",
    "Clarify",
    "ParseResult",
    "Rejected",
    "TradeOffer",
    "validate",
    "validate_in_game",
]
