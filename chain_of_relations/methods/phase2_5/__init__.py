"""Phase 2.5 constraint-first routing."""

from .router import ConstraintFirstDecision, ConstraintFirstRouter

__all__ = ["ConstraintFirstDecision", "ConstraintFirstRouter"]


def __getattr__(name: str):
	if name == "ConstraintFirstRoutingAgent":
		from .agent import ConstraintFirstRoutingAgent

		return ConstraintFirstRoutingAgent
	raise AttributeError(name)
