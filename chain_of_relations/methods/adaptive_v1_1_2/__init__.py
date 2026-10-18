"""Rule-aware Adaptive Routing v1.1.2."""

from chain_of_relations.methods.adaptive_v1_1_2.router import (
	AdaptiveRouter,
	RouterConfig,
	RouterDecision,
)

__all__ = ["AdaptiveRouter", "RouterConfig", "RouterDecision", "AdaptiveRoutingAgent"]


def __getattr__(name: str):
	if name == "AdaptiveRoutingAgent":
		from chain_of_relations.methods.adaptive_v1_1_2.agent import AdaptiveRoutingAgent

		return AdaptiveRoutingAgent
	raise AttributeError(name)
