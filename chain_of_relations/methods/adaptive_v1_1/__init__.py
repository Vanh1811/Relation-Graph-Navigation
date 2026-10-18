"""Adaptive Branch Routing v1.1.

The execution agent is imported lazily so the pure routing validator can be
unit-tested without importing the optional OpenAI client.
"""

from chain_of_relations.methods.adaptive_v1_1.router import AdaptiveRouter

__all__ = ["AdaptiveRouter", "AdaptiveRoutingAgent"]


def __getattr__(name: str):
	if name == "AdaptiveRoutingAgent":
		from chain_of_relations.methods.adaptive_v1_1.agent import AdaptiveRoutingAgent

		return AdaptiveRoutingAgent
	raise AttributeError(name)
