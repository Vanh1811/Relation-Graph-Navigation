"""Complementary-property branch validation for Adaptive v1.1.4."""

from chain_of_relations.methods.adaptive_v1_1_4.router import (
	AdaptiveRouter,
	RouterConfig,
	RouterDecision,
	extract_requested_answer_target,
)

__all__ = [
	"AdaptiveRouter",
	"RouterConfig",
	"RouterDecision",
	"AdaptiveRoutingAgent",
	"extract_requested_answer_target",
]


def __getattr__(name: str):
	if name == "AdaptiveRoutingAgent":
		from chain_of_relations.methods.adaptive_v1_1_4.agent import AdaptiveRoutingAgent

		return AdaptiveRoutingAgent
	raise AttributeError(name)
