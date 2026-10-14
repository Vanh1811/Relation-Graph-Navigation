"""State objects for the Phase 1 two-branch prototype."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BranchState:
	branch_id: str
	relation_chain: list[str] = field(default_factory=list)
	status: str = "ACTIVE"
	history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RelationGraphState:
	shared_answer_variable: str | None = "?answer"
	branches: list[BranchState] = field(default_factory=list)
	merge_operator: str = "AND"

	def __post_init__(self) -> None:
		if len(self.branches) != 2:
			raise ValueError("RelationGraphState requires exactly two branches")
		if self.merge_operator != "AND":
			raise ValueError("Branch-v1 supports only the AND merge operator")
