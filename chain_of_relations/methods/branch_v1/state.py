"""State objects shared by the frozen Branch-v1 and adaptive routing runs."""

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
		if len(self.branches) not in (2, 3):
			raise ValueError("RelationGraphState requires two or three branches")
		if self.merge_operator != "AND":
			raise ValueError("Branch execution supports only the AND merge operator")
