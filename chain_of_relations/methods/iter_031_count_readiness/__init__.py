"""Gold-free contracts for Iteration 31 COUNT execution readiness."""

from .path_completeness import (
    PathCompletenessResult,
    PathEvidence,
    PathStep,
    check_path_completeness,
)
from .type_resolution import (
    TargetTypeEvidence,
    TargetTypeResolution,
    resolve_target_type,
)
from .runtime_contract import (
    CONTRACT_VERSION,
    AdapterOutcome,
    CountPlanContract,
    ReadOnlyQueryBudget,
    RuntimeCallsite,
    callsite_from_cor_shape,
    execute_bounded_select,
    is_single_read_only_count_select,
    run_count_adapter,
)

__all__ = [
    "PathCompletenessResult",
    "PathEvidence",
    "PathStep",
    "AdapterOutcome",
    "CONTRACT_VERSION",
    "CountPlanContract",
    "ReadOnlyQueryBudget",
    "RuntimeCallsite",
    "TargetTypeEvidence",
    "TargetTypeResolution",
    "check_path_completeness",
    "callsite_from_cor_shape",
    "execute_bounded_select",
    "is_single_read_only_count_select",
    "resolve_target_type",
    "run_count_adapter",
]
