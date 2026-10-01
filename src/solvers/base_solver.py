from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    status: str                          # "optimal", "feasible", "infeasible", "error"
    objective_value: Optional[float]
    solution: Dict[str, Any] = field(default_factory=dict)
    solve_time_s: float = 0.0
    gap: Optional[float] = None          # MIP gap, None if not applicable

    @property
    def is_feasible(self) -> bool:
        return self.status in ("optimal", "feasible")

    def __str__(self) -> str:
        gap_str = f"  gap={self.gap:.4%}" if self.gap is not None else ""
        return (
            f"SolverResult(status={self.status}, "
            f"obj={self.objective_value}, "
            f"time={self.solve_time_s:.2f}s{gap_str})"
        )


class BaseSolver(ABC):
    """Abstract base for all OR solvers in the squad-planning pipeline."""

    def __init__(self, name: str, time_limit_s: float = 300.0) -> None:
        self.name = name
        self.time_limit_s = time_limit_s
        self._result: Optional[SolverResult] = None
        self._logger = logging.getLogger(f"{__name__}.{name}")

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_model(self) -> None:
        """Construct the optimization model (variables, constraints, objective)."""

    @abstractmethod
    def solve(self) -> SolverResult:
        """Invoke the solver and return a raw SolverResult."""

    # ------------------------------------------------------------------ #
    # Concrete orchestration
    # ------------------------------------------------------------------ #

    def run(self) -> SolverResult:
        """Build and solve the model with automatic timing and structured logging."""
        self._logger.info("[%s] Building model …", self.name)
        t_build_start = time.perf_counter()

        try:
            self.build_model()
        except Exception as exc:
            self._logger.exception("[%s] Model build failed: %s", self.name, exc)
            return SolverResult(status="error", objective_value=None)

        build_time = time.perf_counter() - t_build_start
        self._logger.info("[%s] Model built in %.2fs. Solving …", self.name, build_time)

        t_solve_start = time.perf_counter()
        try:
            result = self.solve()
        except Exception as exc:
            self._logger.exception("[%s] Solve failed: %s", self.name, exc)
            result = SolverResult(
                status="error",
                objective_value=None,
                solve_time_s=time.perf_counter() - t_solve_start,
            )

        result.solve_time_s = time.perf_counter() - t_solve_start
        self._result = result

        self._logger.info(
            "[%s] Done — status=%s  obj=%s  time=%.2fs",
            self.name,
            result.status,
            result.objective_value,
            result.solve_time_s,
        )
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def last_result(self) -> Optional[SolverResult]:
        return self._result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, time_limit={self.time_limit_s}s)"
