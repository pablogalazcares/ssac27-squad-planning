from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    import pulp
except Exception as exc:  # pragma: no cover - depende del entorno
    pulp = None  # type: ignore[assignment]
    _PULP_IMPORT_ERROR = exc
else:
    _PULP_IMPORT_ERROR = None

from ..solvers.base_solver import BaseSolver, SolverResult

POSITION_GROUPS = ("GK", "DEF", "MID", "FWD")


@dataclass(slots=True)
class SeasonBudget:
    transfer_budget_m: float
    wage_budget_k: float
    max_hires: int


class MultiSeasonSquadPlanner(BaseSolver):
    """
    Optimizador MIP multi-temporada con restricciones equivalentes a legacy.

    Variables:
    - in_squad[p,t] : jugador p en plantilla temporada t
    - hire[p,t]     : jugador p contratado en transición t-1 -> t
    - sell[p,t]     : jugador p vendido/sale en transición t-1 -> t

    Restricciones principales:
    - Presupuesto fichajes por temporada.
    - Masa salarial máxima por temporada.
    - Cupos por posición GK/DEF/MID/FWD.
    - Máximo extranjeros y mínimo sub-23.
    - Máximo contrataciones por temporada.
    - Si un jugador sale, no puede volver.
    - Si se contrata, debe permanecer al menos 2 temporadas.
    - Edad máxima en plantilla.
    - Mejora monotónica de score agregado (opcional).
    """

    def __init__(
        self,
        player_panel: pd.DataFrame,
        config: Mapping[str, Any],
        name: str = "multi_season_squad_planner",
        time_limit_s: float = 300.0,
    ) -> None:
        super().__init__(name=name, time_limit_s=time_limit_s)
        self._ensure_pulp_available()

        self.panel = player_panel.copy()
        self.config = dict(config)

        self._validate_panel()
        self.seasons = sorted(self.panel["season_idx"].astype(int).unique().tolist())
        self.players = sorted(self.panel["player_id"].astype(int).unique().tolist())

        if 0 not in self.seasons:
            raise ValueError("player_panel must include baseline season_idx=0")

        self._position_limits = self._parse_position_limits()
        self._max_foreign = int(self.config.get("max_foreign_players", 17))
        self._min_u23 = int(self.config.get("min_u23_players", 4))
        self._max_age = float(self.config.get("max_age", 35.0))
        self._enforce_monotonic_score = bool(self.config.get("enforce_monotonic_score", True))
        self._enforce_min_two_years = bool(self.config.get("enforce_min_two_years", True))
        self._enforce_no_rehire = bool(self.config.get("enforce_no_rehire", True))

        self._season_budgets = self._parse_season_budgets()

        self.model: pulp.LpProblem | None = None  # type: ignore[type-arg]
        self._vars: dict[str, dict[tuple[int, int], pulp.LpVariable]] = {}  # type: ignore[type-arg]

    def build_model(self) -> None:
        self._ensure_pulp_available()
        assert pulp is not None

        model = pulp.LpProblem(self.name, pulp.LpMaximize)

        idx = [(p, t) for p in self.players for t in self.seasons]
        idx_transitions = [(p, t) for p in self.players for t in self.seasons if t > 0]

        in_squad = pulp.LpVariable.dicts("in_squad", idx, lowBound=0, upBound=1, cat="Binary")
        hire = pulp.LpVariable.dicts("hire", idx_transitions, lowBound=0, upBound=1, cat="Binary")
        sell = pulp.LpVariable.dicts("sell", idx_transitions, lowBound=0, upBound=1, cat="Binary")

        self._vars = {"in_squad": in_squad, "hire": hire, "sell": sell}

        # Objetivo: maximizar score total proyectado en horizontes futuros,
        # opcionalmente descontado temporalmente (discount_factor < 1 aplica
        # una tasa de descuento anual: δ^(t-1) para el periodo t).
        discount = float(self.config.get("discount_factor", 1.0))
        model += pulp.lpSum(
            (discount ** (t - 1)) * self._value(p, t, "score_pred") * in_squad[(p, t)]
            for p in self.players
            for t in self.seasons
            if t > 0
        ), "maximize_total_projected_score"

        # Estado inicial fijado.
        for p in self.players:
            is_initial = int(self._value(p, 0, "is_initial_squad") > 0.5)
            model += in_squad[(p, 0)] == is_initial, f"initial_state_p{p}"

        # Restricciones por temporada.
        for t in self.seasons:
            # Tamaño plantilla total.
            squad_size_expr = pulp.lpSum(in_squad[(p, t)] for p in self.players)

            cfg_size_min = self.config.get("squad_size_min")
            cfg_size_max = self.config.get("squad_size_max")
            # All operational constraints apply only for t>0.
            # t=0 is observed historical data — squad composition is fixed by
            # initial_state constraints and should not be subject to planning
            # restrictions (size cap, wage budget, age limit, foreign quota).
            # This lets clubs with >25 players or exceeded wage caps at t=0
            # release/restructure naturally from t=1 onwards.
            if t > 0:
                if cfg_size_min is not None:
                    model += squad_size_expr >= int(cfg_size_min), f"squad_min_t{t}"
                if cfg_size_max is not None:
                    model += squad_size_expr <= int(cfg_size_max), f"squad_max_t{t}"

                for pos in POSITION_GROUPS:
                    expr = pulp.lpSum(
                        in_squad[(p, t)]
                        for p in self.players
                        if self._value(p, t, "position_group") == pos
                    )
                    model += expr >= self._position_limits[pos]["min"], f"pos_{pos}_min_t{t}"
                    model += expr <= self._position_limits[pos]["max"], f"pos_{pos}_max_t{t}"

                model += (
                    pulp.lpSum(
                        int(self._value(p, t, "is_foreign_epl") > 0.5) * in_squad[(p, t)]
                        for p in self.players
                    )
                    <= self._max_foreign
                ), f"foreign_max_t{t}"

                model += (
                    pulp.lpSum(
                        int(self._value(p, t, "is_u23") > 0.5) * in_squad[(p, t)]
                        for p in self.players
                    )
                    >= self._min_u23
                ), f"u23_min_t{t}"

                wage_budget = self._season_budgets[t].wage_budget_k
                model += (
                    pulp.lpSum(
                        self._value(p, t, "wage_k") * in_squad[(p, t)]
                        for p in self.players
                    )
                    <= wage_budget
                ), f"wage_budget_t{t}"

                for p in self.players:
                    model += (
                        self._value(p, t, "age") * in_squad[(p, t)] <= self._max_age
                    ), f"max_age_p{p}_t{t}"

            for p in self.players:
                model += (
                    in_squad[(p, t)] <= int(self._value(p, t, "is_available") > 0.5)
                ), f"availability_p{p}_t{t}"

        # Restricciones de transición (t>0).
        for t in self.seasons:
            if t == 0:
                continue
            for p in self.players:
                model += (
                    in_squad[(p, t)] - in_squad[(p, t - 1)] <= hire[(p, t)]
                ), f"hire_link_p{p}_t{t}"
                model += (
                    in_squad[(p, t - 1)] - in_squad[(p, t)] <= sell[(p, t)]
                ), f"sell_link_p{p}_t{t}"
                model += hire[(p, t)] + sell[(p, t)] <= 1, f"no_simultaneous_p{p}_t{t}"

            # Máximo contrataciones.
            model += (
                pulp.lpSum(hire[(p, t)] for p in self.players)
                <= self._season_budgets[t].max_hires
            ), f"max_hires_t{t}"

            # Presupuesto fichajes.
            model += (
                pulp.lpSum(
                    self._value(p, t, "market_value_m") * hire[(p, t)]
                    for p in self.players
                )
                <= self._season_budgets[t].transfer_budget_m
            ), f"transfer_budget_t{t}"

        # Si se vende, no puede regresar luego.
        if self._enforce_no_rehire:
            for p in self.players:
                for t_sell in self.seasons:
                    if t_sell == 0:
                        continue
                    for t_hire in self.seasons:
                        if t_hire > t_sell:
                            model += (
                                hire[(p, t_hire)] <= 1 - sell[(p, t_sell)]
                            ), f"no_rehire_p{p}_sell{t_sell}_hire{t_hire}"

        # Si se contrata, se queda al menos 2 temporadas.
        if self._enforce_min_two_years:
            for p in self.players:
                for t in self.seasons:
                    if t == 0:
                        continue
                    if (t + 1) in self.seasons:
                        model += (
                            hire[(p, t)] <= in_squad[(p, t + 1)]
                        ), f"min_two_years_p{p}_t{t}"

        # Mejora de score agregado año a año (opcional).
        if self._enforce_monotonic_score:
            for t in self.seasons:
                if (t + 1) not in self.seasons or t == 0:
                    continue
                left = pulp.lpSum(
                    self._value(p, t, "score_pred") * in_squad[(p, t)]
                    for p in self.players
                )
                right = pulp.lpSum(
                    self._value(p, t + 1, "score_pred") * in_squad[(p, t + 1)]
                    for p in self.players
                )
                model += left <= right, f"monotonic_score_t{t}_to_t{t+1}"

        self.model = model

    def solve(self) -> SolverResult:
        self._ensure_pulp_available()
        assert pulp is not None

        if self.model is None:
            raise RuntimeError("Model not built. Run build_model() first.")

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=float(self.time_limit_s))
        self.model.solve(solver)

        raw_status = pulp.LpStatus.get(self.model.status, "Undefined")
        status = self._map_status(raw_status, self.model)
        objective = pulp.value(self.model.objective)

        solution = self._extract_solution() if status in {"optimal", "feasible"} else {}
        return SolverResult(
            status=status,
            objective_value=float(objective) if objective is not None else None,
            solution=solution,
            gap=None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_solution(self) -> dict[str, Any]:
        if self.model is None:
            return {}

        in_squad = self._vars["in_squad"]
        hire = self._vars["hire"]
        sell = self._vars["sell"]

        def _b(var: pulp.LpVariable) -> int:  # type: ignore[type-arg]
            val = var.value()
            return int(val is not None and val >= 0.5)

        roster_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for t in self.seasons:
            selected = []
            transfer_spend = 0.0
            wage_mass = 0.0
            total_score = 0.0
            hires_count = 0
            sells_count = 0
            foreign_count = 0
            u23_count = 0

            # Conteo correcto de altas y bajas: comparar in_squad[t] vs in_squad[t-1].
            # El flag sell del modelo solo está disponible para jugadores con in_squad[t]=1,
            # por lo que no captura jugadores que realmente salen (in_squad[t]=0).
            if t > 0:
                for p in self.players:
                    prev = _b(in_squad[(p, t - 1)])
                    curr = _b(in_squad[(p, t)])
                    if prev == 0 and curr == 1:
                        hires_count += 1
                    elif prev == 1 and curr == 0:
                        sells_count += 1

            for p in self.players:
                in_val = _b(in_squad[(p, t)])
                if not in_val:
                    continue

                name = str(self._value(p, t, "player_name"))
                team_name = str(self._value(p, t, "team_name")) if "team_name" in self.panel.columns else ""
                pos_group = str(self._value(p, t, "position_group"))
                score = float(self._value(p, t, "score_pred"))
                mv = float(self._value(p, t, "market_value_m"))
                wage = float(self._value(p, t, "wage_k"))
                age = float(self._value(p, t, "age"))
                is_foreign = int(self._value(p, t, "is_foreign_epl") > 0.5)
                is_u23 = int(self._value(p, t, "is_u23") > 0.5)

                # hire=1 si el jugador es nuevo en la plantilla (vino del pool)
                hire_val = 1 if (t > 0 and _b(in_squad[(p, t - 1)]) == 0) else 0
                # sell=0 aquí: los que salen no aparecen en el roster (in_squad=0)
                sell_val = 0

                if hire_val:
                    transfer_spend += mv

                wage_mass += wage
                total_score += score
                foreign_count += is_foreign
                u23_count += is_u23

                selected.append(
                    {
                        "season_idx": t,
                        "season_name": self._season_name(t),
                        "player_id": p,
                        "player_name": name,
                        "team_name": team_name,
                        "position_group": pos_group,
                        "score_pred": score,
                        "market_value_m": mv,
                        "wage_k": wage,
                        "age": age,
                        "is_foreign": is_foreign,
                        "is_u23": is_u23,
                        "hire": hire_val,
                        "sell": sell_val,
                        "is_initial_squad": int(self._value(p, 0, "is_initial_squad") > 0.5),
                    }
                )

            roster_rows.extend(selected)

            summary_rows.append(
                {
                    "season_idx": t,
                    "season_name": self._season_name(t),
                    "n_players": len(selected),
                    "team_score": round(total_score, 3),
                    "transfer_spend_m": round(transfer_spend, 3),
                    "transfer_budget_m": self._season_budgets[t].transfer_budget_m,
                    "wage_mass_k": round(wage_mass, 3),
                    "wage_budget_k": self._season_budgets[t].wage_budget_k,
                    "n_hires": hires_count,
                    "max_hires": self._season_budgets[t].max_hires,
                    "n_sells": sells_count,
                    "n_foreign": foreign_count,
                    "max_foreign": self._max_foreign,
                    "n_u23": u23_count,
                    "min_u23": self._min_u23,
                }
            )

        return {
            "roster": roster_rows,
            "season_summary": summary_rows,
        }

    def _validate_panel(self) -> None:
        required = [
            "player_id",
            "player_name",
            "season_idx",
            "season_name",
            "position_group",
            "score_pred",
            "market_value_m",
            "wage_k",
            "age",
            "is_foreign_epl",
            "is_u23",
            "is_initial_squad",
            "is_available",
        ]
        missing = [col for col in required if col not in self.panel.columns]
        if missing:
            # Compatibilidad hacia atrás con paneles antiguos.
            if missing == ["is_available"]:
                self.panel["is_available"] = True
            else:
                raise ValueError(f"player_panel missing required columns: {missing}")

        self.panel["season_idx"] = pd.to_numeric(self.panel["season_idx"], errors="coerce").astype(int)
        self.panel["player_id"] = pd.to_numeric(self.panel["player_id"], errors="coerce").astype(int)

        for col in ["score_pred", "market_value_m", "wage_k", "age"]:
            self.panel[col] = pd.to_numeric(self.panel[col], errors="coerce")

        if self.panel[["score_pred", "market_value_m", "wage_k", "age"]].isna().any().any():
            raise ValueError("player_panel contains NaN in numeric optimization columns")

        # Unicidad jugador-temporada.
        dup = self.panel.duplicated(subset=["player_id", "season_idx"], keep=False)
        if dup.any():
            repeated = self.panel.loc[dup, ["player_id", "season_idx"]].head(10).to_dict("records")
            raise ValueError(
                "player_panel has duplicated player_id-season_idx combinations. "
                f"Examples: {repeated}"
            )

    def _parse_position_limits(self) -> dict[str, dict[str, int]]:
        default_limits = {
            "GK": {"min": 2, "max": 3},
            "DEF": {"min": 7, "max": 10},
            "MID": {"min": 6, "max": 10},
            "FWD": {"min": 4, "max": 8},
        }

        cfg_limits = self.config.get("position_limits", {})
        limits: dict[str, dict[str, int]] = {}
        for pos in POSITION_GROUPS:
            src = cfg_limits.get(pos, default_limits[pos]) if isinstance(cfg_limits, Mapping) else default_limits[pos]
            limits[pos] = {"min": int(src.get("min", default_limits[pos]["min"])), "max": int(src.get("max", default_limits[pos]["max"]))}
            if limits[pos]["min"] > limits[pos]["max"]:
                raise ValueError(f"position_limits invalid for {pos}: min > max")
        return limits

    def _parse_season_budgets(self) -> dict[int, SeasonBudget]:
        seasons = sorted(self.panel["season_idx"].astype(int).unique().tolist())

        transfer_cfg = self.config.get("transfer_budget_m", 40.0)
        wage_cfg = self.config.get("wage_budget_k", 850.0)
        hires_cfg = self.config.get("max_hires_per_season", 4)

        budgets: dict[int, SeasonBudget] = {}
        for t in seasons:
            transfer = self._season_value(transfer_cfg, t, default=40.0)
            wage = self._season_value(wage_cfg, t, default=850.0)
            hires = int(self._season_value(hires_cfg, t, default=4.0))
            budgets[t] = SeasonBudget(
                transfer_budget_m=float(transfer),
                wage_budget_k=float(wage),
                max_hires=int(max(0, hires)),
            )
        return budgets

    @staticmethod
    def _season_value(cfg: Any, season_idx: int, default: float) -> float:
        if isinstance(cfg, Mapping):
            if season_idx in cfg:
                return float(cfg[season_idx])
            key = str(season_idx)
            if key in cfg:
                return float(cfg[key])
            return float(default)
        if isinstance(cfg, (list, tuple)):
            if season_idx < len(cfg):
                return float(cfg[season_idx])
            return float(cfg[-1])
        return float(cfg)

    def _value(self, player_id: int, season_idx: int, column: str) -> Any:
        row = self.panel.loc[
            (self.panel["player_id"] == player_id) & (self.panel["season_idx"] == season_idx),
            column,
        ]
        if row.empty:
            raise KeyError(f"Missing value for player_id={player_id}, season_idx={season_idx}, column={column}")
        return row.iloc[0]

    def _season_name(self, season_idx: int) -> str:
        row = self.panel.loc[self.panel["season_idx"] == season_idx, "season_name"]
        if row.empty:
            return str(season_idx)
        return str(row.iloc[0])

    @staticmethod
    def _map_status(raw_status: str, model: pulp.LpProblem) -> str:  # type: ignore[type-arg]
        if raw_status == "Optimal":
            return "optimal"
        if raw_status in {"Infeasible", "Unbounded"}:
            return "infeasible"
        has_incumbent = any(var.value() is not None for var in model.variables())
        if raw_status in {"Undefined", "Not Solved"} and has_incumbent:
            return "feasible"
        if has_incumbent:
            return "feasible"
        return "error"

    @staticmethod
    def _ensure_pulp_available() -> None:
        if pulp is None:
            raise ImportError("PuLP no está disponible. Instala dependencias.") from _PULP_IMPORT_ERROR


__all__ = ["MultiSeasonSquadPlanner", "SeasonBudget", "POSITION_GROUPS"]
