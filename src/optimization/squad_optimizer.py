from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import pulp
except Exception as exc:  # pragma: no cover - depende del entorno
    pulp = None  # type: ignore[assignment]
    _PULP_IMPORT_ERROR = exc
else:
    _PULP_IMPORT_ERROR = None

from ..models.squad import Player
from ..solvers.base_solver import BaseSolver, SolverResult


POSITION_GROUPS = ("GK", "DEF", "MID", "FWD")
POSITION_BUCKET_MAP = {
    "GK": "GK",
    "DEF": "DEF",
    "CB": "DEF",
    "LCB": "DEF",
    "RCB": "DEF",
    "LB": "DEF",
    "RB": "DEF",
    "LWB": "DEF",
    "RWB": "DEF",
    "MID": "MID",
    "CDM": "MID",
    "DM": "MID",
    "CM": "MID",
    "CAM": "MID",
    "AM": "MID",
    "LM": "MID",
    "RM": "MID",
    "FWD": "FWD",
    "LW": "FWD",
    "RW": "FWD",
    "CF": "FWD",
    "ST": "FWD",
    "SS": "FWD",
}


@dataclass(frozen=True, slots=True)
class _PlayerContext:
    player: Player
    position_group: str
    is_current: bool
    is_market: bool
    is_academy: bool
    is_foreign: bool
    is_expiring: bool


class SquadOptimizer(BaseSolver):
    """
    Modelo MIP de una temporada para planificación de plantilla.

    Variables binarias de decisión por jugador:
    - hire[j]    : contratar
    - fire[j]    : despedir/salir del plantel
    - renew[j]   : renovar contrato
    - promote[j] : promover desde cantera
    """

    def __init__(
        self,
        players: Sequence[Player | Mapping[str, Any]],
        config: Mapping[str, Any],
        name: str = "squad_optimizer",
        time_limit_s: float = 300.0,
    ) -> None:
        super().__init__(name=name, time_limit_s=time_limit_s)
        self._ensure_pulp_available()

        self.config: Dict[str, Any] = dict(config)
        self._player_contexts = self._normalize_players(players, self.config)

        self._transfer_budget_m = self._get_required_float("transfer_budget_m")
        self._max_wage_mass_k = self._get_required_float("max_wage_mass_k", "wage_budget_k")
        self._max_foreign_players = self._get_required_int(
            "max_foreign_players", "max_foreigners"
        )
        self._min_u23_players = self._get_required_int("min_u23_players", "min_sub23_players")
        self._position_limits = self._parse_position_limits()

        self._squad_size_min: Optional[int] = self._get_optional_int("squad_size_min")
        self._squad_size_max: Optional[int] = self._get_optional_int("squad_size_max")
        if (
            self._squad_size_min is not None
            and self._squad_size_max is not None
            and self._squad_size_min > self._squad_size_max
        ):
            raise ValueError("squad_size_min cannot be greater than squad_size_max")

        self.model: Optional[pulp.LpProblem] = None  # type: ignore[type-arg]
        self._variables: Dict[str, Dict[str, pulp.LpVariable]] = {}  # type: ignore[type-arg]

    def build_model(self) -> None:
        self._ensure_pulp_available()
        assert pulp is not None

        if not self._player_contexts:
            raise ValueError("At least one player is required to build the optimization model")

        player_ids = [ctx.player.id for ctx in self._player_contexts]
        problem = pulp.LpProblem(self.name, pulp.LpMaximize)

        select = pulp.LpVariable.dicts("select", player_ids, lowBound=0, upBound=1, cat="Binary")
        hire = pulp.LpVariable.dicts("hire", player_ids, lowBound=0, upBound=1, cat="Binary")
        fire = pulp.LpVariable.dicts("fire", player_ids, lowBound=0, upBound=1, cat="Binary")
        renew = pulp.LpVariable.dicts("renew", player_ids, lowBound=0, upBound=1, cat="Binary")
        promote = pulp.LpVariable.dicts(
            "promote", player_ids, lowBound=0, upBound=1, cat="Binary"
        )
        self._variables = {
            "select": select,
            "hire": hire,
            "fire": fire,
            "renew": renew,
            "promote": promote,
        }

        problem += pulp.lpSum(
            ctx.player.performance_score * select[ctx.player.id] for ctx in self._player_contexts
        ), "maximize_squad_performance"

        for ctx in self._player_contexts:
            pid = ctx.player.id

            if ctx.is_current:
                problem += hire[pid] == 0, f"no_hire_current_{pid}"
                problem += promote[pid] == 0, f"no_promote_current_{pid}"
                if ctx.is_expiring:
                    problem += fire[pid] == 0, f"no_fire_expiring_{pid}"
                    problem += select[pid] == renew[pid], f"renew_link_{pid}"
                else:
                    problem += renew[pid] == 0, f"no_renew_nonexpiring_{pid}"
                    problem += select[pid] + fire[pid] == 1, f"retain_or_fire_{pid}"
                continue

            if ctx.is_market:
                problem += fire[pid] == 0, f"no_fire_market_{pid}"
                problem += renew[pid] == 0, f"no_renew_market_{pid}"
                problem += promote[pid] == 0, f"no_promote_market_{pid}"
                problem += select[pid] == hire[pid], f"hire_link_{pid}"
                continue

            if ctx.is_academy:
                problem += fire[pid] == 0, f"no_fire_academy_{pid}"
                problem += renew[pid] == 0, f"no_renew_academy_{pid}"
                problem += hire[pid] == 0, f"no_hire_academy_{pid}"
                problem += select[pid] == promote[pid], f"promote_link_{pid}"
                continue

            problem += hire[pid] == 0, f"no_hire_other_{pid}"
            problem += fire[pid] == 0, f"no_fire_other_{pid}"
            problem += renew[pid] == 0, f"no_renew_other_{pid}"
            problem += promote[pid] == 0, f"no_promote_other_{pid}"
            problem += select[pid] == 0, f"not_select_other_{pid}"

        problem += (
            pulp.lpSum(ctx.player.value_m * hire[ctx.player.id] for ctx in self._player_contexts)
            <= self._transfer_budget_m
        ), "transfer_budget_limit"

        problem += (
            pulp.lpSum(ctx.player.wage_k * select[ctx.player.id] for ctx in self._player_contexts)
            <= self._max_wage_mass_k
        ), "wage_budget_limit"

        for group_name, limits in self._position_limits.items():
            group_expr = pulp.lpSum(
                select[ctx.player.id]
                for ctx in self._player_contexts
                if ctx.position_group == group_name
            )
            problem += group_expr >= limits["min"], f"position_{group_name}_min"
            problem += group_expr <= limits["max"], f"position_{group_name}_max"

        problem += (
            pulp.lpSum(
                int(ctx.is_foreign) * select[ctx.player.id] for ctx in self._player_contexts
            )
            <= self._max_foreign_players
        ), "foreign_players_limit"

        problem += (
            pulp.lpSum(
                select[ctx.player.id] for ctx in self._player_contexts if ctx.player.age < 23
            )
            >= self._min_u23_players
        ), "u23_minimum"

        if self._squad_size_min is not None:
            problem += (
                pulp.lpSum(select[ctx.player.id] for ctx in self._player_contexts)
                >= self._squad_size_min
            ), "squad_size_min"
        if self._squad_size_max is not None:
            problem += (
                pulp.lpSum(select[ctx.player.id] for ctx in self._player_contexts)
                <= self._squad_size_max
            ), "squad_size_max"

        self.model = problem

    def solve(self) -> SolverResult:
        self._ensure_pulp_available()
        assert pulp is not None

        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() before solve().")

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=float(self.time_limit_s))
        self.model.solve(solver)

        raw_status = pulp.LpStatus.get(self.model.status, "Undefined")
        mapped_status = self._map_status(raw_status)
        objective = pulp.value(self.model.objective)
        solution = self._extract_solution()

        return SolverResult(
            status=mapped_status,
            objective_value=None if objective is None else float(objective),
            solution=solution,
            gap=None,
        )

    def _extract_solution(self) -> Dict[str, Any]:
        if self.model is None or not self._variables:
            return {}

        def _b(var: pulp.LpVariable) -> int:  # type: ignore[type-arg]
            value = var.value()
            return int(value is not None and value >= 0.5)

        selected: list[str] = []
        hired: list[str] = []
        fired: list[str] = []
        renewed: list[str] = []
        promoted: list[str] = []

        total_transfer_spend = 0.0
        total_wage = 0.0
        position_counts = {group: 0 for group in POSITION_GROUPS}
        foreign_count = 0
        u23_count = 0

        for ctx in self._player_contexts:
            pid = ctx.player.id
            select_val = _b(self._variables["select"][pid])
            hire_val = _b(self._variables["hire"][pid])
            fire_val = _b(self._variables["fire"][pid])
            renew_val = _b(self._variables["renew"][pid])
            promote_val = _b(self._variables["promote"][pid])

            if select_val:
                selected.append(pid)
                total_wage += ctx.player.wage_k
                position_counts[ctx.position_group] += 1
                foreign_count += int(ctx.is_foreign)
                u23_count += int(ctx.player.age < 23)

            if hire_val:
                hired.append(pid)
                total_transfer_spend += ctx.player.value_m
            if fire_val:
                fired.append(pid)
            if renew_val:
                renewed.append(pid)
            if promote_val:
                promoted.append(pid)

        return {
            "selected_players": selected,
            "hired_players": hired,
            "fired_players": fired,
            "renewed_players": renewed,
            "promoted_players": promoted,
            "position_counts": position_counts,
            "foreign_players_count": foreign_count,
            "u23_players_count": u23_count,
            "total_transfer_spend_m": round(total_transfer_spend, 4),
            "transfer_budget_m": self._transfer_budget_m,
            "total_wage_mass_k": round(total_wage, 4),
            "max_wage_mass_k": self._max_wage_mass_k,
        }

    def _map_status(self, raw_status: str) -> str:
        if raw_status == "Optimal":
            return "optimal"
        if raw_status in {"Infeasible", "Unbounded"}:
            return "infeasible"

        has_incumbent = any(var.value() is not None for var in self.model.variables()) if self.model else False
        if has_incumbent:
            return "feasible"
        if raw_status == "Not Solved":
            return "error"
        return "error"

    def _normalize_players(
        self,
        players: Sequence[Player | Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> list[_PlayerContext]:
        if not isinstance(players, Sequence):
            raise TypeError("players must be a sequence of Player or mapping objects")

        current_ids = set(self._to_str_set(config.get("current_squad_ids", [])))
        market_ids = set(self._to_str_set(config.get("market_player_ids", [])))
        academy_ids = set(self._to_str_set(config.get("academy_player_ids", [])))
        foreign_ids = set(self._to_str_set(config.get("foreign_player_ids", [])))
        expiring_ids = set(self._to_str_set(config.get("expiring_player_ids", [])))

        contexts: list[_PlayerContext] = []
        seen_ids: set[str] = set()

        for raw in players:
            if isinstance(raw, Player):
                player = raw
                metadata: Mapping[str, Any] = {}
                default_current = True
            elif isinstance(raw, Mapping):
                player = self._player_from_mapping(raw)
                metadata = raw
                source = str(raw.get("source", "")).strip().lower()
                source_is_market = source == "market"
                source_is_academy = source == "academy"
                source_is_current = source == "current"
                default_current = source_is_current or (
                    not source_is_market
                    and not source_is_academy
                    and not bool(raw.get("is_market", False))
                    and not bool(raw.get("is_academy", False))
                )
            else:
                raise TypeError("Each player must be Player or mapping")

            if player.id in seen_ids:
                raise ValueError(f"Duplicated player id detected: {player.id!r}")
            seen_ids.add(player.id)

            source = str(metadata.get("source", "")).strip().lower()
            is_current = bool(
                metadata.get("is_current", source == "current" or player.id in current_ids or default_current)
            )
            is_market = bool(metadata.get("is_market", source == "market" or player.id in market_ids))
            is_academy = bool(
                metadata.get("is_academy", source == "academy" or player.id in academy_ids)
            )
            is_foreign = bool(
                metadata.get(
                    "is_foreign", metadata.get("foreign", player.id in foreign_ids)
                )
            )
            is_expiring = bool(
                metadata.get("is_expiring", player.id in expiring_ids or player.is_expiring)
            )

            if is_current and (is_market or is_academy):
                raise ValueError(
                    f"Player {player.id!r} cannot be both current squad and market/academy"
                )
            if is_market and is_academy:
                raise ValueError(f"Player {player.id!r} cannot be both market and academy")

            position_group = self._map_position_group(player.position)
            contexts.append(
                _PlayerContext(
                    player=player,
                    position_group=position_group,
                    is_current=is_current,
                    is_market=is_market,
                    is_academy=is_academy,
                    is_foreign=is_foreign,
                    is_expiring=is_expiring,
                )
            )

        return contexts

    def _parse_position_limits(self) -> Dict[str, Dict[str, int]]:
        raw_limits = self.config.get("position_limits")
        if not isinstance(raw_limits, Mapping):
            raise ValueError(
                "config['position_limits'] is required and must be a mapping with GK/DEF/MID/FWD"
            )

        limits: Dict[str, Dict[str, int]] = {}
        player_count = len(self._player_contexts)

        for group in POSITION_GROUPS:
            group_limit = raw_limits.get(group)
            if not isinstance(group_limit, Mapping):
                raise ValueError(
                    f"position_limits[{group!r}] must be a mapping with 'min' and 'max'"
                )
            min_value = group_limit.get("min")
            max_value = group_limit.get("max")
            if min_value is None or max_value is None:
                raise ValueError(f"position_limits[{group!r}] must define both 'min' and 'max'")

            min_int = int(min_value)
            max_int = int(max_value)
            if min_int < 0:
                raise ValueError(f"position_limits[{group!r}]['min'] must be >= 0")
            if max_int < 0:
                raise ValueError(f"position_limits[{group!r}]['max'] must be >= 0")
            if min_int > max_int:
                raise ValueError(f"position_limits[{group!r}] has min greater than max")
            if max_int > player_count:
                raise ValueError(
                    f"position_limits[{group!r}]['max'] cannot exceed total players ({player_count})"
                )

            limits[group] = {"min": min_int, "max": max_int}

        return limits

    def _get_required_float(self, *keys: str) -> float:
        value = self._get_any_key(*keys)
        if value is None:
            key_list = ", ".join(keys)
            raise ValueError(f"Missing required numeric config parameter. Expected one of: {key_list}")
        parsed = float(value)
        if parsed < 0:
            raise ValueError(f"Config parameter {keys[0]!r} must be >= 0")
        return parsed

    def _get_required_int(self, *keys: str) -> int:
        value = self._get_any_key(*keys)
        if value is None:
            key_list = ", ".join(keys)
            raise ValueError(f"Missing required integer config parameter. Expected one of: {key_list}")
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"Config parameter {keys[0]!r} must be >= 0")
        return parsed

    def _get_optional_int(self, key: str) -> Optional[int]:
        if key not in self.config:
            return None
        value = int(self.config[key])
        if value < 0:
            raise ValueError(f"Config parameter {key!r} must be >= 0")
        return value

    def _get_any_key(self, *keys: str) -> Any:
        for key in keys:
            if key in self.config:
                return self.config[key]
        return None

    def _player_from_mapping(self, raw: Mapping[str, Any]) -> Player:
        required = {
            "id",
            "name",
            "position",
            "age",
            "value_m",
            "wage_k",
            "performance_score",
            "injury_risk",
            "seasons_remaining",
        }
        missing = [field for field in required if field not in raw]
        if missing:
            raise ValueError(f"Player mapping is missing required fields: {missing}")

        return Player(
            id=str(raw["id"]),
            name=str(raw["name"]),
            position=str(raw["position"]),
            age=int(raw["age"]),
            value_m=float(raw["value_m"]),
            wage_k=float(raw["wage_k"]),
            performance_score=float(raw["performance_score"]),
            injury_risk=float(raw["injury_risk"]),
            seasons_remaining=int(raw["seasons_remaining"]),
        )

    def _map_position_group(self, position: str) -> str:
        normalized = position.strip().upper()
        if normalized not in POSITION_BUCKET_MAP:
            supported = ", ".join(sorted(POSITION_BUCKET_MAP))
            raise ValueError(f"Unsupported position {position!r}. Supported values: {supported}")
        return POSITION_BUCKET_MAP[normalized]

    def _to_str_set(self, value: Any) -> set[str]:
        if value is None:
            return set()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("Expected a sequence of player ids in config")
        return {str(item) for item in value}

    def _ensure_pulp_available(self) -> None:
        if pulp is None:
            raise ImportError(
                "PuLP is not available. Install dependencies and retry."
            ) from _PULP_IMPORT_ERROR
