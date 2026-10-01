from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class Player:
    id: str
    name: str
    position: str  # GK, CB, LB, RB, CM, CAM, CDM, LW, RW, ST
    age: int
    value_m: float
    wage_k: float
    performance_score: float  # 0–100
    injury_risk: float  # 0–1
    seasons_remaining: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.performance_score <= 100.0:
            raise ValueError(f"performance_score must be in [0, 100], got {self.performance_score}")
        if not 0.0 <= self.injury_risk <= 1.0:
            raise ValueError(f"injury_risk must be in [0, 1], got {self.injury_risk}")
        if self.age < 15 or self.age > 45:
            raise ValueError(f"Unrealistic age: {self.age}")

    @property
    def is_young(self) -> bool:
        return self.age < 23

    @property
    def is_expiring(self) -> bool:
        return self.seasons_remaining <= 1


@dataclass
class Squad:
    players: List[Player] = field(default_factory=list)
    wage_budget: float = 0.0       # weekly total in k€
    transfer_budget: float = 0.0   # in M€
    season: int = 1

    # ------------------------------------------------------------------ #
    # Computed properties
    # ------------------------------------------------------------------ #

    @property
    def total_wage(self) -> float:
        return sum(p.wage_k for p in self.players)

    @property
    def total_value(self) -> float:
        return sum(p.value_m for p in self.players)

    @property
    def age_profile(self) -> Dict[str, float]:
        if not self.players:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        ages = [p.age for p in self.players]
        return {
            "mean": sum(ages) / len(ages),
            "min": float(min(ages)),
            "max": float(max(ages)),
        }

    @property
    def positions_count(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.players:
            counts[p.position] = counts.get(p.position, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # Feasibility checks
    # ------------------------------------------------------------------ #

    def is_wage_feasible(self) -> bool:
        return self.total_wage <= self.wage_budget

    def is_size_valid(self, min_size: int = 18, max_size: int = 25) -> bool:
        return min_size <= len(self.players) <= max_size

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_by_position(self, position: str) -> List[Player]:
        return [p for p in self.players if p.position == position]

    def get_by_id(self, player_id: str) -> Optional[Player]:
        for p in self.players:
            if p.id == player_id:
                return p
        return None

    def add_player(self, player: Player) -> None:
        if any(p.id == player.id for p in self.players):
            raise ValueError(f"Player {player.id} already in squad")
        self.players.append(player)

    def remove_player(self, player_id: str) -> Player:
        for i, p in enumerate(self.players):
            if p.id == player_id:
                return self.players.pop(i)
        raise KeyError(f"Player {player_id} not found in squad")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict:
        profile = self.age_profile
        return {
            "season": self.season,
            "squad_size": len(self.players),
            "wage_budget_k": self.wage_budget,
            "total_wage_k": round(self.total_wage, 2),
            "wage_utilisation_pct": round(
                100 * self.total_wage / self.wage_budget, 1
            ) if self.wage_budget else 0.0,
            "transfer_budget_m": self.transfer_budget,
            "total_value_m": round(self.total_value, 2),
            "mean_age": round(profile["mean"], 1),
            "positions": self.positions_count,
            "wage_feasible": self.is_wage_feasible(),
            "size_valid": self.is_size_valid(),
            "expiring_contracts": sum(1 for p in self.players if p.is_expiring),
            "young_players": sum(1 for p in self.players if p.is_young),
        }
