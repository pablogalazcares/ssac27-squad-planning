"""
P12 — Experimento E5b: Decision Regret (20 clubes EPL 2023/24)
==============================================================
Objetivo:
    Comparar los fichajes recomendados por el MIP en t=1 con los fichajes
    realmente ejecutados por cada club en el mercado de verano 2024.

    Para los 20 clubes EPL 2023/24 y los dos escenarios (I uniforme,
    II club-específico), calcula:
      - n_real_buys      : número de fichajes reales (Transfermarkt)
      - n_mip_hires_t1   : número de fichajes recomendados por el MIP en t=1
      - n_overlap        : intersección por nombre normalizado
      - overlap_pct      : % de overlap sobre los fichajes reales
      - lock_in_*        : columnas de lock-in importadas de P11

Diferencia con P11:
    P11 ejecuta el MIP. P12 NO ejecuta el MIP — carga los rosters guardados
    por P11 y solo agrega el análisis de overlap con datos reales.
    Esto evita duplicar ~30 min de cómputo y garantiza consistencia.

Diferencia con P12c:
    P12 mide overlap por NOMBRE de jugador.
    P12c mide Q/M (score por euro) — métrica continua, no binaria.

Outputs:
    outputs/pipeline/p12_e5b/
        e5b_summary_scenario_I.csv   — 20 clubes, Escenario I
        e5b_summary_scenario_II.csv  — 20 clubes, Escenario II
        e5b_regret_report.txt        — informe narrativo

Upstream:
    P11: outputs/pipeline/p11_e5a/scenario_*/{slug}/result_full_h5.json
    P11: outputs/pipeline/p11_e5a/e5a_summary_scenario_*.csv (lock-in info)
    EXTERNAL: data/external/transfermarkt/epl_summer_2024_transfers.json
"""

from __future__ import annotations

import json
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (  # noqa: E402
    REPO,
    OPT_PANEL as PANEL_PATH,
    TM_2024_PATH as TM_PATH,
    P11_OUT,
    P12_OUT as OUT_DIR,
)

sys.path.insert(0, str(REPO))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lista canónica de los 20 clubes EPL 2023/24 (idéntica a P11.EPL_2324_CLUBS)
EPL_2324_CLUBS = [
    "Arsenal", "Aston Villa", "AFC Bournemouth", "Brentford",
    "Brighton & Hove Albion", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Liverpool", "Luton Town", "Manchester City",
    "Manchester United", "Newcastle United", "Nottingham Forest",
    "Sheffield United", "Tottenham Hotspur", "West Ham United",
    "Wolverhampton Wanderers",
]

# Mapeo de nombres EPL 23/24 (P10) → nombre que aparece en TM JSON
# Se construye sobre la marcha buscando substrings; ver _match_tm_club_name().
TM_CLUB_HINTS = {
    "AFC Bournemouth":         "Bournemouth",
    "Arsenal":                 "Arsenal",
    "Aston Villa":             "Aston Villa",
    "Brentford":               "Brentford",
    "Brighton & Hove Albion":  "Brighton",
    "Burnley":                 "Burnley",
    "Chelsea":                 "Chelsea",
    "Crystal Palace":          "Crystal Palace",
    "Everton":                 "Everton",
    "Fulham":                  "Fulham",
    "Liverpool":               "Liverpool",
    "Luton Town":              "Luton",
    "Manchester City":         "Manchester City",
    "Manchester United":       "Manchester United",
    "Newcastle United":        "Newcastle",
    "Nottingham Forest":       "Nottingham Forest",
    "Sheffield United":        "Sheffield",
    "Tottenham Hotspur":       "Tottenham",
    "West Ham United":         "West Ham",
    "Wolverhampton Wanderers": "Wolverhampton",
}


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("&", "and").replace(".", "")


def load_real_transfers(club_name: str) -> pd.DataFrame:
    """Carga fichajes reales del verano 2024 para un club (entradas, no préstamos)."""
    data = json.loads(Path(TM_PATH).read_text())
    df   = pd.DataFrame(data)
    hint = TM_CLUB_HINTS.get(club_name, club_name)
    mask = (
        df["club_name"].str.contains(hint, case=False, na=False)
        & (df["direction"] == "in")
        & (~df["is_loan"].fillna(False))
        & (df["transfer_fee_m"].fillna(0) > 0)
    )
    return df[mask][["player_name", "transfer_fee_m", "from_club"]].reset_index(drop=True)


def load_mip_hires_t1(scenario: str, club_name: str) -> list[str]:
    """
    Carga los nombres de jugadores recomendados por el MIP en t=1 desde el
    output de P11 (Full H=5).
    """
    slug = _slugify(club_name)
    result_path = P11_OUT / f"scenario_{scenario}" / slug / "result_full_h5.json"
    if not result_path.exists():
        print(f"  [WARN] {result_path} no existe — devolviendo lista vacía.")
        return []
    with open(result_path) as f:
        data = json.load(f)
    sol = data.get("solution") or {}
    roster = sol.get("roster") if isinstance(sol, dict) else None
    if not roster:
        return []
    roster_df = pd.DataFrame(roster) if not isinstance(roster, pd.DataFrame) else roster
    if "season_idx" not in roster_df.columns:
        return []
    t1 = roster_df[roster_df["season_idx"] == 1]
    if "hire" in t1.columns:
        return t1[t1["hire"] == 1]["player_name"].astype(str).tolist()
    return t1["player_name"].astype(str).tolist() if "player_name" in t1.columns else []


def load_p11_summary(scenario: str) -> pd.DataFrame:
    """Carga el resumen P11 con scores y lock-in info."""
    p = P11_OUT / f"e5a_summary_scenario_{scenario}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} no encontrado — correr P11 con --scenario {scenario} primero."
        )
    return pd.read_csv(p)


def compute_overlap(mip_hires: list[str], real_buys: pd.DataFrame) -> dict:
    """Calcula overlap por nombre normalizado."""
    mip_set  = {_normalize(n) for n in mip_hires}
    real_set = {_normalize(n) for n in real_buys["player_name"].tolist()}
    overlap  = mip_set & real_set
    return {
        "n_mip_hires_t1":  len(mip_set),
        "n_real_buys":     len(real_set),
        "n_overlap":       len(overlap),
        "overlap_pct":     round(len(overlap) / max(len(real_set), 1) * 100, 1),
        "overlap_players": sorted(overlap),
        "mip_hires":       sorted(mip_set),
        "real_buys":       sorted(real_set),
    }


def run_scenario(scenario: str) -> pd.DataFrame:
    print("=" * 60)
    print(f"P12 — Decision Regret | Escenario {scenario}")
    print("=" * 60)

    p11_summary = load_p11_summary(scenario)
    p11_indexed = p11_summary.set_index("club_name") if "club_name" in p11_summary.columns \
                  else p11_summary

    rows = []
    for club in EPL_2324_CLUBS:
        try:
            real_buys = load_real_transfers(club)
            mip_hires = load_mip_hires_t1(scenario, club)
            overlap   = compute_overlap(mip_hires, real_buys)

            # Importar info de lock-in desde P11
            p11_row = p11_indexed.loc[club] if club in p11_indexed.index else None

            entry = {
                "club_name":         club,
                "scenario":          scenario,
                "n_real_buys":       overlap["n_real_buys"],
                "n_mip_hires_t1":    overlap["n_mip_hires_t1"],
                "n_overlap":         overlap["n_overlap"],
                "overlap_pct":       overlap["overlap_pct"],
                "overlap_players":   overlap["overlap_players"],
                "mip_hires_t1":      overlap["mip_hires"],
                "real_buys":         overlap["real_buys"],
            }
            if p11_row is not None:
                entry.update({
                    "h5_score":               p11_row.get("h5_score", np.nan),
                    "rolling_h3_score":       p11_row.get("rolling_h3_score", np.nan),
                    "myopic_score":           p11_row.get("myopic_score", np.nan),
                    "delta_pct_h5_myop":      p11_row.get("delta_pct_h5_myop", np.nan),
                    "delta_pct_roll_myop":    p11_row.get("delta_pct_roll_myop", np.nan),
                    "lock_in_h5":             p11_row.get("lock_in_h5", False),
                    "lock_in_h5_reason":      p11_row.get("lock_in_h5_reason", None),
                    "lock_in_rolling":        p11_row.get("lock_in_rolling", False),
                    "lock_in_rolling_reason": p11_row.get("lock_in_rolling_reason", None),
                    "lock_in_myopic":         p11_row.get("lock_in_myopic", False),
                    "lock_in_myopic_reason":  p11_row.get("lock_in_myopic_reason", None),
                })
            rows.append(entry)
            print(f"  {club:30s}  real={overlap['n_real_buys']:2d}  "
                  f"mip={overlap['n_mip_hires_t1']:2d}  "
                  f"overlap={overlap['n_overlap']:2d}  ({overlap['overlap_pct']:5.1f}%)")
        except Exception as exc:
            print(f"  [ERROR] {club}: {exc}")
            rows.append({"club_name": club, "scenario": scenario, "error": str(exc)})

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / f"e5b_summary_scenario_{scenario}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Output: {out_path}")
    return df


def main() -> dict[str, pd.DataFrame]:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["I", "II", "BOTH"], default="BOTH")
    args = parser.parse_args()

    results: dict[str, pd.DataFrame] = {}
    if args.scenario in ("I", "BOTH"):
        results["I"] = run_scenario("I")
    if args.scenario in ("II", "BOTH"):
        results["II"] = run_scenario("II")
    return results


if __name__ == "__main__":
    main()
