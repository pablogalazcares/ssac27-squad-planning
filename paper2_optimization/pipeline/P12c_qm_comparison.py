"""
P12c — Q/M Cross-club Comparison (H4)
======================================
Objetivo:
    Para los 20 clubes EPL 2023/24 y los dos escenarios (I uniforme, II
    club-específico), comparar el ratio Q/M (score proyectado por euro
    gastado) del MIP vs los fichajes reales de verano 2024.

    Q/M = Σ_p score_h1_p / Σ_p fee_p, donde p son los fichajes del conjunto.

Definición operacional:
    - Q_MIP  = sum(score_h1) sobre jugadores recomendados por MIP en t=1
    - M_MIP  = sum(mv_h1)    sobre los mismos jugadores
    - Q_real = sum(score_h1) sobre jugadores realmente comprados que están
               en el universo C (proyectables)
    - M_real = sum(fee_TM)   sobre los mismos jugadores (solo proyectables)

    Los jugadores reales comprados que NO están en C se excluyen del cálculo
    (limitación de scope, no afectan ni numerador ni denominador de Q/M_real).

Hipótesis H4: ≥80% de los 20 clubes con Q/M_MIP > Q/M_real, en ambos escenarios.

Outputs:
    outputs/pipeline/p12c_qm_per_club.csv
        Una fila por (club, escenario) con:
          n_real_total, n_real_in_C, n_real_not_in_C,
          q_mip, m_mip, qm_mip,
          q_real_proj, m_real_proj, qm_real_proj,
          delta_pct_qm, fee_lost_total_m (gasto no analizable)

Upstream:
    P10: outputs/pipeline/p10_optimizer_panel.parquet (universo C + scores h1 + mv h1)
    P11: outputs/pipeline/p11_e5a/scenario_{I,II}/{slug}/result_full_h5.json
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
    REPO, OUT_DIR,
    OPT_PANEL as PANEL_PATH,
    TM_2024_PATH as TM_PATH,
    P11_OUT,
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


EPL_2324_CLUBS = [
    "Arsenal", "Aston Villa", "AFC Bournemouth", "Brentford",
    "Brighton & Hove Albion", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Liverpool", "Luton Town", "Manchester City",
    "Manchester United", "Newcastle United", "Nottingham Forest",
    "Sheffield United", "Tottenham Hotspur", "West Ham United",
    "Wolverhampton Wanderers",
]

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
    "Luton Town":               "Luton",
    "Manchester City":         "Manchester City",
    "Manchester United":       "Manchester United",
    "Newcastle United":         "Newcastle",
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


def load_universe_lookup(panel: pd.DataFrame) -> dict:
    """
    Construye un lookup name_normalized → {player_id, score_h1, mv_h1}
    a partir del panel P10 (universo C).
    """
    lkp = {}
    for _, r in panel.iterrows():
        key = _normalize(r["player_name"])
        lkp[key] = {
            "player_id": int(r["player_id"]),
            "score_h1":  float(r.get("score_h1", np.nan)),
            "mv_h1":     float(r.get("mv_h1",    np.nan)),  # euros
        }
    return lkp


def load_real_transfers(club_name: str) -> pd.DataFrame:
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
    slug = _slugify(club_name)
    result_path = P11_OUT / f"scenario_{scenario}" / slug / "result_full_h5.json"
    if not result_path.exists():
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


def compute_qm(scenario: str, panel: pd.DataFrame, universe_lkp: dict) -> pd.DataFrame:
    rows = []
    for club in EPL_2324_CLUBS:
        try:
            real = load_real_transfers(club)
            mip_names = load_mip_hires_t1(scenario, club)

            # ── Q/M MIP ──────────────────────────────────────────────────────
            q_mip = 0.0
            m_mip_euros = 0.0
            for name in mip_names:
                key = _normalize(name)
                info = universe_lkp.get(key)
                if info is None or np.isnan(info["score_h1"]) or np.isnan(info["mv_h1"]):
                    continue
                q_mip += info["score_h1"]
                m_mip_euros += info["mv_h1"]
            m_mip_m = m_mip_euros / 1_000_000.0  # → M€
            qm_mip = (q_mip / m_mip_m) if m_mip_m > 0 else np.nan

            # ── Q/M real (solo jugadores en C) ───────────────────────────────
            n_real_total    = len(real)
            n_real_in_C     = 0
            n_real_not_in_C = 0
            q_real          = 0.0
            m_real_m        = 0.0
            fee_lost_m      = 0.0  # gasto en jugadores no proyectables
            real_in_C_names: list[str] = []
            real_not_in_C_names: list[str] = []
            for _, r in real.iterrows():
                key = _normalize(r["player_name"])
                fee = float(r["transfer_fee_m"])
                if key in universe_lkp and not np.isnan(universe_lkp[key]["score_h1"]):
                    n_real_in_C += 1
                    q_real += universe_lkp[key]["score_h1"]
                    m_real_m += fee
                    real_in_C_names.append(r["player_name"])
                else:
                    n_real_not_in_C += 1
                    fee_lost_m += fee
                    real_not_in_C_names.append(r["player_name"])
            qm_real = (q_real / m_real_m) if m_real_m > 0 else np.nan
            delta_pct_qm = ((qm_mip - qm_real) / qm_real * 100) \
                if qm_real and qm_real > 0 and not np.isnan(qm_mip) else np.nan

            rows.append({
                "club_name":           club,
                "scenario":            scenario,
                "n_real_total":        n_real_total,
                "n_real_in_C":         n_real_in_C,
                "n_real_not_in_C":     n_real_not_in_C,
                "q_mip":               round(q_mip, 2),
                "m_mip_m":             round(m_mip_m, 3),
                "qm_mip":              round(qm_mip, 3) if not np.isnan(qm_mip) else np.nan,
                "q_real_proj":         round(q_real, 2),
                "m_real_proj_m":       round(m_real_m, 3),
                "qm_real_proj":        round(qm_real, 3) if not np.isnan(qm_real) else np.nan,
                "delta_pct_qm":        round(delta_pct_qm, 2) if not np.isnan(delta_pct_qm) else np.nan,
                "fee_lost_total_m":    round(fee_lost_m, 3),
                "real_in_C_names":     real_in_C_names,
                "real_not_in_C_names": real_not_in_C_names,
            })
            print(f"  {club:30s} n_in_C={n_real_in_C}/{n_real_total} "
                  f"Q/M_MIP={qm_mip:.3f}  Q/M_real={qm_real if not np.isnan(qm_real) else 'NA'}  "
                  f"Δ%={delta_pct_qm if not np.isnan(delta_pct_qm) else 'NA'}")
        except Exception as exc:
            print(f"  [ERROR] {club}: {exc}")
            rows.append({"club_name": club, "scenario": scenario, "error": str(exc)})

    return pd.DataFrame(rows)


def evaluate_h4(df: pd.DataFrame) -> dict:
    """Evalúa H4 sobre el DataFrame de salida."""
    out = {}
    for scenario in ("I", "II"):
        sub = df[df["scenario"] == scenario].dropna(subset=["qm_mip", "qm_real_proj"])
        n = len(sub)
        if n == 0:
            out[scenario] = {"n": 0, "n_mip_wins": 0, "pct_mip_wins": np.nan,
                             "median_delta_pct": np.nan, "pass": False}
            continue
        n_wins = (sub["qm_mip"] > sub["qm_real_proj"]).sum()
        pct = n_wins / n * 100
        median_d = sub["delta_pct_qm"].median()
        out[scenario] = {
            "n":               int(n),
            "n_mip_wins":      int(n_wins),
            "pct_mip_wins":    round(pct, 1),
            "median_delta_pct": round(float(median_d), 2),
            "pass":            bool(pct >= 80),
        }
    out["h4_pass_both"] = bool(out["I"].get("pass") and out["II"].get("pass"))
    return out


def main() -> pd.DataFrame:
    print("=" * 60)
    print("P12c — Q/M cross-club comparison (H4)")
    print("=" * 60)

    assert PANEL_PATH.exists(), \
        f"[ERROR] {PANEL_PATH} no encontrado — correr P10 primero."

    panel = pd.read_parquet(PANEL_PATH)
    universe_lkp = load_universe_lookup(panel)
    print(f"  Universo C cargado: {len(universe_lkp)} jugadores")

    all_rows = []
    for scenario in ("I", "II"):
        if not (P11_OUT / f"scenario_{scenario}").exists():
            print(f"  [WARN] Escenario {scenario} no encontrado en P11 — saltando")
            continue
        print(f"\n  --- Escenario {scenario} ---")
        df_s = compute_qm(scenario, panel, universe_lkp)
        all_rows.append(df_s)

    if not all_rows:
        raise RuntimeError("No hay outputs de P11 para procesar.")

    df_all = pd.concat(all_rows, ignore_index=True)
    out_path = OUT_DIR / "p12c_qm_per_club.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n  Output: {out_path}")

    # Evaluación H4
    h4 = evaluate_h4(df_all)
    h4_path = OUT_DIR / "p12c_h4_evaluation.csv"
    pd.DataFrame([
        {"scenario": "I",  **h4.get("I", {})},
        {"scenario": "II", **h4.get("II", {})},
        {"scenario": "BOTH", "pass": h4.get("h4_pass_both", False)},
    ]).to_csv(h4_path, index=False)

    print(f"\n  === H4 — Q/M MIP > Q/M real ===")
    for sc in ("I", "II"):
        info = h4.get(sc, {})
        print(f"  Escenario {sc}: {info.get('n_mip_wins', 0)}/{info.get('n', 0)} "
              f"({info.get('pct_mip_wins', 'NA')}%) | mediana Δ% = "
              f"{info.get('median_delta_pct', 'NA')}% | {'PASS' if info.get('pass') else 'FAIL'}")
    print(f"  H4 (ambos escenarios): "
          f"{'PASS' if h4.get('h4_pass_both') else 'FAIL'}")

    return df_all


if __name__ == "__main__":
    main()
