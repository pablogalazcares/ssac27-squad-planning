"""
P10 — Panel Unificado del Optimizador
========================================
Objetivo:
    Integrar todos los inputs necesarios del optimizador MIP en un único
    DataFrame plano por (player_id, horizon), con tests de completitud y
    caracterización de la calidad de cada fuente.

    El panel resultante es el input directo de P11 (E5a, 20 clubes) y P12 (E5b).
    GARANTIZA que ningún player-horizon tiene NaN en las variables críticas.

Completitud requerida (paper §8, variables del MIP):
    Por jugador:
      - player_id, player_name, position_group, team_name
      - age_base
    Por jugador × horizonte h=0..H:
      - score_{p,h}         ← P07
      - mv_{p,h}            ← P08 (en €)
      - wage_annual_{p,h}   ← P09 (en £/año)

Caracterización de calidad (columnas *_source):
    score_source:   "observed" | "projected" | "imputed_pos"
    mv_source:      "observed" | "f3_from_t0" | "imputed_pos" | "f3_recursive"
    wage_source:    "direct" | "direct_t0" | "fuzzy" | "imputed_pos_score" | "imputed_global"

Tests de completitud (inline):
    - N jugadores = exactamente los del universo P06
    - Ningún NaN en score, mv, wage para cualquier horizonte
    - player_id sin duplicados
    - position_group ∈ {GK, DEF, MID, FWD}
    - age_base ∈ [15, 45]
    - Distribución posicional: MID < 35%

Tests de plausibilidad:
    - score ∈ [0, 100]
    - mv > 0
    - wage > 0
    - Para h=0: mv_h0 y wage_h0 consistentes con EDA (rango EPL conocido)

Outputs:
    outputs/pipeline/p10_optimizer_panel.parquet
    outputs/pipeline/p10_quality_report.csv  — una fila por jugador, fuentes de cada var

Upstream:
    P06: outputs/pipeline/p06_projection_universe.parquet
    P07: outputs/pipeline/p07_projected_scores.parquet
    P08: outputs/pipeline/p08_mv_panel.parquet
    P09: outputs/pipeline/p09_wage_panel.parquet
         outputs/pipeline/p09_wage_coverage.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (  # noqa: E402
    REPO, OUT_DIR, UNIVERSE_PATH, MV_PANEL, WAGE_PANEL, WAGE_COVERAGE,
    PROJ_PATH as PROJ_SCORES,
)

HORIZON = 5


def completeness_check(
    universe: pd.DataFrame,
    proj: pd.DataFrame,
    mv: pd.DataFrame,
    wages: pd.DataFrame,
    horizon: int,
) -> None:
    """
    Test previo que verifica que todos los datasets tienen los mismos jugadores
    y que las columnas necesarias están presentes. Falla rápido si hay inconsistencia.
    """
    N = len(universe)
    ids_universe = set(universe["player_id"].unique())
    ids_proj     = set(proj["player_id"].unique())
    ids_mv       = set(mv["player_id"].unique())
    ids_wage     = set(wages["player_id"].unique())

    # Cobertura de cada source
    missing_proj  = ids_universe - ids_proj
    missing_mv    = ids_universe - ids_mv
    missing_wage  = ids_universe - ids_wage

    assert len(missing_proj) == 0, \
        (f"[ERROR] {len(missing_proj)} jugadores del universo sin scores proyectados.\n"
         f"Ejemplo: {list(missing_proj)[:5]}\nCorrer P07.")

    assert len(missing_mv) == 0, \
        (f"[ERROR] {len(missing_mv)} jugadores del universo sin panel MV.\n"
         f"Ejemplo: {list(missing_mv)[:5]}\nCorrer P08.")

    assert len(missing_wage) == 0, \
        (f"[ERROR] {len(missing_wage)} jugadores del universo sin panel salarial.\n"
         f"Ejemplo: {list(missing_wage)[:5]}\nCorrer P09.")

    # Columnas requeridas
    score_cols = [f"score_h{h}" for h in range(horizon + 1)]
    mv_cols    = [f"mv_h{h}" for h in range(horizon + 1)]
    wage_cols  = [f"wage_annual_h{h}" for h in range(horizon + 1)]

    for col in score_cols:
        assert col in proj.columns, f"[ERROR] Columna faltante en P07: {col}"
    for col in mv_cols:
        assert col in mv.columns, f"[ERROR] Columna faltante en P08: {col}"
    for col in wage_cols:
        assert col in wages.columns, f"[ERROR] Columna faltante en P09: {col}"

    print(f"  [COMPLETENESS OK] {N} jugadores cubiertos en P07 + P08 + P09")


def build_panel(
    universe: pd.DataFrame,
    proj: pd.DataFrame,
    mv: pd.DataFrame,
    wages: pd.DataFrame,
    wage_cov: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Integra todos los inputs en un único panel wide por jugador.

    Returns:
        (panel, quality_report)
        panel: una fila por jugador, columnas score/mv/wage para h=0..H
        quality_report: una fila por jugador con fuentes de cada variable
    """
    # Base: universe
    base_cols = ["player_id", "player_name", "position_group", "team_name",
                 "age_base", "set_label"]
    panel = universe[base_cols].copy()

    # Scores proyectados
    score_cols = [f"score_h{h}" for h in range(horizon + 1)]
    panel = panel.merge(proj[["player_id"] + score_cols], on="player_id", how="left")

    # MV panel
    mv_cols = [f"mv_h{h}" for h in range(horizon + 1)]
    panel = panel.merge(mv[["player_id"] + mv_cols], on="player_id", how="left")

    # Wage panel
    wage_cols = [f"wage_annual_h{h}" for h in range(horizon + 1)]
    panel = panel.merge(wages[["player_id"] + wage_cols], on="player_id", how="left")

    # ── Quality report ────────────────────────────────────────────────────────
    qr = universe[["player_id", "player_name", "position_group", "set_label"]].copy()

    # Score source — set_label='B' (jugadores en 23/24) vs 'E' (solo en 22/23)
    qr["score_source"] = np.where(
        qr["set_label"] == "B", "observed_2324",
        np.where(qr["set_label"] == "E", "projected_from_2223", "imputed_pos")
    )

    # MV source
    qr = qr.merge(universe[["player_id", "mv_base"]].rename(
        columns={"mv_base": "mv_h0_input"}
    ), on="player_id", how="left")
    qr["mv_source_h0"] = np.where(
        qr["mv_h0_input"].notna(),
        np.where(qr["set_label"] == "B", "observed_2324", "f3_from_2223"),
        "imputed_pos",
    )
    for h in range(1, horizon + 1):
        qr[f"mv_source_h{h}"] = "f3_recursive"

    # Wage source
    if wage_cov is not None and "wage_source" in wage_cov.columns:
        qr = qr.merge(wage_cov[["player_id", "wage_source"]], on="player_id", how="left")
        qr["wage_source"] = qr["wage_source"].fillna("imputed_global")
    else:
        qr["wage_source"] = "unknown"

    # Limpiar columnas auxiliares del quality report
    qr = qr.drop(columns=["mv_h0_input", "set_label"], errors="ignore")

    return panel.drop(columns=["set_label"], errors="ignore"), qr


def validate_panel(panel: pd.DataFrame, horizon: int) -> None:
    """Tests de regresión del panel unificado."""
    N = len(panel)

    assert panel["player_id"].duplicated().sum() == 0, \
        "player_id duplicados en panel final"

    assert panel["position_group"].isin({"GK", "DEF", "MID", "FWD"}).all(), \
        f"position_group con valores inesperados: " \
        f"{panel[~panel['position_group'].isin({'GK','DEF','MID','FWD'})]['position_group'].unique()}"

    mid_pct = (panel["position_group"] == "MID").mean()
    assert mid_pct < 0.35, \
        f"MID={mid_pct:.1%} > 35% — bug de imputación posicional"

    valid_age = panel["age_base"].dropna()
    assert (valid_age >= 15).all() and (valid_age <= 45).all(), \
        "age_base fuera de [15, 45]"

    # Completitud de columnas críticas
    for h in range(horizon + 1):
        for col in [f"score_h{h}", f"mv_h{h}", f"wage_annual_h{h}"]:
            n_nan = panel[col].isna().sum()
            assert n_nan == 0, \
                f"[ERROR] {n_nan} NaN en columna {col} — completitud rota"
            vals = panel[col]
            assert (vals >= 0).all(), f"Valores negativos en {col}"
            if col.startswith("score"):
                assert (vals <= 100).all(), f"score > 100 en {col}"
            elif col.startswith("mv"):
                assert (vals > 0).all(), f"MV = 0 en {col}"
            elif col.startswith("wage"):
                assert (vals > 0).all(), f"Wage = 0 en {col}"

    print(f"  [TESTS OK] Panel completo: {N} jugadores × {horizon+1} horizontes")
    print(f"  Distribución posicional:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        n = (panel["position_group"] == pos).sum()
        print(f"    {pos}: {n} ({n/N:.1%})")

    # Estadísticas descriptivas del panel h=0
    print(f"\n  Estadísticas h=0:")
    print(f"    score_h0:      mean={panel['score_h0'].mean():.1f}, "
          f"std={panel['score_h0'].std():.1f}")
    print(f"    mv_h0:         mean={panel['mv_h0'].mean()/1e6:.1f}M€, "
          f"median={panel['mv_h0'].median()/1e6:.1f}M€")
    print(f"    wage_h0:       mean={panel['wage_annual_h0'].mean()/1e6:.2f}M£, "
          f"median={panel['wage_annual_h0'].median()/1e6:.2f}M£")

    # Club wage bills (check feasibility del constraint salarial)
    if "team_name" in panel.columns:
        club_bills = (panel.groupby("team_name")["wage_annual_h0"].sum() / 1e6).sort_values(ascending=False)
        print(f"\n  Club wage bills h=0 (£M/año): "
              f"max={club_bills.max():.1f}, median={club_bills.median():.1f}, "
              f"min={club_bills.min():.1f}")


def main(horizon: int = HORIZON) -> pd.DataFrame:
    print("=" * 60)
    print(f"P10 — Panel Unificado del Optimizador (H={horizon})")
    print("=" * 60)

    for path, name in [(UNIVERSE_PATH, "P06"), (PROJ_SCORES, "P07"),
                       (MV_PANEL, "P08"), (WAGE_PANEL, "P09")]:
        assert path.exists(), \
            f"[ERROR] Input no encontrado: {path} ({name}). Correr step anterior."

    universe = pd.read_parquet(UNIVERSE_PATH)
    proj     = pd.read_parquet(PROJ_SCORES)
    mv       = pd.read_parquet(MV_PANEL)
    wages    = pd.read_parquet(WAGE_PANEL)
    wage_cov = pd.read_csv(WAGE_COVERAGE) if WAGE_COVERAGE.exists() else None

    print(f"\n  Universo:        {len(universe)} jugadores")
    print(f"  Scores P07:      {len(proj)} jugadores")
    print(f"  MV panel P08:    {len(mv)} jugadores")
    print(f"  Wage panel P09:  {len(wages)} jugadores")

    # Test previo de completitud
    print(f"\n  Test de completitud...")
    completeness_check(universe, proj, mv, wages, horizon)

    # Construir panel
    print(f"\n  Construyendo panel unificado...")
    panel, quality_report = build_panel(universe, proj, mv, wages, wage_cov, horizon)

    # Validar
    validate_panel(panel, horizon)

    # Guardar
    out_path = OUT_DIR / "p10_optimizer_panel.parquet"
    panel.to_parquet(out_path, index=False)

    qr_path = OUT_DIR / "p10_quality_report.csv"
    quality_report.to_csv(qr_path, index=False)

    print(f"\n  Outputs:")
    print(f"    {out_path}")
    print(f"    {qr_path}")
    return panel


if __name__ == "__main__":
    main()
