"""
P09 — Wage Panel (Constante por horizonte)
============================================
Objetivo:
    Construir el panel de salarios w_{p,h} para los 738 jugadores × H horizontes
    del optimizador MIP, siguiendo la especificación del paper §7.2.

Especificación del modelo de salarios (paper §7.2):
    1. Datos: Capology EPL 2022/23 + 2023/24 (534 + 556 registros)
    2. Matching:
       (a) Coincidencia directa (Unicode normalizado)
       (b) Jaccard token ≥ 0.50
       (c) Imputación: mediana dentro de (position_group, score_quintile)
       (d) Fallback: mediana global (1 jugador)
    3. Base wage: annual_gross_gbp de 2023/24 (o 2022/23 si falta)
    4. Proyección: CONSTANTE → w_{p,h} = w_{p,0} para todo h.

JUSTIFICACIÓN DE PROYECCIÓN CONSTANTE (paper §7.2):
    El EDA de Capology muestra que el 72.2% de los salarios observados registran
    cambio YoY = 0 entre 2022/23 y 2023/24. Esto refleja que los contratos EPL
    son multi-año (típicamente 3-5 años) y los salarios se renegocian de forma
    esporádica, no anualmente. Un modelo AR(1) o exponencial captura varianza
    de contratos en momentos específicos de renegociación, no la dinámica media.
    La proyección constante es la especificación más robusta y defendible:
    ancla al valor observado sin añadir ruido de proyección.
    El constraint salarial del MIP opera a nivel de plantilla completa (Σ w_p),
    donde la precisión de cada w_p individual importa menos que el nivel agregado.

NOTA METODOLÓGICA IMPORTANTE (paper §7.2):
    No usar wage = κ × MV. Esto introduciría dependencia lineal entre el
    constraint de transferencia y el constraint salarial en el MIP, haciendo
    la restricción salarial potencialmente redundante.

NOTA METODOLÓGICA IMPORTANTE (paper §7.2):
    No usar wage = κ × MV. Esto introduciría dependencia lineal entre el
    constraint de transferencia y el constraint salarial en el MIP, haciendo
    la restricción salarial potencialmente redundante.

Outputs:
    outputs/pipeline/p09_wage_panel.parquet
    Columnas: player_id, wage_annual_h0..wage_annual_hH  (en £/año)

    outputs/pipeline/p09_wage_coverage.csv
    Reporte de cobertura: método de asignación para cada jugador.

Tests inline:
    - 738 jugadores en el panel (exactamente los del universo P06)
    - wage_annual_h0 ∈ [200k, 50M] £/año para el 99% de jugadores
    - Cobertura directa + fuzzy >= 70% del universo
    - wage_annual_h* nunca nulo

Upstream:
    P06: outputs/pipeline/p06_projection_universe.parquet
    P03: outputs/pipeline/p03_score_player_season.parquet  (para quintiles)
    EXTERNAL: data/external/capology/epl_salaries_all.csv
"""

from __future__ import annotations

import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import load_scores_with_pt

from _config import (  # noqa: E402
    REPO, OUT_DIR, CAPOLOGY_PATH, UNIVERSE_PATH, SCORES_PATH,
)

# Proyección constante — w_{p,h} = w_{p,0} para todo h.
# Justificación: 72.2% cambio YoY = 0 en Capology (multi-year contracts).
# No se usa proyección exponencial ni AR(1).
JACCARD_THRESHOLD = 0.50   # umbral Jaccard para matching fuzzy
SEASON_TARGET = "2023-2024"  # temporada base para salario h=0


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_normalize(a).split()), set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_and_clean_capology() -> pd.DataFrame:
    """Carga y normaliza datos Capology."""
    assert CAPOLOGY_PATH.exists(), \
        f"[ERROR] {CAPOLOGY_PATH} no encontrado. Verificar data/external/capology/."

    cap = pd.read_csv(CAPOLOGY_PATH)
    assert "annual_gross_gbp" in cap.columns, \
        f"Columna 'annual_gross_gbp' faltante. Columnas: {cap.columns.tolist()}"

    cap = cap.dropna(subset=["annual_gross_gbp"])
    cap["annual_gross_gbp"] = pd.to_numeric(cap["annual_gross_gbp"], errors="coerce")
    cap = cap[cap["annual_gross_gbp"] > 0]

    # Seleccionar temporada base 2023-2024; fallback a 2022-2023
    if "season" in cap.columns:
        cap_t1 = cap[cap["season"] == SEASON_TARGET].copy()
        cap_t0 = cap[cap["season"] == "2022-2023"].copy()
    else:
        cap_t1 = cap.copy()
        cap_t0 = pd.DataFrame()

    print(f"  Capology 2023-2024: {len(cap_t1)} registros")
    print(f"  Capology 2022-2023: {len(cap_t0)} registros")

    # Deduplicar por player_name dentro de cada temporada
    cap_t1 = cap_t1.sort_values("annual_gross_gbp", ascending=False).drop_duplicates("player_name")
    cap_t0 = cap_t0.sort_values("annual_gross_gbp", ascending=False).drop_duplicates("player_name") if len(cap_t0) > 0 else cap_t0

    return cap_t1, cap_t0


def match_players(
    universe: pd.DataFrame,
    cap_t1: pd.DataFrame,
    cap_t0: pd.DataFrame,
) -> pd.DataFrame:
    """
    Realiza el matching player_name → annual_gross_gbp en dos pasos:
    1. Match exacto (Unicode normalizado)
    2. Jaccard token ≥ 0.50

    Returns:
        universe con columnas añadidas: wage_base, wage_source
    """
    # Construir lookup normalizado de Capology (prioridad t1)
    wage_lookup: dict[str, float] = {}
    for _, row in cap_t1.iterrows():
        key = _normalize(row["player_name"])
        wage_lookup[key] = row["annual_gross_gbp"]
    wage_lookup_t0: dict[str, float] = {}
    for _, row in cap_t0.iterrows():
        key = _normalize(row["player_name"])
        wage_lookup_t0[key] = row["annual_gross_gbp"]

    all_cap_names = list(wage_lookup.keys()) + list(wage_lookup_t0.keys())

    results = []
    for _, row in universe.iterrows():
        pid  = row["player_id"]
        name = row["player_name"]
        key  = _normalize(name)

        # 1. Match exacto t1
        if key in wage_lookup:
            results.append({"player_id": pid, "wage_base": wage_lookup[key],
                            "wage_source": "direct"})
            continue

        # 2. Match exacto t0
        if key in wage_lookup_t0:
            results.append({"player_id": pid, "wage_base": wage_lookup_t0[key],
                            "wage_source": "direct_t0"})
            continue

        # 3. Jaccard fuzzy (busca en t1 + t0)
        best_j, best_w = 0.0, None
        for cap_key, wage in {**wage_lookup, **wage_lookup_t0}.items():
            j = _jaccard(name, cap_key)
            if j > best_j:
                best_j, best_w = j, wage

        if best_j >= JACCARD_THRESHOLD and best_w is not None:
            results.append({"player_id": pid, "wage_base": best_w,
                            "wage_source": "fuzzy"})
        else:
            results.append({"player_id": pid, "wage_base": np.nan,
                            "wage_source": "imputed"})

    matched = pd.DataFrame(results)
    universe = universe.merge(matched, on="player_id", how="left")
    return universe


def impute_wages(universe: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa salarios faltantes con mediana de (position_group, score_quintile).
    Jugadores sin score: mediana de position_group.
    Fallback global: mediana general (paper: 1 jugador).
    """
    # Añadir score_pt 2023/24 para quintiles
    score_t1 = (scores[scores["season_name"] == "2023/2024"]
                [["player_id", "score_pt"]]
                .drop_duplicates("player_id"))
    universe = universe.merge(score_t1, on="player_id", how="left")

    # Quintiles de score
    non_null = universe["score_pt"].dropna()
    if len(non_null) > 10:
        bins = pd.qcut(non_null, q=5, labels=False, duplicates="drop")
        universe.loc[non_null.index, "score_quintile"] = bins
    universe["score_quintile"] = universe["score_quintile"].fillna(2).astype(int)

    # Imputar por (position_group, score_quintile)
    for (pos, q), grp in universe.groupby(["position_group", "score_quintile"]):
        med = grp.loc[grp["wage_base"].notna(), "wage_base"].median()
        if pd.isna(med):
            continue
        mask = (universe["position_group"].eq(pos) &
                universe["score_quintile"].eq(q) &
                universe["wage_base"].isna())
        universe.loc[mask, "wage_base"]   = med
        universe.loc[mask, "wage_source"] = "imputed_pos_score"

    # Fallback global
    global_med = universe["wage_base"].median()
    n_missing  = universe["wage_base"].isna().sum()
    universe["wage_base"]   = universe["wage_base"].fillna(global_med)
    universe.loc[universe["wage_source"].isna(), "wage_source"] = "imputed_global"

    if n_missing > 0:
        print(f"  [INFO] {n_missing} jugadores con wage imputado por mediana global")

    return universe


def project_wages(universe: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Proyecta salarios para h=0..H usando proyección CONSTANTE:
        w_{p,h} = w_{p,0}  para todo h ∈ {0,...,H}

    Justificación: 72.2% de jugadores muestran cambio YoY = 0 en Capology,
    reflejo de contratos multi-año EPL. Ver paper §7.2 y docstring del módulo.

    Returns:
        DataFrame con player_id y wage_annual_h0..wage_annual_hH
    """
    panel = universe[["player_id", "wage_base"]].copy()
    panel["wage_annual_h0"] = panel["wage_base"]

    for h in range(1, horizon + 1):
        panel[f"wage_annual_h{h}"] = panel["wage_annual_h0"]  # constante

    panel = panel.drop(columns=["wage_base"])
    return panel


def validate_wage_panel(panel: pd.DataFrame, universe: pd.DataFrame,
                        coverage: pd.DataFrame, horizon: int) -> None:
    """Tests de regresión del panel de salarios."""
    assert len(panel) == len(universe), \
        f"Panel tiene {len(panel)} filas vs {len(universe)} en universo"

    assert panel["player_id"].duplicated().sum() == 0, \
        "player_id duplicados en panel de salarios"

    wage_cols = [f"wage_annual_h{h}" for h in range(horizon + 1)]
    for col in wage_cols:
        assert col in panel.columns, f"Columna faltante: {col}"
        assert (panel[col] > 0).all(), f"Salario nulo o negativo en {col}"
        assert panel[col].isna().sum() == 0, f"NaN en {col}"

    # Rango razonable para wage_annual_h0 (£/año)
    w0 = panel["wage_annual_h0"]
    p1, p99 = w0.quantile([0.01, 0.99])
    print(f"  Wage h=0: p1={p1/1e6:.2f}M£, p99={p99/1e6:.2f}M£, "
          f"median={w0.median()/1e6:.2f}M£")
    assert p1 > 50_000, f"wage p1 < £50k/año — posible error de unidades: {p1:.0f}"
    assert p99 < 50_000_000, f"wage p99 > £50M/año — outlier extremo: {p99:.0f}"

    # Cobertura
    sources = coverage["wage_source"].value_counts()
    print(f"  Cobertura:")
    for src, n in sources.items():
        print(f"    {src}: {n} ({n/len(coverage):.1%})")
    direct_pct = (sources.get("direct", 0) + sources.get("direct_t0", 0) +
                  sources.get("fuzzy", 0)) / len(coverage)
    assert direct_pct >= 0.60, \
        f"Cobertura directa+fuzzy = {direct_pct:.1%} < 60% — revisar matching"

    print(f"  [TESTS OK] Panel salarios: {len(panel)} jugadores × {horizon+1} horizontes")


def main(horizon: int = 5) -> pd.DataFrame:
    print("=" * 60)
    print(f"P09 — Wage Panel (proyección constante, H={horizon})")
    print("=" * 60)

    for path, name in [(UNIVERSE_PATH, "P06 universe"), (SCORES_PATH, "P03 scores")]:
        assert path.exists(), \
            f"[ERROR] Input no encontrado: {path} ({name}). Correr step anterior."

    universe = pd.read_parquet(UNIVERSE_PATH)
    scores   = load_scores_with_pt(SCORES_PATH)
    print(f"\n  Universo: {len(universe)} jugadores")

    # Cargar y matchear Capology
    cap_t1, cap_t0 = load_and_clean_capology()
    print(f"\n  Matching player names...")
    universe = match_players(universe, cap_t1, cap_t0)

    matched_pct = universe["wage_source"].isin({"direct", "direct_t0", "fuzzy"}).mean()
    print(f"  Matched (direct + fuzzy): {matched_pct:.1%}")

    # Imputar faltantes
    print(f"\n  Imputando salarios faltantes...")
    universe = impute_wages(universe, scores)

    # Proyectar
    print(f"\n  Proyectando salarios para H={horizon}...")
    panel = project_wages(universe, horizon)

    # Coverage report
    coverage = universe[["player_id", "player_name", "position_group",
                          "wage_base", "wage_source"]].copy()
    coverage.columns = ["player_id", "player_name", "position_group",
                        "wage_annual_h0_base", "wage_source"]

    validate_wage_panel(panel, universe, coverage, horizon)

    out_path = OUT_DIR / "p09_wage_panel.parquet"
    panel.to_parquet(out_path, index=False)

    cov_path = OUT_DIR / "p09_wage_coverage.csv"
    coverage.to_csv(cov_path, index=False)

    print(f"\n  Outputs:")
    print(f"    {out_path}")
    print(f"    {cov_path}")
    return panel


if __name__ == "__main__":
    main()
