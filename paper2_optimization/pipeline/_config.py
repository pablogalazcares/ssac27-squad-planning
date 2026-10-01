"""
_config.py — Configuración de rutas del pipeline INDEPENDIENTE del Paper 2.

Este módulo es la ÚNICA fuente de rutas para los scripts P08–P14 del Paper 2.
Aísla por completo la frontera con el Paper 1: la única dependencia con
``outputs/pipeline/`` está encapsulada en ``PROJECTIONS_DIR``.

Frontera de entrada (producida por Paper 1 / fuentes crudas):
    - p07_projected_scores.parquet  (proyecciones h0–h5)  ← input principal
    - p06_projection_universe.parquet (universo C)
    - p03_score_player_season.parquet (scores históricos)
    - p05_selected_config.csv (α del modelo seleccionado)
    - df_final_market_value.csv, capology, transfermarkt (externos)

Outputs del Paper 2 → paper2_optimization/outputs/ (aislados de Paper 1).
"""
from __future__ import annotations

from pathlib import Path

# repo root: paper2_optimization/pipeline/_config.py → parents[2]
REPO = Path(__file__).resolve().parents[2]

# ── Frontera: fuente de proyecciones (Paper 1). Reapuntar aquí para cambiar de
#    proveedor de proyecciones (hoy el pipeline P01–P07). ──────────────────────
PROJECTIONS_DIR = REPO / "outputs" / "pipeline"

# ── Datos externos crudos (fuentes de datos, no "otro paper") ─────────────────
DATA_DIR = REPO / "data"

# ── Outputs propios del Paper 2 ───────────────────────────────────────────────
OUT_DIR = REPO / "paper2_optimization" / "outputs"
FIG_DIR = REPO / "paper2_optimization" / "figures"
TAB_DIR = REPO / "paper2_optimization" / "tables"

# ── Inputs de frontera ────────────────────────────────────────────────────────
SCORES_PATH     = PROJECTIONS_DIR / "p03_score_player_season.parquet"
UNIVERSE_PATH   = PROJECTIONS_DIR / "p06_projection_universe.parquet"
PROJ_PATH       = PROJECTIONS_DIR / "p07_projected_scores.parquet"
P05_CONFIG_PATH = PROJECTIONS_DIR / "p05_selected_config.csv"

MV_RAW_PATH   = DATA_DIR / "processed" / "statsbomb_legacy" / "df_final_market_value.csv"
CAPOLOGY_PATH = DATA_DIR / "external" / "capology" / "epl_salaries_all.csv"
TM_DIR        = DATA_DIR / "external" / "transfermarkt"
TM_2024_PATH  = TM_DIR / "epl_summer_2024_transfers.json"

# ── Outputs del Paper 2 (regenerados por este pipeline) ───────────────────────
MV_PANEL      = OUT_DIR / "p08_mv_panel.parquet"
MV_COEFF      = OUT_DIR / "p08_mv_model_coeff.csv"
WAGE_PANEL    = OUT_DIR / "p09_wage_panel.parquet"
WAGE_COVERAGE = OUT_DIR / "p09_wage_coverage.csv"
OPT_PANEL     = OUT_DIR / "p10_optimizer_panel.parquet"
BUDGET_CAPS   = OUT_DIR / "p14_budget_caps.parquet"
P11_OUT       = OUT_DIR / "p11_e5a"
P12_OUT       = OUT_DIR / "p12_e5b"

for _d in (OUT_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)
