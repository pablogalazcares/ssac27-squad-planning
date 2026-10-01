"""
pipeline_utils.py — funciones compartidas entre scripts del pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import REPO, P05_CONFIG_PATH  # noqa: E402,F401


def get_selected_alpha() -> float:
    """Lee α del modelo seleccionado por P05."""
    assert P05_CONFIG_PATH.exists(), (
        f"[ERROR] {P05_CONFIG_PATH} no encontrado — correr P05 primero."
    )
    cfg = pd.read_csv(P05_CONFIG_PATH)
    return float(cfg.iloc[0]["alpha"])


def compose_score_pt(scores: pd.DataFrame, alpha: float | None = None) -> pd.DataFrame:
    """Añade columna score_pt si no existe.
    score_pt = (1-α)*score_OBV + α*score_PSR, con effective α=0 si has_psr=0.
    """
    if "score_pt" in scores.columns:
        return scores
    if alpha is None:
        alpha = get_selected_alpha()
    df = scores.copy()
    eff_alpha = alpha * df.get("has_psr", pd.Series(0, index=df.index)).fillna(0)
    df["score_pt"] = (
        (1 - eff_alpha) * df["score_OBV"].fillna(0)
        + eff_alpha * df["score_PSR"].fillna(0)
    )
    return df


def load_scores_with_pt(path: Path, alpha: float | None = None) -> pd.DataFrame:
    """Carga P03 scores y asegura columna score_pt."""
    scores = pd.read_parquet(path)
    return compose_score_pt(scores, alpha)
