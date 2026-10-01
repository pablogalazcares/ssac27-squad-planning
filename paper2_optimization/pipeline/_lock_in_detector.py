"""
_lock_in_detector.py — Detección y clasificación de planning lock-in
=====================================================================

Lock-in se define como la situación en la que una estrategia de planificación
(típicamente Myopic) genera una solución infeasible o gravemente degradada
en algún step del horizonte H, debido a decisiones tempranas que agotan
recursos críticos (presupuesto de salarios, slots de squad, etc.).

Tipos de lock-in:
    INFEASIBLE_WAGE_CAP    — Step infeasible y la masa salarial es la restricción binding.
    INFEASIBLE_BUDGET      — Step infeasible y el presupuesto de transferencias es binding.
    INFEASIBLE_POSITIONAL  — Step infeasible y las restricciones de squad/posición son binding.
    INFEASIBLE_OTHER       — Step infeasible pero la razón no se pudo determinar.
    COLLAPSE_BELOW_BASELINE— Solución factible pero score_t < 0.5 × baseline_score en algún t.

Uso:
    from _lock_in_detector import detect_lock_in_rolling, detect_lock_in_full_h5

    # Para Rolling/Myopic (con step_stats list):
    info = detect_lock_in_rolling(step_stats, baseline_score, diagnostics_files=...)

    # Para Full H=5 (con result.status + objective):
    info = detect_lock_in_full_h5(status, objective_value, baseline_score, diagnostics_file=...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


COLLAPSE_THRESHOLD = 0.5  # score_t < 0.5 * baseline → COLLAPSE_BELOW_BASELINE


def _classify_infeasibility(diag: dict | None) -> str:
    """
    Mapea el output de diagnose_infeasibility (de P11) a un código de razón.
    """
    if not diag or not isinstance(diag, dict):
        return "INFEASIBLE_OTHER"
    violated = diag.get("violated_constraints") or []
    if not violated:
        return "INFEASIBLE_OTHER"
    # Priorizar wage_budget > transfer_budget > posicional > otros
    text = " ".join(str(v).lower() for v in violated)
    if "wage" in text or "salar" in text:
        return "INFEASIBLE_WAGE_CAP"
    if "budget" in text or "transfer" in text:
        return "INFEASIBLE_BUDGET"
    if "u23" in text or "foreign" in text or "position" in text or "squad" in text:
        return "INFEASIBLE_POSITIONAL"
    return "INFEASIBLE_OTHER"


def _load_diagnostic(path: Path | str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def detect_lock_in_rolling(
    step_stats: list[dict],
    baseline_score: float,
    out_dir: Path | str | None = None,
) -> dict:
    """
    Detecta lock-in en una corrida Rolling/Myopic.

    Parameters
    ----------
    step_stats : list of dict with keys 'step', 'status', 'objective'
        Salida de run_rolling() en P11.
    baseline_score : float
        Score del squad inicial (objetivo de la solución t=0 sin transferencias).
    out_dir : path, optional
        Directorio donde están los archivos step_*_infeasibility_diag.json.

    Returns
    -------
    dict with keys:
        lock_in : bool
        lock_in_step : int | None  (1-indexed)
        lock_in_reason : str | None
    """
    if not step_stats:
        return {"lock_in": False, "lock_in_step": None, "lock_in_reason": None}

    out_path = Path(out_dir) if out_dir else None

    # Pass 1: buscar infeasibility
    for entry in step_stats:
        status = str(entry.get("status", "")).lower()
        obj    = entry.get("objective")
        step   = int(entry.get("step", 0))

        is_infeas = ("infeasible" in status) or (obj is None) or \
                    (isinstance(obj, (int, float)) and obj < -1000)
        if is_infeas:
            diag = _load_diagnostic(
                out_path / f"step_{step}_infeasibility_diag.json"
            ) if out_path else None
            reason = _classify_infeasibility(diag)
            return {
                "lock_in": True,
                "lock_in_step": step + 1,
                "lock_in_reason": reason,
            }

    # Pass 2: buscar collapse below baseline
    if baseline_score and baseline_score > 0:
        threshold = COLLAPSE_THRESHOLD * baseline_score
        for entry in step_stats:
            obj  = entry.get("objective")
            step = int(entry.get("step", 0))
            if isinstance(obj, (int, float)) and obj < threshold:
                return {
                    "lock_in": True,
                    "lock_in_step": step + 1,
                    "lock_in_reason": "COLLAPSE_BELOW_BASELINE",
                }

    return {"lock_in": False, "lock_in_step": None, "lock_in_reason": None}


def detect_lock_in_full_h5(
    status: str,
    objective_value: float | None,
    baseline_score: float,
    diagnostic_path: Path | str | None = None,
) -> dict:
    """
    Detecta lock-in en una corrida Full H=5.

    Parameters
    ----------
    status : str
        Estado del solver (e.g., 'Optimal', 'Infeasible').
    objective_value : float or None
        Valor objetivo total de la solución.
    baseline_score : float
        Score del squad inicial (referencia).
    diagnostic_path : path, optional
        Archivo de diagnóstico de infeasibility si el solver lo generó.

    Returns
    -------
    dict idéntico a detect_lock_in_rolling().
    """
    status_l = str(status or "").lower()
    is_infeas = ("infeasible" in status_l) or (objective_value is None) or \
                (isinstance(objective_value, (int, float)) and objective_value < -1000)

    if is_infeas:
        diag = _load_diagnostic(diagnostic_path)
        reason = _classify_infeasibility(diag)
        return {
            "lock_in": True,
            "lock_in_step": None,  # Full H=5 no tiene step específico
            "lock_in_reason": reason,
        }

    # Full H=5: solo aplica COLLAPSE_BELOW_BASELINE si el objetivo total << H × baseline
    # (no aplica step-wise porque H=5 resuelve de una sola vez)
    if baseline_score and baseline_score > 0 and isinstance(objective_value, (int, float)):
        # Asumimos H=5 horizons; un colapso total sería que el total < 0.5 * 5 * baseline
        if objective_value < COLLAPSE_THRESHOLD * 5 * baseline_score:
            return {
                "lock_in": True,
                "lock_in_step": None,
                "lock_in_reason": "COLLAPSE_BELOW_BASELINE",
            }

    return {"lock_in": False, "lock_in_step": None, "lock_in_reason": None}


def summarize_lock_in(info_dict: dict) -> str:
    """
    Genera un string descriptivo para logging.
    """
    if not info_dict.get("lock_in"):
        return "no-lock-in"
    step = info_dict.get("lock_in_step")
    reason = info_dict.get("lock_in_reason", "UNKNOWN")
    step_str = f"step={step}" if step is not None else "step=N/A"
    return f"LOCK-IN [{reason}, {step_str}]"
