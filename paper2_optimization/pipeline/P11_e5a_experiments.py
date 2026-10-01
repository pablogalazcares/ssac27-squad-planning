"""
P11 — Experimento E5a: 20 clubes EPL (horizonte comparativo)
=============================================================
Objetivo:
    Replicar el Experimento E5a del paper §9 usando el pipeline limpio
    (P01–P10) como fuente de datos, en lugar del pipeline legacy.

    E5a compara tres horizontes de planificación para TODOS los clubes EPL:
      1. Full MIP H=5  : solución global sobre 5 ventanas de fichajes
      2. Rolling H=3   : ventana deslizante de 3 temporadas
      3. Myopic H=1    : decisiones secuenciales de 1 temporada

    Métrica reportada (paper §9):
      H5_score_gain = Σ_{t=1}^{5} squad_score_t (Full MIP)
      Δ% vs Myopic  = (H5_score - Myopic_score) / |Myopic_score| × 100
    Hipótesis H1: Full MIP H=5 supera a Myopic en ≥ 25% del universe.

Presupuestos (E5a — parámetros uniformes, paper §9.1):
    Transfer budget: £80M/temporada (constante)
    Wage budget: £77.2M/año (mediana EPL 2023/24, Capology) = 1,484.6 k��/semana.
                 Binding para clubes de gasto medio-bajo; no binding para top-6.
    Max hires: 5/temporada

Diferencias vs e5_club_experiments.py:
    - Datos de scores: P07 proyección (no pivot E4)
    - MV panel: P08 F3 panel (no e4_mv_dynamic)
    - Wages: P09 Capology (no κ×MV)
    - Posiciones: P01 canónico (sin MID inflation)
    - El formato del panel para MultiSeasonSquadPlanner es idéntico al legacy.

Outputs:
    outputs/pipeline/p11_e5a/
        e5a_summary.csv           — una fila por club con H5/H3/Myopic scores
        e5a_full_h5_{club}.json   — resultado MIP Full H=5 por club
        e5a_rolling_{club}.json   — resultado rolling H=3
        e5a_myopic_{club}.json    — resultado myopic

Tests inline:
    - Todos los 20 clubes EPL tienen solución óptima o near-optimal
    - Δ% H5 vs Myopic: mediana en [-60%, -20%] (paper reporta ~-39.9%)
    - R_solve ≥ 18/20 clubes (90% feasibility)

Upstream:
    P10: outputs/pipeline/p10_optimizer_panel.parquet
    STATIC: data/processed/statsbomb_legacy/df_final_market_value.csv (country)
    EXTERNAL: data/external/transfermarkt/epl_summer_2024_transfers.json

    [Scenario II — por-club budgets]:
    P14: outputs/pipeline/p14_budget_caps.parquet
         Actualmente E5a usa DEFAULT_TRANSFER_M = 80.0 (EUR M, Scenario I uniforme).
         Para Scenario II (20 clubes con budget_cap_i individual), cargar p14_budget_caps.parquet
         y reemplazar DEFAULT_TRANSFER_M por el valor club-específico en build_club_config().
"""

from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (  # noqa: E402
    REPO, MV_RAW_PATH,
    OPT_PANEL as PANEL_PATH,
    BUDGET_CAPS as BUDGET_CAPS_PATH,
    WAGE_PANEL as WAGE_PANEL_PATH,
    P11_OUT as OUT_DIR,
)

# Motor de optimización compartido (src/) — dependencia de código, no de Paper 1.
sys.path.insert(0, str(REPO))

# Detector de lock-in (módulo local en este pipeline).
from _lock_in_detector import (  # noqa
    detect_lock_in_rolling, detect_lock_in_full_h5, summarize_lock_in
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Presupuestos E5a (paper §9.1) ──────────────────────────────────────────────
DEFAULT_TRANSFER_M  = 80.0          # M£/temporada (uniform, Escenario I)
# W_t = mediana de la masa salarial EPL 2023/24 (Capology EDA): £77.2M/año
# = 77_200_000 / 1000 / 52 = 1_484.6 k£/semana
E5A_WAGE_K          = 1_484.6       # k£/semana (£77.2M/año, mediana EPL 2023/24)
MAX_HIRES           = 5
HORIZON             = 5
BASE_YEAR           = 2023          # h=0 → 2023/24

# ── Clubes EPL 2023/24 (los 20 que disputaron la temporada base) ─────────────
EPL_2324_CLUBS = {
    "Arsenal",
    "Aston Villa",
    "AFC Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Liverpool",
    "Luton Town",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sheffield United",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
}

# ── Constraints del MIP (compartidos con legacy) ───────────────────────────────
POSITION_LIMITS = {
    "GK":  {"min": 3, "max": 3},
    "DEF": {"min": 8, "max": 8},
    "MID": {"min": 7, "max": 7},
    "FWD": {"min": 7, "max": 7},
}
SQUAD_MIN, SQUAD_MAX = 25, 25
MAX_FOREIGN, MIN_U23 = 17, 2
MAX_AGE              = 38

UK_NATIONS = {"Inglaterra", "Escocia", "Gales", "Irlanda del Norte",
              "England", "Scotland", "Wales", "Northern Ireland"}

N_WORKERS = 4


# Mapeo nombres P10 (StatsBomb) → P14 (Transfermarkt)
# Necesario porque P14 usa nombres TM con sufijo "FC" / formas cortas.
CLUB_NAME_P10_TO_P14 = {
    "AFC Bournemouth":         "AFC Bournemouth",
    "Arsenal":                  "Arsenal FC",
    "Aston Villa":              "Aston Villa",
    "Brentford":                "Brentford FC",
    "Brighton & Hove Albion":   "Brighton",
    "Burnley":                  "Burnley FC",
    "Chelsea":                  "Chelsea FC",
    "Crystal Palace":           "Crystal Palace",
    "Everton":                  "Everton FC",
    "Fulham":                   "Fulham FC",
    "Liverpool":                "Liverpool FC",
    "Luton Town":               "Luton Town",
    "Manchester City":          "Manchester City",
    "Manchester United":        "Manchester United",
    "Newcastle United":         "Newcastle United",
    "Nottingham Forest":        "Nottingham Forest",
    "Sheffield United":         "Sheffield United",
    "Tottenham Hotspur":        "Tottenham Hotspur",
    "West Ham United":          "West Ham United",
    "Wolverhampton Wanderers":  "Wolverhampton",
}


def load_club_budgets() -> dict[str, float]:
    """Carga B_i (transfer budget en M€) por club desde P14, indexado por nombre P10."""
    if not BUDGET_CAPS_PATH.exists():
        raise FileNotFoundError(
            f"{BUDGET_CAPS_PATH} no encontrado — correr P14 primero para Escenario II."
        )
    caps = pd.read_parquet(BUDGET_CAPS_PATH)
    p14_lkp = dict(zip(caps["club_name"].astype(str),
                       caps["budget_cap_eur_m"].astype(float)))
    # Reindexar por nombre P10
    return {p10_name: p14_lkp[p14_name]
            for p10_name, p14_name in CLUB_NAME_P10_TO_P14.items()
            if p14_name in p14_lkp}


def load_club_wage_bills() -> dict[str, float]:
    """
    Calcula W_i (masa salarial total en k£/semana) por club a partir de p09_wage_panel
    + p10_optimizer_panel (team_name). Para Escenario II.
    """
    if not WAGE_PANEL_PATH.exists() or not PANEL_PATH.exists():
        raise FileNotFoundError("p09_wage_panel.parquet o p10_optimizer_panel.parquet faltante.")
    wages = pd.read_parquet(WAGE_PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    merged = panel[["player_id", "team_name"]].merge(
        wages[["player_id", "wage_annual_h0"]], on="player_id", how="left"
    )
    # Suma anual GBP por club → convertir a k£/semana
    by_club = merged.groupby("team_name")["wage_annual_h0"].sum() / 1000.0 / 52.0
    return by_club.to_dict()


def load_country_lookup() -> dict[int, str]:
    """Carga lookup player_id → country desde df_final_market_value.csv.

    Usa 2023/24 si disponible, con fallback a 2022/23 para jugadores de clubes
    relegados (Leeds, Leicester, etc.) que no aparecen en el CSV de 2023/24.
    """
    mv = pd.read_csv(MV_RAW_PATH)
    mv_22 = mv[mv["season_name"] == "2022/2023"].drop_duplicates("player_id")
    mv_23 = mv[mv["season_name"] == "2023/2024"].drop_duplicates("player_id")
    lkp_22 = dict(zip(mv_22["player_id"].astype(int), mv_22["country"].astype(str)))
    lkp_23 = dict(zip(mv_23["player_id"].astype(int), mv_23["country"].astype(str)))
    # 2023/24 tiene prioridad; 2022/23 cubre clubes relegados
    return {**lkp_22, **lkp_23}


def build_optimizer_panel(
    p10: pd.DataFrame,
    club_name: str,
    country_lkp: dict[int, str],
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Convierte el panel wide de P10 al formato long esperado por MultiSeasonSquadPlanner.

    Format long:
        player_id, player_name, season_idx, season_name, position_group,
        score_pred, market_value_m, wage_k, age, is_foreign_epl, is_u23,
        is_initial_squad, is_available, team_name, country, set_label

    Parameters:
        p10:       p10_optimizer_panel.parquet (format wide, un jugador por fila)
        club_name: nombre del club (para is_initial_squad)
        country_lkp: player_id → country
        horizon:   H
    """
    rows = []
    for _, player in p10.iterrows():
        pid     = int(player["player_id"])
        pname   = str(player["player_name"])
        pos     = str(player["position_group"])
        team    = str(player["team_name"])
        age0    = float(player["age_base"]) if pd.notna(player["age_base"]) else 28.0
        country = country_lkp.get(pid, "Unknown")

        is_for  = int(country not in UK_NATIONS)
        is_init = int(str(team).lower().replace(" ", "") in
                      club_name.lower().replace(" ", "") or
                      club_name.lower().replace(" ", "") in
                      str(team).lower().replace(" ", ""))

        for h in range(horizon + 1):
            score = player.get(f"score_h{h}", np.nan)
            mv    = player.get(f"mv_h{h}", np.nan)
            wage  = player.get(f"wage_annual_h{h}", np.nan)

            if pd.isna(score) or pd.isna(mv) or pd.isna(wage):
                continue

            age_h = age0 + h
            rows.append({
                "player_id":        pid,
                "player_name":      pname,
                "season_idx":       h,
                "season_name":      f"{BASE_YEAR + h}/{BASE_YEAR + h + 1}",
                "position_group":   pos,
                "score_pred":       round(float(score), 3),
                "market_value_m":   round(float(mv) / 1_000_000, 3),
                "wage_k":           round(float(wage) / 1_000 / 52, 2),  # £/año → k£/sem
                "age":              round(age_h, 1),
                "is_foreign_epl":   is_for,
                "is_u23":           int(age_h < 23),
                "is_initial_squad": is_init,
                "is_available":     1,
                "team_name":        team,
                "country":          country,
                "set_label":        str(player.get("set_label", "B")),
            })

    panel = pd.DataFrame(rows).drop_duplicates(subset=["player_id", "season_idx"])
    return panel.reset_index(drop=True)


def make_mip_config(club_cfg: dict, seasons: list[int]) -> dict:
    """Construye config dict para MultiSeasonSquadPlanner."""
    tb = {t: club_cfg["transfer_budget_m"] for t in seasons}
    wb = {t: float(club_cfg["wage_budget_k"]) for t in seasons}
    mh = {t: int(club_cfg["max_hires"]) if t > 0 else 0 for t in seasons}
    return {
        "transfer_budget_m":     tb,
        "wage_budget_k":         wb,
        "max_hires_per_season":  mh,
        "max_foreign_players":   MAX_FOREIGN,
        "min_u23_players":       MIN_U23,
        "max_age":               MAX_AGE,
        "squad_size_min":        SQUAD_MIN,
        "squad_size_max":        SQUAD_MAX,
        "position_limits":       POSITION_LIMITS,
        "enforce_monotonic_score": False,
        "enforce_min_two_years":   False,
        "enforce_no_rehire":       False,
    }


def diagnose_infeasibility(
    panel:      pd.DataFrame,
    config:     dict,
    season_idx: int = 1,
    label:      str = "",
) -> dict:
    """
    Analiza qué restricciones del MIP son inviables para un panel/temporada dado.

    Se llama automáticamente cuando el MIP devuelve infeasible. Identifica la
    causa probable de infeasibilidad chequeando cada constraint de forma independiente.

    Restricciones chequeadas:
      - wage_budget  : ¿existe algún squad de 25 válido por posición dentro del wage cap?
      - transfer_budget : ¿hay jugadores comprables con el presupuesto disponible?
      - position_counts : ¿hay suficientes jugadores por posición?
      - squad_size   : ¿hay al menos 25 jugadores disponibles?
      - foreign_quota: ¿puede formarse un squad con ≤ max_foreign extranjeros?
      - u23_quota    : ¿hay al menos min_u23 jugadores U23 disponibles?

    Nota metodológica:
      - No se usa duración de contrato: el MIP puede liberar cualquier jugador
        en cualquier temporada sin coste adicional.
      - Las liberaciones (sin venta) no generan ingreso de transferencia.
      - El número implícito de bajas por temporada está acotado por max_hires
        (para mantener squad_size_min = squad_size_max = 25): max_sells ≤ max_hires.

    Parameters:
        panel:      panel long del club (columnas: player_id, season_idx, is_initial_squad,
                    wage_k, market_value_m, position_group, age, is_foreign_epl, is_u23)
        config:     dict de configuración del MIP (mismo formato que make_mip_config)
        season_idx: temporada a diagnosticar (default 1 = primera de planificación)
        label:      prefijo para log

    Returns dict con: feasibility_verdict, violated_constraints, details
    """
    prefix = f"  [DIAG {label}]" if label else "  [DIAG]"
    sub = panel[panel["season_idx"] == season_idx].copy()

    if sub.empty:
        return {"feasibility_verdict": "NO_DATA", "violated_constraints": ["no_players"],
                "details": {}}

    # Presupuestos de la temporada
    wage_cfg     = config.get("wage_budget_k", {})
    transfer_cfg = config.get("transfer_budget_m", {})
    max_hires    = config.get("max_hires_per_season", {}).get(season_idx, MAX_HIRES)
    wage_cap     = float(wage_cfg[season_idx]) if isinstance(wage_cfg, dict) else float(wage_cfg)
    transfer_cap = float(transfer_cfg[season_idx]) if isinstance(transfer_cfg, dict) else float(transfer_cfg)
    max_foreign  = int(config.get("max_foreign_players", MAX_FOREIGN))
    min_u23      = int(config.get("min_u23_players", MIN_U23))
    squad_min    = int(config.get("squad_size_min", SQUAD_MIN))
    squad_max    = int(config.get("squad_size_max", SQUAD_MAX))
    pos_limits   = config.get("position_limits", POSITION_LIMITS)

    print(f"{prefix} season_idx={season_idx} | {len(sub)} jugadores disponibles | "
          f"wage_cap={wage_cap:.1f} k£/wk | transfer_cap={transfer_cap:.1f} M£")

    violated = []
    details  = {}

    # ── 1. Conteo por posición ─────────────────────────────────────────────────
    pos_counts = sub.groupby("position_group").size().to_dict()
    print(f"{prefix} Jugadores por posición: {pos_counts}")
    details["pos_counts"] = pos_counts
    for pos, lims in pos_limits.items():
        n_pos = pos_counts.get(pos, 0)
        if n_pos < lims["min"]:
            violated.append(f"pos_{pos}_insuficiente (N={n_pos}, necesita≥{lims['min']})")
            print(f"{prefix} FALLA pos_{pos}: N={n_pos} < min={lims['min']}")

    # ── 2. Squad size total ────────────────────────────────────────────────────
    n_total = len(sub)
    details["n_total"] = n_total
    if n_total < squad_min:
        violated.append(f"squad_size (N={n_total} < {squad_min})")
        print(f"{prefix} FALLA squad_size: N={n_total} < {squad_min}")

    # ── 3. Wage constraint — best-case (jugadores más baratos por posición) ────
    # Para cada posición, seleccionar los N más baratos (según pos_limits["min"])
    min_wage_squad = []
    for pos, lims in pos_limits.items():
        n_needed = lims["min"]
        pos_sub  = sub[sub["position_group"] == pos].nsmallest(n_needed, "wage_k")
        min_wage_squad.append(pos_sub)

    if min_wage_squad:
        cheapest_squad = pd.concat(min_wage_squad)
        min_wage_bill  = cheapest_squad["wage_k"].sum()
        details["min_wage_bill_k"]  = round(min_wage_bill, 1)
        details["wage_cap_k"]       = wage_cap
        details["wage_headroom_k"]  = round(wage_cap - min_wage_bill, 1)
        details["wage_cap_M_annual"] = round(wage_cap * 52 / 1000, 1)
        details["min_wage_M_annual"] = round(min_wage_bill * 52 / 1000, 1)
        print(f"{prefix} Wage mínimo posible (mejor 25): {min_wage_bill:.1f} k£/wk "
              f"(={min_wage_bill * 52 / 1000:.1f} M£/año) vs cap={wage_cap:.1f} k£/wk "
              f"(={wage_cap * 52 / 1000:.1f} M£/año)")
        if min_wage_bill > wage_cap:
            violated.append(
                f"wage_budget (min_possible={min_wage_bill:.1f} > cap={wage_cap:.1f} k£/wk; "
                f"déficit={min_wage_bill - wage_cap:.1f} k£/wk = "
                f"{(min_wage_bill - wage_cap) * 52 / 1000:.1f} M£/año)"
            )
            print(f"{prefix} FALLA wage: incluso el squad más barato posible supera el cap")

    # ── 4. Max hires implícito (=max_sells dado squad_size fijo) ──────────────
    details["max_hires"] = max_hires
    details["implied_max_sells"] = max_hires
    print(f"{prefix} max_hires={max_hires} → max_sells_implied={max_hires} "
          f"(dado squad_size_min=squad_size_max={squad_min})")

    # ── 5. Foreign quota ───────────────────────────────────────────────────────
    n_domestic = (sub["is_foreign_epl"] == 0).sum()
    min_domestic_needed = squad_min - max_foreign
    details["n_domestic"] = int(n_domestic)
    details["max_foreign"] = max_foreign
    if n_domestic < min_domestic_needed:
        violated.append(f"foreign_quota (only {n_domestic} domestic, need ≥{min_domestic_needed})")
        print(f"{prefix} FALLA foreign: {n_domestic} domésticos < {min_domestic_needed} requeridos")

    # ── 6. U23 quota ──────────────────────────────────────────────────────────
    n_u23 = (sub["is_u23"] == 1).sum()
    details["n_u23"] = int(n_u23)
    details["min_u23"] = min_u23
    if n_u23 < min_u23:
        violated.append(f"u23_quota (only {n_u23} U23 available, need ≥{min_u23})")
        print(f"{prefix} FALLA u23: {n_u23} jugadores U23 < mínimo {min_u23}")

    # ── Veredicto ──────────────────────────────────────────────────────────────
    verdict = "FEASIBLE_POSSIBLE" if not violated else "LIKELY_INFEASIBLE"
    print(f"{prefix} Veredicto: {verdict} | Constraints violadas: {violated or ['ninguna']}")
    return {"feasibility_verdict": verdict, "violated_constraints": violated, "details": details}


def run_mip(panel: pd.DataFrame, config: dict, time_limit: float = 300.0):
    """Llama a MultiSeasonSquadPlanner y retorna resultado."""
    from src.optimization.multi_season_planner import MultiSeasonSquadPlanner  # noqa
    planner = MultiSeasonSquadPlanner(
        player_panel=panel, config=config, time_limit_s=time_limit
    )
    return planner.run()


def run_full_h5(panel: pd.DataFrame, club_cfg: dict, out_dir: Path) -> dict:
    """Full MIP sobre H=5 temporadas."""
    seasons = list(range(HORIZON + 1))
    cfg     = make_mip_config(club_cfg, seasons)
    result  = run_mip(panel, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj    = getattr(result, "objective_value", None)
    status = str(getattr(result, "status", "?"))

    # Diagnóstico automático de infeasibilidad
    infeasibility_diag = {}
    if "infeasible" in status.lower() or obj is None or (isinstance(obj, float) and obj < -1000):
        print(f"  [INFO] Full H=5 infeasible — ejecutando diagnóstico de restricciones...")
        infeasibility_diag = diagnose_infeasibility(panel, cfg, season_idx=1,
                                                    label=club_cfg.get("slug", ""))

    with open(out_dir / "result_full_h5.json", "w") as f:
        json.dump({
            "status":              status,
            "total_score":         float(obj) if obj is not None else None,
            "solution":            getattr(result, "solution", {}),
            "infeasibility_diag":  infeasibility_diag,
        }, f, default=str, indent=2)
    # Persistir el diagnóstico de infeasibility en archivo separado para el detector de lock-in
    if infeasibility_diag:
        with open(out_dir / "full_h5_infeasibility_diag.json", "w") as f:
            json.dump(infeasibility_diag, f, default=str, indent=2)
    return result


def run_rolling(
    panel_full: pd.DataFrame,
    club_cfg:   dict,
    out_dir:    Path,
    window:     int,
    label:      str = "",
) -> dict:
    """
    Rolling MIP con ventana deslizante.

    Implementado localmente (sin depender de scripts/experiments/).
    Incluye el squad tracker corregido (set comparison, no sell==1).

    Parameters:
        panel_full: panel long de un club (columnas del formato MIP)
        club_cfg:   dict con transfer_budget_m (scalar), wage_budget_k, max_hires
        out_dir:    directorio de output
        window:     tamaño de la ventana (3 = rolling H=3, 1 = myopic)
        label:      etiqueta para logging
    """
    print(f"  → {label or f'Rolling window={window}'} …", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_squad: set[int] = set(
        panel_full[
            (panel_full["season_idx"] == 0) & (panel_full["is_initial_squad"] == 1)
        ]["player_id"].tolist()
    )
    sold_ever:   set[int] = set()
    all_rosters, all_summaries, step_stats = [], [], []
    total_objective = 0.0

    for step in range(HORIZON):
        orig_t0   = step
        orig_end  = min(step + window, HORIZON)
        orig_seasons = list(range(orig_t0, orig_end + 1))

        sub = panel_full[panel_full["season_idx"].isin(orig_seasons)].copy()
        sub["season_idx"] = sub["season_idx"] - orig_t0
        sub["is_initial_squad"] = sub.apply(
            lambda r: 1 if (int(r["season_idx"]) == 0 and
                            int(r["player_id"]) in current_squad) else 0,
            axis=1,
        )
        sub = sub[~sub["player_id"].isin(sold_ever)].copy()

        sub_seasons = sorted(sub["season_idx"].unique().tolist())
        # Re-índice del presupuesto: misma M£ por cada sub-temporada
        tb = {t: float(club_cfg["transfer_budget_m"]) for t in sub_seasons}
        wb = {t: float(club_cfg["wage_budget_k"]) for t in sub_seasons}
        mh = {t: int(club_cfg["max_hires"]) if t > 0 else 0 for t in sub_seasons}
        cfg_step = {
            "transfer_budget_m":    tb,
            "wage_budget_k":        wb,
            "max_hires_per_season": mh,
            "max_foreign_players":  MAX_FOREIGN,
            "min_u23_players":      MIN_U23,
            "max_age":              MAX_AGE,
            "squad_size_min":       SQUAD_MIN,
            "squad_size_max":       SQUAD_MAX,
            "position_limits":      POSITION_LIMITS,
            "enforce_monotonic_score": False,
            "enforce_min_two_years":   False,
            "enforce_no_rehire":       False,
        }

        result = run_mip(sub, cfg_step, time_limit=180.0)

        # Diagnóstico automático si el step es infeasible
        step_status = str(getattr(result, "status", "?"))
        if "infeasible" in step_status.lower() or result.objective_value is None or \
                (isinstance(result.objective_value, float) and result.objective_value < -1000):
            print(f"    [INFO] Step {step+1} infeasible — ejecutando diagnóstico...")
            diag = diagnose_infeasibility(sub, cfg_step, season_idx=1,
                                          label=f"step{step+1}")
            diag_path = out_dir / f"step_{step}_infeasibility_diag.json"
            with open(diag_path, "w") as _f:
                json.dump(diag, _f, default=str, indent=2)

        # Squad tracker corregido: comparación de conjuntos (no sell==1)
        roster_t1 = pd.DataFrame()
        if result.solution and result.solution.get("roster"):
            roster_df_all = pd.DataFrame(result.solution["roster"])
            roster_t1 = roster_df_all[roster_df_all["season_idx"] == 1]

        players_in_t1 = set(roster_t1["player_id"].tolist()) if not roster_t1.empty else set()
        sold    = current_squad - players_in_t1
        hired   = players_in_t1 - current_squad
        current_squad = players_in_t1
        sold_ever |= sold

        obj = result.objective_value if result.objective_value is not None else 0.0
        total_objective += float(obj)

        # Guardar step
        roster_df  = pd.DataFrame(result.solution.get("roster", []))
        summary_df = pd.DataFrame(result.solution.get("season_summary", []))
        if not roster_df.empty:
            roster_df["season_idx"] += orig_t0
            roster_df["step"] = step
            roster_df.to_csv(out_dir / f"step_{step}_roster.csv", index=False)
        if not summary_df.empty:
            summary_df["season_idx"] += orig_t0
            summary_df["step"] = step
            summary_df.to_csv(out_dir / f"step_{step}_summary.csv", index=False)

        all_rosters.append(roster_df)
        all_summaries.append(summary_df)

        print(f"    Step {step+1}: status={result.status} | "
              f"obj={obj:.1f} | hired={len(hired)} sold={len(sold)}", flush=True)
        step_stats.append({
            "step": step, "orig_season": orig_t0 + 1,
            "status": result.status,
            "n_hired": len(hired), "n_sold": len(sold),
            "objective": obj,
        })

    combined_roster  = pd.concat(all_rosters,   ignore_index=True) if all_rosters   else pd.DataFrame()
    combined_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    # Deduplicar: solo la decisión del step actual (season_idx == step + 1)
    if not combined_summary.empty and "step" in combined_summary.columns:
        mask = (
            (combined_summary["season_idx"] == combined_summary["step"] + 1)
            | ((combined_summary["step"] == 0) & (combined_summary["season_idx"] == 0))
        )
        combined_summary = combined_summary[mask].sort_values("season_idx").reset_index(drop=True)

    if not combined_roster.empty:
        combined_roster.to_csv(out_dir / "roster_combined.csv", index=False)
    if not combined_summary.empty:
        combined_summary.to_csv(out_dir / "season_summary_combined.csv", index=False)

    return {
        "steps":       step_stats,
        "total_score": total_objective,
        "roster":      combined_roster,
        "summary":     combined_summary,
    }


def _realized_score(summary_df: pd.DataFrame, horizon: int = HORIZON) -> float:
    """
    Score realizado para Rolling/Myopic: suma de team_score sobre las temporadas
    realmente comprometidas (season_idx ∈ [1, H]), excluyendo el baseline t=0.

    summary_df debe ser el combined_summary ya deduplicado (devuelto por run_rolling).
    Esto evita el double-counting del objetivo de ventanas overlapping de Rolling.
    """
    if summary_df is None or summary_df.empty or "team_score" not in summary_df.columns:
        return float("nan")
    sub = summary_df[(summary_df["season_idx"] >= 1) & (summary_df["season_idx"] <= horizon)]
    return float(sub["team_score"].sum())


def run_rolling_h3(panel: pd.DataFrame, club_cfg: dict, out_dir: Path) -> dict:
    """Rolling MIP con ventana H=3. Devuelve realized_total (no overlapping window sum)."""
    res = run_rolling(panel, club_cfg, out_dir / "rolling_h3", window=3, label="Rolling H=3")
    realized = _realized_score(res.get("summary"))
    res["realized_score"] = realized
    print(f"    rolling_h3 realized={realized:.2f}  (window-sum={res['total_score']:.2f})")
    # Persistir step_stats detallados para detección de lock-in offline
    with open(out_dir / "rolling_h3_step_stats.json", "w") as f:
        json.dump({"step_stats": res["steps"], "total_score": res["total_score"],
                   "realized_score": realized}, f, default=str, indent=2)
    return res


def run_myopic(panel: pd.DataFrame, club_cfg: dict, out_dir: Path) -> dict:
    """Myopic: H=1, 5 MIPs secuenciales de 1 temporada. window=1 no double-cuenta."""
    res = run_rolling(panel, club_cfg, out_dir / "myopic", window=1, label="Myopic H=1")
    realized = _realized_score(res.get("summary"))
    res["realized_score"] = realized
    print(f"    myopic realized={realized:.2f}  (window-sum={res['total_score']:.2f})")
    with open(out_dir / "myopic_step_stats.json", "w") as f:
        json.dump({"step_stats": res["steps"], "total_score": res["total_score"],
                   "realized_score": realized}, f, default=str, indent=2)
    return res


def _baseline_score(panel: pd.DataFrame) -> float:
    """Score del squad inicial en t=0 (suma de score_pred para is_initial_squad=1)."""
    sub = panel[(panel["season_idx"] == 0) & (panel["is_initial_squad"] == 1)]
    return float(sub["score_pred"].sum())


def _is_club_done(slug: str) -> bool:
    """Retorna True si el club ya tiene los 3 resultados (full_h5 + rolling + myopic)."""
    d = OUT_DIR / slug
    return (d / "result_full_h5.json").exists() and (d / "roster_combined.csv").exists()


def _load_h5_score(slug: str) -> float:
    """Carga total_score del resultado full_h5 guardado."""
    p = OUT_DIR / slug / "result_full_h5.json"
    if not p.exists():
        return np.nan
    with open(p) as f:
        d = json.load(f)
    return float(d.get("total_score") or np.nan)


def run_club(
    club_name: str,
    p10: pd.DataFrame,
    country_lkp: dict[int, str],
    time_limit: float = 300.0,
    scenario: str = "I",
    club_transfer_budget_m: float | None = None,
    club_wage_budget_k: float | None = None,
) -> dict:
    """
    Corre los 3 horizontes para un club. Retorna resumen con score gains
    y detección de lock-in para cada variante.

    Parameters
    ----------
    scenario : 'I' (uniforme) o 'II' (club-específico).
    club_transfer_budget_m : presupuesto en M£ para Escenario II (override).
    club_wage_budget_k     : presupuesto salarial en k£/semana para Escenario II.
    """
    slug = club_name.lower().replace(" ", "_").replace("&", "and").replace(".", "")
    scenario_dir = f"scenario_{scenario}"
    out_dir = OUT_DIR / scenario_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    transfer_m = club_transfer_budget_m if (scenario == "II" and club_transfer_budget_m is not None) \
                  else DEFAULT_TRANSFER_M
    wage_k = club_wage_budget_k if (scenario == "II" and club_wage_budget_k is not None) \
              else E5A_WAGE_K

    club_cfg = {
        "slug":             slug,
        "team_match":       club_name,
        "transfer_budget_m": transfer_m,
        "wage_budget_k":    wage_k,
        "max_hires":        MAX_HIRES,
    }

    panel = build_optimizer_panel(p10, club_name, country_lkp)
    n_init = (panel[panel["season_idx"] == 0]["is_initial_squad"] == 1).sum()
    print(f"  {club_name} [Esc.{scenario}]: {len(panel['player_id'].unique())} jugadores, "
          f"{n_init} en squad inicial | B_i={transfer_m:.1f}M£ | W_i={wage_k:.1f}k£/wk")

    if n_init < 15:
        print(f"  [SKIP] {club_name} tiene solo {n_init} jugadores en squad inicial — "
              f"insuficiente para restricciones del MIP")
        return {"club_name": club_name, "status": "SKIP_NO_SQUAD",
                "scenario": scenario,
                "h5_score": np.nan, "rolling_h3_score": np.nan, "myopic_score": np.nan}

    baseline_score = _baseline_score(panel)

    variants_to_run = [
        ("full_h5",    run_full_h5),
        ("rolling_h3", run_rolling_h3),
        ("myopic",     run_myopic),
    ]

    results = {}
    for variant, fn in variants_to_run:
        try:
            res = fn(panel, club_cfg, out_dir)
            if variant == "full_h5":
                # Full H=5: objective_value YA es sum sobre 5 seasons realizadas
                score = _extract_total_score(res)
            else:
                # Rolling/Myopic: usar realized_score (sum team_score temporadas comprometidas)
                # NO usar total_score (sum de window-objectives → double-count en Rolling).
                score = res.get("realized_score", np.nan) if isinstance(res, dict) else np.nan
            results[variant] = {"result": res, "total_score": score}
            print(f"    {variant}: score={score:.2f}")
        except Exception as exc:
            print(f"    [ERROR] {variant} — {exc}")
            results[variant] = {"result": None, "total_score": np.nan}

    h5    = results.get("full_h5",    {}).get("total_score", np.nan)
    roll  = results.get("rolling_h3", {}).get("total_score", np.nan)
    myop  = results.get("myopic",     {}).get("total_score", np.nan)

    # ── Detección de lock-in ──────────────────────────────────────────────────
    # Full H=5
    h5_diag_path = out_dir / "full_h5_infeasibility_diag.json"
    h5_status = "Optimal"
    if results.get("full_h5", {}).get("result") is not None:
        h5_status = str(getattr(results["full_h5"]["result"], "status", "Optimal"))
    lock_h5 = detect_lock_in_full_h5(h5_status, h5, baseline_score, h5_diag_path)

    # Rolling H=3 — usar step_stats
    rolling_steps = results.get("rolling_h3", {}).get("result", {}).get("steps", []) \
        if isinstance(results.get("rolling_h3", {}).get("result"), dict) else []
    lock_roll = detect_lock_in_rolling(rolling_steps, baseline_score,
                                       out_dir=out_dir / "rolling_h3")

    # Myopic — usar step_stats
    myopic_steps = results.get("myopic", {}).get("result", {}).get("steps", []) \
        if isinstance(results.get("myopic", {}).get("result"), dict) else []
    lock_myop = detect_lock_in_rolling(myopic_steps, baseline_score,
                                       out_dir=out_dir / "myopic")

    print(f"    Lock-in: full_h5={summarize_lock_in(lock_h5)} | "
          f"rolling={summarize_lock_in(lock_roll)} | "
          f"myopic={summarize_lock_in(lock_myop)}")

    delta_h5_vs_myop = ((h5 - myop) / abs(myop) * 100) if not np.isnan(myop) and myop != 0 else np.nan
    delta_roll_vs_myop = ((roll - myop) / abs(myop) * 100) if not np.isnan(myop) and myop != 0 else np.nan

    return {
        "club_name":            club_name,
        "scenario":             scenario,
        "transfer_budget_m":    transfer_m,
        "wage_budget_k":        wage_k,
        "status":               "OK",
        "baseline_score":       baseline_score,
        "h5_score":             h5,
        "rolling_h3_score":     roll,
        "myopic_score":         myop,
        "delta_pct_h5_myop":    delta_h5_vs_myop,
        "delta_pct_roll_myop":  delta_roll_vs_myop,
        "lock_in_h5":           lock_h5["lock_in"],
        "lock_in_h5_step":      lock_h5["lock_in_step"],
        "lock_in_h5_reason":    lock_h5["lock_in_reason"],
        "lock_in_rolling":      lock_roll["lock_in"],
        "lock_in_rolling_step": lock_roll["lock_in_step"],
        "lock_in_rolling_reason": lock_roll["lock_in_reason"],
        "lock_in_myopic":       lock_myop["lock_in"],
        "lock_in_myopic_step":  lock_myop["lock_in_step"],
        "lock_in_myopic_reason": lock_myop["lock_in_reason"],
        "n_players":            len(panel["player_id"].unique()),
        "n_initial_squad":      int(n_init),
    }


def _extract_total_score(result) -> float:
    """
    Extrae el score total acumulado del resultado del MIP.
    Maneja tres tipos de resultado:
      - SolverResult (full MIP): atributo objective_value
      - dict rolling: clave 'total_score' (suma de objective por step)
      - dict full: clave 'total_score' o 'objective'
    """
    if result is None:
        return np.nan
    if isinstance(result, dict):
        for key in ("total_score", "objective_value", "objective"):
            val = result.get(key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                return float(val)
        return np.nan
    # SolverResult object
    obj = getattr(result, "objective_value", None)
    if obj is not None:
        return float(obj)
    return float(getattr(result, "total_score", np.nan))


def run_scenario(scenario: str, p10: pd.DataFrame,
                 country_lkp: dict[int, str]) -> pd.DataFrame:
    """
    Corre P11 para el escenario indicado (I = uniforme, II = club-específico).

    Filtra a los 20 clubes EPL 2023/24 (EPL_2324_CLUBS).
    Para Escenario II, carga B_i y W_i desde P14 + p09_wage_panel.
    """
    print("=" * 60)
    print(f"P11 — Escenario {scenario} sobre los 20 clubes EPL 2023/24")
    print("=" * 60)

    # ── Filtrar a los 20 clubes EPL 23/24 ─────────────────────────────────────
    clubs_in_panel = set(p10["team_name"].dropna().unique().tolist())
    clubs = sorted(EPL_2324_CLUBS & clubs_in_panel)
    missing = EPL_2324_CLUBS - clubs_in_panel
    if missing:
        print(f"  [WARN] Clubes EPL 23/24 NO presentes en P10: {sorted(missing)}")
    print(f"  Clubes a correr: {len(clubs)} (esperados: 20)")
    assert len(clubs) == 20, \
        f"Se esperaban 20 clubes EPL 23/24 pero se encontraron {len(clubs)}"

    # ── Cargar presupuestos club-específicos para Escenario II ───────────────
    budgets_m: dict[str, float] = {}
    wages_k:   dict[str, float] = {}
    if scenario == "II":
        budgets_m = load_club_budgets()
        wages_k   = load_club_wage_bills()
        print(f"  Budgets cargados desde P14 ({len(budgets_m)} clubes)")
        print(f"  Wages calculados desde P09+P10 ({len(wages_k)} clubes)")

    summaries = []
    for club in clubs:
        print(f"\n  Corriendo {club}...")
        kw = {"scenario": scenario}
        if scenario == "II":
            kw["club_transfer_budget_m"] = budgets_m.get(club, DEFAULT_TRANSFER_M)
            kw["club_wage_budget_k"]     = wages_k.get(club, E5A_WAGE_K)
        summary = run_club(club, p10, country_lkp, **kw)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    # ── Tests de regresión ────────────────────────────────────────────────────
    ok_clubs = summary_df[summary_df["status"] == "OK"]
    n_ok = len(ok_clubs)
    print(f"\n  [Esc.{scenario}] Clubes con solución: {n_ok}/{len(clubs)}")

    if n_ok > 0:
        med_h5    = ok_clubs["delta_pct_h5_myop"].median()
        med_roll  = ok_clubs["delta_pct_roll_myop"].median()
        n_h5_pos  = (ok_clubs["delta_pct_h5_myop"] > 0).sum()
        n_roll_pos= (ok_clubs["delta_pct_roll_myop"] > 0).sum()
        print(f"  Mediana Δ% Full H=5 vs Myopic: {med_h5:.2f}%  ({n_h5_pos}/{n_ok} positivos)")
        print(f"  Mediana Δ% Rolling vs Myopic:  {med_roll:.2f}%  ({n_roll_pos}/{n_ok} positivos)")

        # Reporte de lock-in
        for variant in ("h5", "rolling", "myopic"):
            n_lock = ok_clubs[f"lock_in_{variant}"].sum()
            if n_lock > 0:
                print(f"  Lock-in {variant}: {n_lock} clubes")
                lock_sub = ok_clubs[ok_clubs[f"lock_in_{variant}"]][
                    ["club_name", f"lock_in_{variant}_step", f"lock_in_{variant}_reason"]
                ]
                for _, row in lock_sub.iterrows():
                    print(f"    {row['club_name']:30s} step={row[f'lock_in_{variant}_step']} "
                          f"reason={row[f'lock_in_{variant}_reason']}")

    out_path = OUT_DIR / f"e5a_summary_scenario_{scenario}.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n  Output: {out_path}")
    return summary_df


def main() -> dict[str, pd.DataFrame]:
    """Ejecuta P11 para los dos escenarios (I y II) salvo que se pase --scenario."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["I", "II", "BOTH"], default="BOTH",
                        help="Escenario a ejecutar (default: BOTH)")
    args = parser.parse_args()

    assert PANEL_PATH.exists(), \
        f"[ERROR] {PANEL_PATH} no encontrado — correr P10 primero."

    p10 = pd.read_parquet(PANEL_PATH)
    print(f"  Panel P10: {len(p10)} jugadores")
    country_lkp = load_country_lookup()

    results: dict[str, pd.DataFrame] = {}
    if args.scenario in ("I", "BOTH"):
        results["I"] = run_scenario("I", p10, country_lkp)
    if args.scenario in ("II", "BOTH"):
        results["II"] = run_scenario("II", p10, country_lkp)
    return results


if __name__ == "__main__":
    main()
