"""
P14 — Club Transfer Budget Caps
================================
Objetivo:
    Calcular budget_cap_i para cada club EPL: el presupuesto de transferencias
    constante en el tiempo usado en el MIP multi-período.

    Fórmula:
        B_i = max(0, mean_G_i − α × ρ̂_i × mean_I_i)

    Donde:
        mean_G_i   = media histórica de gasto bruto (incoming, non-loan, fee > 0)
        mean_I_i   = media histórica de ingresos por ventas (outgoing, non-loan, fee > 0)
        ρ_raw_i    = mean_I_i / mean_G_i  (ratio de dependencia de ventas)
        ρ̂_i        = shrinkage bayesiano hacia la mediana de liga (k = 1 temporada)
                   = (n_i / (n_i + k)) × ρ_raw_i + (k / (n_i + k)) × ρ_liga
        α          = 0.5  (conservadurismo presupuestario, parámetro de sensibilidad)

    Moneda: EUR millones (canónica del proyecto).

    Fallbacks:
        - n_i = 0 (sin historial): Q25 de la liga  → Ipswich si sube a EPL sin datos
        - n_i = 1 (recién ascendido con 1 temporada): Q25 de la liga
          → Ipswich Town (único verano en EPL: 2024, gasto atípico de debutante)
        - Clubs con n_i ≥ 2 y temporadas con gap (relegación): media solo sobre
          temporadas donde el club estaba en EPL (celdas con G_i = 0 excluidas)

Upstream:
    data/external/transfermarkt/epl_summer_2022_transfers.json
    data/external/transfermarkt/epl_summer_2023_transfers.json
    data/external/transfermarkt/epl_summer_2024_transfers.json

Output:
    outputs/pipeline/p14_budget_caps.csv
    outputs/pipeline/p14_budget_caps.parquet
    Columnas: club_name, n_seasons, mean_G_eur, mean_I_eur, rho_raw, rho_hat,
              budget_cap_eur_m, fallback_used
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import REPO, TM_DIR, OUT_DIR  # noqa: E402
OUT_DIR.mkdir(parents=True, exist_ok=True)

TM_FILES = {
    "2022": TM_DIR / "epl_summer_2022_transfers.json",
    "2023": TM_DIR / "epl_summer_2023_transfers.json",
    "2024": TM_DIR / "epl_summer_2024_transfers.json",
}

ALPHA = 0.5       # conservadurismo presupuestario
K_SHRINKAGE = 1   # peso prior bayesiano (1 temporada equivalente)
MIN_SEASONS_FULL = 2  # n mínimo para no usar fallback Q25

# Club name normalization: algunos nombres difieren entre TM JSONs y StatsBomb
CLUB_NAME_MAP = {
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolverhampton",
    "Leeds United": None,   # relegado 2022/23, no en EPL 2023/24–2024/25
}


def load_transfers() -> pd.DataFrame:
    """Carga y concatena los 3 JSON de transfermarkt en un DataFrame plano."""
    records = []
    for season_str, path in TM_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Transfermarkt JSON no encontrado: {path}\n"
                "Ejecutar el scraper de transfermarkt antes de P14."
            )
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for r in raw:
            records.append({
                "season":       int(season_str),
                "club_name":    r["club_name"],
                "direction":    r["direction"],    # "in" | "out"
                "fee_eur_m":    float(r.get("transfer_fee_m", 0.0)),
                "is_loan":      bool(r.get("is_loan", False)),
            })
    return pd.DataFrame(records)


def normalize_club_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de clubes a los usados en StatsBomb."""
    df = df.copy()
    df["club_name"] = df["club_name"].map(
        lambda x: CLUB_NAME_MAP.get(x, x)
    )
    return df.dropna(subset=["club_name"])  # excluye Leeds, etc.


def compute_club_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa per-club: mean_G (gross spend incoming) y mean_I (transfer income outgoing).
    Solo incluye transferencias no-préstamo con fee > 0.
    """
    # Filtrar: solo permanentes con fee real
    perm = df[~df["is_loan"] & (df["fee_eur_m"] > 0)].copy()

    # EPL clubs con al menos 1 registro
    all_clubs = sorted(df["club_name"].unique())
    results = []

    for club in all_clubs:
        club_df = perm[perm["club_name"] == club]
        incoming = club_df[club_df["direction"] == "in"]
        outgoing = club_df[club_df["direction"] == "out"]

        # Temporadas en las que el club estuvo en EPL (según los JSON)
        seasons_in_epl = sorted(df[df["club_name"] == club]["season"].unique())
        n_seasons = len(seasons_in_epl)

        # Gross spend: suma de incoming per season, media de esas temporadas
        g_per_season = incoming.groupby("season")["fee_eur_m"].sum()
        # Incluir 0 para temporadas en EPL sin compras permanentes
        g_full = pd.Series(
            {s: g_per_season.get(s, 0.0) for s in seasons_in_epl}
        )
        mean_G = g_full.mean()

        # Transfer income: suma de outgoing per season
        i_per_season = outgoing.groupby("season")["fee_eur_m"].sum()
        i_full = pd.Series(
            {s: i_per_season.get(s, 0.0) for s in seasons_in_epl}
        )
        mean_I = i_full.mean()

        results.append({
            "club_name":  club,
            "n_seasons":  n_seasons,
            "mean_G_eur": mean_G,
            "mean_I_eur": mean_I,
        })

    return pd.DataFrame(results)


def apply_bayesian_shrinkage(
    stats: pd.DataFrame,
    alpha: float = ALPHA,
    k: int = K_SHRINKAGE,
) -> pd.DataFrame:
    """
    Aplica shrinkage bayesiano de ρ_raw hacia la media de liga.
    Para clubes con n < MIN_SEASONS_FULL: fallback al Q25 de liga.
    """
    df = stats.copy()

    # ── Liga-wide stats (solo clubes con datos suficientes) ────────────────────
    df_full = df[df["n_seasons"] >= MIN_SEASONS_FULL].copy()
    rho_liga = (
        df_full["mean_I_eur"].sum() / df_full["mean_G_eur"].sum()
        if df_full["mean_G_eur"].sum() > 0 else 0.55
    )
    print(f"  ρ_liga = {rho_liga:.3f}")

    # ── ρ_raw por club ─────────────────────────────────────────────────────────
    df["rho_raw"] = np.where(
        df["mean_G_eur"] > 0,
        df["mean_I_eur"] / df["mean_G_eur"],
        rho_liga,
    )

    # ── Shrinkage bayesiano ────────────────────────────────────────────────────
    df["rho_hat"] = (
        (df["n_seasons"] / (df["n_seasons"] + k)) * df["rho_raw"]
        + (k / (df["n_seasons"] + k)) * rho_liga
    )

    # ── B_market_i ────────────────────────────────────────────────────────────
    df["budget_cap_eur_m"] = np.maximum(
        0.0,
        df["mean_G_eur"] - alpha * df["rho_hat"] * df["mean_I_eur"],
    )

    # ── Fallback Q25 para clubes con n < MIN_SEASONS_FULL ────────────────────
    # Calculado sobre los clubes con datos suficientes (filtrar DESPUÉS de calcular budget_cap)
    q25 = np.percentile(
        df.loc[df["n_seasons"] >= MIN_SEASONS_FULL, "budget_cap_eur_m"].clip(lower=0),
        25,
    )
    print(f"  Q25 liga (clubes n≥{MIN_SEASONS_FULL}) = {q25:.1f} EUR M")

    fallback_mask = df["n_seasons"] < MIN_SEASONS_FULL
    df.loc[fallback_mask, "budget_cap_eur_m"] = q25
    df["fallback_used"] = fallback_mask

    return df, q25


def main() -> None:
    print("=" * 60)
    print("P14 — Club Transfer Budget Caps")
    print(f"       α = {ALPHA}, k = {K_SHRINKAGE}, moneda = EUR M")
    print("=" * 60)

    # A. Cargar y normalizar
    print("\n  Cargando transfermarkt JSONs...")
    raw = load_transfers()
    raw = normalize_club_names(raw)
    print(f"  {len(raw):,} transferencias cargadas; {raw['club_name'].nunique()} clubes")

    # B. Estadísticas por club
    print("\n  Computando mean_G y mean_I por club...")
    stats = compute_club_stats(raw)

    # C. Shrinkage + budget_cap
    print("\n  Aplicando shrinkage bayesiano...")
    caps, q25 = apply_bayesian_shrinkage(stats, alpha=ALPHA, k=K_SHRINKAGE)

    # D. Resumen
    caps_sorted = caps.sort_values("budget_cap_eur_m", ascending=False)
    print("\n  Budget caps por club (EUR M, α=0.5):")
    print(
        caps_sorted[["club_name", "n_seasons", "mean_G_eur", "mean_I_eur",
                      "rho_hat", "budget_cap_eur_m", "fallback_used"]]
        .round(1)
        .to_string(index=False)
    )

    median_cap = caps["budget_cap_eur_m"].median()
    mean_cap   = caps["budget_cap_eur_m"].mean()
    print(f"\n  Mediana liga: {median_cap:.1f} EUR M")
    print(f"  Media liga:   {mean_cap:.1f} EUR M")
    print(f"  Fallback (Q25={q25:.1f} EUR M): {caps['fallback_used'].sum()} clubes → "
          f"{caps.loc[caps['fallback_used'], 'club_name'].tolist()}")

    # E. Guardar
    csv_path     = OUT_DIR / "p14_budget_caps.csv"
    parquet_path = OUT_DIR / "p14_budget_caps.parquet"
    caps.to_csv(csv_path, index=False)
    caps.to_parquet(parquet_path, index=False)
    print(f"\n  Output CSV:     {csv_path}")
    print(f"  Output Parquet: {parquet_path}")


if __name__ == "__main__":
    main()
