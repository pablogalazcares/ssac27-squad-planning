"""
P13_paper2_figures.py — Figuras del Paper 2 (Optimización).

Genera ÚNICAMENTE las figuras que usa el Paper 2, leyendo de la frontera de
proyecciones (Paper 1) + los outputs del propio pipeline del Paper 2.
Escribe a paper2_optimization/figures/ (PDF + SVG).

Figuras:
  Input de proyección (contexto):
    - fig_score_distribution      (de p03 scores)
    - fig_projection_convergence  (de p07 proyecciones)
    - fig_spaghetti_all           (de p07 proyecciones)
  Modelo de valor de mercado (§5):
    - fig_mv_age                  (de MV legacy + universo)
    - fig_mv_position             (de MV legacy + universo)
  Resultados de optimización (§8):
    - fig_e5a_comparison          (de p11_e5a summaries: Δ% por club)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (  # noqa: E402
    FIG_DIR, SCORES_PATH, UNIVERSE_PATH, PROJ_PATH, MV_RAW_PATH, P11_OUT,
)
from pipeline_utils import load_scores_with_pt, get_selected_alpha  # noqa: E402

# ── Estilo global ─────────────────────────────────────────────────────────────
POSITION_COLORS = {"GK": "#2196F3", "DEF": "#4CAF50", "MID": "#FF9800", "FWD": "#F44336"}
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})


def _savefig(name: str) -> None:
    path = FIG_DIR / f"{name}.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close()
    print(f"  -> {path.name}")


def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    window = max(3, int(len(xs) * frac))
    ys_smooth = uniform_filter1d(ys.astype(float), size=window, mode="nearest")
    return xs, ys_smooth


# ══════════════════════════════════════════════════════════════════════════════
# Input de proyección
# ══════════════════════════════════════════════════════════════════════════════

def fig_score_distribution(scores: pd.DataFrame) -> None:
    """Violin plots de score_pt por posición y temporada."""
    data = scores.dropna(subset=["score_pt"])
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.5), sharey=True, sharex=True)
    for ax, pos in zip(axes, POSITION_ORDER):
        sub = data[data["position_group"] == pos]
        vals_22 = sub[sub["season_name"] == "2022/2023"]["score_pt"].values
        vals_23 = sub[sub["season_name"] == "2023/2024"]["score_pt"].values
        for i, (vals, color) in enumerate([(vals_22, "lightblue"), (vals_23, POSITION_COLORS[pos])]):
            if len(vals) < 2:
                continue
            parts = ax.violinplot(vals, positions=[i], showmedians=True, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("black")
        ax.set_title(pos)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["22/23", "23/24"], fontsize=7)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("$\\mathrm{score}_{p,s}$")
    _savefig("fig_score_distribution")


def fig_projection_convergence(proj: pd.DataFrame) -> None:
    """Distribución proyectada por horizonte (boxplots + media/std)."""
    score_cols = sorted([c for c in proj.columns if c.startswith("score_h")],
                        key=lambda c: int(c.replace("score_h", "")))
    H = len(score_cols) - 1
    data = [proj[c].dropna().values for c in score_cols]

    fig = plt.figure(figsize=(11, 4))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    bplot = ax1.boxplot(data, positions=range(H + 1), patch_artist=True, whis=(5, 95),
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="", markersize=0))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, H + 1))
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    ax1.set_xlabel("Planning horizon $k$")
    ax1.set_ylabel("Projected $\\hat{\\mathrm{score}}_{p,t+k}$")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(H + 1))

    ax2 = fig.add_subplot(gs[1])
    means = [np.nanmean(d) for d in data]
    stds = [np.nanstd(d) for d in data]
    ks = range(H + 1)
    ax2.plot(ks, means, "o-", color="#1565C0", label="Mean", linewidth=1.5)
    ax2.plot(ks, stds, "s--", color="#E65100", label="Std", linewidth=1.5)
    ax2.set_xlabel("Horizon $k$")
    ax2.legend()
    ax2.set_ylim(0, None)
    _savefig("fig_projection_convergence")


def fig_spaghetti_all(proj: pd.DataFrame, alpha: float) -> None:
    """Trayectorias individuales de todos los jugadores."""
    score_cols = sorted([c for c in proj.columns if c.startswith("score_h")],
                        key=lambda c: int(c.replace("score_h", "")))
    H = len(score_cols) - 1
    ks = range(H + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for _, row in proj.iterrows():
        pos = row.get("position_group", "MID")
        vals = [row[c] for c in score_cols]
        ax.plot(ks, vals, color=POSITION_COLORS.get(pos, "gray"), alpha=0.08, linewidth=0.4)
    for pos in POSITION_ORDER:
        sub = proj[proj["position_group"] == pos]
        if len(sub) == 0:
            continue
        meds = [sub[c].median() for c in score_cols]
        ax.plot(ks, meds, color=POSITION_COLORS[pos], linewidth=2, linestyle="--", label=pos)
    ax.set_xlabel("Planning horizon $k$")
    ax.set_ylabel("Projected $\\hat{\\mathrm{score}}_{p,t+k}$")
    ax.set_ylim(0, 100)
    ax.legend(title="Position", loc="upper right")
    ax.set_title(f"$N={len(proj)}$ players, $\\alpha={alpha:g}$, $H={H}$")
    _savefig("fig_spaghetti_all")


# ══════════════════════════════════════════════════════════════════════════════
# Modelo de valor de mercado
# ══════════════════════════════════════════════════════════════════════════════

def _mv_with_positions(mv_raw: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    mv = mv_raw.copy()
    mv["market_value"] = pd.to_numeric(mv["market_value"], errors="coerce")
    mv["player_id"] = pd.to_numeric(mv["player_id"], errors="coerce")
    mv = mv.dropna(subset=["player_id", "market_value"])
    mv["player_id"] = mv["player_id"].astype(int)
    return mv.merge(universe[["player_id", "position_group"]], on="player_id", how="inner")


def fig_mv_age(mv: pd.DataFrame) -> None:
    """MV vs edad con LOWESS por grupo posicional (universo del Paper 2)."""
    if "age" not in mv.columns:
        print("  [SKIP] fig_mv_age: columna age no disponible")
        return
    data = mv.dropna(subset=["age"])
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax, pos in zip(axes, POSITION_ORDER):
        sub = data[(data["position_group"] == pos) & (data["market_value"] > 0)]
        if len(sub) < 5:
            continue
        mv_m = sub["market_value"] / 1e6
        ax.scatter(sub["age"], mv_m, alpha=0.2, s=5, color=POSITION_COLORS[pos])
        xs, ys = _lowess(sub["age"].values, mv_m.values)
        ax.plot(xs, ys, color="black", linewidth=1.5)
        ax.set_title(pos)
        ax.set_xlabel("Age")
    axes[0].set_ylabel("Market value (€M)")
    _savefig("fig_mv_age")


def fig_mv_position(mv: pd.DataFrame) -> None:
    """Distribución MV por grupo posicional."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    groups = [mv[mv["position_group"] == p]["market_value"].dropna().values / 1e6
              for p in POSITION_ORDER]
    bplot = ax.boxplot(groups, labels=POSITION_ORDER, patch_artist=True,
                       medianprops=dict(color="black", linewidth=1.5),
                       flierprops=dict(marker=".", markersize=2, alpha=0.2))
    for patch, pos in zip(bplot["boxes"], POSITION_ORDER):
        patch.set_facecolor(POSITION_COLORS[pos])
        patch.set_alpha(0.7)
    ax.set_ylabel("Market value (€M)")
    ax.set_xlabel("Positional group")
    _savefig("fig_mv_position")


# ══════════════════════════════════════════════════════════════════════════════
# Resultados de optimización (E5a)
# ══════════════════════════════════════════════════════════════════════════════

def fig_e5a_comparison() -> None:
    """Δ% (Full H=5 y Rolling H=3 vs Myopic) por club, ambos escenarios."""
    frames = {}
    for sc, label in [("I", "Scenario I (uniform)"), ("II", "Scenario II (club-specific)")]:
        p = P11_OUT / f"e5a_summary_scenario_{sc}.csv"
        if p.exists():
            frames[label] = pd.read_csv(p)
    if not frames:
        print("  [SKIP] fig_e5a_comparison: summaries E5a no disponibles")
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(11, 6), sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for ax, (label, df) in zip(axes, frames.items()):
        df = df.sort_values("delta_pct_h5_myop")
        y = np.arange(len(df))
        ax.barh(y - 0.2, df["delta_pct_h5_myop"], height=0.4,
                color="#1565C0", alpha=0.8, label="Full $H{=}5$")
        ax.barh(y + 0.2, df["delta_pct_roll_myop"], height=0.4,
                color="#E65100", alpha=0.8, label="Rolling $H{=}3$")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(df["club_name"], fontsize=7)
        ax.set_xlabel("$\\Delta\\%$ vs Myopic $H{=}1$")
        ax.set_title(label)
        ax.legend(loc="lower right")
    _savefig("fig_e5a_comparison")


def main() -> None:
    print("\n[P13] Figuras del Paper 2 →", FIG_DIR)
    alpha = get_selected_alpha()

    scores = load_scores_with_pt(SCORES_PATH, alpha)
    universe = pd.read_parquet(UNIVERSE_PATH)
    proj = pd.read_parquet(PROJ_PATH)
    mv_raw = pd.read_csv(MV_RAW_PATH)
    mv = _mv_with_positions(mv_raw, universe)

    print("\n  Input de proyección:")
    fig_score_distribution(scores)
    fig_projection_convergence(proj)
    fig_spaghetti_all(proj, alpha)

    print("\n  Modelo de valor de mercado:")
    fig_mv_age(mv)
    fig_mv_position(mv)

    print("\n  Resultados de optimización:")
    fig_e5a_comparison()

    print("\n[P13] OK")


if __name__ == "__main__":
    main()
