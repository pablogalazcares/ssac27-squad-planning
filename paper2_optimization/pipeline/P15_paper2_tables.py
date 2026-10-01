"""
P15_paper2_tables.py — Tablas LaTeX del Paper 2 generadas desde outputs.

Elimina el hard-code de las tablas de resultados: cada .tex se regenera desde
los parquet/csv del pipeline del Paper 2 y de la frontera de proyecciones.
Escribe a paper2_optimization/tables/.

Tablas generadas (data-driven):
  - tab_ranking_preservation.tex   (de p07: ρ horizonte vs h0)
  - tab_e5a_scenario_I.tex         (de p11 summary escenario I)
  - tab_e5a_scenario_II.tex        (de p11 summary escenario II)
  - tab_lock_in_summary.tex        (de p11 summaries, ambos escenarios)
  - tab_qm_results.tex             (de p12c_h4_evaluation)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import TAB_DIR, PROJ_PATH, P11_OUT, OUT_DIR  # noqa: E402


def _write(name: str, body: str) -> None:
    path = TAB_DIR / f"{name}.tex"
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"  -> {path.name}")


def _scenario_summary(sc: str) -> pd.DataFrame | None:
    p = P11_OUT / f"e5a_summary_scenario_{sc}.csv"
    return pd.read_csv(p) if p.exists() else None


# Clubes objetivo del experimento (20 EPL 2023/24), con los nombres tal como
# aparecen en p14_budget_caps. Southampton/Leicester/Ipswich aportan datos
# históricos de transferencias pero NO son clubes-objetivo del MIP.
TARGET_CLUBS_P14 = {
    "AFC Bournemouth", "Arsenal FC", "Aston Villa", "Brentford FC", "Brighton",
    "Burnley FC", "Chelsea FC", "Crystal Palace", "Everton FC", "Fulham FC",
    "Liverpool FC", "Luton Town", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Sheffield United",
    "Tottenham Hotspur", "West Ham United", "Wolverhampton",
}


# ── tab_budget_caps (Escenario II) ────────────────────────────────────────────
def tab_budget_caps() -> None:
    p = OUT_DIR / "p14_budget_caps.csv"
    if not p.exists():
        print("  [SKIP] tab_budget_caps: p14_budget_caps no disponible")
        return
    df = pd.read_csv(p)
    df = df[df["club_name"].isin(TARGET_CLUBS_P14)].sort_values(
        "budget_cap_eur_m", ascending=False)
    med = df["budget_cap_eur_m"].median()
    fb = df[df["fallback_used"]]
    q25 = fb["budget_cap_eur_m"].iloc[0] if len(fb) else float("nan")
    n_fb = int(df["fallback_used"].sum())

    def _row(r) -> str:
        mark = "$^{\\dagger}$" if r.fallback_used else ""
        if r.fallback_used:
            return (f"{r.club_name}{mark} & {int(r.n_seasons)} & \\multicolumn{{3}}{{c}}{{fallback}} "
                    f"& {r.budget_cap_eur_m:.1f} \\\\")
        return (f"{r.club_name}{mark} & {int(r.n_seasons)} & {r.mean_G_eur:.1f} & {r.mean_I_eur:.1f} "
                f"& {r.rho_hat:.2f} & {r.budget_cap_eur_m:.1f} \\\\")

    lines = "\n".join(_row(r) for r in df.itertuples())
    body = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{Club-specific transfer budgets ($B_i$, \\si{{\\mega\\EUR}}) for the
  20 target clubs in Scenario~II. $n$: EPL seasons in the data window;
  $\\bar{{G}}$: mean gross spend; $\\bar{{I}}$: mean transfer income;
  $\\hat{{\\rho}}$: shrunk sales-dependence ratio. Clubs with $n < 2$
  ($^{{\\dagger}}$, {n_fb} clubs) receive the fallback
  $B_i = Q_{{25}} \\approx \\SI{{{q25:.1f}}}{{\\mega\\EUR}}$ estimated over the full
  Premier League. Median over the target clubs: \\SI{{{med:.1f}}}{{\\mega\\EUR}}.}}
\\label{{tab:budget_caps}}
\\begin{{tabular}}{{lrrrrc}}
\\toprule
\\textbf{{Club}} & $n$ & $\\bar{{G}}$ & $\\bar{{I}}$ & $\\hat{{\\rho}}$ & $B_i$ \\\\
\\midrule
{lines}
\\midrule
\\textbf{{Median (target)}} & & & & & \\textbf{{{med:.1f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write("tab_budget_caps", body)


# ── tab_ranking_preservation ──────────────────────────────────────────────────
def tab_ranking_preservation() -> None:
    proj = pd.read_parquet(PROJ_PATH)
    cols = sorted([c for c in proj.columns if c.startswith("score_h")],
                  key=lambda c: int(c.replace("score_h", "")))
    base = proj["score_h0"]
    rows = []
    for c in cols:
        h = int(c.replace("score_h", ""))
        rho = 1.0 if h == 0 else spearmanr(proj[c], base, nan_policy="omit").statistic
        rows.append((h, proj[c].mean(), proj[c].std(), rho))
    N = len(proj)
    lines = "\n".join(
        f"{h} & {m:.2f} ({s:.2f}) & {rho:.3f} \\\\" for h, m, s, rho in rows
    )
    body = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{Ranking preservation: Spearman $\\rho$ between projected
  scores at horizon $h$ and the anchor at $h=0$
  ({N}-player projection universe $C = A \\cup B$).
  Source: \\textcite{{companion_performance}}.}}
\\label{{tab:ranking_preservation_p2}}
\\begin{{tabular}}{{lrr}}
\\toprule
$h$ & Score mean (std) & $\\rho$ vs.\\ $h=0$ \\\\
\\midrule
{lines}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write("tab_ranking_preservation", body)


# ── tab_e5a_scenario_{I,II} ───────────────────────────────────────────────────
def _e5a_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    locked = df[df["lock_in_h5"]]["club_name"].tolist()
    return {
        "n": n,
        "med_h5": df["delta_pct_h5_myop"].median(),
        "med_roll": df["delta_pct_roll_myop"].median(),
        "h5_gt0": int((df["delta_pct_h5_myop"] > 0).sum()),
        "roll_gt0": int((df["delta_pct_roll_myop"] > 0).sum()),
        "h5_gt2": int((df["delta_pct_h5_myop"] > 2).sum()),
        "roll_gt5": int((df["delta_pct_roll_myop"] > 5).sum()),
        "li_h5": int(df["lock_in_h5"].sum()),
        "li_roll": int(df["lock_in_rolling"].sum()),
        "li_myop": int(df["lock_in_myopic"].sum()),
        "li_h5_names": locked,
    }


def tab_e5a_scenario(sc: str, budget_desc: str) -> None:
    df = _scenario_summary(sc)
    if df is None:
        print(f"  [SKIP] tab_e5a_scenario_{sc}: summary no disponible")
        return
    m = _e5a_metrics(df)
    li_names = ", ".join(m["li_h5_names"]) if m["li_h5_names"] else "--"
    pct = lambda k: f"{100 * k / m['n']:.0f}\\%"
    body = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{Scenario~{sc} aggregate results across all {m['n']} EPL 2023/24 clubs
  ({budget_desc}).}}
\\label{{tab:e5a_scenario_{sc}}}
\\begin{{tabular}}{{lr}}
\\toprule
Metric & Value \\\\
\\midrule
Median $\\Delta\\%$ Full $H{{=}}5$ vs.\\ Myopic & ${m['med_h5']:+.3f}\\%$ \\\\
Median $\\Delta\\%$ Rolling $H{{=}}3$ vs.\\ Myopic & ${m['med_roll']:+.3f}\\%$ \\\\
Clubs with Full $H{{=}}5 >$ Myopic & {m['h5_gt0']}/{m['n']} ({pct(m['h5_gt0'])}) \\\\
Clubs with Rolling $H{{=}}3 >$ Myopic & {m['roll_gt0']}/{m['n']} ({pct(m['roll_gt0'])}) \\\\
Clubs with $\\Delta\\%_{{H5}} > 2\\%$ (H2 criterion) & {m['h5_gt2']}/{m['n']} ({pct(m['h5_gt2'])}) \\\\
Clubs with $\\Delta\\%_{{\\mathrm{{Rolling}}}} > 5\\%$ (H1 criterion) & {m['roll_gt5']}/{m['n']} ({pct(m['roll_gt5'])}) \\\\
Lock-in: Full $H{{=}}5$ & {m['li_h5']} ({li_names}) \\\\
Lock-in: Rolling $H{{=}}3$ & {m['li_roll']} \\\\
Lock-in: Myopic $H{{=}}1$ & {m['li_myop']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write(f"tab_e5a_scenario_{sc}", body)


# ── tab_lock_in_summary ───────────────────────────────────────────────────────
def tab_lock_in_summary() -> None:
    rows = []
    for sc, desc in [("I", "I (uniform budget)"), ("II", "II (club-specific budget)")]:
        df = _scenario_summary(sc)
        if df is None:
            continue
        rows.append((desc, int(df["lock_in_h5"].sum()),
                     int(df["lock_in_rolling"].sum()), int(df["lock_in_myopic"].sum())))
    if not rows:
        print("  [SKIP] tab_lock_in_summary: summaries no disponibles")
        return
    n = len(_scenario_summary("I"))
    lines = "\n".join(f"{d} & {a} & {b} & {c} \\\\" for d, a, b, c in rows)
    body = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{Lock-in counts by planning variant and scenario
  ({n} EPL 2023/24 clubs).
  Full $H{{=}}5$ consistently achieves the lowest lock-in count.}}
\\label{{tab:lock_in_summary}}
\\begin{{tabular}}{{lccc}}
\\toprule
Scenario & Full $H{{=}}5$ & Rolling $H{{=}}3$ & Myopic $H{{=}}1$ \\\\
\\midrule
{lines}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write("tab_lock_in_summary", body)


# ── tab_qm_results (H4) ───────────────────────────────────────────────────────
def tab_qm_results() -> None:
    p = OUT_DIR / "p12c_h4_evaluation.csv"
    if not p.exists():
        print("  [SKIP] tab_qm_results: p12c_h4_evaluation no disponible")
        return
    df = pd.read_csv(p)
    df = df[df["scenario"].isin(["I", "II"])]
    name = {"I": "I (uniform)", "II": "II (club-specific)"}
    lines = "\n".join(
        f"{name[r.scenario]} & {int(r.n)} & {int(r.n_mip_wins)} "
        f"({r.pct_mip_wins:.0f}\\%) & ${r.median_delta_pct:+.1f}\\%$ \\\\"
        for r in df.itertuples()
    )
    body = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{H4 --- Score per million euro: MIP first-window
  recommendations vs.\\ real summer-2024 signings.
  Restricted to clubs with at least one projectable real transfer.}}
\\label{{tab:qm_results}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Scenario & $n$ evaluable & MIP $>$ real & Median $\\Delta\\%$ $Q/M$ \\\\
\\midrule
{lines}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write("tab_qm_results", body)


def main() -> None:
    print("\n[P15] Tablas del Paper 2 →", TAB_DIR)
    tab_ranking_preservation()
    tab_e5a_scenario("I", "uniform budget: \\pounds 80M transfer, \\pounds 77.2M wage cap")
    tab_e5a_scenario("II", "club-specific budgets from \\cref{eq:budget_cap}")
    tab_lock_in_summary()
    tab_qm_results()
    tab_budget_caps()
    print("\n[P15] OK")


if __name__ == "__main__":
    main()
