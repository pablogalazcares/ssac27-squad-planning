# Plan for Five, Sign for Three: Winning on the Pitch Without Breaking the Bank

Replication repository for an abstract submitted to the MIT Sloan Sports Analytics Conference 2027 Research Paper Competition (Soccer track).

> **Status: work in progress.** This repository contains the optimization code and the derived data used for the results reported in the abstract. The full replication package (upstream player-score construction and documentation) will be released together with the full manuscript.

## What this is

A multi-season squad-planning model for the 20 Premier League clubs of 2023/24. Projected player-quality scores (up to five seasons ahead) feed a mixed-integer program that decides, each season from 2024/25 to 2028/29, which players to sign, keep and release from a 759-player market. Constraints: 25-player squad with positional quotas (GK 3, DEF 8, MID 7, FWD 7), at most 17 foreign players, at least 2 under-23 players, maximum age 38, a per-season transfer budget, a wage cap, and at most five signings per window.

Three planners are compared:

| Planner | Description |
|---|---|
| Myopic (H=1) | Greedy: re-optimizes one season at a time |
| Rolling (H=3) | Three-season horizon, re-planned every summer |
| Full (H=5) | Five-season horizon, solved once |

under two budget regimes:

- **Scenario I (uniform):** 80M transfer budget per season and a GBP 77.2M annual wage cap for every club.
- **Scenario II (club-specific):** transfer budgets calibrated from each club's real net spending over three transfer windows (Bayesian shrinkage) and each club's own 2023/24 wage bill.

## Repository layout

```
src/
  optimization/multi_season_planner.py   Multi-period MIP (PuLP + CBC)
  optimization/squad_optimizer.py        Single-season optimizer
  solvers/base_solver.py                 Solver interface
  models/squad.py                        Player / squad data classes
paper2_optimization/
  pipeline/
    P11_e5a_experiments.py               Main experiment: 3 planners x 20 clubs x 2 scenarios
    _lock_in_detector.py                 Infeasibility / collapse detection
    P08-P15                              Market-value model, wage panel, optimizer panel,
                                         budget calibration, figures and tables
    _config.py                           All paths
  outputs/
    p10_optimizer_panel.parquet          Optimizer input: 759 players x (score, market value, wage) for h=0..5
    p09_wage_panel.parquet               Annual wage per player, h=0..5
    p14_budget_caps.(csv|parquet)        Club-specific transfer budgets (Scenario II)
    p11_e5a/e5a_summary_scenario_*.csv   Results reported in the abstract (one row per club)
data/processed/statsbomb_legacy/
  df_final_market_value.csv              Player nationality lookup (player_id, season, country) for the foreign-player rule
```

## Reproducing the main results

Requires Python 3.10+ and the CBC solver. PuLP ships a CBC binary for most platforms; on Apple Silicon without Rosetta, install a native one (`brew install cbc`) and point PuLP's bundled `solverdir/cbc/osx/i64/cbc` to it.

```bash
pip install -r requirements.txt
python paper2_optimization/pipeline/P11_e5a_experiments.py            # both scenarios
python paper2_optimization/pipeline/P11_e5a_experiments.py --scenario II
```

This regenerates `paper2_optimization/outputs/p11_e5a/e5a_summary_scenario_{I,II}.csv` from the shared optimizer panel. Key columns: `h5_score`, `rolling_h3_score`, `myopic_score` (accumulated squad quality over 2024/25 to 2028/29), `delta_pct_h5_myop` (Full vs Myopic, %), and `lock_in_*` (whether and when a planner reached a season with no feasible squad).

Scripts P08, P09, P10 and P14 rebuild the optimizer inputs from upstream sources that are not redistributed here (see below).

## Data sources

| Data | Source | Included here |
|---|---|---|
| Player-quality scores (event data + On-Ball Value) | StatsBomb, EPL 2022/23 and 2023/24 | Only derived, aggregated projected scores (`score_h0..h5`). Raw event data is licensed and not redistributed. |
| Market values | Transfermarkt | Derived projections (`mv_h0..h5`) |
| Wages | Capology (public estimates) | Annual wage per player |
| Transfer history (budget calibration) | Transfermarkt, summers 2022 to 2024 | Aggregated club-level budget caps only |

Currency note: transfer values and budgets are in EUR (Transfermarkt); wages are in GBP (Capology).

## License

Code: MIT (see `LICENSE`). Derived data is provided for research replication only; rights to the underlying source data remain with their respective providers.
