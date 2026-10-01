#!/usr/bin/env bash
# =====================================================================
# run_pipeline.sh — Pipeline INDEPENDIENTE del Paper 2 (Optimización)
#
# Arranca en P08 consumiendo la frontera de proyecciones (Paper 1) y
# regenera todos los outputs/figuras/tablas en paper2_optimization/.
#
# Frontera (configurable en _config.py → PROJECTIONS_DIR):
#   p03_score_player_season, p06_projection_universe, p07_projected_scores,
#   p05_selected_config + externos (MV legacy, Capology, Transfermarkt).
#
# Uso:  bash paper2_optimization/pipeline/run_pipeline.sh
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

# Activar venv si existe
if [[ -f "$REPO/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
fi

run() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "▶ $1"
  echo "════════════════════════════════════════════════════════════════"
  python "$HERE/$1"
}

# Orden de dependencias:
#   P14 (budget caps) → P08 (MV), P09 (wages) → P10 (panel) → P11 (E5a)
#   → P12 (E5b), P12c (Q/M) → P13 (figuras), P15 (tablas)
run P14_budget_caps.py
run P08_mv_model.py
run P09_wage_panel.py
run P10_optimizer_panel.py
run P11_e5a_experiments.py
run P12_e5b_decision_regret.py
run P12c_qm_comparison.py
run P13_paper2_figures.py
run P15_paper2_tables.py

echo ""
echo "✓ Pipeline Paper 2 completo. Outputs en paper2_optimization/outputs/"
