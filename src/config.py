"""Central paths, constants, and RNG seed shared across the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BRIEFS_DIR = REPORTS_DIR / "briefs"
DB_PATH = DATA_PROCESSED / "finance.db"

for d in (DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, BRIEFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42

# Number of real UCI customer profiles to draw the synthetic account/transaction
# overlay for. The full profile table keeps all ~41k real rows; the overlay is
# sampled down to keep the demo fast to build and review.
N_SYNTHETIC_CUSTOMERS = 5000

# 2025 US single-filer marginal tax brackets (simplified, no deductions/credits).
TAX_BRACKETS_SINGLE_2025 = [
    (0, 11_925, 0.10),
    (11_925, 48_475, 0.12),
    (48_475, 103_350, 0.22),
    (103_350, 197_300, 0.24),
    (197_300, 250_525, 0.32),
    (250_525, 626_350, 0.35),
    (626_350, float("inf"), 0.37),
]

IRS_401K_EMPLOYEE_LIMIT_2025 = 23_500
TARGET_HIGH_YIELD_APY = 0.045
EMERGENCY_FUND_TARGET_MONTHS = 3
