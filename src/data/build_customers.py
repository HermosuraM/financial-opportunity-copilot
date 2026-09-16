"""Load the real UCI Bank Marketing dataset and derive a customer profile table.

The raw dataset has no income field, so we derive an income estimate from job
category + age via a documented lookup table. This is a modeling assumption,
not a real income figure -- see README "Data & Assumptions".
"""
import numpy as np
import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, RNG_SEED, TAX_BRACKETS_SINGLE_2025

RAW_CSV = DATA_RAW / "bank-additional" / "bank-additional-full.csv"

# Rough US median base salary by BLS-style occupation grouping, used only to
# seed a plausible income distribution per job category (2024-ish figures,
# intentionally coarse).
JOB_BASE_INCOME = {
    "admin.": 52_000,
    "blue-collar": 45_000,
    "entrepreneur": 85_000,
    "housemaid": 30_000,
    "management": 105_000,
    "retired": 38_000,
    "self-employed": 70_000,
    "services": 42_000,
    "student": 12_000,
    "technician": 60_000,
    "unemployed": 15_000,
    "unknown": 45_000,
}

EDUCATION_MULTIPLIER = {
    "illiterate": 0.8,
    "basic.4y": 0.85,
    "basic.6y": 0.88,
    "basic.9y": 0.92,
    "high.school": 1.0,
    "professional.course": 1.15,
    "university.degree": 1.35,
    "unknown": 1.0,
}


def marginal_tax_bracket(income: float) -> float:
    for lo, hi, rate in TAX_BRACKETS_SINGLE_2025:
        if lo <= income < hi:
            return rate
    return TAX_BRACKETS_SINGLE_2025[-1][2]


def build_customers() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df = df.reset_index(drop=True)
    df.insert(0, "customer_id", df.index + 1)

    rng = np.random.default_rng(RNG_SEED)

    base = df["job"].map(JOB_BASE_INCOME).fillna(JOB_BASE_INCOME["unknown"])
    edu_mult = df["education"].map(EDUCATION_MULTIPLIER).fillna(1.0)
    age_mult = np.clip(1 + (df["age"] - 40) * 0.006, 0.7, 1.4)
    noise = rng.lognormal(mean=0.0, sigma=0.18, size=len(df))
    est_income = (base * edu_mult * age_mult * noise).round(-2).clip(lower=12_000)

    out = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "age": df["age"],
            "job": df["job"],
            "marital": df["marital"],
            "education": df["education"],
            "has_credit_default": (df["default"] == "yes").astype(int),
            "has_housing_loan": (df["housing"] == "yes").astype(int),
            "has_personal_loan": (df["loan"] == "yes").astype(int),
            "contact_type": df["contact"],
            "campaign_contacts": df["campaign"],
            "prior_campaign_outcome": df["poutcome"],
            "est_annual_income": est_income,
        }
    )
    out["marginal_tax_rate"] = out["est_annual_income"].map(marginal_tax_bracket)
    return out


def main() -> None:
    customers = build_customers()
    out_path = DATA_PROCESSED / "customers.csv"
    customers.to_csv(out_path, index=False)
    print(f"Wrote {len(customers):,} customer profiles -> {out_path}")


if __name__ == "__main__":
    main()
