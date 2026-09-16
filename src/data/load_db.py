"""Load processed CSVs into a local SQLite database for SQL-based feature work."""
import sqlite3

import pandas as pd

from src.config import DATA_PROCESSED, DB_PATH

TABLES = {
    "customers": "customers_sample.csv",
    "accounts": "accounts.csv",
    "transactions": "transactions.csv",
    "holdings": "holdings.csv",
}


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        for table, csv_name in TABLES.items():
            df = pd.read_csv(DATA_PROCESSED / csv_name)
            df.to_sql(table, conn, if_exists="replace", index=False)
            conn.execute(f"CREATE INDEX idx_{table}_customer_id ON {table}(customer_id)")
            print(f"Loaded {table:<12} {len(df):>8,} rows")
        conn.commit()
    finally:
        conn.close()
    print(f"SQLite DB ready -> {DB_PATH}")


if __name__ == "__main__":
    main()
