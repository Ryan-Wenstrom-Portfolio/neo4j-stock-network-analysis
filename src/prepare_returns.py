from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Project paths and parameters
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "NASDAQ_100_Data_From_2010.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "log_returns.csv"
)

MIN_COVERAGE = 0.90


# -----------------------------
# 1. Load raw price data
# -----------------------------

df = pd.read_csv(RAW_FILE, sep="\t")
df["Date"] = pd.to_datetime(df["Date"])

print("\n=== PREPARING STOCK RETURN DATA ===")
print(f"Raw rows: {len(df):,}")
print(f"Raw tickers: {df['Name'].nunique()}")


# -----------------------------
# 2. Measure historical coverage
# -----------------------------

total_trading_dates = df["Date"].nunique()

ticker_coverage = (
    df.groupby("Name")["Date"].nunique()
    / total_trading_dates
)

selected_tickers = ticker_coverage[
    ticker_coverage >= MIN_COVERAGE
].index.tolist()

excluded_tickers = ticker_coverage[
    ticker_coverage < MIN_COVERAGE
].index.tolist()


print(f"\nMinimum coverage requirement: {MIN_COVERAGE:.0%}")
print(f"Selected tickers: {len(selected_tickers)}")
print(f"Excluded tickers: {len(excluded_tickers)}")

print("\nExcluded tickers:")
print(", ".join(sorted(excluded_tickers)))


# -----------------------------
# 3. Keep stocks with sufficient history
# -----------------------------

filtered = df[
    df["Name"].isin(selected_tickers)
].copy()

filtered = filtered.sort_values(
    ["Name", "Date"]
)


# -----------------------------
# 4. Calculate daily log returns
# -----------------------------

filtered["log_return"] = (
    filtered
    .groupby("Name")["Adj Close"]
    .transform(
        lambda prices: np.log(
            prices / prices.shift(1)
        )
    )
)


# -----------------------------
# 5. Convert to date x ticker matrix
# -----------------------------

return_matrix = (
    filtered
    .pivot(
        index="Date",
        columns="Name",
        values="log_return",
    )
    .sort_index()
)


# -----------------------------
# 6. Keep dates shared by all selected stocks
# -----------------------------

clean_returns = return_matrix.dropna(
    axis=0,
    how="any",
)


# -----------------------------
# 7. Validate final dataset
# -----------------------------

print("\n=== CLEAN RETURN MATRIX ===")

print(
    f"Rows / trading dates: "
    f"{clean_returns.shape[0]:,}"
)

print(
    f"Stock columns: "
    f"{clean_returns.shape[1]}"
)

print(
    f"Date range: "
    f"{clean_returns.index.min().date()} "
    f"to "
    f"{clean_returns.index.max().date()}"
)

print(
    f"Missing values: "
    f"{clean_returns.isna().sum().sum():,}"
)

print(
    f"Minimum selected ticker coverage: "
    f"{ticker_coverage[selected_tickers].min():.2%}"
)


# -----------------------------
# 8. Save processed dataset
# -----------------------------

clean_returns.to_csv(
    OUTPUT_FILE,
    index_label="Date",
)

print("\nSaved cleaned log-return matrix to:")
print(OUTPUT_FILE)