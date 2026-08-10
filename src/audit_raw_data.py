from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "NASDAQ_100_Data_From_2010.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ticker_coverage.csv"


# The source file uses tabs even though its extension is .csv
df = pd.read_csv(RAW_FILE, sep="\t")
df["Date"] = pd.to_datetime(df["Date"])


print("\n=== RAW DATASET AUDIT ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Unique tickers: {df['Name'].nunique()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

print("\nColumns:")
for column in df.columns:
    print(f" - {column}")


duplicate_count = df.duplicated(subset=["Date", "Name"]).sum()
print(f"\nDuplicate Date + Name rows: {duplicate_count:,}")

missing_adj_close = df["Adj Close"].isna().sum()
print(f"Missing Adj Close values: {missing_adj_close:,}")


coverage = (
    df.groupby("Name")
    .agg(
        row_count=("Date", "size"),
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        adj_close_non_null=("Adj Close", "count"),
    )
    .reset_index()
    .sort_values(["row_count", "Name"], ascending=[False, True])
)

coverage["first_date"] = coverage["first_date"].dt.date
coverage["last_date"] = coverage["last_date"].dt.date


print(f"\nTicker coverage saved to:")
print(OUTPUT_FILE)

print("\nTop 10 tickers by observation count:")
print(coverage.head(10).to_string(index=False))

print("\nBottom 20 tickers by observation count:")
print(coverage.tail(20).to_string(index=False))

total_trading_dates = df["Date"].nunique()

coverage["coverage_pct"] = (
    coverage["row_count"] / total_trading_dates * 100
)

coverage.to_csv(OUTPUT_FILE, index=False)

print("\n=== COVERAGE SUMMARY ===")
print(f"Total unique trading dates: {total_trading_dates:,}")

print("\nBottom 25 tickers by historical coverage:")
print(
    coverage
    .sort_values("coverage_pct")
    .head(25)
    [["Name", "row_count", "first_date", "coverage_pct"]]
    .to_string(index=False)
)