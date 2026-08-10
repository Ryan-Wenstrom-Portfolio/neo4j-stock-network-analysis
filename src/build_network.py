from pathlib import Path

import pandas as pd


# -----------------------------
# Project paths and parameters
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "log_returns.csv"
)

STOCKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stocks.csv"
)

EDGES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edges.csv"
)

CORRELATION_THRESHOLD = 0.55


# -----------------------------
# 1. Load cleaned return matrix
# -----------------------------

returns = pd.read_csv(
    RETURNS_FILE,
    index_col="Date",
)

print("\n=== BUILDING STOCK CORRELATION NETWORK ===")
print(f"Trading dates: {len(returns):,}")
print(f"Stocks: {len(returns.columns)}")


# -----------------------------
# 2. Calculate Pearson correlations
# -----------------------------

correlations = returns.corr(method="pearson")


# -----------------------------
# 3. Create node table
# -----------------------------

stocks = pd.DataFrame(
    {
        "ticker": sorted(returns.columns)
    }
)


# -----------------------------
# 4. Create edge table
# -----------------------------

edges = []

tickers = list(correlations.columns)

for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):

        stock_1 = tickers[i]
        stock_2 = tickers[j]

        correlation = correlations.loc[
            stock_1,
            stock_2,
        ]

        if correlation > CORRELATION_THRESHOLD:
            edges.append(
                {
                    "source": stock_1,
                    "target": stock_2,
                    "weight": correlation,
                }
            )


edges = pd.DataFrame(
    edges,
    columns=["source", "target", "weight"],
)

edges = edges.sort_values(
    ["source", "target"]
).reset_index(drop=True)


# -----------------------------
# 5. Basic graph diagnostics
# -----------------------------

degree_counts = pd.concat(
    [
        edges["source"],
        edges["target"],
    ]
).value_counts()

isolates = [
    ticker
    for ticker in stocks["ticker"]
    if ticker not in degree_counts.index
]

node_count = len(stocks)
edge_count = len(edges)

possible_edges = (
    node_count * (node_count - 1) / 2
)

density = (
    edge_count / possible_edges
    if possible_edges
    else 0
)

average_degree = (
    (2 * edge_count) / node_count
    if node_count
    else 0
)


# -----------------------------
# 6. Save Neo4j input files
# -----------------------------

stocks.to_csv(
    STOCKS_FILE,
    index=False,
)

edges.to_csv(
    EDGES_FILE,
    index=False,
)


# -----------------------------
# 7. Report results
# -----------------------------

print(
    f"\nCorrelation threshold: "
    f">{CORRELATION_THRESHOLD}"
)

print(f"Nodes: {node_count}")
print(f"Edges: {edge_count}")
print(f"Isolated stocks: {len(isolates)}")
print(f"Graph density: {density:.4%}")
print(f"Average degree: {average_degree:.2f}")

print("\nIsolated stocks:")
print(", ".join(sorted(isolates)))

print("\nSaved:")
print(STOCKS_FILE)
print(EDGES_FILE)