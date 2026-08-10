from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
VISUALS_DIR = ROOT / "visuals"

EDGES_FILE = DATA_DIR / "edges.csv"
METRICS_FILE = DATA_DIR / "node_metrics.csv"
RETURNS_FILE = DATA_DIR / "log_returns.csv"

VISUALS_DIR.mkdir(exist_ok=True)


# ============================================================
# Company names
# ============================================================

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "ADBE": "Adobe Inc.",
    "ADI": "Analog Devices, Inc.",
    "ADP": "Automatic Data Processing, Inc.",
    "ADSK": "Autodesk, Inc.",
    "AEP": "American Electric Power Company, Inc.",
    "ALGN": "Align Technology, Inc.",
    "AMAT": "Applied Materials, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "AMGN": "Amgen Inc.",
    "AMZN": "Amazon.com, Inc.",
    "ANSS": "ANSYS, Inc.",
    "ASML": "ASML Holding N.V.",
    "ATVI": "Activision Blizzard, Inc.",
    "AVGO": "Broadcom Inc.",
    "BIDU": "Baidu, Inc.",
    "BIIB": "Biogen Inc.",
    "BKNG": "Booking Holdings Inc.",
    "CDNS": "Cadence Design Systems, Inc.",
    "CERN": "Cerner Corporation",
    "CHKP": "Check Point Software Technologies Ltd.",
    "CHTR": "Charter Communications, Inc.",
    "CMCSA": "Comcast Corporation",
    "COST": "Costco Wholesale Corporation",
    "CPRT": "Copart, Inc.",
    "CSCO": "Cisco Systems, Inc.",
    "CSX": "CSX Corporation",
    "CTAS": "Cintas Corporation",
    "CTSH": "Cognizant Technology Solutions Corporation",
    "DLTR": "Dollar Tree, Inc.",
    "DXCM": "DexCom, Inc.",
    "EA": "Electronic Arts Inc.",
    "EBAY": "eBay Inc.",
    "EXC": "Exelon Corporation",
    "FAST": "Fastenal Company",
    "FISV": "Fiserv, Inc.",
    "GILD": "Gilead Sciences, Inc.",
    "GOOG": "Alphabet Inc. Class C",
    "GOOGL": "Alphabet Inc. Class A",
    "HON": "Honeywell International Inc.",
    "IDXX": "IDEXX Laboratories, Inc.",
    "ILMN": "Illumina, Inc.",
    "INCY": "Incyte Corporation",
    "INTC": "Intel Corporation",
    "INTU": "Intuit Inc.",
    "ISRG": "Intuitive Surgical, Inc.",
    "KDP": "Keurig Dr Pepper Inc.",
    "KLAC": "KLA Corporation",
    "LRCX": "Lam Research Corporation",
    "LULU": "lululemon athletica inc.",
    "MAR": "Marriott International, Inc.",
    "MCHP": "Microchip Technology Incorporated",
    "MDLZ": "Mondelez International, Inc.",
    "MELI": "MercadoLibre, Inc.",
    "MNST": "Monster Beverage Corporation",
    "MRVL": "Marvell Technology, Inc.",
    "MSFT": "Microsoft Corporation",
    "MTCH": "Match Group, Inc.",
    "MU": "Micron Technology, Inc.",
    "NFLX": "Netflix, Inc.",
    "NTES": "NetEase, Inc.",
    "NVDA": "NVIDIA Corporation",
    "NXPI": "NXP Semiconductors N.V.",
    "ORLY": "O'Reilly Automotive, Inc.",
    "PAYX": "Paychex, Inc.",
    "PCAR": "PACCAR Inc",
    "PEP": "PepsiCo, Inc.",
    "QCOM": "QUALCOMM Incorporated",
    "REGN": "Regeneron Pharmaceuticals, Inc.",
    "ROST": "Ross Stores, Inc.",
    "SBUX": "Starbucks Corporation",
    "SGEN": "Seagen Inc.",
    "SIRI": "Sirius XM Holdings Inc.",
    "SNPS": "Synopsys, Inc.",
    "SWKS": "Skyworks Solutions, Inc.",
    "TCOM": "Trip.com Group Limited",
    "TMUS": "T-Mobile US, Inc.",
    "TSLA": "Tesla, Inc.",
    "TXN": "Texas Instruments Incorporated",
    "VRSK": "Verisk Analytics, Inc.",
    "VRSN": "VeriSign, Inc.",
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "WBA": "Walgreens Boots Alliance, Inc.",
    "XEL": "Xcel Energy Inc.",
    "XLNX": "Xilinx, Inc.",
}


# ============================================================
# Visual settings
# ============================================================

ROLE_COLORS = {
    "Top 10 PageRank": "#F28E2B",
    "Direct Neighbor": "#4E79A7",
    "Connected Stock": "#B07AA1",
    "Isolated Stock": "#E15759",
}

STATIC_SIZES = {
    "Top 10 PageRank": 940,
    "Direct Neighbor": 830,
    "Connected Stock": 750,
    "Isolated Stock": 670,
}

INTERACTIVE_SIZES = {
    "Top 10 PageRank": 30,
    "Direct Neighbor": 27,
    "Connected Stock": 24,
    "Isolated Stock": 21,
}

ROLE_ORDER = [
    "Top 10 PageRank",
    "Direct Neighbor",
    "Connected Stock",
    "Isolated Stock",
]


# ============================================================
# Load and prepare graph
# ============================================================

def load_data():
    edges = pd.read_csv(EDGES_FILE)
    metrics = pd.read_csv(METRICS_FILE)
    returns = pd.read_csv(RETURNS_FILE, index_col="Date")
    return edges, metrics, returns


def build_graph(edges, metrics):
    graph = nx.Graph()
    graph.add_nodes_from(metrics["ticker"])

    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.weight)

    return graph


def classify_nodes(graph, metrics):
    isolated = {n for n in graph if graph.degree(n) == 0}
    connected = set(graph) - isolated

    top = set(
        metrics.loc[metrics["ticker"].isin(connected)]
        .nlargest(10, "pagerank_weighted")["ticker"]
    )

    neighbors = set()
    for node in top:
        neighbors.update(graph.neighbors(node))
    neighbors -= top

    other = connected - top - neighbors

    role_nodes = {
        "Top 10 PageRank": top,
        "Direct Neighbor": neighbors,
        "Connected Stock": other,
        "Isolated Stock": isolated,
    }

    role_map = {
        ticker: role
        for role, nodes in role_nodes.items()
        for ticker in nodes
    }

    return connected, isolated, role_nodes, role_map


# ============================================================
# Layout
# ============================================================

def separate_nodes(
    positions,
    min_distance=0.155,
    iterations=450,
    pull_strength=0.010,
):
    """Prevent overlapping nodes while retaining spring-layout structure."""
    nodes = list(positions)

    current = {
        node: np.array(positions[node], dtype=float)
        for node in nodes
    }

    anchors = {
        node: current[node].copy()
        for node in nodes
    }

    for _ in range(iterations):
        movement = {
            node: np.zeros(2)
            for node in nodes
        }

        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                delta = current[node_a] - current[node_b]
                distance = np.linalg.norm(delta)

                if distance == 0:
                    delta = np.array([0.001, 0.001])
                    distance = np.linalg.norm(delta)

                if distance < min_distance:
                    push = (
                        delta
                        / distance
                        * (min_distance - distance)
                        * 0.58
                    )
                    movement[node_a] += push
                    movement[node_b] -= push

        for node in nodes:
            anchor_pull = (
                anchors[node] - current[node]
            ) * pull_strength

            current[node] += movement[node] + anchor_pull

    return {
        node: tuple(coords)
        for node, coords in current.items()
    }


def build_layout(graph, connected, isolated):
    core = graph.subgraph(connected).copy()

    positions = nx.spring_layout(
        core,
        seed=42,
        weight="weight",
        k=0.52,
        iterations=900,
        scale=1.0,
    )

    positions = {
        node: (x * 1.22, y * 0.95)
        for node, (x, y) in positions.items()
    }

    positions = separate_nodes(positions)

    xs = np.array([x for x, _ in positions.values()])
    ys = np.array([y for _, y in positions.values()])

    center_x = (xs.min() + xs.max()) / 2
    center_y = (ys.min() + ys.max()) / 2

    positions = {
        node: (x - center_x, y - center_y)
        for node, (x, y) in positions.items()
    }

    isolates = sorted(isolated)

    angles = np.linspace(
        0,
        2 * np.pi,
        len(isolates),
        endpoint=False,
    )

    for ticker, angle in zip(isolates, angles):
        positions[ticker] = (
            1.78 * np.cos(angle),
            1.13 * np.sin(angle),
        )

    return core, positions


def edge_widths(core):
    weights = [
        core[u][v]["weight"]
        for u, v in core.edges
    ]

    minimum = min(weights)
    maximum = max(weights)

    widths = [
        0.50
        + 1.85
        * (
            (weight - minimum)
            / (maximum - minimum)
        )
        for weight in weights
    ]

    return np.array(weights), widths


# ============================================================
# Static network PNG
# ============================================================

def create_static_network(
    graph,
    core,
    positions,
    role_nodes,
    role_map,
):
    fig = plt.figure(figsize=(18, 10))

    fig.suptitle(
        "NASDAQ Stock Correlation Network",
        fontsize=21,
        y=0.975,
    )

    fig.text(
        0.57,
        0.937,
        (
            "Structural roles based on weighted PageRank "
            "and historical Pearson correlations > 0.55"
        ),
        ha="center",
        fontsize=11,
    )

    # Legend
    ax_info = fig.add_axes(
        [0.022, 0.20, 0.18, 0.62]
    )
    ax_info.axis("off")

    ax_info.text(
        0,
        0.97,
        "Network Roles",
        fontsize=15,
        fontweight="bold",
    )

    descriptions = {
        "Top 10 PageRank": "Highest weighted PageRank",
        "Direct Neighbor": "Connected to a Top-10 stock",
        "Connected Stock": "Other stock in the core",
        "Isolated Stock": "No correlation above 0.55",
    }

    for i, role in enumerate(ROLE_ORDER):
        y = 0.82 - i * 0.19

        ax_info.scatter(
            0.06,
            y,
            s=STATIC_SIZES[role],
            color=ROLE_COLORS[role],
            edgecolor="white",
            linewidth=1.4,
        )

        ax_info.text(
            0.16,
            y + 0.026,
            role,
            fontsize=11,
            fontweight="bold",
            va="center",
        )

        ax_info.text(
            0.16,
            y - 0.030,
            descriptions[role],
            fontsize=8.5,
            va="center",
        )

    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    # Network
    ax = fig.add_axes(
        [0.19, 0.075, 0.79, 0.825]
    )
    ax.axis("off")

    _, widths = edge_widths(core)

    nx.draw_networkx_edges(
        core,
        positions,
        ax=ax,
        width=widths,
        alpha=0.31,
        edge_color="#707070",
    )

    for role in reversed(ROLE_ORDER):
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=sorted(role_nodes[role]),
            node_size=STATIC_SIZES[role],
            node_color=ROLE_COLORS[role],
            edgecolors="white",
            linewidths=1.2,
            ax=ax,
        )

    label_styles = {
        "Top 10 PageRank": (7.5, "bold"),
        "Direct Neighbor": (6.3, "normal"),
        "Connected Stock": (6.0, "normal"),
        "Isolated Stock": (5.7, "normal"),
    }

    for ticker in graph:
        x, y = positions[ticker]
        size, weight = label_styles[role_map[ticker]]

        ax.text(
            x,
            y,
            ticker,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color="white",
            zorder=10,
        )

    ax.set_xlim(-1.98, 1.98)
    ax.set_ylim(-1.28, 1.28)
    ax.set_aspect("equal", adjustable="box")

    fig.text(
        0.59,
        0.025,
        (
            "Each edge represents a historical Pearson correlation "
            "greater than 0.55. Structural centrality describes this "
            "constructed network and does not imply causal influence "
            "or investment quality."
        ),
        ha="center",
        fontsize=9,
    )

    fig.savefig(
        VISUALS_DIR / "correlation_network.png",
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Interactive network HTML
# ============================================================

def add_interactive_edges(
    figure,
    core,
    positions,
):
    weights, _ = edge_widths(core)

    low_cutoff, high_cutoff = np.quantile(
        weights,
        [0.33, 0.66],
    )

    groups = {
        "Lower": {
            "x": [],
            "y": [],
            "width": 0.8,
            "color": "rgba(110,110,110,0.18)",
        },
        "Middle": {
            "x": [],
            "y": [],
            "width": 1.2,
            "color": "rgba(100,100,100,0.25)",
        },
        "Higher": {
            "x": [],
            "y": [],
            "width": 1.8,
            "color": "rgba(90,90,90,0.34)",
        },
    }

    for source, target, data in core.edges(data=True):
        weight = data["weight"]

        if weight <= low_cutoff:
            group = groups["Lower"]
        elif weight <= high_cutoff:
            group = groups["Middle"]
        else:
            group = groups["Higher"]

        x0, y0 = positions[source]
        x1, y1 = positions[target]

        group["x"].extend([x0, x1, None])
        group["y"].extend([y0, y1, None])

    for group in groups.values():
        figure.add_trace(
            go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="lines",
                line={
                    "width": group["width"],
                    "color": group["color"],
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )


def hover_text(
    ticker,
    role,
    row,
):
    company = COMPANY_NAMES.get(
        ticker,
        "Company name unavailable",
    )

    degree = int(round(row["degree"]))
    weighted = float(row["pagerank_weighted"])
    unweighted = float(row["pagerank_unweighted"])

    status = (
        "Isolated at correlation > 0.55"
        if role == "Isolated Stock"
        else "46-stock connected core"
    )

    return (
        "<span style='font-size:18px'>"
        f"<b>{ticker}</b>"
        "</span><br>"
        "<span style='font-size:14px'>"
        f"{company}"
        "</span><br><br>"
        f"<b>Network role:</b> {role}<br>"
        f"<b>Weighted PageRank:</b> {weighted:.4f}<br>"
        f"<b>Unweighted PageRank:</b> {unweighted:.4f}<br>"
        f"<b>Degree / neighbors:</b> {degree}<br>"
        f"<b>Network status:</b> {status}"
    )


def create_interactive_network(
    core,
    positions,
    role_nodes,
    metrics,
):
    figure = go.Figure()
    lookup = metrics.set_index("ticker")

    add_interactive_edges(
        figure,
        core,
        positions,
    )

    for role in ROLE_ORDER:
        nodes = sorted(role_nodes[role])

        node_x = [
            positions[ticker][0]
            for ticker in nodes
        ]

        node_y = [
            positions[ticker][1]
            for ticker in nodes
        ]

        tooltips = [
            hover_text(
                ticker,
                role,
                lookup.loc[ticker],
            )
            for ticker in nodes
        ]

        figure.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                name=role,
                text=nodes,
                textposition="middle center",
                textfont={
                    "color": "white",
                    "size": 9,
                },
                marker={
                    "size": INTERACTIVE_SIZES[role],
                    "color": ROLE_COLORS[role],
                    "line": {
                        "color": "white",
                        "width": 1.2,
                    },
                },
                hovertext=tooltips,
                hoverinfo="text",
            )
        )

    figure.update_layout(
        title={
            "text": (
                "<b>NASDAQ Stock Correlation Network</b><br>"
                "<sup>"
                "Structural roles based on weighted PageRank "
                "and historical Pearson correlations > 0.55"
                "</sup>"
            ),
            "x": 0.58,
            "xanchor": "center",
        },
        width=1500,
        height=850,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={
            "l": 210,
            "r": 40,
            "t": 100,
            "b": 80,
        },
        legend={
            "title": {
                "text": "<b>Network Roles</b>"
            },
            "x": -0.14,
            "y": 0.95,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "borderwidth": 0,
            "font": {
                "size": 12
            },
        },
        xaxis={
            "visible": False,
            "range": [-2.02, 2.02],
            "fixedrange": False,
        },
        yaxis={
            "visible": False,
            "range": [-1.30, 1.30],
            "scaleanchor": "x",
            "scaleratio": 1,
            "fixedrange": False,
        },
        hoverlabel={
            "bgcolor": "white",
            "font_size": 13,
            "font_family": "Arial",
        },
        annotations=[
            {
                "text": (
                    "Hover over any stock for PageRank, degree, "
                    "role, and network details. "
                    "Scroll to zoom and drag to pan."
                ),
                "x": 0.58,
                "y": -0.055,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 11,
                    "color": "#555555",
                },
            }
        ],
    )

    figure.write_html(
        VISUALS_DIR
        / "correlation_network_interactive.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
        },
    )


# ============================================================
# Degree vs. PageRank
# ============================================================

def create_degree_chart(metrics):
    correlation = (
        metrics[
            ["degree", "pagerank_weighted"]
        ]
        .corr()
        .iloc[0, 1]
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    x = metrics["degree"].to_numpy()
    y = metrics["pagerank_weighted"].to_numpy()

    ax.scatter(x, y, alpha=0.72, s=55)

    slope, intercept = np.polyfit(x, y, 1)
    trend_x = np.linspace(x.min(), x.max(), 100)

    ax.plot(
        trend_x,
        slope * trend_x + intercept,
        linestyle="--",
        linewidth=1.2,
        alpha=0.55,
    )

    offsets = {
        "TXN": (7, 2),
        "ADI": (7, 5),
        "MSFT": (7, 5),
        "HON": (7, 5),
        "PAYX": (7, 5),
    }

    lookup = metrics.set_index("ticker")

    for ticker, offset in offsets.items():
        row = lookup.loc[ticker]

        ax.annotate(
            ticker,
            (
                row["degree"],
                row["pagerank_weighted"],
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )

    fig.suptitle(
        "PageRank Closely Tracks Network Connectivity",
        fontsize=15,
        y=0.97,
    )

    ax.set_title(
        (
            "Pearson correlation across "
            f"85 stocks = {correlation:.3f}"
        ),
        fontsize=10,
        pad=10,
    )

    ax.set_xlabel("Degree Centrality")
    ax.set_ylabel("Weighted PageRank")
    ax.grid(alpha=0.18)

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    fig.savefig(
        VISUALS_DIR / "degree_vs_pagerank.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Threshold sensitivity
# ============================================================

def threshold_data(returns):
    correlations = returns.corr()
    tickers = list(correlations.columns)

    results = []

    for threshold in [0.50, 0.55, 0.60, 0.65]:
        edge_count = 0
        connected = set()

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                if correlations.iloc[i, j] > threshold:
                    edge_count += 1
                    connected.add(tickers[i])
                    connected.add(tickers[j])

        results.append({
            "threshold": threshold,
            "edges": edge_count,
            "isolates": len(tickers) - len(connected),
        })

    return pd.DataFrame(results)


def create_threshold_chart(returns):
    sensitivity = threshold_data(returns)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    edge_line = ax1.plot(
        sensitivity["threshold"],
        sensitivity["edges"],
        marker="o",
        linewidth=2.2,
        color="#4E79A7",
        label="Edges",
    )[0]

    ax1.set_xlabel("Correlation Threshold")
    ax1.set_ylabel(
        "Number of Edges",
        color="#4E79A7",
    )
    ax1.tick_params(
        axis="y",
        labelcolor="#4E79A7",
    )

    ax2 = ax1.twinx()

    isolate_line = ax2.plot(
        sensitivity["threshold"],
        sensitivity["isolates"],
        marker="s",
        linestyle="--",
        linewidth=2.2,
        color="#E15759",
        label="Isolated Stocks",
    )[0]

    ax2.set_ylabel(
        "Isolated Stocks",
        color="#E15759",
    )
    ax2.tick_params(
        axis="y",
        labelcolor="#E15759",
    )

    ax1.axvline(
        0.55,
        linestyle=":",
        linewidth=1.4,
        color="#666666",
        alpha=0.8,
    )

    ax1.text(
        0.552,
        415,
        "Selected threshold",
        rotation=90,
        fontsize=8,
        color="#555555",
        va="top",
    )

    for row in sensitivity.itertuples(index=False):
        ax1.annotate(
            str(row.edges),
            (row.threshold, row.edges),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

        ax2.annotate(
            str(row.isolates),
            (row.threshold, row.isolates),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax1.legend(
        [edge_line, isolate_line],
        ["Edges", "Isolated Stocks"],
        loc="center right",
    )

    ax1.set_title(
        (
            "Network Structure Is Sensitive "
            "to the Correlation Threshold"
        ),
        fontsize=14,
        pad=12,
    )

    ax1.grid(alpha=0.18)
    fig.tight_layout()

    fig.savefig(
        VISUALS_DIR / "threshold_sensitivity.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    edges, metrics, returns = load_data()
    graph = build_graph(edges, metrics)

    (
        connected,
        isolated,
        role_nodes,
        role_map,
    ) = classify_nodes(graph, metrics)

    core, positions = build_layout(
        graph,
        connected,
        isolated,
    )

    create_static_network(
        graph,
        core,
        positions,
        role_nodes,
        role_map,
    )

    create_interactive_network(
        core,
        positions,
        role_nodes,
        metrics,
    )

    create_degree_chart(metrics)
    create_threshold_chart(returns)

    print("\nCreated portfolio visuals:")
    print(" - visuals/correlation_network.png")
    print(" - visuals/correlation_network_interactive.html")
    print(" - visuals/degree_vs_pagerank.png")
    print(" - visuals/threshold_sensitivity.png")


if __name__ == "__main__":
    main()