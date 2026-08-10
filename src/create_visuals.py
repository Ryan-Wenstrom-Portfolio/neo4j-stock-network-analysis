from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EDGES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edges.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "node_metrics.csv"
)

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "log_returns.csv"
)

VISUALS_DIR = PROJECT_ROOT / "visuals"

VISUALS_DIR.mkdir(exist_ok=True)


# ============================================================
# Load data
# ============================================================

edges = pd.read_csv(EDGES_FILE)

metrics = pd.read_csv(METRICS_FILE)

returns = pd.read_csv(
    RETURNS_FILE,
    index_col="Date",
)

metric_lookup = metrics.set_index("ticker")

# ============================================================
# Dataset-period company names
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
# Helper: separate overlapping connected nodes
# ============================================================

def separate_nodes(
    positions,
    min_distance=0.155,
    iterations=450,
    pull_strength=0.010,
):
    """
    Preserve the relationship-driven spring layout while
    enforcing a minimum distance between node centers.
    """

    nodes = list(positions.keys())

    current = {
        node: np.array(
            positions[node],
            dtype=float,
        )
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

        for i in range(len(nodes)):

            for j in range(i + 1, len(nodes)):

                node_a = nodes[i]
                node_b = nodes[j]

                delta = (
                    current[node_a]
                    - current[node_b]
                )

                distance = np.linalg.norm(delta)

                if distance == 0:

                    delta = np.array(
                        [0.001, 0.001]
                    )

                    distance = np.linalg.norm(
                        delta
                    )

                if distance < min_distance:

                    direction = (
                        delta / distance
                    )

                    overlap = (
                        min_distance
                        - distance
                    )

                    push = (
                        direction
                        * overlap
                        * 0.58
                    )

                    movement[node_a] += push
                    movement[node_b] -= push

        for node in nodes:

            anchor_pull = (
                anchors[node]
                - current[node]
            ) * pull_strength

            current[node] += (
                movement[node]
                + anchor_pull
            )

    return {
        node: tuple(current[node])
        for node in nodes
    }


# ============================================================
# Build graph
# ============================================================

graph = nx.Graph()

graph.add_nodes_from(
    metrics["ticker"]
)

for row in edges.itertuples(index=False):

    graph.add_edge(
        row.source,
        row.target,
        weight=row.weight,
    )


# ============================================================
# Structural roles
# ============================================================

isolated_stocks = {
    node
    for node in graph.nodes
    if graph.degree(node) == 0
}


connected_stocks = {
    node
    for node in graph.nodes
    if graph.degree(node) > 0
}


top_pagerank = set(
    metrics.loc[
        metrics["ticker"].isin(
            connected_stocks
        )
    ]
    .nlargest(
        10,
        "pagerank_weighted",
    )["ticker"]
)


top_neighbors = set()

for node in top_pagerank:

    top_neighbors.update(
        graph.neighbors(node)
    )


top_neighbors -= top_pagerank


remaining_connected = (
    connected_stocks
    - top_pagerank
    - top_neighbors
)


# ============================================================
# Role metadata
# ============================================================

ROLE_COLORS = {
    "Top 10 PageRank": "#F28E2B",
    "Direct Neighbor": "#4E79A7",
    "Connected Stock": "#B07AA1",
    "Isolated Stock": "#E15759",
}


# Static PNG node sizes.
# Orange > Blue > Purple > Red.

STATIC_NODE_SIZES = {
    "Top 10 PageRank": 940,
    "Direct Neighbor": 830,
    "Connected Stock": 750,
    "Isolated Stock": 670,
}


# Interactive HTML node sizes in pixels.

INTERACTIVE_NODE_SIZES = {
    "Top 10 PageRank": 30,
    "Direct Neighbor": 27,
    "Connected Stock": 24,
    "Isolated Stock": 21,
}


def get_role(ticker):

    if ticker in top_pagerank:
        return "Top 10 PageRank"

    if ticker in top_neighbors:
        return "Direct Neighbor"

    if ticker in remaining_connected:
        return "Connected Stock"

    return "Isolated Stock"


# ============================================================
# Network layout
# ============================================================

core = graph.subgraph(
    connected_stocks
).copy()


# ------------------------------------------------------------
# Initial relationship-driven force layout
# ------------------------------------------------------------

initial_core_position = nx.spring_layout(
    core,
    seed=42,
    weight="weight",
    k=0.52,
    iterations=900,
    scale=1.0,
)


# Expand connected core before collision separation.

for node in initial_core_position:

    x, y = initial_core_position[node]

    initial_core_position[node] = (
        x * 1.22,
        y * 0.95,
    )


# ------------------------------------------------------------
# Collision separation
# ------------------------------------------------------------

core_position = separate_nodes(
    initial_core_position,
    min_distance=0.155,
    iterations=450,
    pull_strength=0.010,
)


# ------------------------------------------------------------
# Recenter connected component
# ------------------------------------------------------------

core_x = np.array(
    [
        coordinates[0]
        for coordinates in core_position.values()
    ]
)

core_y = np.array(
    [
        coordinates[1]
        for coordinates in core_position.values()
    ]
)


center_x = (
    core_x.min()
    + core_x.max()
) / 2

center_y = (
    core_y.min()
    + core_y.max()
) / 2


for node in core_position:

    x, y = core_position[node]

    core_position[node] = (
        x - center_x,
        y - center_y,
    )


# ============================================================
# Add isolate ring
# ============================================================

position = dict(
    core_position
)


sorted_isolates = sorted(
    isolated_stocks
)


angles = np.linspace(
    0,
    2 * np.pi,
    len(sorted_isolates),
    endpoint=False,
)


# Larger ring preserves space after increasing node sizes.

for ticker, angle in zip(
    sorted_isolates,
    angles,
):

    position[ticker] = (
        1.78 * np.cos(angle),
        1.13 * np.sin(angle),
    )


# ============================================================
# 1. STATIC NETWORK — PNG
# ============================================================

fig = plt.figure(
    figsize=(18, 10)
)


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


# ------------------------------------------------------------
# Network Roles panel
# ------------------------------------------------------------

ax_info = fig.add_axes(
    [
        0.022,
        0.20,
        0.18,
        0.62,
    ]
)

ax_info.axis("off")


ax_info.text(
    0,
    0.97,
    "Network Roles",
    fontsize=15,
    fontweight="bold",
)


legend_items = [
    (
        "Top 10 PageRank",
        "Highest weighted PageRank",
    ),
    (
        "Direct Neighbor",
        "Connected to a Top-10 stock",
    ),
    (
        "Connected Stock",
        "Other stock in the core",
    ),
    (
        "Isolated Stock",
        "No correlation above 0.55",
    ),
]


for i, (
    role,
    description,
) in enumerate(legend_items):

    y = 0.82 - i * 0.19

    ax_info.scatter(
        0.06,
        y,
        s=STATIC_NODE_SIZES[role],
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
        description,
        fontsize=8.5,
        va="center",
    )


ax_info.set_xlim(
    0,
    1,
)

ax_info.set_ylim(
    0,
    1,
)


# ------------------------------------------------------------
# Main network panel
# ------------------------------------------------------------

ax_network = fig.add_axes(
    [
        0.19,
        0.075,
        0.79,
        0.825,
    ]
)

ax_network.axis("off")


# ------------------------------------------------------------
# Relationship widths
# ------------------------------------------------------------

edge_weights = [
    core[u][v]["weight"]
    for u, v in core.edges
]


minimum_weight = min(
    edge_weights
)

maximum_weight = max(
    edge_weights
)


edge_widths = [
    0.50
    + 1.85
    * (
        (weight - minimum_weight)
        /
        (maximum_weight - minimum_weight)
    )
    for weight in edge_weights
]


nx.draw_networkx_edges(
    core,
    position,
    ax=ax_network,
    width=edge_widths,
    alpha=0.31,
    edge_color="#707070",
)


# ------------------------------------------------------------
# Draw each node category
# ------------------------------------------------------------

role_node_sets = {
    "Top 10 PageRank": top_pagerank,
    "Direct Neighbor": top_neighbors,
    "Connected Stock": remaining_connected,
    "Isolated Stock": isolated_stocks,
}


# Draw smaller categories first so larger central nodes
# visually remain on top.

draw_order = [
    "Isolated Stock",
    "Connected Stock",
    "Direct Neighbor",
    "Top 10 PageRank",
]


for role in draw_order:

    nx.draw_networkx_nodes(
        graph,
        position,
        nodelist=sorted(
            role_node_sets[role]
        ),
        node_size=STATIC_NODE_SIZES[
            role
        ],
        node_color=ROLE_COLORS[
            role
        ],
        edgecolors="white",
        linewidths=1.2,
        ax=ax_network,
    )


# ------------------------------------------------------------
# Label every stock ticker
# ------------------------------------------------------------

for ticker in graph.nodes:

    x, y = position[ticker]

    role = get_role(
        ticker
    )

    if role == "Top 10 PageRank":

        font_size = 7.5
        font_weight = "bold"

    elif role == "Direct Neighbor":

        font_size = 6.3
        font_weight = "normal"

    elif role == "Connected Stock":

        font_size = 6.0
        font_weight = "normal"

    else:

        font_size = 5.7
        font_weight = "normal"

    ax_network.text(
        x,
        y,
        ticker,
        ha="center",
        va="center",
        fontsize=font_size,
        fontweight=font_weight,
        color="white",
        zorder=10,
    )


# ------------------------------------------------------------
# Plot bounds
# ------------------------------------------------------------

ax_network.set_xlim(
    -1.98,
    1.98,
)

ax_network.set_ylim(
    -1.28,
    1.28,
)

ax_network.set_aspect(
    "equal",
    adjustable="box",
)


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


plt.savefig(
    VISUALS_DIR
    / "correlation_network.png",
    dpi=260,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 2. INTERACTIVE NETWORK — HTML
# ============================================================

interactive_figure = go.Figure()


# ------------------------------------------------------------
# Interactive edges
# ------------------------------------------------------------

# Split relationships into three strength groups so stronger
# correlations remain visually more prominent.

edge_weight_array = np.array(
    edge_weights
)

low_cutoff, high_cutoff = np.quantile(
    edge_weight_array,
    [
        0.33,
        0.66,
    ],
)


edge_groups = {
    "Lower": {
        "x": [],
        "y": [],
        "width": 0.8,
        "color": (
            "rgba(110,110,110,0.18)"
        ),
    },
    "Middle": {
        "x": [],
        "y": [],
        "width": 1.2,
        "color": (
            "rgba(100,100,100,0.25)"
        ),
    },
    "Higher": {
        "x": [],
        "y": [],
        "width": 1.8,
        "color": (
            "rgba(90,90,90,0.34)"
        ),
    },
}


for source, target, data in core.edges(
    data=True
):

    weight = data["weight"]

    if weight <= low_cutoff:
        group = "Lower"

    elif weight <= high_cutoff:
        group = "Middle"

    else:
        group = "Higher"

    x0, y0 = position[
        source
    ]

    x1, y1 = position[
        target
    ]

    edge_groups[group]["x"].extend(
        [
            x0,
            x1,
            None,
        ]
    )

    edge_groups[group]["y"].extend(
        [
            y0,
            y1,
            None,
        ]
    )


for group in edge_groups.values():

    interactive_figure.add_trace(
        go.Scatter(
            x=group["x"],
            y=group["y"],
            mode="lines",
            line=dict(
                width=group[
                    "width"
                ],
                color=group[
                    "color"
                ],
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )


# ------------------------------------------------------------
# Interactive nodes
# ------------------------------------------------------------

for role in [
    "Top 10 PageRank",
    "Direct Neighbor",
    "Connected Stock",
    "Isolated Stock",
]:

    nodes = sorted(
        role_node_sets[role]
    )

    node_x = []
    node_y = []
    node_text = []
    hover_text = []


    for ticker in nodes:

        x, y = position[
            ticker
        ]

        node_x.append(
            x
        )

        node_y.append(
            y
        )

        node_text.append(
            ticker
        )


        row = metric_lookup.loc[
            ticker
        ]


        degree = int(
            round(
                row["degree"]
            )
        )


        weighted_pr = float(
            row[
                "pagerank_weighted"
            ]
        )


        unweighted_pr = float(
            row[
                "pagerank_unweighted"
            ]
        )


        component_id = int(
            round(
                row[
                    "component_id"
                ]
            )
        )


        # Company name will appear automatically if a
        # verified company_name column is later added.

                # ----------------------------------------------------
        # Recruiter-friendly hover information
        # ----------------------------------------------------

        company_name = COMPANY_NAMES.get(
            ticker,
            "Company name unavailable",
        )


        if ticker in isolated_stocks:

            network_status = (
                "Isolated at correlation > 0.55"
            )

        else:

            network_status = (
                "46-stock connected core"
            )


        hover_text.append(
            f"<span style='font-size:18px'>"
            f"<b>{ticker}</b>"
            f"</span>"
            f"<br>"
            f"<span style='font-size:14px'>"
            f"{company_name}"
            f"</span>"
            f"<br><br>"
            f"<b>Network role:</b> {role}"
            f"<br>"
            f"<b>Weighted PageRank:</b> "
            f"{weighted_pr:.4f}"
            f"<br>"
            f"<b>Unweighted PageRank:</b> "
            f"{unweighted_pr:.4f}"
            f"<br>"
            f"<b>Degree / neighbors:</b> "
            f"{degree}"
            f"<br>"
            f"<b>Network status:</b> "
            f"{network_status}"
        )


    interactive_figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            name=role,
            text=node_text,
            textposition="middle center",
            textfont=dict(
                color="white",
                size=9,
            ),
            marker=dict(
                size=INTERACTIVE_NODE_SIZES[
                    role
                ],
                color=ROLE_COLORS[
                    role
                ],
                line=dict(
                    color="white",
                    width=1.2,
                ),
            ),
            hovertext=hover_text,
            hoverinfo="text",
        )
    )


# ------------------------------------------------------------
# Interactive chart design
# ------------------------------------------------------------

interactive_figure.update_layout(

    title=dict(
        text=(
            "<b>NASDAQ Stock Correlation Network</b>"
            "<br>"
            "<sup>"
            "Structural roles based on weighted PageRank "
            "and historical Pearson correlations > 0.55"
            "</sup>"
        ),
        x=0.58,
        xanchor="center",
    ),

    width=1500,
    height=850,

    paper_bgcolor="white",
    plot_bgcolor="white",

    margin=dict(
        l=210,
        r=40,
        t=100,
        b=80,
    ),

    legend=dict(
        title=dict(
            text="<b>Network Roles</b>"
        ),
        x=-0.14,
        y=0.95,
        xanchor="left",
        yanchor="top",
        bgcolor=(
            "rgba(255,255,255,0.85)"
        ),
        borderwidth=0,
        font=dict(
            size=12
        ),
    ),

    xaxis=dict(
        visible=False,
        range=[
            -2.02,
            2.02,
        ],
        fixedrange=False,
    ),

    yaxis=dict(
        visible=False,
        range=[
            -1.30,
            1.30,
        ],
        scaleanchor="x",
        scaleratio=1,
        fixedrange=False,
    ),

    hoverlabel=dict(
        bgcolor="white",
        font_size=13,
        font_family="Arial",
    ),

    annotations=[
        dict(
            text=(
                "Hover over any stock for PageRank, "
                "degree, role, and component details. "
                "Scroll to zoom and drag to pan."
            ),
            x=0.58,
            y=-0.055,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=11,
                color="#555555",
            ),
        )
    ],
)


interactive_figure.write_html(
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
# 3. DEGREE VS. WEIGHTED PAGERANK
# ============================================================

correlation = (
    metrics[
        [
            "degree",
            "pagerank_weighted",
        ]
    ]
    .corr()
    .iloc[0, 1]
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.scatter(
    metrics["degree"],
    metrics["pagerank_weighted"],
    alpha=0.72,
    s=55,
)


# ------------------------------------------------------------
# Linear reference trend
# ------------------------------------------------------------

x_values = metrics[
    "degree"
].to_numpy()

y_values = metrics[
    "pagerank_weighted"
].to_numpy()


slope, intercept = np.polyfit(
    x_values,
    y_values,
    1,
)


trend_x = np.linspace(
    x_values.min(),
    x_values.max(),
    100,
)


trend_y = (
    slope * trend_x
    + intercept
)


ax.plot(
    trend_x,
    trend_y,
    linestyle="--",
    linewidth=1.2,
    alpha=0.55,
)


# ------------------------------------------------------------
# Selected annotations
# ------------------------------------------------------------

annotation_offsets = {
    "TXN": (7, 2),
    "ADI": (7, 5),
    "MSFT": (7, 5),
    "HON": (7, 5),
    "PAYX": (7, 5),
}


for ticker, offset in annotation_offsets.items():

    row = metrics.loc[
        metrics["ticker"] == ticker
    ].iloc[0]

    ax.annotate(
        ticker,
        (
            row["degree"],
            row[
                "pagerank_weighted"
            ],
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


ax.set_xlabel(
    "Degree Centrality"
)

ax.set_ylabel(
    "Weighted PageRank"
)

ax.grid(
    alpha=0.18
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)


plt.savefig(
    VISUALS_DIR
    / "degree_vs_pagerank.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 4. CORRELATION THRESHOLD SENSITIVITY
# ============================================================

correlations = returns.corr()


thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
]


results = []

tickers = list(
    correlations.columns
)


for threshold in thresholds:

    threshold_edges = []

    for i in range(
        len(tickers)
    ):

        for j in range(
            i + 1,
            len(tickers),
        ):

            if (
                correlations.iloc[
                    i,
                    j,
                ]
                > threshold
            ):

                threshold_edges.append(
                    (
                        tickers[i],
                        tickers[j],
                    )
                )


    connected = set()

    for source, target in threshold_edges:

        connected.add(
            source
        )

        connected.add(
            target
        )


    isolates = (
        len(tickers)
        - len(connected)
    )


    results.append(
        {
            "threshold": threshold,
            "edges": len(
                threshold_edges
            ),
            "isolates": isolates,
        }
    )


sensitivity = pd.DataFrame(
    results
)


fig, ax1 = plt.subplots(
    figsize=(9, 6)
)


edge_line = ax1.plot(
    sensitivity[
        "threshold"
    ],
    sensitivity[
        "edges"
    ],
    marker="o",
    linewidth=2.2,
    color="#4E79A7",
    label="Edges",
)[0]


ax1.set_xlabel(
    "Correlation Threshold"
)


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
    sensitivity[
        "threshold"
    ],
    sensitivity[
        "isolates"
    ],
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


# ------------------------------------------------------------
# Highlight selected threshold
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Value labels
# ------------------------------------------------------------

for row in sensitivity.itertuples(
    index=False
):

    ax1.annotate(
        str(row.edges),
        (
            row.threshold,
            row.edges,
        ),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

    ax2.annotate(
        str(row.isolates),
        (
            row.threshold,
            row.isolates,
        ),
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )


ax1.legend(
    [
        edge_line,
        isolate_line,
    ],
    [
        "Edges",
        "Isolated Stocks",
    ],
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


ax1.grid(
    alpha=0.18
)


plt.tight_layout()


plt.savefig(
    VISUALS_DIR
    / "threshold_sensitivity.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Finished
# ============================================================

print(
    "\nCreated portfolio visuals:"
)

print(
    " - visuals/correlation_network.png"
)

print(
    " - visuals/correlation_network_interactive.html"
)

print(
    " - visuals/degree_vs_pagerank.png"
)

print(
    " - visuals/threshold_sensitivity.png"
)