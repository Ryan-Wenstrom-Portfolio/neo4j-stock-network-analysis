// ============================================================
// Stock Network Analytics with Neo4j
// ============================================================
// Graph model:
//   (:Stock {ticker})
//   (:Stock)-[:SIMILAR_TO {weight}]->(:Stock)
//
// SIMILAR_TO relationships represent historical Pearson
// correlations greater than 0.55. Although stored once in Neo4j,
// correlations are symmetric, so GDS projects them as undirected.
// ============================================================


// ------------------------------------------------------------
// 1. Schema
// ------------------------------------------------------------

CREATE CONSTRAINT stock_ticker_unique IF NOT EXISTS
FOR (s:Stock)
REQUIRE s.ticker IS UNIQUE;


// ------------------------------------------------------------
// 2. Load stock nodes
// ------------------------------------------------------------

LOAD CSV WITH HEADERS FROM 'file:///stocks.csv' AS row
MERGE (:Stock {ticker: row.ticker});


// ------------------------------------------------------------
// 3. Load correlation relationships
// ------------------------------------------------------------

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row

MATCH (source:Stock {ticker: row.source})
MATCH (target:Stock {ticker: row.target})

MERGE (source)-[r:SIMILAR_TO]->(target)

SET r.weight = toFloat(row.weight);


// ------------------------------------------------------------
// 4. Validate the stored Neo4j graph
// ------------------------------------------------------------

MATCH (s:Stock)
WITH count(s) AS nodes

MATCH ()-[r:SIMILAR_TO]->()

RETURN
    nodes,
    count(r) AS relationships;


// Validate relationship weights.

MATCH ()-[r:SIMILAR_TO]->()

RETURN
    min(r.weight) AS minimum_weight,
    max(r.weight) AS maximum_weight,

    sum(
        CASE
            WHEN r.weight IS NULL
            THEN 1
            ELSE 0
        END
    ) AS missing_weights,

    sum(
        CASE
            WHEN r.weight <= 0.55 OR r.weight > 1
            THEN 1
            ELSE 0
        END
    ) AS invalid_weights;


// Count stocks with no correlation relationships.

MATCH (s:Stock)
WHERE NOT (s)--()

RETURN count(s) AS isolated_stocks;


// ------------------------------------------------------------
// 5. Create the GDS in-memory graph
// ------------------------------------------------------------

// Remove an existing projection so this analysis can be rerun.

CALL gds.graph.drop(
    'stockGraph',
    false
);


// Project correlations as undirected and retain their weights.

CALL gds.graph.project(
    'stockGraph',
    'Stock',
    {
        SIMILAR_TO: {
            orientation: 'UNDIRECTED',
            properties: 'weight'
        }
    }
)

YIELD
    graphName,
    nodeCount,
    relationshipCount

RETURN
    graphName,
    nodeCount,
    relationshipCount;


// ------------------------------------------------------------
// 6. Connected-component structure
// ------------------------------------------------------------

CALL gds.wcc.stream('stockGraph')

YIELD
    nodeId,
    componentId

WITH
    componentId,
    count(*) AS component_size

WITH collect(component_size) AS sizes

RETURN
    size(sizes) AS component_count,

    reduce(
        max_size = 0,
        size IN sizes |
        CASE
            WHEN size > max_size
            THEN size
            ELSE max_size
        END
    ) AS largest_component;


// ------------------------------------------------------------
// 7. Degree centrality
// ------------------------------------------------------------

// Store degree only in the GDS in-memory graph.

CALL gds.degree.mutate(
    'stockGraph',
    {
        mutateProperty: 'degree'
    }
)

YIELD nodePropertiesWritten

RETURN nodePropertiesWritten;


// Top stocks by degree.

CALL gds.graph.nodeProperty.stream(
    'stockGraph',
    'degree'
)

YIELD
    nodeId,
    propertyValue AS degree

RETURN
    gds.util.asNode(nodeId).ticker AS ticker,
    degree

ORDER BY
    degree DESC,
    ticker ASC

LIMIT 15;


// ------------------------------------------------------------
// 8. Unweighted PageRank
// ------------------------------------------------------------

// Every correlation relationship is treated equally.

CALL gds.pageRank.mutate(
    'stockGraph',
    {
        mutateProperty: 'pagerank_unweighted'
    }
)

YIELD nodePropertiesWritten

RETURN nodePropertiesWritten;


// Top stocks by unweighted PageRank.

CALL gds.graph.nodeProperty.stream(
    'stockGraph',
    'pagerank_unweighted'
)

YIELD
    nodeId,
    propertyValue AS pagerank

RETURN
    gds.util.asNode(nodeId).ticker AS ticker,
    pagerank

ORDER BY pagerank DESC

LIMIT 15;


// ------------------------------------------------------------
// 9. Weighted PageRank
// ------------------------------------------------------------

// Correlation magnitude is used as the relationship weight.

CALL gds.pageRank.mutate(
    'stockGraph',
    {
        relationshipWeightProperty: 'weight',
        mutateProperty: 'pagerank_weighted'
    }
)

YIELD nodePropertiesWritten

RETURN nodePropertiesWritten;


// Top stocks by weighted PageRank.

CALL gds.graph.nodeProperty.stream(
    'stockGraph',
    'pagerank_weighted'
)

YIELD
    nodeId,
    propertyValue AS pagerank

RETURN
    gds.util.asNode(nodeId).ticker AS ticker,
    pagerank

ORDER BY pagerank DESC

LIMIT 15;


// ------------------------------------------------------------
// 10. Compare centrality measures
// ------------------------------------------------------------

// Pearson similarity across all 85 stocks quantifies how closely
// degree, unweighted PageRank, and weighted PageRank move together.

CALL gds.graph.nodeProperty.stream(
    'stockGraph',
    'degree'
)

YIELD
    nodeId,
    propertyValue AS degree

WITH
    collect(degree) AS degree_scores,

    collect(
        gds.util.nodeProperty(
            'stockGraph',
            nodeId,
            'pagerank_unweighted'
        )
    ) AS unweighted_scores,

    collect(
        gds.util.nodeProperty(
            'stockGraph',
            nodeId,
            'pagerank_weighted'
        )
    ) AS weighted_scores

RETURN
    gds.similarity.pearson(
        degree_scores,
        unweighted_scores
    ) AS degree_vs_unweighted_pagerank,

    gds.similarity.pearson(
        degree_scores,
        weighted_scores
    ) AS degree_vs_weighted_pagerank,

    gds.similarity.pearson(
        unweighted_scores,
        weighted_scores
    ) AS unweighted_vs_weighted_pagerank;

    // ------------------------------------------------------------
// 11. Export-ready node metrics
// ------------------------------------------------------------

// Combine connected-component membership with the centrality
// properties already stored in the GDS in-memory graph.
//
// Run this query after Sections 1-10, then export the result
// from Neo4j as data/processed/node_metrics.csv.

CALL gds.wcc.stream('stockGraph')

YIELD
    nodeId,
    componentId

RETURN
    gds.util.asNode(nodeId).ticker AS ticker,

    gds.util.nodeProperty(
        'stockGraph',
        nodeId,
        'degree'
    ) AS degree,

    gds.util.nodeProperty(
        'stockGraph',
        nodeId,
        'pagerank_unweighted'
    ) AS pagerank_unweighted,

    gds.util.nodeProperty(
        'stockGraph',
        nodeId,
        'pagerank_weighted'
    ) AS pagerank_weighted,

    componentId AS component_id

ORDER BY
    pagerank_weighted DESC,
    ticker ASC;