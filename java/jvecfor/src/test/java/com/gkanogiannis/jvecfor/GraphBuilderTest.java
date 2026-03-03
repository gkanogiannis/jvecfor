package com.gkanogiannis.jvecfor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;

class GraphBuilderTest {

    /**
     * Hand-crafted knnIndices for deterministic testing (no jvector dependency).
     *
     * <pre>
     * Node 0: [1, 2, 3]   Node 3: [0, 1, 2]
     * Node 1: [0, 2, 3]   Node 4: [3, 5, 0]
     * Node 2: [0, 1, 3]   Node 5: [4, 3, 2]
     * </pre>
     *
     * Edges generated (j > i, j in knn[i]): (0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(4,5)
     */
    private static final int[][] KNN = {
        {1, 2, 3},
        {0, 2, 3},
        {0, 1, 3},
        {0, 1, 2},
        {3, 5, 0},
        {4, 3, 2}
    };

    // --- neighborsToKNNGraph tests ---

    @Test
    void neighborsToKNNGraph_outputShapeAndDimensions() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToKNNGraph(KNN, 1);

        assertEquals(6, graph.n, "n must equal number of nodes.");
        assertEquals(graph.i.length, graph.j.length, "i and j arrays must have equal length.");
        assertEquals(graph.i.length, graph.x.length, "i and x arrays must have equal length.");
        assertTrue(graph.i.length > 0, "Graph must have at least one edge.");
    }

    @Test
    void neighborsToKNNGraph_noSelfLoops() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToKNNGraph(KNN, 1);

        for (int idx = 0; idx < graph.i.length; idx++) {
            assertTrue(graph.i[idx] != graph.j[idx], "Self-loop found at index " + idx + ".");
        }
    }

    @Test
    void neighborsToKNNGraph_upperTriangleOnly() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToKNNGraph(KNN, 1);

        for (int idx = 0; idx < graph.i.length; idx++) {
            assertTrue(
                    graph.i[idx] < graph.j[idx],
                    "Edge (" + graph.i[idx] + "," + graph.j[idx] + ") violates i < j.");
        }
    }

    @Test
    void neighborsToKNNGraph_allWeightsAreOne() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToKNNGraph(KNN, 1);

        for (int idx = 0; idx < graph.x.length; idx++) {
            assertEquals(
                    1.0, graph.x[idx], 1e-9, "KNN edge weight must be 1.0 at index " + idx + ".");
        }
    }

    @Test
    void neighborsToKNNGraph_noDuplicateEdges() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToKNNGraph(KNN, 1);

        Set<Long> seen = new HashSet<>();
        for (int idx = 0; idx < graph.i.length; idx++) {
            long key = ((long) graph.i[idx] << 32) | (graph.j[idx] & 0xFFFFFFFFL);
            assertTrue(
                    seen.add(key), "Duplicate edge (" + graph.i[idx] + "," + graph.j[idx] + ").");
        }
    }

    // --- neighborsToSNNGraph tests ---

    @Test
    void neighborsToSNNGraph_number_outputShapeAndDimensions() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToSNNGraph(KNN, "number", 1);

        assertEquals(6, graph.n, "n must equal number of nodes.");
        assertEquals(graph.i.length, graph.j.length, "i and j arrays must have equal length.");
        assertEquals(graph.i.length, graph.x.length, "i and x arrays must have equal length.");
        assertTrue(graph.i.length > 0, "Graph must have at least one edge.");
    }

    @Test
    void neighborsToSNNGraph_number_noSelfLoops() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToSNNGraph(KNN, "number", 1);

        for (int idx = 0; idx < graph.i.length; idx++) {
            assertTrue(graph.i[idx] != graph.j[idx], "Self-loop found at index " + idx + ".");
        }
    }

    @Test
    void neighborsToSNNGraph_number_upperTriangleOnly() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToSNNGraph(KNN, "number", 1);

        for (int idx = 0; idx < graph.i.length; idx++) {
            assertTrue(
                    graph.i[idx] < graph.j[idx],
                    "Edge (" + graph.i[idx] + "," + graph.j[idx] + ") violates i < j.");
        }
    }

    @Test
    void neighborsToSNNGraph_number_weightsAreSharedNeighborCounts() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToSNNGraph(KNN, "number", 1);

        // For edge (0,1): KNN(0)={1,2,3}, KNN(1)={0,2,3}
        //   actual shared={2,3}=2, +1 (1∈KNN(0), self-rank), +1 (0∈KNN(1), self-rank) => weight=4
        double edgeWeight01 = findEdgeWeight(graph, 0, 1);
        assertEquals(4.0, edgeWeight01, 1e-9, "Edge (0,1) should have shared-neighbor weight 4.");

        // For edge (4,5): KNN(4)={3,5,0}, KNN(5)={4,3,2}
        //   actual shared={3}=1, +1 (5∈KNN(4), self-rank), +1 (4∈KNN(5), self-rank) => weight=3
        double edgeWeight45 = findEdgeWeight(graph, 4, 5);
        assertEquals(3.0, edgeWeight45, 1e-9, "Edge (4,5) should have shared-neighbor weight 3.");
    }

    @Test
    void neighborsToSNNGraph_jaccard_weightsInUnitInterval() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToSNNGraph(KNN, "jaccard", 1);

        for (int idx = 0; idx < graph.x.length; idx++) {
            assertTrue(
                    graph.x[idx] > 0.0 && graph.x[idx] <= 1.0,
                    "Jaccard weight out of (0,1] at index " + idx + ": " + graph.x[idx]);
        }
    }

    @Test
    void neighborsToSNNGraph_rank_weightsPositive() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToSNNGraph(KNN, "rank", 1);

        for (int idx = 0; idx < graph.x.length; idx++) {
            assertTrue(graph.x[idx] > 0.0, "Rank weight must be positive at index " + idx + ".");
        }
    }

    @Test
    void neighborsToSNNGraph_noDuplicateEdges() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.neighborsToSNNGraph(KNN, "number", 1);

        Set<Long> seen = new HashSet<>();
        for (int idx = 0; idx < graph.i.length; idx++) {
            long key = ((long) graph.i[idx] << 32) | (graph.j[idx] & 0xFFFFFFFFL);
            assertTrue(
                    seen.add(key), "Duplicate edge (" + graph.i[idx] + "," + graph.j[idx] + ").");
        }
    }

    @Test
    void neighborsToSNNGraph_unknownTypeThrows() {
        GraphBuilder builder = new GraphBuilder();
        assertThrows(
                IllegalArgumentException.class,
                () -> builder.neighborsToSNNGraph(KNN, "unknown_type", 1));
    }

    @Test
    void neighborsToSNNGraph_multiThread_sameResultAsSingleThread() {
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph single = builder.neighborsToSNNGraph(KNN, "number", 1);
        GraphBuilder.SNNGraph multi = builder.neighborsToSNNGraph(KNN, "number", 4);

        // Compare edge sets (order may differ between runs)
        assertEquals(
                toEdgeSet(single),
                toEdgeSet(multi),
                "Multi-thread result must equal single-thread result.");
    }

    // --- makeKNNGraph tests (data → ANN → KNN graph) ---

    @Test
    void makeKNNGraph_outputShapeAndDimensions() {
        int n = 20, dims = 4, k = 3;
        double[][] data = makeGrid(n, dims);
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.makeKNNGraph(data, k, "euclidean", 1);

        assertEquals(n, graph.n, "n must equal number of nodes.");
        assertEquals(graph.i.length, graph.j.length, "i and j arrays must have equal length.");
        assertEquals(graph.i.length, graph.x.length, "i and x arrays must have equal length.");
        assertTrue(graph.i.length > 0, "Graph must have at least one edge.");
    }

    @Test
    void makeKNNGraph_allWeightsAreOne() {
        double[][] data = makeGrid(20, 4);
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.makeKNNGraph(data, 3, "euclidean", 1);

        for (int idx = 0; idx < graph.x.length; idx++) {
            assertEquals(1.0, graph.x[idx], 1e-9, "KNN edge weight must be 1.0.");
        }
    }

    @Test
    void makeKNNGraph_upperTriangleNoSelfLoops() {
        double[][] data = makeGrid(20, 4);
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.makeKNNGraph(data, 3, "euclidean", 1);

        for (int idx = 0; idx < graph.i.length; idx++) {
            assertTrue(graph.i[idx] < graph.j[idx], "Edge must satisfy i < j.");
        }
    }

    // --- makeSNNGraph tests (data → ANN → SNN graph) ---

    @Test
    void makeSNNGraph_outputShapeAndDimensions() {
        int n = 20, dims = 4, k = 3;
        double[][] data = makeGrid(n, dims);
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.makeSNNGraph(data, k, "euclidean", "number", 1);

        assertEquals(n, graph.n, "n must equal number of nodes.");
        assertEquals(graph.i.length, graph.j.length, "i and j arrays must have equal length.");
        assertEquals(graph.i.length, graph.x.length, "i and x arrays must have equal length.");
        assertTrue(graph.i.length > 0, "Graph must have at least one edge.");
    }

    @Test
    void makeSNNGraph_jaccard_weightsInUnitInterval() {
        double[][] data = makeGrid(20, 4);
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.makeSNNGraph(data, 3, "euclidean", "jaccard", 1);

        for (int idx = 0; idx < graph.x.length; idx++) {
            assertTrue(
                    graph.x[idx] > 0.0 && graph.x[idx] <= 1.0,
                    "Jaccard weight out of (0,1] at index " + idx + ": " + graph.x[idx]);
        }
    }

    @Test
    void makeSNNGraph_rank_weightsPositive() {
        double[][] data = makeGrid(20, 4);
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.makeSNNGraph(data, 3, "euclidean", "rank", 1);

        for (int idx = 0; idx < graph.x.length; idx++) {
            assertTrue(graph.x[idx] > 0.0, "Rank weight must be positive.");
        }
    }

    @Test
    void makeSNNGraph_upperTriangleNoSelfLoops() {
        double[][] data = makeGrid(20, 4);
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.makeSNNGraph(data, 3, "euclidean", "number", 1);

        for (int idx = 0; idx < graph.i.length; idx++) {
            assertTrue(graph.i[idx] < graph.j[idx], "Edge must satisfy i < j.");
        }
    }

    @Test
    void makeSNNGraph_cosineMetric_producesEdges() {
        double[][] data = makeGrid(20, 4);
        GraphBuilder builder = new GraphBuilder();
        GraphBuilder.SNNGraph graph = builder.makeSNNGraph(data, 3, "cosine", "jaccard", 1);

        assertTrue(graph.i.length > 0, "Cosine SNN graph must have at least one edge.");
    }

    // --- helpers ---

    private static double[][] makeGrid(int n, int dims) {
        double[][] data = new double[n][dims];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dims; d++) {
                data[i][d] = i * 0.1 + d;
            }
        }
        return data;
    }

    private static double findEdgeWeight(GraphBuilder.SNNGraph graph, int src, int dst) {
        for (int idx = 0; idx < graph.i.length; idx++) {
            if (graph.i[idx] == src && graph.j[idx] == dst) {
                return graph.x[idx];
            }
        }
        throw new AssertionError("Edge (" + src + "," + dst + ") not found in graph.");
    }

    private static Set<String> toEdgeSet(GraphBuilder.SNNGraph graph) {
        Set<String> edges = new HashSet<>();
        for (int idx = 0; idx < graph.i.length; idx++) {
            edges.add(graph.i[idx] + ":" + graph.j[idx] + ":" + graph.x[idx]);
        }
        return edges;
    }
}
