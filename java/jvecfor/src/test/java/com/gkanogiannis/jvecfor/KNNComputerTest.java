package com.gkanogiannis.jvecfor;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class KNNComputerTest {

    private static double[][] makeGrid(int n, int dims) {
        double[][] data = new double[n][dims];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dims; d++) {
                data[i][d] = i * 0.1 + d;
            }
        }
        return data;
    }

    // -----------------------------------------------------------------------
    // computeANN tests (approximate, jvector HNSW-DiskANN
    // -----------------------------------------------------------------------

    @Test
    void computeANN_euclidean_returnsCorrectShape() {
        int n = 20, dims = 4, k = 3;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        assertEquals(n, result.indices.length, "Row count must equal n.");
        assertEquals(n, result.distances.length, "Row count must equal n.");
        for (int i = 0; i < n; i++) {
            assertEquals(k, result.indices[i].length, "Each row must have k neighbors.");
            assertEquals(k, result.distances[i].length, "Each row must have k distances.");
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"euclidean", "cosine"})
    void computeANN_selfNotInNeighbors(String metric) {
        int n = 30, dims = 5, k = 5;
        double[][] data = new double[n][dims];
        for (int i = 0; i < n / 2; i++) {
            data[i][0] = 10.0 + i * 0.01;
            data[i][1] = i * 0.001;
        }
        for (int i = n / 2; i < n; i++) {
            data[i][0] = (i - n / 2) * 0.001;
            data[i][1] = 10.0 + (i - n / 2) * 0.01;
        }
        KNNComputer computer = new KNNComputer(dims, metric);
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] != i, "Node " + i + " must not be its own neighbor.");
            }
        }
    }

    @Test
    void computeANN_validIndicesInRange() {
        int n = 25, dims = 3, k = 4;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeANN(data, k, 2);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] >= 0 && result.indices[i][j] < n,
                        "Neighbor index out of range at [" + i + "][" + j + "].");
            }
        }
    }

    @Test
    void computeANN_distancesNonNegative() {
        int n = 20, dims = 4, k = 3;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(result.distances[i][j] >= 0.0, "Distance must be non-negative.");
            }
        }
    }

    @Test
    void computeANN_clusteredData_neighborsFromSameCluster() {
        int clusterSize = 10, dims = 2, k = 4;
        int n = 2 * clusterSize;
        double[][] data = new double[n][dims];
        for (int i = 0; i < clusterSize; i++) {
            data[i][0] = i * 0.01;
            data[i][1] = i * 0.01;
        }
        for (int i = clusterSize; i < n; i++) {
            data[i][0] = 100.0 + (i - clusterSize) * 0.01;
            data[i][1] = 100.0 + (i - clusterSize) * 0.01;
        }
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        for (int i = 0; i < clusterSize; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] < clusterSize,
                        "Cluster-0 point " + i + " has neighbor in cluster 1.");
            }
        }
        for (int i = clusterSize; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] >= clusterSize,
                        "Cluster-1 point " + i + " has neighbor in cluster 0.");
            }
        }
    }

    @Test
    void computeANN_unknownMetricThrows() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new KNNComputer(4, "manhattan"),
                "Unknown metric should throw IllegalArgumentException.");
    }

    // -----------------------------------------------------------------------
    // New parameter tests: m, oversampleFactor, verbose
    // -----------------------------------------------------------------------

    @Test
    void computeANN_higherM_producesValidResult() {
        int n = 30, dims = 5, k = 4;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean", 0, 32);
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        assertEquals(n, result.indices.length);
        for (int i = 0; i < n; i++) {
            assertEquals(k, result.indices[i].length);
            for (int j = 0; j < k; j++) {
                assertTrue(result.indices[i][j] != i, "Self must not be neighbor.");
                assertTrue(
                        result.indices[i][j] >= 0 && result.indices[i][j] < n,
                        "Neighbor index out of range.");
            }
        }
    }

    @Test
    void computeANN_oversample_returnsExactlyKNeighbors() {
        int n = 30, dims = 5, k = 4;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean", 0, 16, 2.0f);
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        assertEquals(n, result.indices.length);
        for (int i = 0; i < n; i++) {
            assertEquals(k, result.indices[i].length, "Must return exactly k neighbors.");
            for (int j = 0; j < k; j++) {
                assertTrue(result.indices[i][j] != i, "Self must not be neighbor.");
                assertTrue(
                        result.indices[i][j] >= 0 && result.indices[i][j] < n,
                        "Neighbor index out of range.");
            }
        }
    }

    @Test
    void computeANN_oversampleClustered_clusterPurityMaintained() {
        int clusterSize = 15, dims = 2, k = 5;
        int n = 2 * clusterSize;
        double[][] data = new double[n][dims];
        for (int i = 0; i < clusterSize; i++) {
            data[i][0] = i * 0.01;
            data[i][1] = i * 0.01;
        }
        for (int i = clusterSize; i < n; i++) {
            data[i][0] = 100.0 + (i - clusterSize) * 0.01;
            data[i][1] = 100.0 + (i - clusterSize) * 0.01;
        }
        KNNComputer computer = new KNNComputer(dims, "euclidean", 0, 16, 2.0f);
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        for (int i = 0; i < clusterSize; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(result.indices[i][j] < clusterSize, "Cluster-0 neighbor in cluster 1.");
            }
        }
        for (int i = clusterSize; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(result.indices[i][j] >= clusterSize, "Cluster-1 neighbor in cluster 0.");
            }
        }
    }

    @Test
    void computeANN_verboseFalse_noException() {
        double[][] data = makeGrid(20, 4);
        KNNComputer computer = new KNNComputer(4, "euclidean", 0, 16, 1.0f, false);
        assertDoesNotThrow(() -> computer.computeANN(data, 3, 1));
    }

    @Test
    void computeANN_verboseTrue_noException() {
        double[][] data = makeGrid(20, 4);
        KNNComputer computer = new KNNComputer(4, "euclidean", 0, 16, 1.0f, true);
        assertDoesNotThrow(() -> computer.computeANN(data, 3, 1));
    }

    // -----------------------------------------------------------------------
    // PQ (Product Quantization) tests
    // -----------------------------------------------------------------------

    @Test
    void computeANN_pq_validShape() {
        int n = 30, dims = 10, k = 4, pqSubspaces = 5;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean", 0, 16, 1.0f, false, pqSubspaces);
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        assertEquals(n, result.indices.length, "Row count must equal n.");
        assertEquals(n, result.distances.length, "Row count must equal n.");
        for (int i = 0; i < n; i++) {
            assertEquals(k, result.indices[i].length, "Each row must have k neighbors.");
            assertEquals(k, result.distances[i].length, "Each row must have k distances.");
        }
    }

    @Test
    void computeANN_pq_selfNotInNeighbors() {
        int n = 30, dims = 10, k = 4, pqSubspaces = 5;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean", 0, 16, 1.0f, false, pqSubspaces);
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] != i, "Node " + i + " must not be its own neighbor.");
            }
        }
    }

    @Test
    void computeANN_pq_clusteredData_clusterPurity() {
        int clusterSize = 15, dims = 4, k = 4, pqSubspaces = 2;
        int n = 2 * clusterSize;
        double[][] data = new double[n][dims];
        for (int i = 0; i < clusterSize; i++) {
            data[i][0] = i * 0.01;
            data[i][1] = i * 0.01;
        }
        for (int i = clusterSize; i < n; i++) {
            data[i][0] = 100.0 + (i - clusterSize) * 0.01;
            data[i][1] = 100.0 + (i - clusterSize) * 0.01;
        }
        KNNComputer computer = new KNNComputer(dims, "euclidean", 0, 16, 1.0f, false, pqSubspaces);
        KNNComputer.KNNResult result = computer.computeANN(data, k, 1);

        for (int i = 0; i < clusterSize; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] < clusterSize,
                        "Cluster-0 point " + i + " has neighbor in cluster 1.");
            }
        }
        for (int i = clusterSize; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] >= clusterSize,
                        "Cluster-1 point " + i + " has neighbor in cluster 0.");
            }
        }
    }

    @Test
    void computeANN_multiThread_validShape() {
        int n = 40, dims = 5, k = 4;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeANN(data, k, 4);

        assertEquals(n, result.indices.length);
        assertEquals(n, result.distances.length);
        for (int i = 0; i < n; i++) {
            assertEquals(k, result.indices[i].length);
            assertEquals(k, result.distances[i].length);
            for (int j = 0; j < k; j++) {
                assertTrue(result.indices[i][j] != i, "Self must not be neighbor.");
            }
        }
    }

    // -----------------------------------------------------------------------
    // computeKNN tests (exact, VP-tree)
    // -----------------------------------------------------------------------

    @Test
    void computeKNN_euclidean_returnsCorrectShape() {
        int n = 20, dims = 4, k = 3;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeKNN(data, k, 1);

        assertEquals(n, result.indices.length, "Row count must equal n.");
        assertEquals(n, result.distances.length, "Row count must equal n.");
        for (int i = 0; i < n; i++) {
            assertEquals(k, result.indices[i].length, "Each row must have k neighbors.");
            assertEquals(k, result.distances[i].length, "Each row must have k distances.");
        }
    }

    @Test
    void computeKNN_selfNotInNeighbors() {
        int n = 20, dims = 4, k = 3;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeKNN(data, k, 1);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] != i, "Node " + i + " must not be its own neighbor.");
            }
        }
    }

    @Test
    void computeKNN_euclidean_exactDistances() {
        // 4 collinear points: 0,1,10,11 — true NN for point 1 is {0 (d=1), 2 (d=9)}
        double[][] data = {{0.0}, {1.0}, {10.0}, {11.0}};
        KNNComputer computer = new KNNComputer(1, "euclidean");
        KNNComputer.KNNResult result = computer.computeKNN(data, 2, 1);

        // Point 1 should have point 0 as nearest neighbor at distance 1.0
        int[] neighbors1 = result.indices[1];
        double[] dists1 = result.distances[1];
        assertEquals(0, neighbors1[0], "Nearest neighbor of point 1 must be point 0.");
        assertEquals(1.0, dists1[0], 1e-9, "Distance from point 1 to point 0 must be 1.0.");
        assertEquals(9.0, dists1[1], 1e-9, "Distance from point 1 to point 2 must be 9.0.");

        // Point 2 should have point 3 as nearest neighbor at distance 1.0
        assertEquals(3, result.indices[2][0], "Nearest neighbor of point 2 must be point 3.");
        assertEquals(
                1.0, result.distances[2][0], 1e-9, "Distance from point 2 to point 3 must be 1.0.");
    }

    @Test
    void computeKNN_validIndicesInRange() {
        int n = 25, dims = 3, k = 4;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeKNN(data, k, 2);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] >= 0 && result.indices[i][j] < n,
                        "Neighbor index out of range at [" + i + "][" + j + "].");
            }
        }
    }

    @Test
    void computeKNN_distancesNonNegative() {
        int n = 20, dims = 4, k = 3;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "euclidean");
        KNNComputer.KNNResult result = computer.computeKNN(data, k, 1);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(result.distances[i][j] >= 0.0, "Distance must be non-negative.");
            }
        }
    }

    @Test
    void computeKNN_dotProductThrows() {
        double[][] data = makeGrid(10, 4);
        KNNComputer computer = new KNNComputer(4, "dot_product");
        assertThrows(
                UnsupportedOperationException.class,
                () -> computer.computeKNN(data, 3, 1),
                "dot_product is not a proper metric and should throw.");
    }

    @Test
    void computeKNN_cosine_selfNotInNeighbors() {
        int n = 20, dims = 4, k = 3;
        double[][] data = makeGrid(n, dims);
        KNNComputer computer = new KNNComputer(dims, "cosine");
        KNNComputer.KNNResult result = computer.computeKNN(data, k, 1);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                assertTrue(
                        result.indices[i][j] != i, "Node " + i + " must not be its own neighbor.");
            }
        }
    }
}
