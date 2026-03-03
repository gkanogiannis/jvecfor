package com.gkanogiannis.jvecfor;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.SearchResult.NodeScore;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * k-nearest-neighbor computation: approximate (ANN) via jvector HNSW-DiskANN exact (KNN) via
 * Vantage-Point tree.
 */
public class KNNComputer {

    @FunctionalInterface
    private interface DistanceFn {
        double distance(double[] a, double[] b);
    }

    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int dimensions;
    private final VectorSimilarityFunction similarityFunction;
    private final int efSearch; // 0 = auto: max(k+1, 3k); use --ef-search for higher recall
    private final int M; // HNSW-DiskANN max connections per node (M); default 16
    private final float
            oversampleFactor; // 1.0 = disabled; >1 multiplies ef to fetch more candidates
    private final boolean verbose; // if false, suppress jvector build progress logging
    private final int pqSubspaces; // 0 = disabled; >0 enables PQ with this many subspaces

    /** Canonical constructor. */
    public KNNComputer(
            int dimensions,
            String metric,
            int efSearch,
            int M,
            float oversampleFactor,
            boolean verbose,
            int pqSubspaces) {
        this.dimensions = dimensions;
        this.similarityFunction = parseMetric(metric);
        this.efSearch = efSearch;
        this.M = M;
        this.oversampleFactor = oversampleFactor;
        this.verbose = verbose;
        this.pqSubspaces = pqSubspaces;
    }

    public KNNComputer(
            int dimensions,
            String metric,
            int efSearch,
            int M,
            float oversampleFactor,
            boolean verbose) {
        this(dimensions, metric, efSearch, M, oversampleFactor, verbose, 0);
    }

    public KNNComputer(int dimensions, String metric, int efSearch, int m, float oversampleFactor) {
        this(dimensions, metric, efSearch, m, oversampleFactor, false, 0);
    }

    public KNNComputer(int dimensions, String metric, int efSearch, int m) {
        this(dimensions, metric, efSearch, m, 1.0f, false, 0);
    }

    public KNNComputer(int dimensions, String metric, int efSearch) {
        this(dimensions, metric, efSearch, 16, 1.0f, false, 0);
    }

    public KNNComputer(int dimensions, String metric) {
        this(dimensions, metric, 0, 16, 1.0f, false, 0);
    }

    /**
     * Compute <em>approximate</em> k-nearest neighbors using jvector's HNSW-DiskANN index. The
     * index is built and searched concurrently with {@code numThreads} threads via a shared {@link
     * ForkJoinPool}.
     *
     * <p>The effective beam width used during search is {@code efSearch} when set explicitly, or
     * {@code max(k+1, 3k)} when using the auto default (0). When {@code oversampleFactor > 1}, the
     * beam is widened to {@code ceil(ef * oversampleFactor)} candidates and the top k are returned,
     * improving recall at proportionally higher search cost.
     *
     * <p>Graph connectivity is controlled by {@code m} (default 16). Higher values (e.g. 32)
     * improve recall for high-dimensional data at the cost of ~2x more memory and a slower build.
     *
     * @param data Matrix (n_points x n_dimensions)
     * @param k Number of neighbors
     * @param numThreads Thread count for both index build and search
     * @return KNNResult with indices and similarity scores
     */
    public KNNResult computeANN(double[][] data, int k, int numThreads) {
        int n = data.length;
        List<VectorFloat<?>> vectors = convertToVectorFloats(data);
        ListRandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, dimensions);

        // Effective beam width: explicit override or auto formula
        int ef = (efSearch > 0) ? efSearch : Math.max(k + 1, 3 * k);
        // Widen beam for oversampling: fetches more candidates, returns top k
        int efFinal = (oversampleFactor > 1.0f) ? (int) Math.ceil(ef * oversampleFactor) : ef;

        int[][] indices = new int[n][k];
        double[][] distances = new double[n][k];

        // Suppress jvector build logging unless verbose mode is on
        if (!verbose) {
            System.setProperty("org.slf4j.simpleLogger.log.io.github.jbellis.jvector", "warn");
        }

        try (ForkJoinPool pool = new ForkJoinPool(numThreads)) {
            // Phase 1: build HNSW-DiskANN index with all n nodes inserted concurrently
            BuildScoreProvider bsp =
                    BuildScoreProvider.randomAccessScoreProvider(ravv, similarityFunction);
            ImmutableGraphIndex index;
            try (GraphIndexBuilder builder =
                    new GraphIndexBuilder(bsp, n, M, 200, 1.2f, 1.2f, true, false, pool, pool)) {
                pool.submit(
                                () ->
                                        IntStream.range(0, n)
                                                .parallel()
                                                .forEach(
                                                        i ->
                                                                builder.addGraphNode(
                                                                        i, vectors.get(i))))
                        .get();
                builder.cleanup();
                index = builder.getGraph();
            }

            // Optional: build PQ codebook and encode all vectors for fast approximate scoring
            final PQVectors pqVectors;
            if (pqSubspaces > 0) {
                int clusters = Math.min(256, n);
                ProductQuantization pq =
                        ProductQuantization.compute(
                                ravv, pqSubspaces, clusters, true, 0.2f, pool, pool);
                pqVectors = pq.encodeAll(ravv, pool);
            } else {
                pqVectors = null;
            }
            final PQVectors finalPqVectors = pqVectors;

            // Phase 2: find k neighbors for every node concurrently.
            // Each worker thread owns one GraphSearcher instance (not thread-safe) via ThreadLocal,
            // avoiding N heap allocations of NodeQueue + IntHashSet for large n.
            final ImmutableGraphIndex finalIndex = index;
            final int finalEf = efFinal;
            ConcurrentLinkedQueue<GraphSearcher> allSearchers = new ConcurrentLinkedQueue<>();
            ThreadLocal<GraphSearcher> tlSearcher =
                    ThreadLocal.withInitial(
                            () -> {
                                GraphSearcher gs = new GraphSearcher(finalIndex);
                                allSearchers.add(gs);
                                return gs;
                            });

            pool.submit(
                            () ->
                                    IntStream.range(0, n)
                                            .parallel()
                                            .forEach(
                                                    idx -> {
                                                        GraphSearcher searcher = tlSearcher.get();
                                                        VectorFloat<?> query = vectors.get(idx);
                                                        DefaultSearchScoreProvider ssp;
                                                        if (finalPqVectors != null) {
                                                            var approxFn =
                                                                    finalPqVectors
                                                                            .precomputedScoreFunctionFor(
                                                                                    query,
                                                                                    similarityFunction);
                                                            var exactFn =
                                                                    DefaultSearchScoreProvider
                                                                            .exact(
                                                                                    query,
                                                                                    similarityFunction,
                                                                                    ravv)
                                                                            .exactScoreFunction();
                                                            ssp =
                                                                    new DefaultSearchScoreProvider(
                                                                            approxFn, exactFn);
                                                        } else {
                                                            ssp =
                                                                    DefaultSearchScoreProvider
                                                                            .exact(
                                                                                    query,
                                                                                    similarityFunction,
                                                                                    ravv);
                                                        }
                                                        SearchResult result =
                                                                searcher.search(
                                                                        ssp, finalEf, Bits.ALL);
                                                        NodeScore[] nodes = result.getNodes();
                                                        int filled = 0;
                                                        for (int j = 0;
                                                                j < nodes.length && filled < k;
                                                                j++) {
                                                            if (nodes[j].node == idx) continue;
                                                            indices[idx][filled] = nodes[j].node;
                                                            distances[idx][filled] =
                                                                    (double) nodes[j].score;
                                                            filled++;
                                                        }
                                                    }))
                    .get();

            // Close all per-thread GraphSearcher views
            for (GraphSearcher gs : allSearchers) {
                try {
                    gs.close();
                } catch (IOException ignored) {
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to build graph index", e);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
        return new KNNResult(indices, distances);
    }

    /**
     * Compute <em>exact</em> k-nearest neighbors using a Vantage-Point tree. Returns true metric
     * distances (Euclidean L2 for {@code euclidean}, angular distance for {@code cosine}).
     *
     * <p>{@code dot_product} is not a proper metric and is not supported; use {@code euclidean} or
     * {@code cosine} when calling exact KNN.
     *
     * @param data Matrix (n_points x n_dimensions)
     * @param k Number of neighbors
     * @param numThreads Thread count
     * @return KNNResult with indices and exact metric distances
     * @throws UnsupportedOperationException if the configured metric is {@code dot_product}
     */
    public KNNResult computeKNN(double[][] data, int k, int numThreads) {
        DistanceFn distFn = toDistanceFn(similarityFunction); // throws for dot_product
        int n = data.length;
        VpTree vpTree = new VpTree(data, distFn);
        int[][] indices = new int[n][k];
        double[][] distances = new double[n][k];

        try (ForkJoinPool pool = new ForkJoinPool(numThreads)) {
            pool.submit(
                            () ->
                                    IntStream.range(0, n)
                                            .parallel()
                                            .forEach(
                                                    idx -> {
                                                        List<long[]> results =
                                                                vpTree.search(data[idx], k, idx);
                                                        for (int j = 0; j < results.size(); j++) {
                                                            indices[idx][j] =
                                                                    (int) results.get(j)[0];
                                                            distances[idx][j] =
                                                                    Double.longBitsToDouble(
                                                                            results.get(j)[1]);
                                                        }
                                                    }))
                    .get();
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
        return new KNNResult(indices, distances);
    }

    // -----------------------------------------------------------------------
    // Distance helpers
    // -----------------------------------------------------------------------

    /**
     * Map the configured jvector similarity function to a proper metric distance function suitable
     * for VP-tree search.
     */
    private static DistanceFn toDistanceFn(VectorSimilarityFunction sim) {
        return switch (sim) {
            case EUCLIDEAN -> KNNComputer::euclideanDist;
            case COSINE -> KNNComputer::cosineAngularDist;
            default ->
                    throw new UnsupportedOperationException(
                            sim
                                    + " is not a proper metric; use euclidean or cosine for exact"
                                    + " KNN.");
        };
    }

    /** Euclidean (L2) distance. */
    private static double euclideanDist(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    /**
     * Angular distance = arccos(cosine_similarity). Satisfies the triangle inequality and is the
     * proper metric counterpart of cosine similarity.
     */
    private static double cosineAngularDist(double[] a, double[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0 || normB == 0.0) return 0.0;
        double cosine = dot / Math.sqrt(normA * normB);
        cosine = Math.max(-1.0, Math.min(1.0, cosine)); // clamp for numerical safety
        return Math.acos(cosine);
    }

    // -----------------------------------------------------------------------
    // Vantage-Point tree (exact metric KNN)
    // -----------------------------------------------------------------------

    /**
     * Vantage-Point tree for exact nearest-neighbor search in metric spaces.
     *
     * <p>Build: O(n log²n). Query: O(k log n) average.
     *
     * <p>Reference: P. N. Yianilos, "Data structures and algorithms for nearest neighbor search in
     * general metric spaces", SODA 1993.
     */
    private static final class VpTree {

        private static final class VpNode {
            final int index; // index into data[]
            final double threshold; // median distance from this VP to its subtree
            final VpNode near; // points within threshold
            final VpNode far; // points beyond threshold

            VpNode(int index, double threshold, VpNode near, VpNode far) {
                this.index = index;
                this.threshold = threshold;
                this.near = near;
                this.far = far;
            }
        }

        private final double[][] data;
        private final DistanceFn distFn;
        private final VpNode root;

        VpTree(double[][] data, DistanceFn distFn) {
            this.data = data;
            this.distFn = distFn;
            int[] pts = new int[data.length];
            for (int i = 0; i < pts.length; i++) pts[i] = i;
            this.root = build(pts, 0, pts.length, new Random(42));
        }

        private VpNode build(int[] pts, int lo, int hi, Random rng) {
            if (lo >= hi) return null;

            // Choose vantage point uniformly at random and move it to pts[lo]
            int vpPos = lo + rng.nextInt(hi - lo);
            int tmp = pts[lo];
            pts[lo] = pts[vpPos];
            pts[vpPos] = tmp;
            int vp = pts[lo];

            int remaining = hi - lo - 1;
            if (remaining == 0) return new VpNode(vp, 0.0, null, null);

            // Compute distances from vp to all remaining points
            double[] dists = new double[remaining];
            for (int i = 0; i < remaining; i++) {
                dists[i] = distFn.distance(data[vp], data[pts[lo + 1 + i]]);
            }

            // Indirect sort: order[] maps sorted rank → original position in dists[]
            Integer[] order = new Integer[remaining];
            for (int i = 0; i < remaining; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(dists[a], dists[b]));

            // Rearrange pts[lo+1 .. hi-1] in distance-sorted order
            int[] arranged = new int[remaining];
            for (int i = 0; i < remaining; i++) arranged[i] = pts[lo + 1 + order[i]];
            System.arraycopy(arranged, 0, pts, lo + 1, remaining);

            // Split: near = [lo+1, split), far = [split, hi); threshold = median distance
            int split = lo + 1 + remaining / 2;
            double threshold = dists[order[remaining / 2]];

            VpNode near = build(pts, lo + 1, split, rng);
            VpNode far = build(pts, split, hi, rng);
            return new VpNode(vp, threshold, near, far);
        }

        /**
         * Find the k nearest neighbors of {@code query}, excluding the point at {@code excludeIdx}.
         *
         * @return list of {@code long[]{index, Double.doubleToRawLongBits(distance)}}, sorted by
         *     ascending distance, length ≤ k
         */
        List<long[]> search(double[] query, int k, int excludeIdx) {
            // Max-heap: the element with the largest distance sits at the top so we can prune
            PriorityQueue<long[]> maxHeap =
                    new PriorityQueue<>(
                            (a, b) ->
                                    Double.compare(
                                            Double.longBitsToDouble(b[1]),
                                            Double.longBitsToDouble(a[1])));
            double[] tau = {Double.MAX_VALUE}; // current k-th distance (radius of acceptance)
            searchNode(root, query, k, excludeIdx, maxHeap, tau);

            // Drain heap into a list and sort ascending
            List<long[]> result = new ArrayList<>(maxHeap);
            result.sort(
                    (a, b) ->
                            Double.compare(
                                    Double.longBitsToDouble(a[1]), Double.longBitsToDouble(b[1])));
            return result;
        }

        private void searchNode(
                VpNode node,
                double[] query,
                int k,
                int excludeIdx,
                PriorityQueue<long[]> maxHeap,
                double[] tau) {
            if (node == null) return;

            double d = distFn.distance(query, data[node.index]);

            // Add this vantage point to candidates if it is close enough and not excluded
            if (node.index != excludeIdx && d < tau[0]) {
                maxHeap.offer(new long[] {node.index, Double.doubleToRawLongBits(d)});
                if (maxHeap.size() > k) maxHeap.poll(); // evict the farthest
                if (maxHeap.size() == k) {
                    tau[0] = Double.longBitsToDouble(maxHeap.peek()[1]);
                }
            }

            // Visit subtrees in order of likelihood; re-evaluate conditions after each visit
            // because tau may have shrunk.
            if (d < node.threshold) {
                // Query is on the "near" side → search near first
                if (d - tau[0] <= node.threshold) {
                    searchNode(node.near, query, k, excludeIdx, maxHeap, tau);
                }
                if (d + tau[0] >= node.threshold) {
                    searchNode(node.far, query, k, excludeIdx, maxHeap, tau);
                }
            } else {
                // Query is on the "far" side → search far first
                if (d + tau[0] >= node.threshold) {
                    searchNode(node.far, query, k, excludeIdx, maxHeap, tau);
                }
                if (d - tau[0] <= node.threshold) {
                    searchNode(node.near, query, k, excludeIdx, maxHeap, tau);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Misc private helpers
    // -----------------------------------------------------------------------

    private VectorSimilarityFunction parseMetric(String metric) {
        return switch (metric.toLowerCase()) {
            case "euclidean" -> VectorSimilarityFunction.EUCLIDEAN;
            case "cosine" -> VectorSimilarityFunction.COSINE;
            case "dot_product" -> VectorSimilarityFunction.DOT_PRODUCT;
            default -> throw new IllegalArgumentException("Unknown metric: " + metric);
        };
    }

    private List<VectorFloat<?>> convertToVectorFloats(double[][] data) {
        List<VectorFloat<?>> vectors = new ArrayList<>(data.length);
        for (double[] row : data) {
            float[] frow = new float[row.length];
            for (int i = 0; i < row.length; i++) {
                frow[i] = (float) row[i];
            }
            vectors.add(VTS.createFloatVector(frow));
        }
        return vectors;
    }

    /** Holds the k-NN result: neighbor indices and similarity scores or distances. */
    public static class KNNResult {
        public final int[][] indices;
        public final double[][] distances;

        public KNNResult(int[][] indices, double[][] distances) {
            this.indices = indices;
            this.distances = distances;
        }
    }
}
