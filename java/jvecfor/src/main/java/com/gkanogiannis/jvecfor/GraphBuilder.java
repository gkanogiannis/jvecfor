package com.gkanogiannis.jvecfor;

import com.gkanogiannis.jvecfor.KNNComputer.KNNResult;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Fast KNN and SNN graph construction from neighbor indices.
 *
 * <p>This is where we get huge speedup over R implementations!
 */
public class GraphBuilder {

    private static final class ReverseKNNResult {
        // CSR flat layout: data[offsets[s]..offsets[s+1]) holds interleaved (node,rank) pairs
        // for all i where s ∈ KNN(i). Indices step by 2: data[m]=node, data[m+1]=rank.
        final int[] data; // interleaved pairs: [node0,rank0, node1,rank1, …] for all s
        final int[]
                offsets; // CSR offsets (length n+1); data slice for s is [offsets[s], offsets[s+1])

        ReverseKNNResult(int[] data, int[] offsets) {
            this.data = data;
            this.offsets = offsets;
        }
    }

    private static final class TaskResult {
        final int[] rowI, colJ;
        final double[] weights;
        final int count;

        TaskResult(int[] rowI, int[] colJ, double[] weights, int count) {
            this.rowI = rowI;
            this.colJ = colJ;
            this.weights = weights;
            this.count = count;
        }
    }

    /**
     * Build a k-nearest-neighbour graph directly from raw data (cells × features/PCs). Computes
     * approximate nearest neighbors via jvector HNSW-DiskANN then builds an unweighted
     * upper-triangle graph. Analogous to {@code bluster::makeKNNGraph} in R.
     *
     * @param data Matrix (n_cells × n_features)
     * @param k Number of neighbors
     * @param metric Similarity metric: {@code "euclidean"}, {@code "cosine"}, or {@code
     *     "dot_product"}
     * @param numThreads Thread count
     * @return Sparse graph as triplets (i, j, weight=1.0)
     */
    public SNNGraph makeKNNGraph(double[][] data, int k, String metric, int numThreads) {
        return makeKNNGraph(data, k, metric, numThreads, 0);
    }

    public SNNGraph makeKNNGraph(
            double[][] data, int k, String metric, int numThreads, int efSearch) {
        int dims = data[0].length;
        KNNComputer computer = new KNNComputer(dims, metric, efSearch);
        KNNResult knn = computer.computeANN(data, k, numThreads);
        return neighborsToKNNGraph(knn.indices, numThreads);
    }

    /**
     * Build a shared-nearest-neighbour graph directly from raw data (cells × features/PCs).
     * Computes approximate nearest neighbors via jvector HNSW-DiskANN then builds a weighted SNN
     * graph. Analogous to {@code bluster::makeSNNGraph} in R.
     *
     * @param data Matrix (n_cells × n_features)
     * @param k Number of neighbors
     * @param metric Similarity metric: {@code "euclidean"}, {@code "cosine"}, or {@code
     *     "dot_product"}
     * @param type SNN weighting: {@code "rank"}, {@code "jaccard"}, or {@code "number"}
     * @param numThreads Thread count
     * @return Sparse graph as triplets (i, j, weight)
     */
    public SNNGraph makeSNNGraph(
            double[][] data, int k, String metric, String type, int numThreads) {
        return makeSNNGraph(data, k, metric, type, numThreads, 0);
    }

    public SNNGraph makeSNNGraph(
            double[][] data, int k, String metric, String type, int numThreads, int efSearch) {
        int dims = data[0].length;
        KNNComputer computer = new KNNComputer(dims, metric, efSearch);
        KNNResult knn = computer.computeANN(data, k, numThreads);
        return neighborsToSNNGraph(knn.indices, type, numThreads);
    }

    /**
     * Build a simple k-nearest-neighbour graph (unweighted, upper triangle) from k-NN indices. Each
     * edge (i, j) with i &lt; j has weight 1.0.
     *
     * @param knnIndices k-NN indices (n_points x k)
     * @param numThreads Thread count for parallel construction
     * @return Sparse graph as triplets (i, j, weight=1.0)
     */
    public SNNGraph neighborsToKNNGraph(int[][] knnIndices, int numThreads) {
        int n = knnIndices.length;

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        int blockSize = Math.max(1, (n + numThreads - 1) / numThreads);
        List<Future<TaskResult>> futures = new ArrayList<>();

        for (int start = 0; start < n; start += blockSize) {
            final int blockStart = start;
            final int blockEnd = Math.min(start + blockSize, n);

            futures.add(
                    executor.submit(
                            (Callable<TaskResult>)
                                    () -> {
                                        int edgeCap = 16;
                                        int[] localI = new int[edgeCap];
                                        int[] localJ = new int[edgeCap];
                                        double[] localX = new double[edgeCap];
                                        int edgeCount = 0;

                                        for (int i = blockStart; i < blockEnd; i++) {
                                            for (int j : knnIndices[i]) {
                                                if (j <= i) continue;
                                                if (edgeCount >= edgeCap) {
                                                    edgeCap *= 2;
                                                    localI = Arrays.copyOf(localI, edgeCap);
                                                    localJ = Arrays.copyOf(localJ, edgeCap);
                                                    localX = Arrays.copyOf(localX, edgeCap);
                                                }
                                                localI[edgeCount] = i;
                                                localJ[edgeCount] = j;
                                                localX[edgeCount] = 1.0;
                                                edgeCount++;
                                            }
                                        }

                                        return new TaskResult(
                                                Arrays.copyOf(localI, edgeCount),
                                                Arrays.copyOf(localJ, edgeCount),
                                                Arrays.copyOf(localX, edgeCount),
                                                edgeCount);
                                    }));
        }

        executor.shutdown();
        return mergeResults(futures, n);
    }

    /**
     * Build SNN (Shared Nearest Neighbor) graph from k-NN indices.
     *
     * <p>Uses reverse k-NN enumeration to find ALL pairs (i,j) with non-empty k-NN intersection,
     * not just direct k-NN edges. This matches the algorithm in bluster::makeSNNGraph (R).
     *
     * @param knnIndices k-NN indices (n_points x k)
     * @param type Weighting scheme: "rank", "jaccard", "number"
     * @param numThreads Thread count for parallel construction
     * @return Sparse graph as triplets (i, j, weight)
     */
    public SNNGraph neighborsToSNNGraph(int[][] knnIndices, String type, int numThreads) {
        int n = knnIndices.length;
        int k = knnIndices[0].length;

        // Early type validation (synchronous, before threads)
        final String typeLower = type.toLowerCase();
        if (!typeLower.equals("rank")
                && !typeLower.equals("jaccard")
                && !typeLower.equals("number"))
            throw new IllegalArgumentException("Unknown SNN type: " + type);

        // Build reverse k-NN with pre-computed ranks (O(1) rank lookup in inner loop)
        final ReverseKNNResult rev = buildReverseKNNWithRanks(knnIndices, n);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        int blockSize = Math.max(1, (n + numThreads - 1) / numThreads);
        List<Future<TaskResult>> futures = new ArrayList<>();

        for (int start = 0; start < n; start += blockSize) {
            final int blockStart = start;
            final int blockEnd = Math.min(start + blockSize, n);

            futures.add(
                    executor.submit(
                            (Callable<TaskResult>)
                                    () -> {
                                        // L1-resident open-addressing hash table.
                                        // Size: nextPow2(4*k²) slots. For k=15 → 1024 slots =
                                        // 16 KB → always fits in L1 regardless of n.
                                        // Stride-4 slot layout: [key(j+1), sharedCnt, minRankSum,
                                        // pad]; key=0 means empty. Hash: (j*GOLDEN)&htMask.
                                        // JIT proves slot∈[0,htMask] via & ops → eliminates all
                                        // bounds checks on ht[base..base+2].
                                        int htCap =
                                                Math.max(
                                                        16,
                                                        Integer.highestOneBit(k * k * 4 - 1) << 1);
                                        int htMask = htCap - 1;
                                        int[] ht = new int[htCap << 2]; // htCap*4 ints

                                        // Touched slot indices for O(k²) reset and emission
                                        int tCap = k * k + 2 * k + 16;
                                        int[] touchedSlots = new int[tCap];
                                        int tCount = 0;

                                        // Thread-local edge buffers (grow by doubling)
                                        int edgeCap = 16;
                                        int[] localI = new int[edgeCap];
                                        int[] localJ = new int[edgeCap];
                                        double[] localX = new double[edgeCap];
                                        int edgeCount = 0;

                                        for (int i = blockStart; i < blockEnd; i++) {
                                            // Reset only touched slots (O(k²), not O(htCap))
                                            for (int t = 0; t < tCount; t++)
                                                ht[touchedSlots[t] << 2] = 0; // clear key
                                            tCount = 0;

                                            int[] iKNN = knnIndices[i];

                                            // Main loop: transitive shared neighbors via s
                                            for (int si = 0; si < iKNN.length; si++) {
                                                int s = iKNN[si];
                                                int rankI = si + 1;
                                                int rStart = rev.offsets[s],
                                                        rEnd = rev.offsets[s + 1];
                                                for (int m = rStart; m < rEnd; m += 2) {
                                                    int j = rev.data[m];
                                                    if (j <= i) continue;
                                                    int rankJ = rev.data[m + 1];
                                                    int slot = htProbe(ht, htMask, j);
                                                    int base = slot << 2;
                                                    if (ht[base] == 0) {
                                                        ht[base] = j + 1;
                                                        ht[base + 1] = 0; // reset sharedCnt
                                                        if (tCount >= tCap) {
                                                            tCap *= 2;
                                                            touchedSlots =
                                                                    Arrays.copyOf(
                                                                            touchedSlots, tCap);
                                                        }
                                                        touchedSlots[tCount++] = slot;
                                                        ht[base + 2] = Integer.MAX_VALUE;
                                                    }
                                                    ht[base + 1]++;
                                                    int rs = rankI + rankJ;
                                                    if (rs < ht[base + 2]) ht[base + 2] = rs;
                                                }
                                            }

                                            // Case 1: j ∈ KNN(i), j > i → rankJ = 0 (self-rank)
                                            for (int si = 0; si < iKNN.length; si++) {
                                                int j = iKNN[si];
                                                if (j <= i) continue;
                                                int slot = htProbe(ht, htMask, j);
                                                int base = slot << 2;
                                                if (ht[base] == 0) {
                                                    ht[base] = j + 1;
                                                    ht[base + 1] = 0; // reset sharedCnt
                                                    if (tCount >= tCap) {
                                                        tCap *= 2;
                                                        touchedSlots =
                                                                Arrays.copyOf(touchedSlots, tCap);
                                                    }
                                                    touchedSlots[tCount++] = slot;
                                                    ht[base + 2] = Integer.MAX_VALUE;
                                                }
                                                ht[base + 1]++;
                                                int rankI = si + 1; // rankJ = 0 (self-rank of j)
                                                if (rankI < ht[base + 2]) ht[base + 2] = rankI;
                                            }

                                            // Case 2: i ∈ KNN(j), j > i → rankI = 0 (self-rank)
                                            int rStartI = rev.offsets[i],
                                                    rEndI = rev.offsets[i + 1];
                                            for (int m = rStartI; m < rEndI; m += 2) {
                                                int j = rev.data[m];
                                                if (j <= i) continue;
                                                int rankJ =
                                                        rev.data[m + 1]; // rankI = 0 (self-rank)
                                                int slot = htProbe(ht, htMask, j);
                                                int base = slot << 2;
                                                if (ht[base] == 0) {
                                                    ht[base] = j + 1;
                                                    ht[base + 1] = 0; // reset sharedCnt
                                                    if (tCount >= tCap) {
                                                        tCap *= 2;
                                                        touchedSlots =
                                                                Arrays.copyOf(touchedSlots, tCap);
                                                    }
                                                    touchedSlots[tCount++] = slot;
                                                    ht[base + 2] = Integer.MAX_VALUE;
                                                }
                                                ht[base + 1]++;
                                                if (rankJ < ht[base + 2]) ht[base + 2] = rankJ;
                                            }

                                            // Emit edges for all touched slots
                                            for (int t = 0; t < tCount; t++) {
                                                int base = touchedSlots[t] << 2;
                                                int j = ht[base] - 1;
                                                int cnt = ht[base + 1];
                                                int minRS = ht[base + 2];
                                                double weight =
                                                        switch (typeLower) {
                                                            case "rank" ->
                                                                    Math.max(
                                                                            k - (minRS / 2.0),
                                                                            1e-6);
                                                            case "jaccard" ->
                                                                    (double) cnt
                                                                            / (2 * (k + 1) - cnt);
                                                            case "number" -> (double) cnt;
                                                            default ->
                                                                    throw new IllegalArgumentException(
                                                                            "Unknown SNN type: "
                                                                                    + typeLower);
                                                        };
                                                if (weight > 0) {
                                                    if (edgeCount >= edgeCap) {
                                                        edgeCap *= 2;
                                                        localI = Arrays.copyOf(localI, edgeCap);
                                                        localJ = Arrays.copyOf(localJ, edgeCap);
                                                        localX = Arrays.copyOf(localX, edgeCap);
                                                    }
                                                    localI[edgeCount] = i;
                                                    localJ[edgeCount] = j;
                                                    localX[edgeCount] = weight;
                                                    edgeCount++;
                                                }
                                            }
                                        }

                                        return new TaskResult(
                                                Arrays.copyOf(localI, edgeCount),
                                                Arrays.copyOf(localJ, edgeCount),
                                                Arrays.copyOf(localX, edgeCount),
                                                edgeCount);
                                    }));
        }

        executor.shutdown();
        return mergeResults(futures, n);
    }

    /**
     * Open-addressing hash probe (linear probing, Fibonacci hash). Returns the slot index for j in
     * the hash table ht. The slot is either an existing entry with key==j+1, or an empty slot
     * (key==0) for first insertion. slot is always ∈ [0,htMask], so callers can safely access
     * ht[slot&lt;&lt;2], ht[(slot&lt;&lt;2)+1], ht[(slot&lt;&lt;2)+2] without bounds checks after
     * JIT inlines this method.
     */
    private static int htProbe(int[] ht, int htMask, int j) {
        int jKey = j + 1;
        int slot = (j * 0x9E3779B9) & htMask; // Fibonacci hashing
        int key;
        while ((key = ht[slot << 2]) != 0 && key != jKey) slot = (slot + 1) & htMask;
        return slot;
    }

    /**
     * Build reverse k-NN index as a flat CSR structure with interleaved (node,rank) pairs. For each
     * shared neighbor s, data[offsets[s]..offsets[s+1]) contains alternating node/rank values for
     * every i where s ∈ KNN(i). Flat layout eliminates per-s pointer dereferences and keeps
     * (node,rank) pairs in the same cache line.
     */
    private ReverseKNNResult buildReverseKNNWithRanks(int[][] knnIndices, int n) {
        // Pass 1: count entries per s
        int[] counts = new int[n];
        for (int[] row : knnIndices) for (int s : row) counts[s]++;

        // Pass 2: exclusive prefix sum for CSR offsets (×2 for interleaved node+rank pairs)
        int[] offsets = new int[n + 1];
        for (int s = 0; s < n; s++) offsets[s + 1] = offsets[s] + counts[s] * 2;

        // Pass 3: fill data; reuse counts[] as per-s write cursor (in ints, step 2 per pair)
        int[] data = new int[offsets[n]];
        Arrays.fill(counts, 0);
        for (int i = 0; i < knnIndices.length; i++)
            for (int si = 0; si < knnIndices[i].length; si++) {
                int s = knnIndices[i][si];
                int idx = offsets[s] + counts[s];
                data[idx] = i; // node
                data[idx + 1] = si + 1; // 1-indexed rank of s in KNN(i)
                counts[s] += 2;
            }
        return new ReverseKNNResult(data, offsets);
    }

    private SNNGraph mergeResults(List<Future<TaskResult>> futures, int n) {
        List<TaskResult> results = new ArrayList<>(futures.size());
        try {
            for (Future<TaskResult> f : futures) {
                try {
                    results.add(f.get());
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause();
                    if (cause instanceof RuntimeException re) throw re;
                    throw new RuntimeException(cause);
                }
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        int total = 0;
        for (TaskResult r : results) total += r.count;
        int[] finalI = new int[total];
        int[] finalJ = new int[total];
        double[] finalX = new double[total];
        int offset = 0;
        for (TaskResult r : results) {
            System.arraycopy(r.rowI, 0, finalI, offset, r.count);
            System.arraycopy(r.colJ, 0, finalJ, offset, r.count);
            System.arraycopy(r.weights, 0, finalX, offset, r.count);
            offset += r.count;
        }
        return new SNNGraph(finalI, finalJ, finalX, n);
    }

    public static class SNNGraph {
        public final int[] i; // Row indices
        public final int[] j; // Column indices
        public final double[] x; // Edge weights
        public final int n; // Number of nodes

        public SNNGraph(int[] i, int[] j, double[] x, int n) {
            this.i = i;
            this.j = j;
            this.x = x;
            this.n = n;
        }
    }
}
