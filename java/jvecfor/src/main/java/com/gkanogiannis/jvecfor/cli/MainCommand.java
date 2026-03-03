package com.gkanogiannis.jvecfor.cli;

import com.gkanogiannis.jvecfor.KNNComputer;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * CLI entry point. Computes k-nearest-neighbour indices and distances from a vector matrix supplied
 * via {@code -i <file>} or stdin (rows=cells, cols=features, tab-separated, no header), writing an
 * R-friendly interleaved n×(k + k) tab-separated matrix to stdout.
 *
 * <p>Output format: each row contains k 1-based indices followed (when {@code --output-dist} is
 * set) by k float distances.
 */
@Command(
        name = "jvecfor",
        mixinStandardHelpOptions = true,
        version = "jvecfor 0.1.0",
        description = "Compute KNN indices from a vector matrix (rows=cells, cols=features).")
public class MainCommand implements Callable<Integer> {

    @Option(
            names = {"-k", "--k"},
            description = "Number of neighbors (default: ${DEFAULT-VALUE}).",
            defaultValue = "15")
    private int k;

    @Option(
            names = {"-m", "--metric"},
            description =
                    "Similarity metric: euclidean, cosine, dot_product (default:"
                            + " ${DEFAULT-VALUE}).",
            defaultValue = "euclidean")
    private String metric;

    @Option(
            names = {"-t", "--threads"},
            description = "Thread count (default: available processors).")
    private int numThreads = Runtime.getRuntime().availableProcessors();

    @Option(
            names = {"--output-dist"},
            description = "Include distances in stdout output.")
    private boolean outputDist;

    @Option(
            names = {"-g", "--graph-type"},
            description =
                    "Computation method: ann (approximate, jvector HNSW-DiskANN default),"
                            + " knn (exact, VP-tree).",
            defaultValue = "ann")
    private String graphType;

    @Option(
            names = {"--ef-search"},
            description =
                    "HNSW-DiskANN search beam width for ANN (0 = auto: max(k+1, 3*k),"
                            + " default: ${DEFAULT-VALUE}).",
            defaultValue = "0")
    private int efSearch;

    @Option(
            names = {"-M"},
            description =
                    "HNSW-DiskANN max connections per node (default: ${DEFAULT-VALUE})."
                            + " Higher values (e.g. 32) improve recall for high-dimensional data"
                            + " at the cost of more memory and a slower build.",
            defaultValue = "16")
    private int M;

    @Option(
            names = {"--oversample-factor"},
            description =
                    "Oversampling multiplier for ANN beam width (default: ${DEFAULT-VALUE})."
                            + " When > 1.0, fetches ceil(ef * factor) candidates and returns top k."
                            + " E.g. 2.0 doubles the candidates searched. No effect on exact KNN.",
            defaultValue = "1.0")
    private float oversampleFactor;

    @Option(
            names = {"--verbose"},
            description = "Enable HNSW-DiskANN build progress logging (default: ${DEFAULT-VALUE}).",
            defaultValue = "false")
    private boolean verbose;

    @Option(
            names = {"--pq-subspaces"},
            description =
                    "Enable Product Quantization for ANN search with this many subspaces"
                            + " (0 = disabled, default: ${DEFAULT-VALUE})."
                            + " Typical: dims/2 (e.g. 25 for 50-dim PCA)."
                            + " Reduces search time ~4-8x with minimal recall loss via"
                            + " approximate graph traversal + exact reranking.",
            defaultValue = "0")
    private int pqSubspaces;

    @Option(
            names = {"-i", "--input"},
            description =
                    "Input TSV file (rows=observations, cols=features, no header)."
                            + " If omitted, reads from stdin.")
    private String inputFile;

    @Override
    public Integer call() throws IOException {
        double[][] data = readInput();
        if (data.length == 0) {
            System.err.println("Error: no data rows in input.");
            return 1;
        }
        int dims = data[0].length;
        KNNComputer.KNNResult result;
        if ("knn".equalsIgnoreCase(graphType)) {
            result = new KNNComputer(dims, metric).computeKNN(data, k, numThreads);
        } else { // "ann" (default)
            result =
                    new KNNComputer(
                                    dims,
                                    metric,
                                    efSearch,
                                    M,
                                    oversampleFactor,
                                    verbose,
                                    pqSubspaces)
                            .computeANN(data, k, numThreads);
        }
        writeOutput(result, outputDist);
        return 0;
    }

    private double[][] readInput() throws IOException {
        List<double[]> rows = new ArrayList<>();
        BufferedReader br =
                (inputFile != null)
                        ? Files.newBufferedReader(Path.of(inputFile), StandardCharsets.UTF_8)
                        : new BufferedReader(
                                new InputStreamReader(System.in, StandardCharsets.UTF_8));
        try (br) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] parts = line.split("\t");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) row[i] = Double.parseDouble(parts[i].trim());
                rows.add(row);
            }
        }
        return rows.toArray(new double[0][]);
    }

    private void writeOutput(KNNComputer.KNNResult result, boolean includeDist) {
        try (PrintWriter w = stdoutWriter()) {
            for (int i = 0; i < result.indices.length; i++) {
                StringBuilder sb = new StringBuilder();
                for (int jj = 0; jj < result.indices[i].length; jj++) {
                    if (jj > 0) sb.append('\t');
                    sb.append(result.indices[i][jj] + 1); // +1 for R 1-based indexing
                }
                if (includeDist) {
                    for (int jj = 0; jj < result.distances[i].length; jj++) {
                        sb.append('\t');
                        sb.append(String.format("%.6f", result.distances[i][jj]));
                    }
                }
                w.println(sb);
            }
        }
    }

    private PrintWriter stdoutWriter() {
        return new PrintWriter(System.out) {
            @Override
            public void close() {
                flush();
            }
        };
    }
}
