package com.gkanogiannis.jvecfor.cli;

import com.gkanogiannis.jvecfor.KNNComputer;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
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
                    "Input file (rows=observations, cols=features)."
                            + " If omitted, reads from stdin.")
    private String inputFile;

    @Option(
            names = {"--format"},
            description =
                    "Input format: tsv (tab-separated, default), mtx (MatrixMarket"
                            + " coordinate), bin (row-major binary doubles).",
            defaultValue = "tsv")
    private String format;

    @Option(
            names = {"--output-format"},
            description =
                    "Output format: text (tab-separated, default), bin (binary"
                            + " int32 indices + float64 distances).",
            defaultValue = "text")
    private String outputFormat;

    @Override
    public Integer call() throws IOException {
        double[][] data;
        switch (format.toLowerCase()) {
            case "mtx":
                data = readMTXInput();
                break;
            case "bin":
                data = readBinaryInput();
                break;
            case "tsv":
            default:
                data = readInput();
                break;
        }
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
        if ("bin".equalsIgnoreCase(outputFormat)) {
            writeBinaryOutput(result, outputDist);
        } else {
            writeOutput(result, outputDist);
        }
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

    private double[][] readMTXInput() throws IOException {
        BufferedReader br =
                (inputFile != null)
                        ? Files.newBufferedReader(Path.of(inputFile), StandardCharsets.UTF_8)
                        : new BufferedReader(
                                new InputStreamReader(System.in, StandardCharsets.UTF_8));
        try (br) {
            // Read and validate header
            String line = br.readLine();
            if (line == null || !line.startsWith("%%MatrixMarket"))
                throw new IOException("Invalid MatrixMarket header");

            // Skip comment lines
            while ((line = br.readLine()) != null && line.startsWith("%")) {
                // skip
            }

            // Parse dimensions: nrow ncol nnz
            if (line == null) throw new IOException("Missing MTX dimension line");
            String[] dims = line.trim().split("\\s+");
            int nrow = Integer.parseInt(dims[0]);
            int ncol = Integer.parseInt(dims[1]);

            // Allocate dense array (zero-initialized by Java)
            double[][] data = new double[nrow][ncol];

            // Read coordinate entries: row col value (1-indexed)
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] parts = line.trim().split("\\s+");
                int r = Integer.parseInt(parts[0]) - 1;
                int c = Integer.parseInt(parts[1]) - 1;
                double v = Double.parseDouble(parts[2]);
                data[r][c] = v;
            }
            return data;
        }
    }

    private double[][] readBinaryInput() throws IOException {
        DataInputStream dis;
        if (inputFile != null) {
            dis = new DataInputStream(new BufferedInputStream(new FileInputStream(inputFile)));
        } else {
            dis = new DataInputStream(new BufferedInputStream(System.in));
        }
        try (dis) {
            int nrow = dis.readInt();
            int ncol = dis.readInt();
            double[][] data = new double[nrow][ncol];
            for (int i = 0; i < nrow; i++) {
                for (int j = 0; j < ncol; j++) {
                    data[i][j] = dis.readDouble();
                }
            }
            return data;
        }
    }

    private void writeBinaryOutput(KNNComputer.KNNResult result, boolean includeDist)
            throws IOException {
        DataOutputStream dos =
                new DataOutputStream(new BufferedOutputStream(System.out));
        int n = result.indices.length;
        int kk = result.indices[0].length;
        dos.writeInt(n);
        dos.writeInt(kk);
        for (int i = 0; i < n; i++) {
            for (int jj = 0; jj < kk; jj++) {
                dos.writeInt(result.indices[i][jj] + 1); // 1-based for R
            }
        }
        if (includeDist) {
            for (int i = 0; i < n; i++) {
                for (int jj = 0; jj < kk; jj++) {
                    dos.writeDouble(result.distances[i][jj]);
                }
            }
        }
        dos.flush();
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
