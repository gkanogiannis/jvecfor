package com.gkanogiannis.jvecfor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class AppTest {

    // -- Helper: generate TSV string for n points in d dims -------------------

    private static String makeTsv(int n, int d) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                if (j > 0) sb.append('\t');
                sb.append(i * 0.1 + j);
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    // -- Helper: generate binary input bytes (big-endian int32 header + row-major float64)

    private static byte[] makeBinary(int n, int d) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeInt(n);
        dos.writeInt(d);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                dos.writeDouble(i * 0.1 + j);
            }
        }
        dos.flush();
        return baos.toByteArray();
    }

    // -- Helper: generate MTX string (MatrixMarket coordinate format)

    private static String makeMtx(int n, int d) {
        StringBuilder sb = new StringBuilder();
        sb.append("%%MatrixMarket matrix coordinate real general\n");
        // Count non-zero entries (all entries in our grid are non-zero for d>=1)
        int nnz = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double val = i * 0.1 + j;
                if (val != 0.0) nnz++;
            }
        }
        sb.append(n).append(' ').append(d).append(' ').append(nnz).append('\n');
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double val = i * 0.1 + j;
                if (val != 0.0) {
                    sb.append(i + 1).append(' ').append(j + 1).append(' ').append(val).append('\n');
                }
            }
        }
        return sb.toString();
    }

    // =========================================================================
    // Existing basic CLI tests
    // =========================================================================

    @Test
    void helpOptionExitsZero() {
        int exitCode = App.run(new String[] {"--help"});
        assertEquals(0, exitCode, "Help option should exit with code 0.");
    }

    @Test
    void unknownOptionFails() {
        int exitCode = App.run(new String[] {"--unknown-option"});
        assertNotEquals(0, exitCode, "Unknown option should fail.");
    }

    @Test
    void stdinMode_validData_exitsZero() throws Exception {
        String tsv = "1.0\t2.0\n3.0\t4.0\n5.0\t6.0\n7.0\t8.0\n";
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(tsv.getBytes(StandardCharsets.UTF_8)));
            int exitCode = App.run(new String[] {"-k", "2"});
            assertEquals(0, exitCode, "Valid stdin data should exit 0.");
        } finally {
            System.setIn(original);
        }
    }

    @Test
    void stdinMode_emptyInput_exitsNonZero() throws Exception {
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(new byte[0]));
            int exitCode = App.run(new String[] {"-k", "2"});
            assertNotEquals(0, exitCode, "Empty stdin should fail.");
        } finally {
            System.setIn(original);
        }
    }

    // =========================================================================
    // TSV file input (--format tsv -i <file>)
    // =========================================================================

    @Test
    void tsvFileInput_exitsZero(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);
        int exitCode =
                App.run(
                        new String[] {
                            "-k", "2", "--format", "tsv", "-i", file.toString()
                        });
        assertEquals(0, exitCode, "TSV file input should exit 0.");
    }

    // =========================================================================
    // Binary input (--format bin)
    // =========================================================================

    @Test
    void binaryStdinInput_exitsZero() throws Exception {
        byte[] bin = makeBinary(10, 3);
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(bin));
            int exitCode = App.run(new String[] {"-k", "2", "--format", "bin"});
            assertEquals(0, exitCode, "Binary stdin input should exit 0.");
        } finally {
            System.setIn(original);
        }
    }

    @Test
    void binaryFileInput_exitsZero(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.bin");
        Files.write(file, makeBinary(10, 3));
        int exitCode =
                App.run(
                        new String[] {
                            "-k", "2", "--format", "bin", "-i", file.toString()
                        });
        assertEquals(0, exitCode, "Binary file input should exit 0.");
    }

    @Test
    void binaryInput_exactKNN_exitsZero() throws Exception {
        byte[] bin = makeBinary(10, 3);
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(bin));
            int exitCode =
                    App.run(new String[] {"-k", "2", "--format", "bin", "-g", "knn"});
            assertEquals(0, exitCode, "Binary input with exact KNN should exit 0.");
        } finally {
            System.setIn(original);
        }
    }

    // =========================================================================
    // MTX input (--format mtx)
    // =========================================================================

    @Test
    void mtxStdinInput_exitsZero() throws Exception {
        String mtx = makeMtx(10, 3);
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(mtx.getBytes(StandardCharsets.UTF_8)));
            int exitCode = App.run(new String[] {"-k", "2", "--format", "mtx"});
            assertEquals(0, exitCode, "MTX stdin input should exit 0.");
        } finally {
            System.setIn(original);
        }
    }

    @Test
    void mtxFileInput_exitsZero(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.mtx");
        Files.writeString(file, makeMtx(10, 3), StandardCharsets.UTF_8);
        int exitCode =
                App.run(
                        new String[] {
                            "-k", "2", "--format", "mtx", "-i", file.toString()
                        });
        assertEquals(0, exitCode, "MTX file input should exit 0.");
    }

    @Test
    void mtxInput_withCommentLines_exitsZero() throws Exception {
        // MTX with comment lines between header and dimensions
        String mtx =
                "%%MatrixMarket matrix coordinate real general\n"
                        + "% This is a comment\n"
                        + "% Another comment\n"
                        + "4 2 4\n"
                        + "1 1 1.0\n"
                        + "2 1 2.0\n"
                        + "3 2 3.0\n"
                        + "4 2 4.0\n";
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(mtx.getBytes(StandardCharsets.UTF_8)));
            int exitCode = App.run(new String[] {"-k", "2", "--format", "mtx"});
            assertEquals(0, exitCode, "MTX with comments should exit 0.");
        } finally {
            System.setIn(original);
        }
    }

    @Test
    void mtxInput_sparseData_exitsZero() throws Exception {
        // Sparse 6x4 matrix with only a few non-zero entries
        String mtx =
                "%%MatrixMarket matrix coordinate real general\n"
                        + "6 4 6\n"
                        + "1 1 1.0\n"
                        + "2 2 2.0\n"
                        + "3 3 3.0\n"
                        + "4 4 4.0\n"
                        + "5 1 5.0\n"
                        + "6 2 6.0\n";
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(mtx.getBytes(StandardCharsets.UTF_8)));
            int exitCode = App.run(new String[] {"-k", "2", "--format", "mtx"});
            assertEquals(0, exitCode, "Sparse MTX input should exit 0.");
        } finally {
            System.setIn(original);
        }
    }

    // =========================================================================
    // Binary output (--output-format bin)
    // =========================================================================

    @Test
    void binaryOutput_producesValidHeader(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);

        // Capture stdout
        PrintStream originalOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured));
            int exitCode =
                    App.run(
                            new String[] {
                                "-k", "2",
                                "-i", file.toString(),
                                "--output-format", "bin"
                            });
            assertEquals(0, exitCode, "Binary output should exit 0.");
        } finally {
            System.setOut(originalOut);
        }

        // Parse binary output: n (int32) + k (int32) + n*k int32 indices
        byte[] out = captured.toByteArray();
        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(out));
        int n = dis.readInt();
        int k = dis.readInt();
        assertEquals(10, n, "Output n must match input rows.");
        assertEquals(2, k, "Output k must match requested k.");
        // Verify we can read all n*k indices
        for (int i = 0; i < n * k; i++) {
            int idx = dis.readInt();
            assertTrue(idx >= 1 && idx <= n, "Index must be 1-based and in range.");
        }
    }

    @Test
    void binaryOutput_withDistances(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);

        PrintStream originalOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured));
            int exitCode =
                    App.run(
                            new String[] {
                                "-k", "2",
                                "-i", file.toString(),
                                "--output-format", "bin",
                                "--output-dist"
                            });
            assertEquals(0, exitCode, "Binary output with distances should exit 0.");
        } finally {
            System.setOut(originalOut);
        }

        byte[] out = captured.toByteArray();
        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(out));
        int n = dis.readInt();
        int k = dis.readInt();
        assertEquals(10, n);
        assertEquals(2, k);
        // Read indices
        for (int i = 0; i < n * k; i++) {
            dis.readInt();
        }
        // Read distances — should have n*k doubles
        for (int i = 0; i < n * k; i++) {
            double d = dis.readDouble();
            assertTrue(d >= 0.0, "Distance must be non-negative.");
        }
        // Should be at end of stream
        assertEquals(0, dis.available(), "No extra bytes expected.");
    }

    @Test
    void binaryOutput_withoutDistances_noExtraBytes(@TempDir Path tmpDir)
            throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);

        PrintStream originalOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured));
            int exitCode =
                    App.run(
                            new String[] {
                                "-k", "2",
                                "-i", file.toString(),
                                "--output-format", "bin"
                                // no --output-dist
                            });
            assertEquals(0, exitCode);
        } finally {
            System.setOut(originalOut);
        }

        byte[] out = captured.toByteArray();
        // Expected size: 4 (n) + 4 (k) + 10*2*4 (indices) = 88 bytes
        int expectedSize = 4 + 4 + 10 * 2 * 4;
        assertEquals(expectedSize, out.length, "Binary output without distances.");
    }

    // =========================================================================
    // Text output with --output-dist
    // =========================================================================

    @Test
    void textOutput_withDistances_hasCorrectColumns(@TempDir Path tmpDir)
            throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);

        PrintStream originalOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured));
            int exitCode =
                    App.run(
                            new String[] {
                                "-k", "2",
                                "-i", file.toString(),
                                "--output-dist"
                            });
            assertEquals(0, exitCode);
        } finally {
            System.setOut(originalOut);
        }

        String output = captured.toString(StandardCharsets.UTF_8);
        String[] lines = output.strip().split("\n");
        assertEquals(10, lines.length, "Output must have n lines.");
        // Each line should have k indices + k distances = 2*k columns
        for (String line : lines) {
            String[] cols = line.split("\t");
            assertEquals(4, cols.length, "Each line must have 2*k=4 columns.");
        }
    }

    // =========================================================================
    // Cross-format consistency: binary input produces same results as TSV
    // =========================================================================

    @Test
    void binaryAndTsvInput_sameResults(@TempDir Path tmpDir) throws Exception {
        int n = 10, d = 3;

        // TSV run
        Path tsvFile = tmpDir.resolve("data.tsv");
        Files.writeString(tsvFile, makeTsv(n, d), StandardCharsets.UTF_8);

        PrintStream originalOut = System.out;
        ByteArrayOutputStream tsvOut = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(tsvOut));
            App.run(
                    new String[] {
                        "-k", "2", "-g", "knn",
                        "--format", "tsv",
                        "-i", tsvFile.toString(),
                        "--output-dist"
                    });
        } finally {
            System.setOut(originalOut);
        }

        // Binary run
        Path binFile = tmpDir.resolve("data.bin");
        Files.write(binFile, makeBinary(n, d));

        ByteArrayOutputStream binOut = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(binOut));
            App.run(
                    new String[] {
                        "-k", "2", "-g", "knn",
                        "--format", "bin",
                        "-i", binFile.toString(),
                        "--output-dist"
                    });
        } finally {
            System.setOut(originalOut);
        }

        // Both should produce identical text output (exact KNN is deterministic)
        assertEquals(
                tsvOut.toString(StandardCharsets.UTF_8),
                binOut.toString(StandardCharsets.UTF_8),
                "Binary and TSV input must produce identical exact KNN results.");
    }

    // =========================================================================
    // Metric and graph-type CLI options
    // =========================================================================

    @Test
    void cosineMetric_exitsZero(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);
        int exitCode =
                App.run(
                        new String[] {
                            "-k", "2", "-m", "cosine", "-i", file.toString()
                        });
        assertEquals(0, exitCode, "Cosine metric should exit 0.");
    }

    @Test
    void dotProductMetric_ann_exitsZero(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);
        int exitCode =
                App.run(
                        new String[] {
                            "-k", "2", "-m", "dot_product", "-g", "ann",
                            "-i", file.toString()
                        });
        assertEquals(0, exitCode, "Dot-product ANN should exit 0.");
    }

    @Test
    void exactKNN_exitsZero(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);
        int exitCode =
                App.run(
                        new String[] {
                            "-k", "2", "-g", "knn", "-i", file.toString()
                        });
        assertEquals(0, exitCode, "Exact KNN should exit 0.");
    }

    @Test
    void verboseFlag_exitsZero(@TempDir Path tmpDir) throws Exception {
        Path file = tmpDir.resolve("data.tsv");
        Files.writeString(file, makeTsv(10, 3), StandardCharsets.UTF_8);
        int exitCode =
                App.run(
                        new String[] {
                            "-k", "2", "--verbose", "-i", file.toString()
                        });
        assertEquals(0, exitCode, "Verbose flag should exit 0.");
    }
}
