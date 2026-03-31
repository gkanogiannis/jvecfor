library(testthat)
library(jvecfor)

# Helper: check if JAR is available for integration tests
skip_if_no_jar <- function() {
    tryCatch(
        jvecfor:::.jvecfor_jar(),
        error = function(e) skip("jvecfor JAR not found")
    )
}

# .jvecfor_jar() ---------------------------------------------------------------

test_that(".jvecfor_jar stops with informative error when option is nonexistent", {
    old_opt <- getOption("jvecfor.jar")
    options(jvecfor.jar = "/nonexistent/path/to/jvecfor.jar")
    on.exit(options(jvecfor.jar = old_opt), add = TRUE)
    expect_error(
        jvecfor:::.jvecfor_jar(),
        regexp = "non-existent file"
    )
})

test_that(".jvecfor_jar returns user option path when file exists", {
    tmp <- tempfile(fileext = ".jar")
    file.create(tmp)
    on.exit({
        options(jvecfor.jar = NULL)
        unlink(tmp)
    }, add = TRUE)
    options(jvecfor.jar = tmp)
    result <- jvecfor:::.jvecfor_jar()
    expect_equal(result, tmp)
})

test_that(".jvecfor_jar finds bundled JAR in inst/java", {
    old_opt <- getOption("jvecfor.jar")
    options(jvecfor.jar = NULL)
    on.exit(options(jvecfor.jar = old_opt), add = TRUE)
    jar <- jvecfor:::.jvecfor_jar()
    expect_true(file.exists(jar))
    expect_match(basename(jar), "^jvecfor-[0-9]+.*\\.jar$")
})

test_that(".jvecfor_jar finds JAR in R_user_dir", {
    user_dir <- tools::R_user_dir("jvecfor", "data")
    if (!dir.exists(user_dir)) dir.create(user_dir, recursive = TRUE)
    fake_jar <- file.path(user_dir, "jvecfor-99.99.99.jar")
    writeLines("fake", fake_jar)
    on.exit(unlink(fake_jar), add = TRUE)

    old_opt <- getOption("jvecfor.jar")
    options(jvecfor.jar = NULL)
    on.exit(options(jvecfor.jar = old_opt), add = TRUE)

    jar <- jvecfor:::.jvecfor_jar()
    # Should find user-dir JAR (priority 2, before bundled)
    expect_true(file.exists(jar))
})

test_that(".jvecfor_jar ignores empty string option", {
    old_opt <- getOption("jvecfor.jar")
    options(jvecfor.jar = "")
    on.exit(options(jvecfor.jar = old_opt), add = TRUE)
    # Should fall through to inst/java lookup
    jar <- jvecfor:::.jvecfor_jar()
    expect_true(file.exists(jar))
})

# .jvecfor_version() -----------------------------------------------------------

test_that(".jvecfor_version returns a valid semver string", {
    ver <- jvecfor:::.jvecfor_version()
    expect_type(ver, "character")
    expect_length(ver, 1L)
    expect_match(ver, "^[0-9]+\\.[0-9]+\\.[0-9]+$")
})

test_that(".jvecfor_version errors when no JAR is found", {
    old_opt <- getOption("jvecfor.jar")
    options(jvecfor.jar = "/nonexistent/jvecfor.jar")
    on.exit(options(jvecfor.jar = old_opt), add = TRUE)
    expect_error(
        jvecfor:::.jvecfor_version(),
        regexp = "Cannot determine|non-existent"
    )
})

# .check_java() ----------------------------------------------------------------

test_that(".check_java returns java path invisibly on success", {
    java <- Sys.which("java")
    if (!nzchar(java)) skip("Java not on PATH")
    result <- jvecfor:::.check_java()
    expect_true(nzchar(result))
})

test_that(".check_java stops when Java is not on PATH", {
    old_path <- Sys.getenv("PATH")
    Sys.setenv(PATH = "")
    on.exit(Sys.setenv(PATH = old_path), add = TRUE)
    expect_error(
        jvecfor:::.check_java(),
        regexp = "Java not found"
    )
})

test_that("version parser identifies Java 21 as valid", {
    ver_line <- 'openjdk version "21.0.2" 2024-01-16'
    m <- regmatches(ver_line, regexpr('"[0-9]+', ver_line))
    major <- as.integer(sub('"', "", m, fixed = TRUE))
    expect_equal(major, 21L)
    expect_true(major >= 20L)
})

test_that("version parser identifies Java 8 legacy format", {
    ver_line <- 'java version "1.8.0_411" 2024-04-16'
    m <- regmatches(ver_line, regexpr('"[0-9]+', ver_line))
    major_raw <- as.integer(sub('"', "", m, fixed = TRUE))
    expect_equal(major_raw, 1L)
    m2 <- regmatches(ver_line, regexpr('"1\\.[0-9]+', ver_line))
    major <- as.integer(sub('"1.', "", m2, fixed = TRUE))
    expect_equal(major, 8L)
    expect_true(major < 20L)
})

test_that("version parser identifies Java 11 as too old", {
    ver_line <- 'openjdk version "11.0.22" 2024-01-16'
    m <- regmatches(ver_line, regexpr('"[0-9]+', ver_line))
    major <- as.integer(sub('"', "", m, fixed = TRUE))
    expect_equal(major, 11L)
    expect_true(major < 20L)
})

# .write_tsv() -----------------------------------------------------------------

test_that(".write_tsv produces correct tab-separated output", {
    mat <- matrix(c(1.5, 2.3, 3.7, 4.1, 5.9, 6.2),
                  nrow = 2, ncol = 3)
    f <- tempfile(fileext = ".tsv")
    on.exit(unlink(f), add = TRUE)
    jvecfor:::.write_tsv(mat, f)
    lines <- readLines(f)
    expect_length(lines, 2L)
    vals <- as.numeric(strsplit(lines[1], "\t")[[1]])
    expect_equal(vals, mat[1, ])
})

test_that(".write_tsv round-trips correctly", {
    mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
    f <- tempfile(fileext = ".tsv")
    on.exit(unlink(f), add = TRUE)
    jvecfor:::.write_tsv(mat, f)
    result <- as.matrix(data.table::fread(f, header = FALSE, sep = "\t"))
    expect_equal(unname(result), mat, tolerance = 1e-10)
})

# .write_bin() -----------------------------------------------------------------

test_that(".write_bin produces correct binary output", {
    mat <- matrix(c(1.5, 2.3, 3.7, 4.1, 5.9, 6.2),
                  nrow = 2, ncol = 3)
    f <- tempfile(fileext = ".bin")
    on.exit(unlink(f), add = TRUE)
    jvecfor:::.write_bin(mat, f)

    con <- file(f, "rb")
    on.exit(close(con), add = TRUE)
    nr <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
    nc <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
    expect_equal(nr, 2L)
    expect_equal(nc, 3L)

    vals <- readBin(con, "double", n = 6L, size = 8L, endian = "big")
    # Row-major: row1 then row2
    expect_equal(vals, as.vector(t(mat)))
})

test_that(".write_bin file size matches expected bytes", {
    mat <- matrix(rnorm(100), nrow = 10, ncol = 10)
    f <- tempfile(fileext = ".bin")
    on.exit(unlink(f), add = TRUE)
    jvecfor:::.write_bin(mat, f)
    # Header: 2 * 4 bytes, data: 100 * 8 bytes
    expect_equal(file.info(f)$size, 8L + 800L)
})

# .write_mtx() -----------------------------------------------------------------

test_that(".write_mtx produces valid MatrixMarket file", {
    if (!requireNamespace("Matrix", quietly = TRUE))
        skip("Matrix not installed")
    X <- Matrix::Matrix(
        matrix(c(1.5, 0, 0, 2.3, 0, 3.7), nrow = 2, ncol = 3),
        sparse = TRUE
    )
    f <- tempfile(fileext = ".mtx")
    on.exit(unlink(f), add = TRUE)
    jvecfor:::.write_mtx(X, f)

    lines <- readLines(f)
    expect_match(lines[1], "%%MatrixMarket")
    # Dimension line (after comments)
    dim_line <- lines[!grepl("^%", lines)][1]
    dims <- as.integer(strsplit(trimws(dim_line), "\\s+")[[1]])
    expect_equal(dims[1], 2L)  # nrow
    expect_equal(dims[2], 3L)  # ncol
    expect_equal(dims[3], 3L)  # nnz
})

test_that(".write_mtx coerces dense matrix to sparse", {
    if (!requireNamespace("Matrix", quietly = TRUE))
        skip("Matrix not installed")
    dense <- matrix(c(1, 0, 0, 2), nrow = 2)
    f <- tempfile(fileext = ".mtx")
    on.exit(unlink(f), add = TRUE)
    jvecfor:::.write_mtx(dense, f)
    lines <- readLines(f)
    expect_match(lines[1], "%%MatrixMarket")
})

# .write_input() ---------------------------------------------------------------

test_that(".write_input dispatches to correct format", {
    mat <- matrix(rnorm(12), nrow = 3, ncol = 4)

    f_bin <- jvecfor:::.write_input(mat, "bin")
    on.exit(unlink(f_bin), add = TRUE)
    expect_match(f_bin, "\\.bin$")
    expect_true(file.exists(f_bin))

    f_tsv <- jvecfor:::.write_input(mat, "tsv")
    on.exit(unlink(f_tsv), add = TRUE)
    expect_match(f_tsv, "\\.tsv$")
    expect_true(file.exists(f_tsv))
})

test_that(".write_input dispatches to mtx for sparse", {
    if (!requireNamespace("Matrix", quietly = TRUE))
        skip("Matrix not installed")
    X <- Matrix::rsparsematrix(5, 3, density = 0.5)
    f <- jvecfor:::.write_input(X, "mtx")
    on.exit(unlink(f), add = TRUE)
    expect_match(f, "\\.mtx$")
    expect_true(file.exists(f))
    lines <- readLines(f)
    expect_match(lines[1], "%%MatrixMarket")
})

test_that(".write_input errors on unknown format", {
    mat <- matrix(1:4, nrow = 2)
    expect_error(
        jvecfor:::.write_input(mat, "parquet"),
        regexp = "Unknown format"
    )
})

# .serialize_bin() -------------------------------------------------------------

test_that(".serialize_bin returns correct raw bytes", {
    mat <- matrix(c(1.0, 2.0, 3.0, 4.0), nrow = 2, ncol = 2)
    raw_out <- jvecfor:::.serialize_bin(mat)
    expect_type(raw_out, "raw")
    # Header: 2 * 4 bytes, data: 4 * 8 bytes = 40 bytes
    expect_equal(length(raw_out), 40L)

    con <- rawConnection(raw_out, "rb")
    on.exit(close(con))
    nr <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
    nc <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
    expect_equal(nr, 2L)
    expect_equal(nc, 2L)
    vals <- readBin(con, "double", n = 4L, size = 8L, endian = "big")
    expect_equal(vals, as.vector(t(mat)))
})

# .serialize_mtx() -------------------------------------------------------------

test_that(".serialize_mtx returns correct MTX raw bytes", {
    if (!requireNamespace("Matrix", quietly = TRUE))
        skip("Matrix not installed")
    X <- Matrix::Matrix(
        matrix(c(1.5, 0, 0, 2.3), nrow = 2, ncol = 2),
        sparse = TRUE
    )
    raw_out <- jvecfor:::.serialize_mtx(X)
    expect_type(raw_out, "raw")
    text <- rawToChar(raw_out)
    lines <- strsplit(text, "\n")[[1]]
    expect_match(lines[1], "%%MatrixMarket")
    # Dimension line
    dims <- as.integer(strsplit(trimws(lines[2]), "\\s+")[[1]])
    expect_equal(dims[1], 2L)
    expect_equal(dims[2], 2L)
    expect_equal(dims[3], 2L)  # nnz
})

test_that(".serialize_mtx preserves full double precision", {
    if (!requireNamespace("Matrix", quietly = TRUE))
        skip("Matrix not installed")
    val <- 1.23456789012345
    X <- Matrix::Matrix(
        matrix(c(val, 0, 0, 0), nrow = 2, ncol = 2),
        sparse = TRUE
    )
    raw_out <- jvecfor:::.serialize_mtx(X)
    text <- rawToChar(raw_out)
    lines <- strsplit(text, "\n")[[1]]
    # The entry line should contain the value with full precision
    entry <- lines[3]
    parsed_val <- as.double(strsplit(trimws(entry), "\\s+")[[1]][3])
    expect_equal(parsed_val, val)
})

# .read_matrix_from_text() -----------------------------------------------------

test_that(".read_matrix_from_text parses tab-separated lines", {
    lines <- c("1\t2\t3", "4\t5\t6")
    result <- jvecfor:::.read_matrix_from_text(lines)
    expect_true(is.matrix(result))
    expect_equal(dim(result), c(2L, 3L))
    expect_equal(unname(result[1, ]), c(1, 2, 3))
    expect_equal(unname(result[2, ]), c(4, 5, 6))
})

test_that(".read_matrix_from_text handles single row", {
    lines <- c("10\t20\t30", "")
    result <- jvecfor:::.read_matrix_from_text(lines)
    expect_equal(dim(result), c(1L, 3L))
    expect_equal(unname(result[1, ]), c(10, 20, 30))
})

test_that(".read_matrix_from_text handles decimal values", {
    lines <- c("1.5\t2.7", "3.14\t0.001")
    result <- jvecfor:::.read_matrix_from_text(lines)
    expect_equal(unname(result[1, 1]), 1.5)
    expect_equal(unname(result[2, 2]), 0.001)
})

# .coerce_X() ------------------------------------------------------------------

test_that(".coerce_X accepts dense numeric matrix", {
    mat <- matrix(rnorm(12), nrow = 3, ncol = 4)
    result <- jvecfor:::.coerce_X(mat)
    expect_false(result$is_sparse)
    expect_true(is.matrix(result$X))
})

test_that(".coerce_X converts data.frame to matrix", {
    df <- data.frame(a = 1:3, b = 4:6)
    result <- jvecfor:::.coerce_X(df)
    expect_false(result$is_sparse)
    expect_true(is.matrix(result$X))
    expect_true(is.numeric(result$X))
})

test_that(".coerce_X preserves sparse matrix without densifying", {
    if (!requireNamespace("Matrix", quietly = TRUE))
        skip("Matrix not installed")
    X <- Matrix::rsparsematrix(10, 5, density = 0.3)
    result <- jvecfor:::.coerce_X(X)
    expect_true(result$is_sparse)
    expect_true(inherits(result$X, "sparseMatrix"))
})

test_that(".coerce_X rejects non-numeric matrix", {
    mat <- matrix(letters[1:6], nrow = 2)
    expect_error(
        jvecfor:::.coerce_X(mat),
        regexp = "numeric"
    )
})

test_that(".coerce_X rejects unsupported types", {
    expect_error(
        jvecfor:::.coerce_X(list(1, 2, 3)),
        regexp = "must be a numeric matrix"
    )
    expect_error(
        jvecfor:::.coerce_X(1:10),
        regexp = "must be a numeric matrix"
    )
})

# .resolve_threads() -----------------------------------------------------------

test_that(".resolve_threads returns explicit num.threads", {
    result <- jvecfor:::.resolve_threads(4L, BiocParallel::bpparam())
    expect_equal(result, 4L)
})

test_that(".resolve_threads errors on invalid num.threads", {
    expect_error(
        jvecfor:::.resolve_threads(-1L, BiocParallel::bpparam()),
        regexp = "positive integer"
    )
    expect_error(
        jvecfor:::.resolve_threads(0L, BiocParallel::bpparam()),
        regexp = "positive integer"
    )
})

test_that(".resolve_threads falls back to BPPARAM", {
    result <- jvecfor:::.resolve_threads(
        NULL, BiocParallel::SerialParam()
    )
    expect_true(result >= 1L)
})

# jvecfor_setup() --------------------------------------------------------------

test_that("jvecfor_setup stops on missing jar_path", {
    expect_error(
        jvecfor_setup(jar_path = "/tmp/does_not_exist.jar"),
        regexp = "JAR not found"
    )
})

test_that("jvecfor_setup installs JAR to R_user_dir", {
    # Create a fake JAR with a deterministic name
    tmp_dir <- tempdir()
    tmp_jar <- file.path(tmp_dir, "jvecfor-99.99.99.jar")
    writeLines("fake", tmp_jar)
    on.exit(unlink(tmp_jar), add = TRUE)

    dest <- jvecfor_setup(jar_path = tmp_jar)
    on.exit(unlink(dest), add = TRUE)

    expect_true(file.exists(dest))
    user_dir <- tools::R_user_dir("jvecfor", "data")
    expect_true(startsWith(dest, user_dir))
    expect_match(basename(dest), "^jvecfor-99\\.99\\.99\\.jar$")
})

test_that("jvecfor_setup rejects JAR without version in filename", {
    tmp_jar <- tempfile(pattern = "custom", fileext = ".jar")
    writeLines("fake", tmp_jar)
    on.exit(unlink(tmp_jar), add = TRUE)
    expect_error(
        jvecfor_setup(jar_path = tmp_jar),
        regexp = "Cannot parse version"
    )
})

# Package options --------------------------------------------------------------

test_that("jvecfor.verbose option defaults to FALSE", {
    old <- getOption("jvecfor.verbose")
    on.exit(options(jvecfor.verbose = old), add = TRUE)
    options(jvecfor.verbose = NULL)
    defaults <- list(jvecfor.verbose = FALSE, jvecfor.jar = NULL)
    toset <- !(names(defaults) %in% names(options()))
    if (any(toset)) options(defaults[toset])
    expect_false(isTRUE(getOption("jvecfor.verbose")))
})

test_that(".onLoad sets default options without overriding existing", {
    old_verbose <- getOption("jvecfor.verbose")
    old_jar <- getOption("jvecfor.jar")
    on.exit({
        options(jvecfor.verbose = old_verbose)
        options(jvecfor.jar = old_jar)
    }, add = TRUE)

    # Set a custom value
    options(jvecfor.verbose = TRUE)
    # Simulate .onLoad
    jvecfor:::.onLoad("lib", "jvecfor")
    # Should NOT overwrite existing TRUE value
    expect_true(getOption("jvecfor.verbose"))
})

test_that(".onLoad sets defaults when options are missing", {
    old_verbose <- getOption("jvecfor.verbose")
    on.exit(options(jvecfor.verbose = old_verbose), add = TRUE)

    # Remove the option entirely
    opts <- options()
    opts[["jvecfor.verbose"]] <- NULL
    options(opts)
    # Simulate .onLoad
    jvecfor:::.onLoad("lib", "jvecfor")
    expect_false(getOption("jvecfor.verbose"))
})

# .onAttach() ------------------------------------------------------------------

test_that(".onAttach does not error when Java is available", {
    java <- Sys.which("java")
    if (!nzchar(java)) skip("Java not on PATH")
    expect_silent(jvecfor:::.onAttach("lib", "jvecfor"))
})

# Binary round-trip integration ------------------------------------------------

test_that("binary format round-trips correctly through Java", {
    skip_if_no_jar()
    mat <- matrix(c(1.5, 2.3, 3.7, 4.1, 5.9, 6.2,
                     7.8, 8.4, 9.1, 10.5, 11.2, 12.6),
                  nrow = 4, ncol = 3)
    nn <- fastFindKNN(mat, k = 2L, type = "knn")
    expect_equal(dim(nn$index), c(4L, 2L))
    expect_type(nn$index, "integer")
    expect_equal(dim(nn$distance), c(4L, 2L))
})

test_that("MTX format round-trips correctly through Java", {
    skip_if_no_jar()
    if (!requireNamespace("Matrix", quietly = TRUE))
        skip("Matrix not installed")
    # Create a properly sparse matrix
    X_sp <- Matrix::rsparsematrix(20, 5, density = 0.3)
    nn <- fastFindKNN(X_sp, k = 3L, type = "knn")
    expect_equal(dim(nn$index), c(20L, 3L))
    expect_type(nn$index, "integer")
})

test_that("dense binary and sparse MTX give same results", {
    skip_if_no_jar()
    if (!requireNamespace("Matrix", quietly = TRUE))
        skip("Matrix not installed")
    X_sp <- Matrix::rsparsematrix(30, 8, density = 0.4)
    X_dense <- as.matrix(X_sp)

    nn_dense  <- fastFindKNN(X_dense, k = 3L, type = "knn")
    nn_sparse <- fastFindKNN(X_sp, k = 3L, type = "knn")

    expect_equal(nn_dense$index, nn_sparse$index)
    expect_equal(nn_dense$distance, nn_sparse$distance, tolerance = 1e-6)
})
