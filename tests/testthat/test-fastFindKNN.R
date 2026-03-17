library(testthat)
library(jvecfor)

# Skip helpers -----------------------------------------------------------------

skip_if_no_java <- function() {
    skip_if_not(nzchar(Sys.which("java")), "Java not found on PATH")
}

skip_if_no_jar <- function() {
    skip_if_no_java()
    jar_ok <- tryCatch(
        { jvecfor:::.jvecfor_jar(); TRUE },
        error = function(e) FALSE
    )
    skip_if_not(jar_ok, "jvecfor JAR not installed (run jvecfor_setup())")
}

set.seed(42)
X_small <- matrix(rnorm(20 * 5), nrow = 20, ncol = 5)

# Basic correctness ------------------------------------------------------------

test_that("basic ann returns correct dimensions and types", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, type = "ann")
    expect_equal(dim(nn$index),    c(20L, 3L))
    expect_equal(dim(nn$distance), c(20L, 3L))
    expect_type(nn$index,    "integer")
    expect_type(nn$distance, "double")
})

test_that("basic knn returns correct dimensions and types", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, type = "knn")
    expect_equal(dim(nn$index),    c(20L, 3L))
    expect_equal(dim(nn$distance), c(20L, 3L))
    expect_type(nn$index,    "integer")
    expect_type(nn$distance, "double")
})

test_that("ann and knn return same-shape output", {
    skip_if_no_jar()
    nn_ann <- fastFindKNN(X_small, k = 3L, type = "ann")
    nn_knn <- fastFindKNN(X_small, k = 3L, type = "knn")
    expect_equal(dim(nn_ann$index),    dim(nn_knn$index))
    expect_equal(dim(nn_ann$distance), dim(nn_knn$distance))
})

test_that("get.distance=FALSE returns NULL distance", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, get.distance = FALSE)
    expect_null(nn$distance)
    expect_equal(dim(nn$index), c(20L, 3L))
})

test_that("output is 1-indexed (no zeros, all in 1..nrow(X))", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L)
    expect_true(all(nn$index >= 1L))
    expect_true(all(nn$index <= nrow(X_small)))
})

test_that("k=1 returns single neighbor column", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 1L)
    expect_equal(dim(nn$index), c(20L, 1L))
})

test_that("cosine metric runs without error", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, metric = "cosine")
    expect_equal(dim(nn$index), c(20L, 3L))
})

# Parameter tuning -------------------------------------------------------------

test_that("M=32 runs without error", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, type = "ann", M = 32L)
    expect_equal(dim(nn$index), c(20L, 3L))
    expect_type(nn$index, "integer")
})

test_that("oversample.factor > 1 runs without error", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, type = "ann", oversample.factor = 2.0)
    expect_equal(dim(nn$index), c(20L, 3L))
})

test_that("pq.subspaces=1 runs without error", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, type = "ann", pq.subspaces = 1L)
    expect_equal(dim(nn$index), c(20L, 3L))
})

test_that("ef.search=50 runs without error", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, type = "ann", ef.search = 50L)
    expect_equal(dim(nn$index), c(20L, 3L))
})

test_that("dot_product metric with type='ann' runs without error", {
    skip_if_no_jar()
    nn <- fastFindKNN(X_small, k = 3L, type = "ann", metric = "dot_product")
    expect_equal(dim(nn$index), c(20L, 3L))
    expect_type(nn$index, "integer")
})

test_that("verbose=TRUE runs without error", {
    skip_if_no_jar()
    expect_no_error(fastFindKNN(X_small, k = 3L, verbose = TRUE))
})

# Pure-R input validation (no Java required) -----------------------------------

test_that("dot_product metric with type='knn' stops with informative error", {
    expect_error(
        fastFindKNN(X_small, k = 3L, type = "knn", metric = "dot_product"),
        regexp = "dot_product|not a proper metric",
        ignore.case = TRUE
    )
})

test_that("k >= nrow(X) stops with error", {
    expect_error(fastFindKNN(X_small, k = 20L), "nrow\\(X\\) must be > k")
    expect_error(fastFindKNN(X_small, k = 25L), "nrow\\(X\\) must be > k")
})

test_that("non-numeric X stops with error", {
    X_char <- matrix(as.character(X_small), nrow = 20)
    expect_error(fastFindKNN(X_char, k = 3L), "numeric")
})

test_that("list input stops with informative error", {
    expect_error(fastFindKNN(list(a = 1:5), k = 3L), "numeric matrix")
})

test_that("k < 1 stops with error", {
    expect_error(fastFindKNN(X_small, k = 0L), "k.*must be a positive integer")
})

test_that("negative ef.search stops with error", {
    expect_error(
        fastFindKNN(X_small, k = 3L, ef.search = -1L),
        "ef.search.*must be"
    )
})

test_that("M < 2 stops with error", {
    expect_error(fastFindKNN(X_small, k = 3L, M = 1L), "'M' must be")
})

test_that("oversample.factor < 1 stops with error", {
    expect_error(
        fastFindKNN(X_small, k = 3L, oversample.factor = 0.5),
        "oversample.factor.*must be"
    )
})

test_that("pq.subspaces exceeding ncol(X) stops with error", {
    expect_error(
        fastFindKNN(X_small, k = 3L, pq.subspaces = 100L),
        "pq.subspaces.*cannot exceed"
    )
})

test_that("sparse matrix input is coerced and accepted", {
    skip_if_no_jar()
    if (!requireNamespace("Matrix", quietly = TRUE)) skip("Matrix not installed")
    X_sp <- Matrix::Matrix(X_small, sparse = TRUE)
    nn <- fastFindKNN(X_sp, k = 3L)
    expect_equal(dim(nn$index), c(20L, 3L))
    expect_type(nn$index, "integer")
})

# Java version regex unit tests ------------------------------------------------

test_that("version parser identifies Java 8 / 11 as too old (regex unit test)", {
    # Validates the regex logic in .check_java() without invoking Java.
    # Java 11 modern format: "11.0.22"
    ver_line_11 <- 'openjdk version "11.0.22" 2024-01-16'
    m <- regmatches(ver_line_11, regexpr('"[0-9]+', ver_line_11))
    major_11 <- suppressWarnings(as.integer(sub('"', "", m, fixed = TRUE)))
    expect_equal(major_11, 11L)
    expect_true(major_11 < 20L)

    # Java 8 legacy format: "1.8.0_411"
    ver_line_8 <- 'java version "1.8.0_411" 2024-04-16'
    m8 <- regmatches(ver_line_8, regexpr('"[0-9]+', ver_line_8))
    major_raw <- suppressWarnings(as.integer(sub('"', "", m8, fixed = TRUE)))
    expect_equal(major_raw, 1L)
    # After the legacy branch: extract feature version
    m8b <- regmatches(ver_line_8, regexpr('"1\\.[0-9]+', ver_line_8))
    major_8 <- suppressWarnings(as.integer(sub('"1.', "", m8b, fixed = TRUE)))
    expect_equal(major_8, 8L)
    expect_true(major_8 < 20L)
})
