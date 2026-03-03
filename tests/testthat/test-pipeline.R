library(testthat)
library(jvecfor)

# Skip helpers -----------------------------------------------------------------

skip_if_no_jar <- function() {
    skip_if_not(nzchar(Sys.which("java")), "Java not found on PATH")
    jar_ok <- tryCatch(
        { jvecfor:::.jvecfor_jar(); TRUE },
        error = function(e) FALSE
    )
    skip_if_not(jar_ok, "jvecfor JAR not installed (run jvecfor_setup())")
}

set.seed(42)
X_small <- matrix(rnorm(20 * 5), nrow = 20, ncol = 5)

# fastMakeSNNGraph -------------------------------------------------------------

test_that("fastMakeSNNGraph returns an igraph object", {
    skip_if_no_jar()
    g <- fastMakeSNNGraph(X_small, k = 3L)
    expect_s3_class(g, "igraph")
})

test_that("fastMakeSNNGraph returns correct vertex count", {
    skip_if_no_jar()
    g <- fastMakeSNNGraph(X_small, k = 3L)
    expect_equal(igraph::vcount(g), nrow(X_small))
})

test_that("fastMakeSNNGraph snn.type='jaccard' runs without error", {
    skip_if_no_jar()
    g <- fastMakeSNNGraph(X_small, k = 3L, snn.type = "jaccard")
    expect_s3_class(g, "igraph")
})

test_that("fastMakeSNNGraph snn.type='number' runs without error", {
    skip_if_no_jar()
    g <- fastMakeSNNGraph(X_small, k = 3L, snn.type = "number")
    expect_s3_class(g, "igraph")
})

test_that("fastMakeSNNGraph returns undirected graph", {
    skip_if_no_jar()
    g <- fastMakeSNNGraph(X_small, k = 3L)
    expect_false(igraph::is_directed(g))
})

# fastMakeKNNGraph -------------------------------------------------------------

test_that("fastMakeKNNGraph returns an igraph object", {
    skip_if_no_jar()
    g <- fastMakeKNNGraph(X_small, k = 3L)
    expect_s3_class(g, "igraph")
})

test_that("fastMakeKNNGraph returns correct vertex count", {
    skip_if_no_jar()
    g <- fastMakeKNNGraph(X_small, k = 3L)
    expect_equal(igraph::vcount(g), nrow(X_small))
})

test_that("fastMakeKNNGraph directed=TRUE returns directed graph", {
    skip_if_no_jar()
    g <- fastMakeKNNGraph(X_small, k = 3L, directed = TRUE)
    expect_true(igraph::is_directed(g))
})

test_that("fastMakeKNNGraph directed=FALSE returns undirected graph", {
    skip_if_no_jar()
    g <- fastMakeKNNGraph(X_small, k = 3L, directed = FALSE)
    expect_false(igraph::is_directed(g))
})

# BPPARAM forwarding -----------------------------------------------------------

test_that("fastMakeSNNGraph accepts explicit BPPARAM", {
    skip_if_no_jar()
    bp <- BiocParallel::SerialParam()
    g <- fastMakeSNNGraph(X_small, k = 3L, BPPARAM = bp)
    expect_s3_class(g, "igraph")
})

test_that("fastMakeKNNGraph accepts explicit BPPARAM", {
    skip_if_no_jar()
    bp <- BiocParallel::SerialParam()
    g <- fastMakeKNNGraph(X_small, k = 3L, BPPARAM = bp)
    expect_s3_class(g, "igraph")
})
