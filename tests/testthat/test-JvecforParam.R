# -- JvecforParam constructor ---------------------------------------------------

test_that("JvecforParam default constructor works", {
    p <- JvecforParam()
    expect_s4_class(p, "JvecforParam")
    expect_s4_class(p, "BiocNeighborParam")
    expect_equal(p@type, "ann")
    expect_equal(p@distance, "Euclidean")
    expect_equal(p@M, 16L)
    expect_equal(p@ef.search, 0L)
    expect_equal(p@oversample.factor, 1.0)
    expect_equal(p@pq.subspaces, 0L)
    expect_false(p@verbose)
})

test_that("JvecforParam custom parameters work", {
    p <- JvecforParam(
        type = "knn", distance = "Cosine",
        M = 32L, ef.search = 100L,
        oversample.factor = 2.0,
        pq.subspaces = 10L, verbose = TRUE
    )
    expect_equal(p@type, "knn")
    expect_equal(p@distance, "Cosine")
    expect_equal(p@M, 32L)
    expect_equal(p@ef.search, 100L)
    expect_equal(p@oversample.factor, 2.0)
    expect_equal(p@pq.subspaces, 10L)
    expect_true(p@verbose)
})

test_that("JvecforParam rejects invalid type", {
    expect_error(JvecforParam(type = "invalid"), "arg")
})

test_that("JvecforParam rejects invalid distance", {
    expect_error(JvecforParam(distance = "Manhattan"), "arg")
})

test_that("JvecforParam coerces numeric parameters", {
    p <- JvecforParam(M = 24, ef.search = 50)
    expect_type(p@M, "integer")
    expect_type(p@ef.search, "integer")
    expect_equal(p@M, 24L)
    expect_equal(p@ef.search, 50L)
})

# -- show method ---------------------------------------------------------------

test_that("show method prints JvecforParam info", {
    p <- JvecforParam(type = "knn", distance = "Cosine", M = 32L)
    out <- capture.output(show(p))
    expect_true(any(grepl("JvecforParam", out)))
    expect_true(any(grepl("Cosine", out)))
    expect_true(any(grepl("knn", out)))
    expect_true(any(grepl("32", out)))
})

# -- distance mapping ---------------------------------------------------------

test_that(".bn_to_jvecfor_distance maps correctly", {
    expect_equal(
        jvecfor:::.bn_to_jvecfor_distance("Euclidean"),
        "euclidean"
    )
    expect_equal(
        jvecfor:::.bn_to_jvecfor_distance("Cosine"),
        "cosine"
    )
})

test_that(".bn_to_jvecfor_distance rejects unsupported metrics", {
    expect_error(
        jvecfor:::.bn_to_jvecfor_distance("Manhattan"),
        "does not support"
    )
})

# -- JvecforIndex / buildIndex -------------------------------------------------

test_that("buildIndex creates JvecforIndex with correct data", {
    m <- matrix(rnorm(100), 10, 10)
    p <- JvecforParam()
    idx <- BiocNeighbors::buildIndex(m, BNPARAM = p)
    expect_s4_class(idx, "JvecforIndex")
    expect_s4_class(idx, "BiocNeighborIndex")
    expect_equal(dim(idx@data), c(10L, 10L))
    expect_identical(idx@param, p)
    expect_null(idx@names)
})

test_that("buildIndex preserves rownames", {
    m <- matrix(rnorm(100), 10, 10)
    rownames(m) <- paste0("cell", seq_len(10))
    idx <- BiocNeighbors::buildIndex(m, BNPARAM = JvecforParam())
    expect_equal(idx@names, paste0("cell", seq_len(10)))
})

test_that("buildIndex handles transposed input", {
    m <- matrix(rnorm(100), 10, 10)
    idx <- BiocNeighbors::buildIndex(m, transposed = TRUE,
                                     BNPARAM = JvecforParam())
    # transposed=TRUE means input is features x obs, so it gets t()
    expect_equal(dim(idx@data), c(10L, 10L))
})

test_that("buildIndex coerces integer to double", {
    m <- matrix(1:100, 10, 10)
    idx <- BiocNeighbors::buildIndex(m, BNPARAM = JvecforParam())
    expect_type(idx@data[1, 1], "double")
})

# -- findKNN via BNPARAM (requires Java) ---------------------------------------

has_java <- nzchar(Sys.which("java"))
has_jar  <- tryCatch(
    {jvecfor:::.jvecfor_jar(); TRUE},
    error = function(e) FALSE
)

test_that("findKNN works with JvecforParam", {
    skip_if_not(has_java && has_jar, "Java/JAR not available")
    set.seed(42)
    m <- matrix(rnorm(500), 50, 10)
    res <- BiocNeighbors::findKNN(m, k = 5,
                                  BNPARAM = JvecforParam())
    expect_type(res, "list")
    expect_equal(dim(res$index), c(50L, 5L))
    expect_equal(dim(res$distance), c(50L, 5L))
    expect_type(res$index[1, 1], "integer")
    expect_type(res$distance[1, 1], "double")
})

test_that("findKNN with subset returns filtered rows", {
    skip_if_not(has_java && has_jar, "Java/JAR not available")
    set.seed(42)
    m <- matrix(rnorm(500), 50, 10)
    res <- BiocNeighbors::findKNN(m, k = 5, subset = 1:10,
                                  BNPARAM = JvecforParam())
    expect_equal(dim(res$index), c(10L, 5L))
    expect_equal(dim(res$distance), c(10L, 5L))
})

test_that("findKNN with get.index=FALSE omits index", {
    skip_if_not(has_java && has_jar, "Java/JAR not available")
    set.seed(42)
    m <- matrix(rnorm(500), 50, 10)
    res <- BiocNeighbors::findKNN(m, k = 5, get.index = FALSE,
                                  BNPARAM = JvecforParam())
    expect_null(res$index)
    expect_equal(dim(res$distance), c(50L, 5L))
})

test_that("findKNN with get.distance=FALSE omits distance", {
    skip_if_not(has_java && has_jar, "Java/JAR not available")
    set.seed(42)
    m <- matrix(rnorm(500), 50, 10)
    res <- BiocNeighbors::findKNN(m, k = 5, get.distance = FALSE,
                                  BNPARAM = JvecforParam())
    expect_equal(dim(res$index), c(50L, 5L))
    expect_null(res$distance)
})

test_that("findKNN with Cosine distance works", {
    skip_if_not(has_java && has_jar, "Java/JAR not available")
    set.seed(42)
    m <- matrix(rnorm(500), 50, 10)
    res <- BiocNeighbors::findKNN(
        m, k = 5,
        BNPARAM = JvecforParam(distance = "Cosine")
    )
    expect_equal(dim(res$index), c(50L, 5L))
})

test_that("findKNN with type='knn' (exact) works", {
    skip_if_not(has_java && has_jar, "Java/JAR not available")
    set.seed(42)
    m <- matrix(rnorm(500), 50, 10)
    res <- BiocNeighbors::findKNN(
        m, k = 5,
        BNPARAM = JvecforParam(type = "knn")
    )
    expect_equal(dim(res$index), c(50L, 5L))
})

test_that("findKNN results match fastFindKNN", {
    skip_if_not(has_java && has_jar, "Java/JAR not available")
    set.seed(42)
    m <- matrix(rnorm(500), 50, 10)
    res_bn <- BiocNeighbors::findKNN(
        m, k = 5,
        BNPARAM = JvecforParam(type = "knn")
    )
    res_ff <- fastFindKNN(m, k = 5, type = "knn", num.threads = 1L)
    expect_equal(res_bn$index, res_ff$index)
    expect_equal(res_bn$distance, res_ff$distance)
})
