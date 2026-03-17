#' Build a Shared Nearest-Neighbor Graph
#'
#' Convenience wrapper that calls \code{fastFindKNN} then delegates graph
#' construction to \code{bluster::neighborsToSNNGraph}.
#'
#' @param X A numeric matrix, \code{data.frame}, or sparse matrix
#'   (\code{Matrix::dgCMatrix}) with rows = cells, cols = features/PCs.
#' @param k Integer. Number of nearest neighbors. Default 15.
#' @param type Character. \code{"ann"} or \code{"knn"}. Default
#'   \code{"ann"}.
#' @param metric Character. \code{"euclidean"}, \code{"cosine"}, or
#'   \code{"dot_product"}. Default \code{"euclidean"}. See
#'   \code{\link{fastFindKNN}} for the \code{dot_product} restriction.
#' @param num.threads Integer or NULL. Number of Java threads. If NULL,
#'   defaults to \code{BiocParallel::bpworkers(BPPARAM)}.
#' @param BPPARAM A \code{\link[BiocParallel]{BiocParallelParam}} object
#'   controlling the thread count. Defaults to
#'   \code{\link[BiocParallel]{bpparam}()}.
#' @param ef.search Integer. HNSW-DiskANN beam width override (0 = auto).
#'   Default 0L.
#' @param M Integer. HNSW-DiskANN max connections per node. Default 16L.
#' @param oversample.factor Numeric. Oversampling multiplier. Default 1.0.
#' @param pq.subspaces Integer. PQ subspaces (0 = disabled). Default 0L.
#' @param verbose Logical. Enable Java verbose logging. Default
#'   \code{getOption("jvecfor.verbose", FALSE)}.
#' @param snn.type Character. SNN weighting scheme passed to
#'   \code{bluster::neighborsToSNNGraph}: \code{"rank"},
#'   \code{"jaccard"}, or \code{"number"}. Default \code{"rank"}.
#' @param ... Additional arguments forwarded to
#'   \code{bluster::neighborsToSNNGraph}.
#'
#' @return An \code{igraph} object (weighted, undirected SNN graph).
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(5000), nrow = 100, ncol = 50)
#'
#' # Full examples require Java >= 20 on PATH
#' g <- fastMakeSNNGraph(X, k = 10)
#' igraph::vcount(g)  # 100
#'
#' # Higher recall with larger beam and more connections
#' g2 <- fastMakeSNNGraph(X, k = 10, M = 32, oversample.factor = 2.0)
#'
#' @export
fastMakeSNNGraph <- function(
    X,
    k                 = 15L,
    type              = c("ann", "knn"),
    metric            = c("euclidean", "cosine", "dot_product"),
    num.threads       = NULL,
    BPPARAM           = BiocParallel::bpparam(),
    ef.search         = 0L,
    M                 = 16L,
    oversample.factor = 1.0,
    pq.subspaces      = 0L,
    verbose           = getOption("jvecfor.verbose", FALSE),
    snn.type          = "rank",
    ...
) {
    type   <- match.arg(type)
    metric <- match.arg(metric)

    nn <- fastFindKNN(
        X,
        k                 = k,
        type              = type,
        metric            = metric,
        num.threads       = num.threads,
        BPPARAM           = BPPARAM,
        ef.search         = ef.search,
        M                 = M,
        oversample.factor = oversample.factor,
        pq.subspaces      = pq.subspaces,
        verbose           = verbose,
        get.distance      = FALSE
    )
    bluster::neighborsToSNNGraph(nn$index, type = snn.type, ...)
}

#' Build a K-Nearest Neighbor Graph
#'
#' Convenience wrapper that calls \code{fastFindKNN} then delegates graph
#' construction to \code{bluster::neighborsToKNNGraph}.
#'
#' @param X A numeric matrix, \code{data.frame}, or sparse matrix
#'   (\code{Matrix::dgCMatrix}) with rows = cells, cols = features/PCs.
#' @param k Integer. Number of nearest neighbors. Default 15.
#' @param type Character. \code{"ann"} or \code{"knn"}. Default
#'   \code{"ann"}.
#' @param metric Character. \code{"euclidean"}, \code{"cosine"}, or
#'   \code{"dot_product"}. Default \code{"euclidean"}. See
#'   \code{\link{fastFindKNN}} for the \code{dot_product} restriction.
#' @param num.threads Integer or NULL. Number of Java threads. If NULL,
#'   defaults to \code{BiocParallel::bpworkers(BPPARAM)}.
#' @param BPPARAM A \code{\link[BiocParallel]{BiocParallelParam}} object
#'   controlling the thread count. Defaults to
#'   \code{\link[BiocParallel]{bpparam}()}.
#' @param ef.search Integer. HNSW-DiskANN beam width override (0 = auto).
#'   Default 0L.
#' @param M Integer. HNSW-DiskANN max connections per node. Default 16L.
#' @param oversample.factor Numeric. Oversampling multiplier. Default 1.0.
#' @param pq.subspaces Integer. PQ subspaces (0 = disabled). Default 0L.
#' @param verbose Logical. Enable Java verbose logging. Default
#'   \code{getOption("jvecfor.verbose", FALSE)}.
#' @param directed Logical. Build directed graph? Default FALSE.
#' @param ... Additional arguments forwarded to
#'   \code{bluster::neighborsToKNNGraph}.
#'
#' @return An \code{igraph} object (KNN graph).
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(5000), nrow = 100, ncol = 50)
#'
#' # Full examples require Java >= 20 on PATH
#' g <- fastMakeKNNGraph(X, k = 10)
#' igraph::vcount(g)  # 100
#'
#' @export
fastMakeKNNGraph <- function(
    X,
    k                 = 15L,
    type              = c("ann", "knn"),
    metric            = c("euclidean", "cosine", "dot_product"),
    num.threads       = NULL,
    BPPARAM           = BiocParallel::bpparam(),
    ef.search         = 0L,
    M                 = 16L,
    oversample.factor = 1.0,
    pq.subspaces      = 0L,
    verbose           = getOption("jvecfor.verbose", FALSE),
    directed          = FALSE,
    ...
) {
    type   <- match.arg(type)
    metric <- match.arg(metric)

    nn <- fastFindKNN(
        X,
        k                 = k,
        type              = type,
        metric            = metric,
        num.threads       = num.threads,
        BPPARAM           = BPPARAM,
        ef.search         = ef.search,
        M                 = M,
        oversample.factor = oversample.factor,
        pq.subspaces      = pq.subspaces,
        verbose           = verbose,
        get.distance      = FALSE
    )
    bluster::neighborsToKNNGraph(nn$index, directed = directed, ...)
}
