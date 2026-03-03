.coerce_X <- function(X) {
    if (inherits(X, c("dgCMatrix", "sparseMatrix"))) {
        X <- as.matrix(X)
    } else if (is.data.frame(X)) {
        X <- as.matrix(X)
    } else if (!is.matrix(X)) {
        stop(
            "X must be a numeric matrix, data.frame, or sparse matrix ",
            "(Matrix::dgCMatrix)."
        )
    }
    if (!is.numeric(X)) stop("X must contain numeric values.")
    X
}

.validate_knn_scalars <- function(
    k, X, metric, type, ef.search, M, oversample.factor, pq.subspaces
) {
    if (metric == "dot_product" && type == "knn") {
        stop(
            "'dot_product' is not a proper metric and cannot be used with ",
            "type='knn' (VP-tree exact search). Use type='ann', or switch to ",
            "metric='euclidean' or metric='cosine' for exact search."
        )
    }

    k <- as.integer(k)
    if (is.na(k) || k < 1L)
        stop("'k' must be a positive integer.")
    if (nrow(X) <= k)
        stop("nrow(X) must be > k (need at least k+1 observations).")

    ef.search         <- as.integer(ef.search)
    M                 <- as.integer(M)
    oversample.factor <- as.numeric(oversample.factor)
    pq.subspaces      <- as.integer(pq.subspaces)

    if (ef.search < 0L)
        stop("'ef.search' must be >= 0 (0 = auto).")
    if (M < 2L)
        stop("'M' must be >= 2.")
    if (oversample.factor < 1.0)
        stop("'oversample.factor' must be >= 1.0.")
    if (pq.subspaces < 0L)
        stop("'pq.subspaces' must be >= 0 (0 = disabled).")
    if (pq.subspaces > 0L && pq.subspaces > ncol(X))
        stop(
            "'pq.subspaces' (", pq.subspaces, ") cannot exceed ",
            "ncol(X) (", ncol(X), ")."
        )

    list(
        k                 = k,
        ef.search         = ef.search,
        M                 = M,
        oversample.factor = oversample.factor,
        pq.subspaces      = pq.subspaces
    )
}

.resolve_threads <- function(num.threads, BPPARAM) {
    if (!is.null(num.threads)) {
        val <- as.integer(num.threads)
        if (is.na(val) || val < 1L)
            stop("'num.threads' must be a positive integer or NULL.")
        val
    } else {
        max(1L, as.integer(BiocParallel::bpworkers(BPPARAM)))
    }
}

.build_java_args <- function(
    jar, k, metric, num_threads, type,
    ef.search, M, oversample.factor, pq.subspaces,
    get.distance, verbose
) {
    args <- c(
        "--add-modules", "jdk.incubator.vector",
        "-jar",  jar,
        "-k",    as.character(k),
        "-m",    metric,
        "-t",    as.character(num_threads),
        "-g",    type,
        "--ef-search",         as.character(ef.search),
        "-M",                  as.character(M),
        "--oversample-factor", as.character(oversample.factor),
        "--pq-subspaces",      as.character(pq.subspaces)
    )
    if (get.distance)    args <- c(args, "--output-dist")
    if (isTRUE(verbose)) args <- c(args, "--verbose")
    args
}

.run_java <- function(java_args, tsv_in, verbose) {
    out <- system2(
        "java",
        args   = java_args,
        stdout = TRUE,
        stderr = if (isTRUE(verbose)) "" else FALSE,
        stdin  = tsv_in
    )
    status <- attr(out, "status")
    if (!is.null(status) && status != 0L) {
        stop(
            "jvecfor exited with status ", status, ". ",
            "Set verbose=TRUE (or options(jvecfor.verbose=TRUE)) to see ",
            "Java stderr."
        )
    }
    out
}

.parse_knn_output <- function(out, k, get.distance) {
    if (get.distance) {
        raw      <- .read_matrix_from_text(out)
        index    <- raw[, seq_len(k), drop = FALSE]
        storage.mode(index) <- "integer"
        distance <- raw[, seq_len(k) + k, drop = FALSE]
    } else {
        index <- .read_matrix_from_text(out)
        storage.mode(index) <- "integer"
        distance <- NULL
    }
    list(index = index, distance = distance)
}

#' Fast K-Nearest Neighbor Search
#'
#' Drop-in replacement for \code{BiocNeighbors::findKNN} using the jvecfor
#' Java library. Supports HNSW-DiskANN approximate search
#' (\code{type="ann"}) and VP-tree exact search (\code{type="knn"}).
#'
#' @param X A numeric matrix, \code{data.frame}, or sparse matrix
#'   (\code{Matrix::dgCMatrix}) with rows = observations, cols = features.
#'   Sparse matrices are coerced to dense before processing.
#' @param k Integer. Number of nearest neighbors to find (excluding self).
#'   Default 15.
#' @param type Character. \code{"ann"} for approximate (HNSW-DiskANN) or
#'   \code{"knn"} for exact (VP-tree). Default \code{"ann"}.
#' @param metric Character. Distance metric: \code{"euclidean"},
#'   \code{"cosine"}, or \code{"dot_product"}. Default \code{"euclidean"}.
#'   \strong{Note:} \code{"dot_product"} is only valid when
#'   \code{type = "ann"}; it is not a proper metric and cannot be used
#'   with the VP-tree exact search.
#' @param num.threads Integer or NULL. Number of Java threads. If NULL,
#'   defaults to \code{BiocParallel::bpworkers(BPPARAM)}.
#' @param BPPARAM A \code{\link[BiocParallel]{BiocParallelParam}} object
#'   controlling the thread count. Defaults to
#'   \code{\link[BiocParallel]{bpparam}()}.
#' @param ef.search Integer. HNSW-DiskANN beam width override (0 = auto:
#'   \code{max(k+1, 3k)}). Only meaningful when \code{type = "ann"}.
#'   Default 0L.
#' @param M Integer. HNSW-DiskANN maximum connections per node. Higher
#'   values (e.g. 32) improve recall for high-dimensional data at the
#'   cost of more memory and a slower build. Only meaningful when
#'   \code{type = "ann"}. Default 16L.
#' @param oversample.factor Numeric. Oversampling multiplier for the ANN
#'   beam width. When > 1.0, fetches
#'   \code{ceil(ef * oversample.factor)} candidates and returns the top
#'   k, improving recall at proportional cost. Only meaningful when
#'   \code{type = "ann"}. Default 1.0.
#' @param pq.subspaces Integer. Number of Product Quantization subspaces
#'   for approximate ANN scoring (0 = disabled). Typical value:
#'   \code{ncol(X) / 2}. Reduces search time approximately 4-8x with
#'   minimal recall loss. Only meaningful when \code{type = "ann"}.
#'   Default 0L.
#' @param get.distance Logical. Return distance matrix alongside index?
#'   Default TRUE.
#' @param verbose Logical. Pass \code{--verbose} to the Java process,
#'   enabling HNSW-DiskANN build progress logging on stderr. Overrides
#'   the \code{jvecfor.verbose} global option when set explicitly.
#'   Default \code{getOption("jvecfor.verbose", FALSE)}.
#'
#' @return A named list:
#'   \describe{
#'     \item{index}{n x k integer matrix of 1-indexed neighbor indices.}
#'     \item{distance}{n x k numeric matrix of distances/similarities, or
#'       NULL if \code{get.distance=FALSE}.}
#'   }
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(200), nrow = 20, ncol = 10)
#'
#' # Full examples require Java >= 25 on PATH
#' nn <- fastFindKNN(X, k = 3)
#' dim(nn$index)    # 20 x 3
#' dim(nn$distance) # 20 x 3
#'
#' # High-recall HNSW-DiskANN with wider beam and more connections
#' nn2 <- fastFindKNN(X, k = 3, M = 32, ef.search = 100,
#'                    oversample.factor = 2.0)
#'
#' # Dot-product similarity (ANN only)
#' nn3 <- fastFindKNN(X, k = 3, metric = "dot_product")
#'
#' @export
fastFindKNN <- function(
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
    get.distance      = TRUE,
    verbose           = getOption("jvecfor.verbose", FALSE)
) {
    type   <- match.arg(type)
    metric <- match.arg(metric)

    X <- .coerce_X(X)
    p <- .validate_knn_scalars(
        k, X, metric, type,
        ef.search, M, oversample.factor, pq.subspaces
    )
    k                 <- p$k
    ef.search         <- p$ef.search
    M                 <- p$M
    oversample.factor <- p$oversample.factor
    pq.subspaces      <- p$pq.subspaces

    num_threads <- .resolve_threads(num.threads, BPPARAM)

    .check_java()
    jar <- .jvecfor_jar()

    tsv_in <- tempfile(fileext = ".tsv")
    on.exit(unlink(tsv_in), add = TRUE)
    .write_tsv(X, tsv_in)

    java_args <- .build_java_args(
        jar, k, metric, num_threads, type,
        ef.search, M, oversample.factor,
        pq.subspaces, get.distance, verbose
    )
    out <- .run_java(java_args, tsv_in, verbose)
    .parse_knn_output(out, k, get.distance)
}
