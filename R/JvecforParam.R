# BiocNeighbors integration ----
# S4 classes and methods for drop-in BNPARAM support.
# Usage: BiocNeighbors::findKNN(X, k, BNPARAM = JvecforParam())

# -- Distance mapping --------------------------------------------------------

.bn_to_jvecfor_distance <- function(distance) {
    map <- c(Euclidean = "euclidean", Cosine = "cosine")
    result <- map[distance]
    if (is.na(result)) {
        stop(
            "jvecfor does not support distance metric '",
            distance, "'. Supported: 'Euclidean', 'Cosine'."
        )
    }
    unname(result)
}

# -- JvecforParam class ------------------------------------------------------

#' JvecforParam: BiocNeighbors Parameter Class for jvecfor
#'
#' A \link[BiocNeighbors:BiocNeighborParam-class]{BiocNeighborParam}
#' subclass for the jvecfor Java backend.
#' Passing a \code{JvecforParam} object as the \code{BNPARAM} argument to
#' \code{\link[BiocNeighbors]{findKNN}} or higher-level functions
#' (e.g. \code{scran::buildSNNGraph}, \code{scater::runUMAP}) routes
#' neighbor search through jvecfor's HNSW-DiskANN or VP-tree engine.
#'
#' @slot type Character. \code{"ann"} (HNSW-DiskANN, default) or
#'   \code{"knn"} (VP-tree exact).
#' @slot M Integer. HNSW max connections per node. Default 16L.
#' @slot ef.search Integer. HNSW beam width (0 = auto). Default 0L.
#' @slot oversample.factor Numeric. Oversampling multiplier (>= 1).
#'   Default 1.0.
#' @slot pq.subspaces Integer. Product-quantization subspaces
#'   (0 = disabled). Default 0L.
#' @slot verbose Logical. Enable Java progress logging. Default FALSE.
#'
#' @section Supported distance metrics:
#' \code{"Euclidean"} and \code{"Cosine"} (title-case, following
#' BiocNeighbors convention). The jvecfor-specific \code{"dot_product"}
#' metric is only available via \code{\link{fastFindKNN}} directly.
#'
#' @section Limitations:
#' \itemize{
#'   \item \code{queryKNN} is not supported. The Java backend performs
#'     self-KNN only (all points query against all points in a single
#'     JVM invocation).
#'   \item The index built by \code{buildIndex} stores the data matrix
#'     in R memory; the actual Java HNSW/VP-tree index is rebuilt each
#'     time \code{findKNN} is called.
#' }
#'
#' @examples
#' library(BiocNeighbors)
#' p <- JvecforParam()
#' p
#'
#' # Custom parameters
#' p2 <- JvecforParam(type = "knn", distance = "Cosine", M = 32L)
#'
#' # Use with BiocNeighbors (requires Java >= 20):
#' # res <- findKNN(X, k = 10, BNPARAM = JvecforParam())
#'
#' @seealso \code{\link{fastFindKNN}} for the standalone function with
#'   full parameter control including \code{dot_product} metric.
#'
#' @importClassesFrom BiocNeighbors BiocNeighborParam BiocNeighborIndex
#' @importFrom BiocNeighbors buildIndex findKnnFromIndex findKNN
#' @importFrom methods setClass setMethod new is show
#'
#' @export
#' @exportClass JvecforParam
setClass("JvecforParam",
    contains = "BiocNeighborParam",
    slots = c(
        type              = "character",
        M                 = "integer",
        ef.search         = "integer",
        oversample.factor = "numeric",
        pq.subspaces      = "integer",
        verbose           = "logical"
    )
)

#' @describeIn JvecforParam Constructor for JvecforParam objects.
#'
#' @param type Character. \code{"ann"} (default) or \code{"knn"}.
#' @param distance Character. \code{"Euclidean"} (default) or
#'   \code{"Cosine"}.
#' @param M Integer. HNSW max connections per node. Default 16L.
#' @param ef.search Integer. HNSW beam width (0 = auto). Default 0L.
#' @param oversample.factor Numeric. Oversampling multiplier. Default 1.0.
#' @param pq.subspaces Integer. PQ subspaces (0 = disabled). Default 0L.
#' @param verbose Logical. Java progress logging. Default FALSE.
#'
#' @return A \code{JvecforParam} object.
#'
#' @export
JvecforParam <- function(
    type              = "ann",
    distance          = "Euclidean",
    M                 = 16L,
    ef.search         = 0L,
    oversample.factor = 1.0,
    pq.subspaces      = 0L,
    verbose           = FALSE
) {
    type     <- match.arg(type, c("ann", "knn"))
    distance <- match.arg(distance, c("Euclidean", "Cosine"))
    new("JvecforParam",
        distance          = distance,
        type              = type,
        M                 = as.integer(M),
        ef.search         = as.integer(ef.search),
        oversample.factor = as.numeric(oversample.factor),
        pq.subspaces      = as.integer(pq.subspaces),
        verbose           = as.logical(verbose)
    )
}

#' @describeIn JvecforParam Print a summary of the parameter object.
#' @param object A \code{JvecforParam} object.
#' @exportMethod show
setMethod("show", "JvecforParam", function(object) {
    cat("JvecforParam\n")
    cat("  distance:", object@distance, "\n")
    cat("  type:", object@type, "\n")
    cat("  M:", object@M, "\n")
    cat("  ef.search:", object@ef.search, "\n")
    cat("  oversample.factor:", object@oversample.factor, "\n")
    cat("  pq.subspaces:", object@pq.subspaces, "\n")
})

# -- JvecforIndex class -------------------------------------------------------

#' JvecforIndex: BiocNeighbors Index Class for jvecfor
#'
#' A \link[BiocNeighbors:BiocNeighborIndex-class]{BiocNeighborIndex}
#' subclass storing the data matrix and \code{JvecforParam} parameters.
#' The actual Java HNSW/VP-tree index is built on-the-fly when
#' \code{\link[BiocNeighbors]{findKNN}} is called.
#'
#' @slot data Numeric matrix (rows = observations, cols = features).
#' @slot param A \code{\link{JvecforParam}} object.
#' @slot names Row names from the original matrix, or NULL.
#'
#' @seealso \code{\link{JvecforParam}}
#'
#' @export
#' @exportClass JvecforIndex
setClass("JvecforIndex",
    contains = "BiocNeighborIndex",
    slots = c(
        data  = "matrix",
        param = "JvecforParam",
        names = "ANY"
    )
)

# -- buildIndex method --------------------------------------------------------

#' @describeIn JvecforParam Build a JvecforIndex from a data matrix.
#'
#' @param X A numeric matrix (rows = observations, cols = features).
#' @param transposed Logical. If TRUE, \code{X} is features-by-obs
#'   and will be transposed. Default FALSE.
#' @param BNPARAM A \code{JvecforParam} object.
#' @param ... Ignored.
#'
#' @return A \code{\linkS4class{JvecforIndex}} object.
#'
#' @exportMethod buildIndex
setMethod("buildIndex", "JvecforParam",
    function(X, BNPARAM, transposed = FALSE, ...) {
        if (transposed) X <- t(X)
        if (!is.matrix(X)) X <- as.matrix(X)
        if (!is.double(X)) storage.mode(X) <- "double"
        new("JvecforIndex",
            data  = X,
            param = BNPARAM,
            names = rownames(X)
        )
    }
)

# -- findKnnFromIndex method --------------------------------------------------

#' @describeIn JvecforParam Find k-nearest neighbors using a
#'   JvecforIndex.
#'
#' @param BNINDEX A \code{\linkS4class{JvecforIndex}} object.
#' @param k Integer. Number of nearest neighbors.
#' @param get.index Logical. Return index matrix? Default TRUE.
#' @param get.distance Logical. Return distance matrix? Default TRUE.
#' @param num.threads Integer. Thread count. Default 1.
#' @param subset Integer vector. Row indices to return results for.
#'   All rows are computed; this filters the output. Default NULL
#'   (all rows).
#'
#' @return A named list with \code{index} (n-by-k integer matrix or
#'   NULL) and \code{distance} (n-by-k numeric matrix or NULL).
#'
#' @exportMethod findKnnFromIndex
setMethod("findKnnFromIndex", "JvecforIndex",
    function(BNINDEX, k, get.index = TRUE, get.distance = TRUE,
             num.threads = 1, subset = NULL, ...) {
        param  <- BNINDEX@param
        metric <- .bn_to_jvecfor_distance(param@distance)

        result <- fastFindKNN(
            X                 = BNINDEX@data,
            k                 = k,
            type              = param@type,
            metric            = metric,
            num.threads       = as.integer(num.threads),
            ef.search         = param@ef.search,
            M                 = param@M,
            oversample.factor = param@oversample.factor,
            pq.subspaces      = param@pq.subspaces,
            get.distance      = !isFALSE(get.distance),
            verbose           = param@verbose
        )

        if (!is.null(subset)) {
            result$index <- result$index[subset, , drop = FALSE]
            if (!is.null(result$distance)) {
                result$distance <-
                    result$distance[subset, , drop = FALSE]
            }
        }

        if (isFALSE(get.index)) {
            result$index <- NULL
        }

        result
    }
)
