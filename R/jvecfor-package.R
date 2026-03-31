#' jvecfor: Fast K-Nearest Neighbor Search for Single-Cell Analysis
#'
#' Drop-in replacement for \code{BiocNeighbors::findKNN} using the jvecfor
#' Java library (HNSW-DiskANN approximate and VP-tree exact methods). Achieves
#' approximately 2x speedup over Annoy-based search at n >= 50K cells.
#' Convenience wrappers delegate SNN/KNN graph construction to the bluster
#' package.
#'
#' @section Main functions:
#' \describe{
#'   \item{\code{\link{fastFindKNN}}}{KNN search -- returns index + distance
#'     matrices.}
#'   \item{\code{\link{fastMakeSNNGraph}}}{KNN -> SNN graph via bluster.}
#'   \item{\code{\link{fastMakeKNNGraph}}}{KNN -> KNN graph via bluster.}
#'   \item{\code{\link{JvecforParam}}}{BiocNeighbors parameter class for
#'     drop-in integration with scran, scater, etc.}
#'   \item{\code{\link{jvecfor_setup}}}{Install a custom jvecfor JAR.}
#' }
#'
#' @section Options:
#' \describe{
#'   \item{\code{jvecfor.verbose}}{Logical. Enable Java/jvecfor progress
#'     logging globally. Default \code{FALSE}.}
#'   \item{\code{jvecfor.jar}}{Character. Path to a custom jvecfor JAR file.
#'     Overrides the bundled JAR in \code{inst/java/}.}
#' }
#'
#' @importFrom BiocParallel bpparam bpworkers
#' @importFrom data.table fread fwrite
#' @importFrom Matrix as.matrix writeMM
#' @importFrom methods as
#' @importFrom processx run
#'
"_PACKAGE"
