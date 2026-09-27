# Build a K-Nearest Neighbor Graph

Convenience wrapper that calls `fastFindKNN` then delegates graph
construction to
[`bluster::neighborsToKNNGraph`](https://rdrr.io/pkg/bluster/man/makeSNNGraph.html).

## Usage

``` r
fastMakeKNNGraph(
  X,
  k = 15L,
  type = c("ann", "knn"),
  metric = c("euclidean", "cosine", "dot_product"),
  num.threads = NULL,
  BPPARAM = BiocParallel::bpparam(),
  ef.search = 0L,
  M = 16L,
  oversample.factor = 1,
  pq.subspaces = 0L,
  verbose = getOption("jvecfor.verbose", FALSE),
  directed = FALSE,
  ...
)
```

## Arguments

- X:

  A numeric matrix, `data.frame`, or sparse matrix (`Matrix::dgCMatrix`)
  with rows = cells, cols = features/PCs.

- k:

  Integer. Number of nearest neighbors. Default 15.

- type:

  Character. `"ann"` or `"knn"`. Default `"ann"`.

- metric:

  Character. `"euclidean"`, `"cosine"`, or `"dot_product"`. Default
  `"euclidean"`. See
  [`fastFindKNN`](https://gkanogiannis.github.io/jvecfor/reference/fastFindKNN.md)
  for the `dot_product` restriction.

- num.threads:

  Integer or NULL. Number of Java threads. If NULL, defaults to
  `BiocParallel::bpworkers(BPPARAM)`.

- BPPARAM:

  A
  [`BiocParallelParam`](https://rdrr.io/pkg/BiocParallel/man/BiocParallelParam-class.html)
  object controlling the thread count. Defaults to
  [`bpparam()`](https://rdrr.io/pkg/BiocParallel/man/register.html).

- ef.search:

  Integer. HNSW-DiskANN beam width override (0 = auto). Default 0L.

- M:

  Integer. HNSW-DiskANN max connections per node. Default 16L.

- oversample.factor:

  Numeric. Oversampling multiplier. Default 1.0.

- pq.subspaces:

  Integer. PQ subspaces (0 = disabled). Default 0L.

- verbose:

  Logical. Enable Java verbose logging. Default
  `getOption("jvecfor.verbose", FALSE)`.

- directed:

  Logical. Build directed graph? Default FALSE.

- ...:

  Additional arguments forwarded to
  [`bluster::neighborsToKNNGraph`](https://rdrr.io/pkg/bluster/man/makeSNNGraph.html).

## Value

An `igraph` object (KNN graph).

## Examples

``` r
set.seed(42)
X <- matrix(rnorm(5000), nrow = 100, ncol = 50)

# Full examples require Java >= 20 on PATH
g <- fastMakeKNNGraph(X, k = 10)
igraph::vcount(g)  # 100
#> [1] 100
```
