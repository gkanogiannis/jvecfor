# Fast K-Nearest Neighbor Search

Drop-in replacement for
[`BiocNeighbors::findKNN`](https://rdrr.io/pkg/BiocNeighbors/man/findKNN.html)
using the jvecfor Java library. Supports HNSW-DiskANN approximate search
(`type="ann"`) and VP-tree exact search (`type="knn"`).

## Usage

``` r
fastFindKNN(
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
  get.distance = TRUE,
  verbose = getOption("jvecfor.verbose", FALSE)
)
```

## Arguments

- X:

  A numeric matrix, `data.frame`, or sparse matrix (`Matrix::dgCMatrix`)
  with rows = observations, cols = features. Sparse matrices are written
  in MatrixMarket format and densified in the Java backend, avoiding
  R-side memory allocation of the full dense matrix.

- k:

  Integer. Number of nearest neighbors to find (excluding self). Default
  15.

- type:

  Character. `"ann"` for approximate (HNSW-DiskANN) or `"knn"` for exact
  (VP-tree). Default `"ann"`.

- metric:

  Character. Distance metric: `"euclidean"`, `"cosine"`, or
  `"dot_product"`. Default `"euclidean"`. **Note:** `"dot_product"` is
  only valid when `type = "ann"`; it is not a proper metric and cannot
  be used with the VP-tree exact search.

- num.threads:

  Integer or NULL. Number of Java threads. If NULL, defaults to
  `BiocParallel::bpworkers(BPPARAM)`.

- BPPARAM:

  A
  [`BiocParallelParam`](https://rdrr.io/pkg/BiocParallel/man/BiocParallelParam-class.html)
  object controlling the thread count. Defaults to
  [`bpparam()`](https://rdrr.io/pkg/BiocParallel/man/register.html).

- ef.search:

  Integer. HNSW-DiskANN beam width override (0 = auto: `max(k+1, 3k)`).
  Only meaningful when `type = "ann"`. Default 0L.

- M:

  Integer. HNSW-DiskANN maximum connections per node. Higher values
  (e.g. 32) improve recall for high-dimensional data at the cost of more
  memory and a slower build. Only meaningful when `type = "ann"`.
  Default 16L.

- oversample.factor:

  Numeric. Oversampling multiplier for the ANN beam width. When \> 1.0,
  fetches `ceil(ef * oversample.factor)` candidates and returns the top
  k, improving recall at proportional cost. Only meaningful when
  `type = "ann"`. Default 1.0.

- pq.subspaces:

  Integer. Number of Product Quantization subspaces for approximate ANN
  scoring (0 = disabled). Typical value: `ncol(X) / 2`. Reduces search
  time approximately 4-8x with minimal recall loss. Only meaningful when
  `type = "ann"`. Default 0L.

- get.distance:

  Logical. Return distance matrix alongside index? Default TRUE.

- verbose:

  Logical. Pass `--verbose` to the Java process, enabling HNSW-DiskANN
  build progress logging on stderr. Overrides the `jvecfor.verbose`
  global option when set explicitly. Default
  `getOption("jvecfor.verbose", FALSE)`.

## Value

A named list:

- index:

  n x k integer matrix of 1-indexed neighbor indices.

- distance:

  n x k numeric matrix of distances/similarities, or NULL if
  `get.distance=FALSE`.

## Examples

``` r
set.seed(42)
X <- matrix(rnorm(200), nrow = 20, ncol = 10)

# Full examples require Java >= 20 on PATH
nn <- fastFindKNN(X, k = 3)
dim(nn$index)    # 20 x 3
#> [1] 20  3
dim(nn$distance) # 20 x 3
#> [1] 20  3

# High-recall HNSW-DiskANN with wider beam and more connections
nn2 <- fastFindKNN(X, k = 3, M = 32, ef.search = 100,
                   oversample.factor = 2.0)

# Dot-product similarity (ANN only)
nn3 <- fastFindKNN(X, k = 3, metric = "dot_product")
```
