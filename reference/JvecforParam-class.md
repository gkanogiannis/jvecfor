# JvecforParam: BiocNeighbors Parameter Class for jvecfor

A
[BiocNeighborParam](https://rdrr.io/pkg/BiocNeighbors/man/BiocNeighborParam.html)
subclass for the jvecfor Java backend. Passing a `JvecforParam` object
as the `BNPARAM` argument to
[`findKNN`](https://rdrr.io/pkg/BiocNeighbors/man/findKNN.html) or
higher-level functions (e.g. `scran::buildSNNGraph`, `scater::runUMAP`)
routes neighbor search through jvecfor's HNSW-DiskANN or VP-tree engine.

## Usage

``` r
JvecforParam(
  type = "ann",
  distance = "Euclidean",
  M = 16L,
  ef.search = 0L,
  oversample.factor = 1,
  pq.subspaces = 0L,
  verbose = FALSE
)

# S4 method for class 'JvecforParam'
show(object)

# S4 method for class 'JvecforParam'
buildIndex(X, BNPARAM, transposed = FALSE, ...)

# S4 method for class 'JvecforIndex'
findKnnFromIndex(
  BNINDEX,
  k,
  get.index = TRUE,
  get.distance = TRUE,
  num.threads = 1,
  subset = NULL,
  ...
)
```

## Arguments

- type:

  Character. `"ann"` (default) or `"knn"`.

- distance:

  Character. `"Euclidean"` (default) or `"Cosine"`.

- M:

  Integer. HNSW max connections per node. Default 16L.

- ef.search:

  Integer. HNSW beam width (0 = auto). Default 0L.

- oversample.factor:

  Numeric. Oversampling multiplier. Default 1.0.

- pq.subspaces:

  Integer. PQ subspaces (0 = disabled). Default 0L.

- verbose:

  Logical. Java progress logging. Default FALSE.

- object:

  A `JvecforParam` object.

- X:

  A numeric matrix (rows = observations, cols = features).

- BNPARAM:

  A `JvecforParam` object.

- transposed:

  Logical. If TRUE, `X` is features-by-obs and will be transposed.
  Default FALSE.

- ...:

  Ignored.

- BNINDEX:

  A
  [`JvecforIndex`](https://gkanogiannis.github.io/jvecfor/reference/JvecforIndex-class.md)
  object.

- k:

  Integer. Number of nearest neighbors.

- get.index:

  Logical. Return index matrix? Default TRUE.

- get.distance:

  Logical. Return distance matrix? Default TRUE.

- num.threads:

  Integer. Thread count. Default 1.

- subset:

  Integer vector. Row indices to return results for. All rows are
  computed; this filters the output. Default NULL (all rows).

## Value

A `JvecforParam` object.

A
[`JvecforIndex`](https://gkanogiannis.github.io/jvecfor/reference/JvecforIndex-class.md)
object.

A named list with `index` (n-by-k integer matrix or NULL) and `distance`
(n-by-k numeric matrix or NULL).

## Methods (by generic)

- `show(JvecforParam)`: Print a summary of the parameter object.

- `buildIndex(JvecforParam)`: Build a JvecforIndex from a data matrix.

- `findKnnFromIndex(JvecforIndex)`: Find k-nearest neighbors using a
  JvecforIndex.

## Functions

- `JvecforParam()`: Constructor for JvecforParam objects.

## Slots

- `type`:

  Character. `"ann"` (HNSW-DiskANN, default) or `"knn"` (VP-tree exact).

- `M`:

  Integer. HNSW max connections per node. Default 16L.

- `ef.search`:

  Integer. HNSW beam width (0 = auto). Default 0L.

- `oversample.factor`:

  Numeric. Oversampling multiplier (\>= 1). Default 1.0.

- `pq.subspaces`:

  Integer. Product-quantization subspaces (0 = disabled). Default 0L.

- `verbose`:

  Logical. Enable Java progress logging. Default FALSE.

## Supported distance metrics

`"Euclidean"` and `"Cosine"` (title-case, following BiocNeighbors
convention). The jvecfor-specific `"dot_product"` metric is only
available via
[`fastFindKNN`](https://gkanogiannis.github.io/jvecfor/reference/fastFindKNN.md)
directly.

## Limitations

- `queryKNN` is not supported. The Java backend performs self-KNN only
  (all points query against all points in a single JVM invocation).

- The index built by `buildIndex` stores the data matrix in R memory;
  the actual Java HNSW/VP-tree index is rebuilt each time `findKNN` is
  called.

## See also

[`fastFindKNN`](https://gkanogiannis.github.io/jvecfor/reference/fastFindKNN.md)
for the standalone function with full parameter control including
`dot_product` metric.

## Examples

``` r
library(BiocNeighbors)
p <- JvecforParam()
p
#> JvecforParam
#>   distance: Euclidean 
#>   type: ann 
#>   M: 16 
#>   ef.search: 0 
#>   oversample.factor: 1 
#>   pq.subspaces: 0 

# Custom parameters
p2 <- JvecforParam(type = "knn", distance = "Cosine", M = 32L)

# Use with BiocNeighbors (requires Java >= 20):
# res <- findKNN(X, k = 10, BNPARAM = JvecforParam())
```
