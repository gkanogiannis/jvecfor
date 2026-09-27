# JvecforIndex: BiocNeighbors Index Class for jvecfor

A
[BiocNeighborIndex](https://rdrr.io/pkg/BiocNeighbors/man/BiocNeighborIndex.html)
subclass storing the data matrix and `JvecforParam` parameters. The
actual Java HNSW/VP-tree index is built on-the-fly when
[`findKNN`](https://rdrr.io/pkg/BiocNeighbors/man/findKNN.html) is
called.

## Slots

- `data`:

  Numeric matrix (rows = observations, cols = features).

- `param`:

  A
  [`JvecforParam`](https://gkanogiannis.github.io/jvecfor/reference/JvecforParam-class.md)
  object.

- `names`:

  Row names from the original matrix, or NULL.

## See also

[`JvecforParam`](https://gkanogiannis.github.io/jvecfor/reference/JvecforParam-class.md)
