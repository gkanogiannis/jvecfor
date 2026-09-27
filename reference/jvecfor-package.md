# jvecfor: Fast K-Nearest Neighbor Search for Single-Cell Analysis

Drop-in replacement for
[`BiocNeighbors::findKNN`](https://rdrr.io/pkg/BiocNeighbors/man/findKNN.html)
using the jvecfor Java library (HNSW-DiskANN approximate and VP-tree
exact methods). Achieves approximately 2x speedup over Annoy-based
search at n \>= 50K cells. Convenience wrappers delegate SNN/KNN graph
construction to the bluster package.

## Main functions

- [`fastFindKNN`](https://gkanogiannis.github.io/jvecfor/reference/fastFindKNN.md):

  KNN search – returns index + distance matrices.

- [`fastMakeSNNGraph`](https://gkanogiannis.github.io/jvecfor/reference/fastMakeSNNGraph.md):

  KNN -\> SNN graph via bluster.

- [`fastMakeKNNGraph`](https://gkanogiannis.github.io/jvecfor/reference/fastMakeKNNGraph.md):

  KNN -\> KNN graph via bluster.

- [`JvecforParam`](https://gkanogiannis.github.io/jvecfor/reference/JvecforParam-class.md):

  BiocNeighbors parameter class for drop-in integration with scran,
  scater, etc.

- [`jvecfor_setup`](https://gkanogiannis.github.io/jvecfor/reference/jvecfor_setup.md):

  Install a custom jvecfor JAR.

## Options

- `jvecfor.verbose`:

  Logical. Enable Java/jvecfor progress logging globally. Default
  `FALSE`.

- `jvecfor.jar`:

  Character. Path to a custom jvecfor JAR file. Overrides the bundled
  JAR in `inst/java/`.

## See also

Useful links:

- <https://github.com/gkanogiannis/jvecfor>

- Report bugs at <https://github.com/gkanogiannis/jvecfor/issues>

## Author

**Maintainer**: Anestis Gkanogiannis <anestis@gkanogiannis.com>
([ORCID](https://orcid.org/0000-0002-6441-0688))
