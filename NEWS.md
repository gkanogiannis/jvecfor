# Changes in jvecfor 0.99.0

* Submitted to Bioconductor.
* Initial implementation of `fastFindKNN()`, drop-in replacement for
  `BiocNeighbors::findKNN` using the jvecfor Java backend (HNSW-DiskANN and VP-tree).
* Added `fastMakeSNNGraph()` and `fastMakeKNNGraph()` convenience wrappers
  delegating graph construction to the bluster package.
* Supports euclidean, cosine, and dot_product distance metrics.
* Multi-threaded index build and search via Java ForkJoinPool.
* Optional Product Quantization (PQ) for approximately 4-8x search speedup.
