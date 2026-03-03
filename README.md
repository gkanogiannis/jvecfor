# jvecfor

<p align="center">
  <img src="logo.svg" width="160" alt="jvecfor logo"/>
</p>

<!-- badges: start -->
[![Bioc history](https://bioconductor.org/shields/years-in-bioc/jvecfor.svg)](https://bioconductor.org/packages/release/bioc/html/jvecfor.html#since)[![Bioc downloads rank](https://bioconductor.org/shields/downloads/release/jvecfor.svg)](http://bioconductor.org/packages/stats/bioc/jvecfor/)
<!-- badges: end -->

> **jvecfor** — a fast R/Bioconductor package for k-nearest-neighbor (KNN) and
> shared-nearest-neighbor (SNN) graph construction in single-cell RNA-seq workflows.
> Drop-in replacement for `BiocNeighbors::findKNN` + `bluster::makeSNNGraph`,
> powered by a Java backend via [jvector](https://github.com/jbellis/jvector) (HNSW-DiskANN),
> SIMD-accelerated with AVX2/AVX-512).

---

## About

`jvecfor` is a R/Bioconductor package that exposes a clean R API
for fast KNN search and graph construction, delegating heavy computation to the `jvecfor` Java library (in `java/`).

- The R package lives in this directory (`./`)
- The Java backend lives in `java/jvecfor/`
- Benchmark scripts live in `benchmark/`

---

## Features

- **`fastFindKNN()`** — drop-in replacement for `BiocNeighbors::findKNN`; returns index + distance matrices
- **`fastMakeSNNGraph()`** — convenience wrapper: KNN → `bluster::neighborsToSNNGraph`
- **`fastMakeKNNGraph()`** — unweighted KNN graph (upper-triangle sparse)
- **Approximate KNN** via HNSW-DiskANN — SIMD-accelerated with AVX2/AVX-512 on supported CPUs
- **Exact KNN** via VP-tree — for ground-truth validation or small datasets
- **Product Quantization (PQ)** — optional ~4–8× search speedup with minimal recall loss
- **Fully parallel** — index build and search both use a shared `ForkJoinPool`
- **faster at n ≥ 50K** than `BiocNeighbors` (Annoy) — ~1.5× at 100K, ~2× at 500K+ (JVM startup overhead dominates at small n)

---

## Requirements

- R ≥ 4.5.0
- Java ≥ 21 (tested on Java 25) — must be on `PATH`
- Bioconductor packages: `bluster`, `BiocNeighbors` (suggested)
- Maven 3.8+ (to build the Java backend from source)

---

## Installation

### 1. Build the Java backend

```bash
make install    # produces inst/java/jvecfor-x.y.z.jar
```

### 2. Install the R package

```r
# From Bioconductor (once submitted):
BiocManager::install("jvecfor")

# From source (development):
devtools::install("path/to/jvecfor")
```

### 3. Copy the JAR into the package

```r
library(jvecfor)
jvecfor_setup()   # auto-finds java/jvecfor/target/jvecfor-x.y.z.jar
# or: jvecfor_setup(jar_path = "path/to/jvecfor-x.y.z.jar")
```

---

## Quick Start

```r
library(jvecfor)

# Simulate 10K cells × 50 PCs
set.seed(42)
pca <- matrix(rnorm(10000 * 50), nrow = 10000)

# Find 15 nearest neighbors (approximate HNSW-DiskANN)
result <- fastFindKNN(pca, k = 15)
str(result)
# List of 2
#  $ index: int [1:10000, 1:15] ...   # 1-indexed neighbor indices
#  $ distance: num [1:10000, 1:15] ...   # distances

# Build SNN graph (Seurat/Bioconductor-compatible)
g <- fastMakeSNNGraph(pca, k = 15, snn.type = "rank")
# Returns an igraph object

# Build KNN graph (unweighted)
g_knn <- fastMakeKNNGraph(pca, k = 15)
```

---

## Full R API

| Function | Description |
|----------|-------------|
| `fastFindKNN(X, k, type, metric, ...)` | KNN search — returns `list(index, distance)` |
| `fastMakeSNNGraph(X, k, ..., snn.type)` | KNN → SNN graph via `bluster::neighborsToSNNGraph` |
| `fastMakeKNNGraph(X, k, ...)` | KNN → unweighted upper-triangle sparse KNN graph |
| `jvecfor_setup(jar_path)` | Copy jvecfor JAR into package installation |

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | `15` | Number of nearest neighbors |
| `type` | `"ann"` | `"ann"` (HNSW-DiskANN approximate) or `"knn"` (VP-tree exact) |
| `metric` | `"euclidean"` | `"euclidean"`, `"cosine"`, or `"dot_product"` (ANN only) |
| `num.threads` | auto | Thread count; NULL = `BiocParallel::bpworkers(BPPARAM)` |
| `ef.search` | `0` | HNSW-DiskANN beam width (0 = auto: `max(k+1, 3k)`) |
| `M` | `16` | HNSW-DiskANN max connections per node (32 improves recall for high-dim) |
| `oversample.factor` | `1.0` | Fetch `ceil(ef × factor)` candidates, return top k |
| `pq.subspaces` | `0` | Product Quantization subspaces (0 = off; try `dims/2`) |
| `verbose` | `FALSE` | Enable Java/jvecfor progress logging |

---

## HNSW-DiskANN Tuning

| Parameter | Effect |
|-----------|--------|
| `M` | Graph connectivity. Default 16; 32 improves recall for high-dimensional data (~2× more memory, slower build). |
| `ef.search` | Beam width during search. Default auto = `max(k+1, 3k)`. Higher → better recall, slower search. |
| `oversample.factor` | Fetch `ceil(ef × factor)` candidates, keep top k. `2.0` doubles candidates; improves recall at proportional cost. |
| `pq.subspaces` | Product Quantization subspaces. Replaces exact distance computations during beam traversal with fast table lookups, then reranks exact. Set to `dims/2` as starting point. |

---

## Java Backend

The Java backend is in `java/jvecfor/`. It is a standalone CLI tool and Java library built on
[jvector](https://github.com/jbellis/jvector) 4.0.0-rc.8.

```bash
# Build
cd java/jvecfor
mvn package -DskipTests        # fast
mvn package                    # build + run all 62 tests

# Direct CLI usage
java --add-modules jdk.incubator.vector \
     -jar java/jvecfor/target/jvecfor-x.y.z.jar \
     -i pca.tsv -k 15 > knn.tsv
```

The fat JAR (`jvecfor-x.y.z.jar`) shades all dependencies and is the only file
needed at runtime.

---

## Benchmarks

The `benchmark/` directory contains R scripts comparing jvecfor against
`BiocNeighbors::findKNN` (Annoy) on the 1.3 M-cell TENxBrainData mouse neuron dataset.

| n cells | jvecfor (HNSW-DiskANN) | BiocNeighbors (Annoy) | Speedup |
|---------|------------------------|----------------------|---------|
| 1K | ~0.43 s | ~0.03 s | 0.07× (JVM startup) |
| 10K | ~0.85 s | ~0.36 s | 0.43× (JVM startup) |
| 50K | ~1.74 s | ~2.05 s | 1.2× |
| 100K | ~2.9 s | ~4.4 s | 1.5× |
| 500K | ~15 s | ~30 s | 2.0× |
| 1.3M | ~49 s | ~90 s | 1.8× |

*(k=15, euclidean, 50 PCs; from `benchmark/benchmark_results_summary.csv`.
JVM startup overhead ~0.4 s dominates at small n; jvecfor is faster at n ≥ 50K.)*

**Memory:** Java uses substantially less RAM at small-to-medium scale
(e.g. 146 MB vs 1,441 MB at 1K; 517 MB vs 1,487 MB at 10K), converging
near 3.2 GB each at 1.3M cells.

![Timing benchmark](benchmark/benchmark_results.png)

![Memory benchmark](benchmark/benchmark_memory.png)

```bash
# Run dev benchmark (20k cells, fast)
cd benchmark
Rscript benchmark20k.R

# Run full benchmark (1.3M cells, slow)
bash run_benchmark.sh
```

## Citation

If you use `fastreeR` in your research, please cite:

> Anestis Gkanogiannis (2016)
> _A scalable assembly-free variable selection algorithm for biomarker discovery from metagenomes_  
> _BMC Bioinformatics_ 17, 311.  
> <https://doi.org/10.1186/s12859-016-1186-3>  
> <https://github.com/gkanogiannis/fastreeR>

---

## Author

Anestis Gkanogiannis  
Bioinformatics/ML Scientist  
Linkedin: [https://www.linkedin.com/in/anestis-gkanogiannis/](https://www.linkedin.com/in/anestis-gkanogiannis/)  
Website: [https://github.com/gkanogiannis](https://github.com/gkanogiannis)  
ORCID: [0000-0002-6441-0688](https://orcid.org/0000-0002-6441-0688)

---

## License

`jvecfor` is licensed under the GNU General Public License v3.0.  
See the [LICENSE](LICENSE.md) file for details.

---
