#!/usr/bin/env Rscript
# ============================================================
#  Benchmark: jvecfor (Java/HNSW-DiskANN) vs BiocNeighbors (R/Annoy)
#  Measures findKNN time + RAM for both methods, and optionally
#  makeSNNGraph (bluster) — controlled by RUN_SNN_GRAPH below.
#
#  Dataset : 1.3 M Mouse Brain Neurons — 10x Genomics
#            TENxBrainData Bioconductor package (Zeisel et al. 2018)
#            https://doi.org/10.1126/science.aaa6090
#
#  Method  :
#    1. Download & log-normalise the full 1.3 M-cell dataset.
#    2. Run truncated SVD (RandomParam) to get 50 PCs  -> cache to disk.
#    3. For each n in N_VALUES, subsample the PCA matrix and run:
#         * R  findKNN  : BiocNeighbors::findKNN(BNPARAM=AnnoyParam())
#         * Java findKNN  : jvecfor CLI -g ann   (jvector HNSW-DiskANN)
#         * (optional) bluster::neighborsToSNNGraph for both
#    4. Save results table (CSV) and log-log wall-clock + memory plots (PDF).
#
#  Note: Java times include ~200-400 ms JVM startup per rep. For n >= 50K
#  this is <5% of total; for small n it dominates. For JVM-startup-free
#  comparison, use the JMH benchmark (AppBenchmark.java).
#
#  Usage   : Rscript benchmark.R
#            (run from the benchmark/ directory; fat jar must exist at
#             ../jvecfor/target/jvecfor-<version>.jar -- run  mvn package -q  first)
# ============================================================

# ── 0. Configuration ─────────────────────────────────────────────────────────

K                <- 15L               # number of neighbors
N_PCS            <- 50L               # PCA dimensions
METRIC           <- "euclidean"       # used by Java side
N_THREADS        <- max(1L, parallel::detectCores() - 1L)
# jvecfor HNSW-DiskANN tuning parameters
HNSW_M           <- 16L               # max connections per node (default 16; 32 improves recall)
HNSW_EF_SEARCH   <- 0L                # search beam width (0 = auto: max(k+1, 3*k) = 45 for k=15)
HNSW_OVERSAMPLE  <- 1.0              # oversampling factor: fetches ceil(ef*factor) candidates
HNSW_PQ_SUBSPACES <- 0L              # Product Quantization subspaces (0 = off; try dims/2, e.g. 25 for 50-dim)

# Set FALSE to skip SNN graph construction and only benchmark findKNN.
RUN_SNN_GRAPH    <- FALSE

# Cell counts to benchmark
N_VALUES  <- as.integer(c(1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 1.3e6))

N_REPS    <- 3L                       # timing repetitions per (n, method, phase)
PLOT_MAX_N   <- 10000L                # skip igraph plot for n above this threshold
PLOT_GRAPHS  <- FALSE                 # set TRUE to save igraph PDFs (slow for large n)

.backend_ver <- read.dcf(file.path("..", "DESCRIPTION"), fields = "Backend")[[1L]]
JAVA_JAR     <- normalizePath(
  file.path("..", "jvecfor", "target", paste0("jvecfor-", .backend_ver, ".jar")),
  mustWork = FALSE
)
PCA_CACHE <- "pca_1M_brain.rds"      # cached PCA matrix (n_cells x N_PCS)

# ── 1. Package setup ─────────────────────────────────────────────────────────

cran_pkgs <- c("ggplot2", "dplyr", "tibble", "scales", "data.table", "igraph")
bioc_pkgs <- c(
    "TENxBrainData",   # 1.3 M mouse neurons (HDF5, lazy-loaded)
    "scuttle",         # logNormCounts
    "BiocSingular",    # runSVD / RandomParam
    "BiocNeighbors",   # findKNN / AnnoyParam
    "BiocParallel",    # MulticoreParam / SnowParam
    "DelayedArray",    # setAutoBPPARAM -- parallelises HDF5 chunk reads
    "bluster"          # neighborsToSNNGraph
)

invisible(lapply(cran_pkgs, function(p) {
    if (!requireNamespace(p, quietly = TRUE)) install.packages(p, quiet = TRUE)
}))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
invisible(lapply(bioc_pkgs, function(p) {
    if (!requireNamespace(p, quietly = TRUE))
        BiocManager::install(p, ask = FALSE, quiet = TRUE)
}))

suppressPackageStartupMessages({
    library(TENxBrainData)
    library(scuttle)
    library(BiocSingular)
    library(BiocNeighbors)
    library(BiocParallel)
    library(DelayedArray)
    library(bluster)
    library(igraph)
    library(ggplot2)
    library(dplyr)
    library(tibble)
    library(scales)
    library(data.table)
})

# ── 2. Parallel backend (defined early so PCA also benefits) ─────────────────

# BPPARAM_IO: used for HDF5-backed DelayedArray operations (logNormCounts, runSVD).
# SnowParam/MulticoreParam both fail on macOS with HDF5 because worker processes
# cannot inherit or reopen HDF5 file handles from the parent. SerialParam is safe
# everywhere; the PCA step is cached after the first run so its speed is irrelevant.
BPPARAM_IO <- SerialParam()

# BPPARAM: used for in-memory operations (findKNN). SnowParam is safe on all
# platforms since no HDF5 handles are involved. MulticoreParam is faster on Linux.
BPPARAM <- if (N_THREADS > 1L) {
    if (.Platform$OS.type == "unix" && Sys.info()[["sysname"]] != "Darwin") {
        MulticoreParam(N_THREADS)
    } else {
        SnowParam(N_THREADS)
    }
} else {
    SerialParam()
}

DelayedArray::setAutoBPPARAM(BPPARAM_IO)

message(sprintf("I/O backend: %s  |  findKNN backend: %s  threads=%d",
                class(BPPARAM_IO), class(BPPARAM), N_THREADS))

# ── 3. Load / compute PCA (one-time; cached afterwards) ──────────────────────

if (file.exists(PCA_CACHE)) {
    message("Loading cached PCA from '", PCA_CACHE, "' ...")
    pca_mat <- readRDS(PCA_CACHE)
} else {
    message("Downloading TENxBrainData (1.3 M mouse neurons) ...")
    message("  This is a ~4 GB HDF5 file; initial download may take several minutes.")
    sce <- TENxBrainData()   # SingleCellExperiment, HDF5-backed, lazy
    message(sprintf("  Loaded: %d cells x %d genes", ncol(sce), nrow(sce)))

    message("Filtering unexpressed genes (rowSums(counts) == 0) ...")
    expressed <- Matrix::rowSums(counts(sce)) > 0
    sce <- sce[expressed, ]
    message(sprintf("  Kept %d / %d genes", sum(expressed), length(expressed)))

    message("Log-normalising ...")
    sce <- scuttle::logNormCounts(sce, BPPARAM = BPPARAM_IO)

    message(sprintf(
        "Running truncated SVD (%d PCs via RandomParam) on %s cells x %d genes ...",
        N_PCS, format(ncol(sce), big.mark = ","), nrow(sce)
    ))
    set.seed(42L)
    svd_res <- BiocSingular::runSVD(
        t(assay(sce, "logcounts")),
        k       = N_PCS,
        BSPARAM = RandomParam(),
        BPPARAM = BPPARAM_IO,
        center  = TRUE,
        scale   = FALSE
    )
    pca_mat <- svd_res$u
    rownames(pca_mat) <- colnames(sce)
    rm(sce); gc()

    message("Saving PCA to '", PCA_CACHE, "' ...")
    saveRDS(pca_mat, PCA_CACHE)
}

n_available <- nrow(pca_mat)
message(sprintf(
    "PCA matrix ready: %s cells x %d PCs",
    format(n_available, big.mark = ","), ncol(pca_mat)
))

# Restrict N_VALUES to what we actually have
N_VALUES <- N_VALUES[N_VALUES <= n_available]

graphs_dir <- "graphs"
dir.create(graphs_dir, showWarnings = FALSE)

# ── Helper: read current RSS of the running R process ─────────────────────────
# Returns current RSS in Mb, or NA on unsupported platforms.
# Linux: reads VmRSS from /proc/self/status
# macOS: queries the current PID via ps
.read_r_rss_mb <- function() {
    sysname <- Sys.info()[["sysname"]]
    if (sysname == "Linux") {
        lines <- readLines("/proc/self/status", warn = FALSE)
        line  <- grep("^VmRSS:", lines, value = TRUE)
        if (!length(line)) return(NA_real_)
        kb <- suppressWarnings(as.numeric(sub("[^0-9]*([0-9]+).*", "\\1", line)))
        if (is.na(kb)) NA_real_ else kb / 1024
    } else if (sysname == "Darwin") {
        out <- system2("ps", c("-o", "rss=", "-p", as.character(Sys.getpid())),
                       stdout = TRUE, stderr = FALSE)
        kb <- suppressWarnings(as.numeric(trimws(tail(out, 1L))))
        if (is.na(kb)) NA_real_ else kb / 1024
    } else {
        NA_real_
    }
}

# ── Helper: measure peak RSS of a Java process via /usr/bin/time ──────────────
# Returns peak RSS in Mb, or NA on unsupported platforms.
# macOS: /usr/bin/time -l  → "NNNN  maximum resident set size" (bytes) on stderr
# Linux: /usr/bin/time -f "%M" → peak RSS in KB as last stderr line
.measure_java_rss_mb <- function(args_vec) {
    sysname  <- Sys.info()[["sysname"]]
    time_bin <- "/usr/bin/time"
    if (!sysname %in% c("Darwin", "Linux") || !file.exists(time_bin)) return(NA_real_)
    if (sysname == "Darwin") {
        stderr_out <- system2(time_bin, c("-l", "java", args_vec),
                              stdout = FALSE, stderr = TRUE)
        line <- grep("maximum resident set size", stderr_out, value = TRUE)
        if (!length(line)) return(NA_real_)
        bytes <- suppressWarnings(
            as.numeric(trimws(strsplit(trimws(line[[1L]]), "\\s+")[[1L]][[1L]]))
        )
        bytes / 1024^2
    } else {
        stderr_out <- system2(time_bin, c("-f", "%M", "java", args_vec),
                              stdout = FALSE, stderr = TRUE)
        kb <- suppressWarnings(as.numeric(trimws(tail(stderr_out, 1L))))
        if (is.na(kb)) NA_real_ else kb / 1024
    }
}

# ── 4. Benchmark loop ─────────────────────────────────────────────────────────

message(sprintf(
    "\nBenchmark parameters: k=%d  metric=%s  threads=%d  HNSW_M=%d  ef_search=%d  oversample=%.2f  pq=%d  snn=%s",
    K, METRIC, N_THREADS, HNSW_M, HNSW_EF_SEARCH, HNSW_OVERSAMPLE, HNSW_PQ_SUBSPACES,
    if (RUN_SNN_GRAPH) "yes" else "no"
))

results <- tibble(
    n        = integer(),
    method   = character(),
    phase    = character(),   # "findKNN" | "makeSNNGraph"
    rep      = integer(),
    time_sec = numeric()
)

# One row per (n, method, phase): peak memory in Mb measured on the last rep.
# findKNN (both R and Java): current process RSS after the run.
# makeSNNGraph (both): gc() heap peak.
mem_results <- tibble(
    n      = integer(),
    method = character(),
    phase  = character(),
    mem_mb = numeric()
)

for (n in N_VALUES) {
    message(sprintf("\n══════  n = %s  ══════", format(n, big.mark = ",")))
    data_sub <- pca_mat[seq_len(n), , drop = FALSE]
    nn_r     <- NULL
    knn_java <- NULL
    g_r      <- NULL
    g_java   <- NULL

    # ── R findKNN (BiocNeighbors/Annoy) ──────────────────────────────────────
    message("  R findKNN (BiocNeighbors/Annoy) ...")
    for (rep in seq_len(N_REPS)) {
        if (rep == N_REPS) gc(reset = TRUE, full = TRUE)
        t0     <- proc.time()[["elapsed"]]
        nn.out <- BiocNeighbors::findKNN(
            data_sub,
            k       = K,
            BNPARAM = AnnoyParam(),
            BPPARAM = BPPARAM
        )
        elapsed <- proc.time()[["elapsed"]] - t0
        message(sprintf("    rep %d  %.2f s", rep, elapsed))
        results <- bind_rows(results, tibble(
            n = n, method = "R (BiocNeighbors/Annoy)",
            phase = "findKNN", rep = rep, time_sec = elapsed
        ))
        if (rep == N_REPS) nn_r <- nn.out
        rm(nn.out)
    }

    # R findKNN memory (RSS via /proc/self/status or ps)
    message("  R findKNN memory (RSS) ...")
    gc(full = TRUE)
    r_rss_mb <- .read_r_rss_mb()
    mem_results <- bind_rows(mem_results, tibble(
        n = n, method = "R (BiocNeighbors/Annoy)", phase = "findKNN",
        mem_mb = r_rss_mb
    ))
    message(sprintf("    RSS: %.0f Mb", r_rss_mb))

    # ── R SNN graph (bluster, using R KNN indices) ────────────────────────────
    if (RUN_SNN_GRAPH) {
        message("  R makeSNNGraph (bluster, R indices) ...")
        for (rep in seq_len(N_REPS)) {
            if (rep == N_REPS) gc(reset = TRUE, full = TRUE)
            t0      <- proc.time()[["elapsed"]]
            g       <- bluster::neighborsToSNNGraph(nn_r$index, type = "rank")
            elapsed <- proc.time()[["elapsed"]] - t0
            message(sprintf("    rep %d  %.2f s", rep, elapsed))
            results <- bind_rows(results, tibble(
                n = n, method = "R (BiocNeighbors/Annoy)",
                phase = "makeSNNGraph", rep = rep, time_sec = elapsed
            ))
            if (rep == N_REPS) {
                g_r <- g
                mi  <- gc(full = TRUE)
                mem_results <- bind_rows(mem_results, tibble(
                    n = n, method = "R (BiocNeighbors/Annoy)", phase = "makeSNNGraph",
                    mem_mb = mi["Vcells", 6] + mi["Ncells", 6]
                ))
            } else rm(g)
        }
    }

    # ── Java findKNN (-g ann) ─────────────────────────────────────────────────
    if (!file.exists(JAVA_JAR)) {
        warning("Fat jar not found: ", JAVA_JAR, "\n  Run: mvn package -q")
    } else {
        message("  Java findKNN (jvecfor/HNSW-DiskANN) ...")
        tsv_in <- tempfile(fileext = ".tsv")
        data.table::fwrite(
            data_sub,
            file      = tsv_in,
            sep       = "\t",
            col.names = FALSE,
            nThread   = N_THREADS
        )

        # no -o: index matrix is streamed from stdout directly into fread (no buffering)
        # Keep args as a vector too — needed for /usr/bin/time RSS measurement.
        java_args_vec <- c("--add-modules", "jdk.incubator.vector",
                           "-jar", JAVA_JAR, "-i", tsv_in,
                           "-k", as.character(K), "-m", METRIC,
                           "-t", as.character(N_THREADS), "-g", "ann",
                           "-M", as.character(HNSW_M),
                           "--ef-search", as.character(HNSW_EF_SEARCH),
                           "--oversample-factor", as.character(HNSW_OVERSAMPLE),
                           "--pq-subspaces", as.character(HNSW_PQ_SUBSPACES))
        java_cmd <- paste(c("java", shQuote(java_args_vec), "2>/dev/null"),
                          collapse = " ")

        knn_java <- NULL
        for (rep in seq_len(N_REPS)) {
            t0      <- proc.time()[["elapsed"]]
            result  <- fread(cmd = java_cmd, header = FALSE)
            elapsed <- proc.time()[["elapsed"]] - t0
            message(sprintf("    rep %d  %.2f s", rep, elapsed))
            results <- bind_rows(results, tibble(
                n = n, method = "Java (jvecfor/HNSW-DiskANN)",
                phase = "findKNN", rep = rep, time_sec = elapsed
            ))
            if (rep == N_REPS) {
                # Bare integer matrix, no dimnames — same structure as nn_r$index.
                knn_java <- matrix(as.integer(unlist(result, use.names = FALSE)),
                                   nrow = nrow(result), ncol = ncol(result))
            }
        }

        # Java peak RSS — measured once after timing reps (does not affect timing).
        message("  Java findKNN memory (RSS via /usr/bin/time) ...")
        java_rss_mb <- .measure_java_rss_mb(java_args_vec)
        mem_results <- bind_rows(mem_results, tibble(
            n = n, method = "Java (jvecfor/HNSW-DiskANN)", phase = "findKNN",
            mem_mb = java_rss_mb
        ))
        message(sprintf("    peak RSS: %.0f Mb", java_rss_mb))

        # Quality check: recall of Java KNN vs R KNN
        if (!is.null(nn_r) && !is.null(knn_java)) {
            recall <- mean(sapply(seq_len(nrow(knn_java)), function(i)
                length(intersect(knn_java[i, ], nn_r$index[i, ])) / K))
            message(sprintf("  KNN recall (Java vs R): %.1f%%", recall * 100))
        }

        unlink(tsv_in)

        # ── Java SNN graph (bluster, using Java KNN indices) ─────────────────
        if (RUN_SNN_GRAPH && !is.null(knn_java)) {
            message("  Java makeSNNGraph (bluster, Java indices) ...")
            for (rep in seq_len(N_REPS)) {
                if (rep == N_REPS) gc(reset = TRUE, full = TRUE)
                t0      <- proc.time()[["elapsed"]]
                g       <- bluster::neighborsToSNNGraph(knn_java, type = "rank")
                elapsed <- proc.time()[["elapsed"]] - t0
                message(sprintf("    rep %d  %.2f s", rep, elapsed))
                results <- bind_rows(results, tibble(
                    n = n, method = "Java (jvecfor/HNSW-DiskANN)",
                    phase = "makeSNNGraph", rep = rep, time_sec = elapsed
                ))
                if (rep == N_REPS) {
                    g_java <- g
                    mi     <- gc(full = TRUE)
                    mem_results <- bind_rows(mem_results, tibble(
                        n = n, method = "Java (jvecfor/HNSW-DiskANN)", phase = "makeSNNGraph",
                        mem_mb = mi["Vcells", 6] + mi["Ncells", 6]
                    ))
                } else rm(g)
            }
        }
    }

    # ── Save outputs for this n ───────────────────────────────────────────────
    tag <- sprintf("n%d", n)
    gf  <- function(fname) file.path(graphs_dir, sprintf("%s_%s.%s", tools::file_path_sans_ext(fname), tag, tools::file_ext(fname)))

    # KNN index matrices (1-indexed, tab-separated, no header)
    if (!is.null(nn_r))     fwrite(as.data.frame(nn_r$index), paste0(gf("knn_R.tsv"),    ".gz"), sep = "\t", col.names = FALSE, compress = "gzip")
    if (!is.null(knn_java)) fwrite(as.data.frame(knn_java),   paste0(gf("knn_Java.tsv"), ".gz"), sep = "\t", col.names = FALSE, compress = "gzip")

    if (RUN_SNN_GRAPH) {
        # SNN graphs: RDS (igraph object) + edge-list TSV (from, to, weight)
        if (!is.null(g_r)) {
            saveRDS(g_r, gf("snn_R.rds"))
            fwrite(igraph::as_data_frame(g_r, what = "edges"), paste0(gf("snn_R_edges.tsv"), ".gz"), sep = "\t", compress = "gzip")
        }
        if (!is.null(g_java)) {
            saveRDS(g_java, gf("snn_Java.rds"))
            fwrite(igraph::as_data_frame(g_java, what = "edges"), paste0(gf("snn_Java_edges.tsv"), ".gz"), sep = "\t", compress = "gzip")
        }

        # Graph comparison statistics
        if (!is.null(g_r) && !is.null(g_java)) {
            edge_key <- function(g) {
                e <- igraph::ends(g, igraph::E(g), names = FALSE)
                paste(pmin(e[, 1L], e[, 2L]), pmax(e[, 1L], e[, 2L]), sep = "-")
            }
            ep_r    <- edge_key(g_r)
            ep_java <- edge_key(g_java)
            n_shared  <- length(intersect(ep_r, ep_java))
            jaccard   <- n_shared / length(union(ep_r, ep_java))
            message(sprintf(
                "  SNN edge Jaccard (R vs Java): %.1f%%  (R:%d  Java:%d  shared:%d edges)",
                jaccard * 100, length(ep_r), length(ep_java), n_shared
            ))
        }

        # igraph plot — BOTH panels use the same layout (computed from g_r with fixed
        # seed) so any visual difference reflects graph structure, not layout randomness.
        if (PLOT_GRAPHS && n <= PLOT_MAX_N && (!is.null(g_r) || !is.null(g_java))) {
            vsize <- max(0.5, 4 - log10(n))
            ref_g <- if (!is.null(g_r)) g_r else g_java
            set.seed(42L)
            lay <- igraph::layout_nicely(ref_g)

            pdf(gf("snn_plot.pdf"), width = 12, height = 6)
            par(mfrow = c(1L, 2L))
            if (!is.null(g_r))
                plot(g_r,    layout = lay, vertex.size = vsize, vertex.label = NA,
                     edge.width = 0.3,
                     main = sprintf("SNN  R/Annoy    n=%s  k=%d", format(n, big.mark = ","), K))
            if (!is.null(g_java))
                plot(g_java, layout = lay, vertex.size = vsize, vertex.label = NA,
                     edge.width = 0.3,
                     main = sprintf("SNN  Java/HNSW-DiskANN n=%s  k=%d", format(n, big.mark = ","), K))
            dev.off()
            message(sprintf("  Saved: %s", gf("snn_plot.pdf")))
        }
    }

    message(sprintf("  Saved: graphs/ outputs for n=%s", format(n, big.mark = ",")))

    rm(data_sub, nn_r, knn_java, g_r, g_java); gc()
}

# ── 5. Summary table ──────────────────────────────────────────────────────────

summary_tbl <- results |>
    group_by(n, method, phase) |>
    summarise(
        median_s = round(median(time_sec), 3),
        min_s    = round(min(time_sec), 3),
        .groups  = "drop"
    ) |>
    left_join(
        mem_results |> mutate(mem_mb = round(mem_mb, 1)),
        by = c("n", "method", "phase")
    )

message("\n── Summary (median wall-clock time + peak memory) ───────────────────────")
print(as.data.frame(summary_tbl), row.names = FALSE)

write.csv(results,     "benchmark_results_raw.csv",     row.names = FALSE)
write.csv(summary_tbl, "benchmark_results_summary.csv", row.names = FALSE)
message("\nSaved: benchmark_results_raw.csv, benchmark_results_summary.csv")

# ── 6. Plot ───────────────────────────────────────────────────────────────────

method_colours <- c(
    "R (BiocNeighbors/Annoy)" = "#E41A1C",
    "Java (jvecfor/HNSW-DiskANN)"     = "#377EB8"
)

params_subtitle <- sprintf(
    "Dataset: 1.3 M mouse neurons (TENxBrainData)  |  k=%d  %d PCs  %d threads  M=%d ef=%d oversample=%.2f  pq=%d",
    K, N_PCS, N_THREADS, HNSW_M, HNSW_EF_SEARCH, HNSW_OVERSAMPLE, HNSW_PQ_SUBSPACES
)

if (RUN_SNN_GRAPH) {
    # Four lines: findKNN only (dashed) + full pipeline (solid), one per method.
    knn_agg <- results |>
        filter(phase == "findKNN") |>
        group_by(n, method) |>
        summarise(median_s = median(time_sec), .groups = "drop") |>
        mutate(pipeline = "findKNN only")

    pipeline_agg <- results |>
        group_by(n, method, rep) |>
        summarise(total_s = sum(time_sec), .groups = "drop") |>
        group_by(n, method) |>
        summarise(median_s = median(total_s), .groups = "drop") |>
        mutate(pipeline = "findKNN + SNN graph")

    plot_data <- bind_rows(knn_agg, pipeline_agg) |>
        mutate(pipeline = factor(pipeline, levels = c("findKNN only", "findKNN + SNN graph")))

    p <- ggplot(plot_data,
                aes(x = n, y = median_s, colour = method, linetype = pipeline, shape = method)) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 3) +
        scale_x_log10(labels = label_number(scale_cut = cut_short_scale()), breaks = N_VALUES) +
        scale_y_log10(labels = label_comma(suffix = " s")) +
        scale_colour_manual(values = method_colours) +
        scale_linetype_manual(values = c("findKNN only" = "dashed", "findKNN + SNN graph" = "solid")) +
        labs(
            title    = "KNN + SNN pipeline: jvecfor (Java/HNSW-DiskANN) vs BiocNeighbors (R/Annoy)",
            subtitle = params_subtitle,
            x = "Cells (n)", y = "Median wall-clock time",
            colour = NULL, linetype = NULL, shape = NULL
        ) +
        theme_bw(base_size = 12) +
        theme(
            legend.position        = "inside",
            legend.position.inside = c(0.22, 0.80),
            legend.background      = element_rect(fill = "white", colour = "grey80")
        )
} else {
    # Two lines: findKNN only, one per method.
    plot_data <- results |>
        filter(phase == "findKNN") |>
        group_by(n, method) |>
        summarise(median_s = median(time_sec), .groups = "drop")

    p <- ggplot(plot_data, aes(x = n, y = median_s, colour = method, shape = method)) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 3) +
        scale_x_log10(labels = label_number(scale_cut = cut_short_scale()), breaks = N_VALUES) +
        scale_y_log10(labels = label_comma(suffix = " s")) +
        scale_colour_manual(values = method_colours) +
        labs(
            title    = "findKNN: jvecfor (Java/HNSW-DiskANN) vs BiocNeighbors (R/Annoy)",
            subtitle = params_subtitle,
            x = "Cells (n)", y = "Median wall-clock time",
            colour = NULL, shape = NULL
        ) +
        theme_bw(base_size = 12) +
        theme(
            legend.position        = "inside",
            legend.position.inside = c(0.22, 0.80),
            legend.background      = element_rect(fill = "white", colour = "grey80")
        )
}

ggsave("benchmark_results.pdf", p, width = 10, height = 5.5)
message("Saved: benchmark_results.pdf")
ggsave("benchmark_results.png", p,
       width = 10, height = 5.5, dpi = 150)
message("Saved: benchmark_results.png")

# ── 7. Memory plot ────────────────────────────────────────────────────────────
#  y-axis: peak RSS in Mb (linear scale)
#  Note: Java = peak process RSS (/usr/bin/time); R = process RSS (ps / /proc)

mem_subtitle <- sprintf(
    "Dataset: 1.3 M mouse neurons (TENxBrainData)  |  k=%d  %d PCs  %d threads  M=%d ef=%d oversample=%.2f  pq=%d",
    K, N_PCS, N_THREADS, HNSW_M, HNSW_EF_SEARCH, HNSW_OVERSAMPLE, HNSW_PQ_SUBSPACES
)

if (RUN_SNN_GRAPH) {
    mem_plot_data <- mem_results |>
        filter(!is.na(mem_mb)) |>
        mutate(phase = factor(phase, levels = c("findKNN", "makeSNNGraph")))

    p_mem <- ggplot(mem_plot_data,
                    aes(x = n, y = mem_mb, colour = method, linetype = phase, shape = method)) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 3) +
        scale_x_log10(labels = label_number(scale_cut = cut_short_scale()), breaks = N_VALUES) +
        scale_y_continuous(labels = label_comma(suffix = " Mb")) +
        scale_colour_manual(values = method_colours) +
        scale_linetype_manual(values = c("findKNN" = "dashed", "makeSNNGraph" = "solid")) +
        labs(
            title    = "Peak memory: jvecfor (Java/HNSW-DiskANN) vs BiocNeighbors (R/Annoy)",
            subtitle = mem_subtitle,
            x = "Cells (n)", y = "Peak memory",
            colour = NULL, linetype = NULL, shape = NULL
        ) +
        theme_bw(base_size = 12) +
        theme(
            legend.position        = "inside",
            legend.position.inside = c(0.22, 0.80),
            legend.background      = element_rect(fill = "white", colour = "grey80")
        )
} else {
    mem_plot_data <- mem_results |>
        filter(!is.na(mem_mb), phase == "findKNN")

    p_mem <- ggplot(mem_plot_data, aes(x = n, y = mem_mb, colour = method, shape = method)) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 3) +
        scale_x_log10(labels = label_number(scale_cut = cut_short_scale()), breaks = N_VALUES) +
        scale_y_continuous(labels = label_comma(suffix = " Mb")) +
        scale_colour_manual(values = method_colours) +
        labs(
            title    = "Peak memory (findKNN): jvecfor (Java/HNSW-DiskANN) vs BiocNeighbors (R/Annoy)",
            subtitle = mem_subtitle,
            x = "Cells (n)", y = "Peak memory",
            colour = NULL, shape = NULL
        ) +
        theme_bw(base_size = 12) +
        theme(
            legend.position        = "inside",
            legend.position.inside = c(0.22, 0.80),
            legend.background      = element_rect(fill = "white", colour = "grey80")
        )
}

ggsave("benchmark_memory.pdf", p_mem, width = 10, height = 5.5)
message("Saved: benchmark_memory.pdf")
ggsave("benchmark_memory.png", p_mem,
       width = 10, height = 5.5, dpi = 150)
message("Saved: benchmark_memory.png")
