.jvecfor_jar <- function() {
    # Priority 1: user-set option
    opt <- getOption("jvecfor.jar")
    if (!is.null(opt) && nzchar(opt)) {
        if (!file.exists(opt))
            stop("jvecfor.jar option points to non-existent file: ", opt)
        return(opt)
    }

    # Priority 2: user-installed JAR in R_user_dir
    user_dir <- tools::R_user_dir("jvecfor", "data")
    if (dir.exists(user_dir)) {
        jars <- dir(user_dir, pattern = "^jvecfor-[0-9]+.*\\.jar$",
                    full.names = TRUE)
        if (length(jars) > 0L) return(jars[[1L]])
    }

    # Priority 3: bundled JAR in inst/java/ of installed package
    java_dir <- system.file("java", package = "jvecfor")
    if (nzchar(java_dir)) {
        jars <- dir(java_dir, pattern = "^jvecfor-[0-9]+.*\\.jar$",
                    full.names = TRUE)
        if (length(jars) > 0L) return(jars[[1L]])
    }

    stop(
        "jvecfor JAR not found. Run jvecfor_setup() to install the JAR, or ",
        "set options(jvecfor.jar = '/path/to/jvecfor-X.Y.Z.jar')."
    )
}

.jvecfor_version <- function() {
    # Extract semver from the installed JAR filename (e.g. "jvecfor-1.0.0.jar")
    jar <- tryCatch(.jvecfor_jar(), error = function(e) NULL)
    if (!is.null(jar)) {
        m <- regmatches(
            basename(jar),
            regexpr("[0-9]+\\.[0-9]+\\.[0-9]+", basename(jar))
        )
        if (length(m) == 1L) return(m)
    }
    stop(
        "Cannot determine jvecfor backend version:",
        "    no JAR found in inst/java/. ",
        "Run jvecfor_setup() first."
    )
}

.check_java <- function() {
    java <- Sys.which("java")
    if (!nzchar(java)) {
        stop(
            "Java not found on PATH. Install Java >= 20 and ensure 'java' ",
            "is on PATH. See https://adoptium.net for distributions."
        )
    }

    # Verify Java >= 20. `java -version` writes to stderr across all JVMs.
    ver_raw <- tryCatch(
        system2(java, args = "-version", stdout = TRUE, stderr = TRUE),
        error = function(e) character(0)
    )
    ver_line <- ver_raw[grepl("version", ver_raw, ignore.case = TRUE)][1L]
    if (!is.na(ver_line)) {
        m <- regmatches(ver_line, regexpr('"[0-9]+', ver_line))
        if (length(m) == 1L) {
            major <- as.integer(sub('"', "", m, fixed = TRUE))
            # Java 8 reports "1.8.0_xxx" -- extract feature version after "1."
            if (!is.na(major) && major == 1L) {
                m2 <- regmatches(ver_line, regexpr('"1\\.[0-9]+', ver_line))
                if (length(m2) == 1L)
                    major <- as.integer(sub('"1.', "", m2, fixed = TRUE))
            }
            if (!is.na(major) && major < 20L) {
                stop(
                    "Java >= 20 is required (found Java ", major, "). ",
                    "Install from https://adoptium.net"
                )
            }
        }
    }
    # If version parsing fails entirely, proceed -- Java itself will error.
    invisible(java)
}

.write_tsv <- function(mat, file) {
    fwrite(
        as.data.frame(mat),
        file      = file,
        sep       = "\t",
        col.names = FALSE,
        nThread   = 1L
    )
}

.write_mtx <- function(X, file) {
    if (!inherits(X, "dgCMatrix")) {
        X <- as(X, "dgCMatrix")
    }
    writeMM(X, file)
}

.write_bin <- function(mat, file) {
    con <- file(file, "wb")
    on.exit(close(con))
    writeBin(as.integer(nrow(mat)), con, size = 4L, endian = "big")
    writeBin(as.integer(ncol(mat)), con, size = 4L, endian = "big")
    # t(mat) then as.vector gives row-major order
    writeBin(as.vector(t(mat)), con, size = 8L, endian = "big")
}

.write_input <- function(X, format) {
    ext <- switch(format,
        mtx = ".mtx",
        bin = ".bin",
        tsv = ".tsv",
        stop("Unknown format: ", format)
    )
    file <- tempfile(fileext = ext)
    switch(format,
        mtx = .write_mtx(X, file),
        bin = .write_bin(X, file),
        tsv = .write_tsv(X, file)
    )
    file
}

.serialize_bin <- function(mat) {
    con <- rawConnection(raw(0L), "wb")
    on.exit(close(con))
    writeBin(as.integer(nrow(mat)), con, size = 4L, endian = "big")
    writeBin(as.integer(ncol(mat)), con, size = 4L, endian = "big")
    writeBin(as.vector(t(mat)), con, size = 8L, endian = "big")
    rawConnectionValue(con)
}

.serialize_mtx <- function(X) {
    if (!inherits(X, "dgCMatrix")) {
        X <- as(X, "dgCMatrix")
    }
    # Convert to triplet form for coordinate output
    Xt <- as(X, "TsparseMatrix")
    nnz <- length(Xt@x)
    header <- paste0(
        "%%MatrixMarket matrix coordinate real general\n",
        nrow(X), " ", ncol(X), " ", nnz, "\n"
    )
    # Xt@i and Xt@j are 0-indexed; MTX is 1-indexed
    # Use %.17g to preserve full double precision
    entries <- paste(
        Xt@i + 1L, Xt@j + 1L, sprintf("%.17g", Xt@x),
        collapse = "\n"
    )
    charToRaw(paste0(header, entries, "\n"))
}

.read_matrix_from_text <- function(lines) {
    as.matrix(fread(
        input  = paste(lines, collapse = "\n"),
        header = FALSE,
        sep    = "\t"
    ))
}

#' Install a Custom jvecfor JAR
#'
#' Copies a custom jvecfor JAR to the user data directory
#' (\code{tools::R_user_dir("jvecfor", "data")}). The bundled JAR in
#' \code{inst/java/} is used by default; call \code{jvecfor_setup()} only
#' to override it with a custom build. Alternatively, set
#' \code{options(jvecfor.jar = "/path/to/jvecfor.jar")} for a session-level
#' override without copying.
#'
#' @param jar_path Path to the jvecfor JAR. If \code{NULL}, auto-searches
#'   \code{java/jvecfor/target/jvecfor-*.jar} relative to the working
#'   directory (works when developing inside the source tree).
#'
#' @return Invisibly returns the path to the installed JAR.
#'
#' @examples
#' # Show where custom JARs are stored
#' tools::R_user_dir("jvecfor", "data")
#'
#' # List bundled JARs
#' dir(system.file("java", package = "jvecfor"), pattern = "*.jar")
#'
#' @export
jvecfor_setup <- function(jar_path = NULL) {
    if (is.null(jar_path)) {
        candidates <- Sys.glob(
            file.path("java", "jvecfor", "target",
                        "jvecfor-[0-9]*.jar")
        )
        candidates <- c(
            candidates,
            Sys.glob(
                file.path("jvecfor", "java", "jvecfor", "target",
                            "jvecfor-[0-9]*.jar")
            )
        )
        candidates <- grep(
            "-jar-with-dependencies|-sources|-javadoc|original-",
            candidates, value = TRUE, invert = TRUE
        )
        candidates <- candidates[file.exists(candidates)]
        if (length(candidates) == 0L) {
            stop(
                "Could not auto-find jvecfor JAR. Build with 'mvn package' ",
                "first (from java/jvecfor/), or pass jar_path explicitly."
            )
        }
        jar_path <- candidates[[1L]]
    }

    if (!file.exists(jar_path)) stop("JAR not found: ", jar_path)

    ver_m <- regmatches(
        basename(jar_path),
        regexpr("[0-9]+\\.[0-9]+\\.[0-9]+", basename(jar_path))
    )
    if (length(ver_m) == 0L)
        stop("Cannot parse version from JAR filename: ", basename(jar_path))

    dest_dir <- tools::R_user_dir("jvecfor", "data")
    if (!dir.exists(dest_dir))
        dir.create(dest_dir, recursive = TRUE)

    dest <- file.path(dest_dir, paste0("jvecfor-", ver_m, ".jar"))
    file.copy(jar_path, dest, overwrite = TRUE)
    message("jvecfor: JAR installed to ", dest)
    invisible(dest)
}

# Package hooks ---------------------------------------------------------------

.onLoad <- function(libname, pkgname) {
    # Set default package options if not already set by the user
    op <- options()
    defaults <- list(
        jvecfor.verbose = FALSE,
        jvecfor.jar     = NULL
    )
    toset <- !(names(defaults) %in% names(op))
    if (any(toset)) options(defaults[toset])
    invisible(NULL)
}

.onAttach <- function(libname, pkgname) {
    # Soft-warn on attach if Java is unavailable rather than failing silently
    java <- Sys.which("java")
    if (!nzchar(java)) {
        packageStartupMessage(
            "jvecfor: Java not found on PATH. ",
            "Install Java >= 20 from https://adoptium.net ",
            "before calling fastFindKNN() or related functions."
        )
    }
    invisible(NULL)
}
