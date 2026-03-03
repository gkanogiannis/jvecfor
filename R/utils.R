.jvecfor_jar <- function() {
    # Priority 1: user-set option
    opt <- getOption("jvecfor.jar")
    if (!is.null(opt) && nzchar(opt)) {
        if (!file.exists(opt))
            stop("jvecfor.jar option points to non-existent file: ", opt)
        return(opt)
    }

    # Priority 2: find any versioned JAR in inst/java/ of installed package
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
            "Java not found on PATH. Install Java >= 21 and ensure 'java' ",
            "is on PATH. See https://adoptium.net for distributions."
        )
    }

    # Verify Java >= 21. `java -version` writes to stderr across all JVMs.
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
            if (!is.na(major) && major < 21L) {
                stop(
                    "Java >= 21 is required (found Java ", major, "). ",
                    "Install from https://adoptium.net"
                )
            }
        }
    }
    # If version parsing fails entirely, proceed -- Java itself will error.
    invisible(java)
}

.write_tsv <- function(mat, file) {
    # nThread = 1L: avoids data.table fork conflicts with the Java ForkJoinPool.
    # The bottleneck is Java computation, not TSV serialisation.
    fwrite(
        as.data.frame(mat),
        file      = file,
        sep       = "\t",
        col.names = FALSE,
        nThread   = 1L
    )
}

.read_matrix_from_text <- function(lines) {
    as.matrix(fread(
        input  = paste(lines, collapse = "\n"),
        header = FALSE,
        sep    = "\t"
    ))
}

#' Copy the jvecfor JAR into the jvecfor package installation directory
#'
#' The jvecfor JAR is already bundled in the installed package. Call
#' \code{jvecfor_setup()} only if you have built a custom JAR and want
#' to replace the bundled one.
#'
#' @param jar_path Path to the jvecfor JAR. If \code{NULL}, auto-searches
#'   \code{jvecfor/target/jvecfor-*.jar} relative to the working directory
#'   (works when developing inside the source tree).
#'
#' @return Invisibly returns the path to the installed JAR.
#'
#' @examples
#' # List all JARs bundled in the package
#' dir(system.file("java", package = "jvecfor"), pattern = "*.jar")
#'
#' @export
jvecfor_setup <- function(jar_path = NULL) {
    dest_dir <- system.file("java", package = "jvecfor")
    if (!nzchar(dest_dir))
        stop("jvecfor package does not appear to be installed.")
    if (is.null(jar_path)) {
        pkg_dir    <- dirname(dest_dir)
        candidates <- Sys.glob(
            file.path(pkg_dir, "..", "java", "jvecfor", "target",
            "jvecfor-[0-9]*.jar")
        )
        candidates <- c(
            candidates,
            Sys.glob(
                file.path("java", "jvecfor", "target",
                "jvecfor-[0-9]*.jar")
            ),
            Sys.glob(
                file.path("jvecfor", "java", "jvecfor", "target",
                "jvecfor-[0-9]*.jar")
            )
        )
        # Keep only plain versioned JARs, not classifier variants
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

    # Extract the version from the source JAR's filename
    ver_m <- regmatches(
        basename(jar_path),
        regexpr("[0-9]+\\.[0-9]+\\.[0-9]+", basename(jar_path))
    )
    if (length(ver_m) == 0L)
        stop("Cannot parse version from JAR filename: ", basename(jar_path))
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
            "Install Java >= 21 from https://adoptium.net ",
            "before calling fastFindKNN() or related functions."
        )
    }
    invisible(NULL)
}
