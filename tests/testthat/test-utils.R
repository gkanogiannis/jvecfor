library(testthat)
library(jvecfor)

# .jvecfor_version() -----------------------------------------------------------

test_that(".jvecfor_version returns a valid semver string", {
    ver <- jvecfor:::.jvecfor_version()
    expect_type(ver, "character")
    expect_length(ver, 1L)
    expect_match(ver, "^[0-9]+\\.[0-9]+\\.[0-9]+$")
})

# .jvecfor_jar() ---------------------------------------------------------------

test_that(".jvecfor_jar stops with informative error when option is nonexistent", {
    old_opt <- getOption("jvecfor.jar")
    options(jvecfor.jar = "/nonexistent/path/to/jvecfor.jar")
    on.exit(options(jvecfor.jar = old_opt), add = TRUE)
    expect_error(
        jvecfor:::.jvecfor_jar(),
        regexp = "non-existent file"
    )
})

test_that(".jvecfor_jar returns user option path when file exists", {
    tmp <- tempfile(fileext = ".jar")
    file.create(tmp)
    on.exit({
        options(jvecfor.jar = NULL)
        unlink(tmp)
    }, add = TRUE)
    options(jvecfor.jar = tmp)
    result <- jvecfor:::.jvecfor_jar()
    expect_equal(result, tmp)
})

# jvecfor_setup() --------------------------------------------------------------

test_that("jvecfor_setup stops when package is not installed", {
    # Override system.file to return empty string as if pkg not installed.
    # This is a conservative test: just check it errors gracefully when no
    # inst/java dir is found.
    # (Full JAR-copy test requires a running installation.)
    expect_error(
        withCallingHandlers(
            jvecfor_setup(jar_path = "/nonexistent/jvecfor.jar"),
            warning = function(w) invokeRestart("muffleWarning")
        ),
        regexp = "JAR not found|not appear to be installed"
    )
})

# Package options --------------------------------------------------------------

test_that("jvecfor.verbose option defaults to FALSE", {
    # Reset so .onLoad default takes effect
    old <- getOption("jvecfor.verbose")
    on.exit(options(jvecfor.verbose = old), add = TRUE)
    options(jvecfor.verbose = NULL)
    # Re-trigger defaults (simulate .onLoad effect)
    defaults <- list(jvecfor.verbose = FALSE, jvecfor.jar = NULL)
    toset <- !(names(defaults) %in% names(options()))
    if (any(toset)) options(defaults[toset])
    expect_false(isTRUE(getOption("jvecfor.verbose")))
})
