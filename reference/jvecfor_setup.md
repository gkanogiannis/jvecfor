# Install a Custom jvecfor JAR

Copies a custom jvecfor JAR to the user data directory
(`tools::R_user_dir("jvecfor", "data")`). The bundled JAR in
`inst/java/` is used by default; call `jvecfor_setup()` only to override
it with a custom build. Alternatively, set
`options(jvecfor.jar = "/path/to/jvecfor.jar")` for a session-level
override without copying.

## Usage

``` r
jvecfor_setup(jar_path = NULL)
```

## Arguments

- jar_path:

  Path to the jvecfor JAR. If `NULL`, auto-searches
  `java/jvecfor/target/jvecfor-*.jar` relative to the working directory
  (works when developing inside the source tree).

## Value

Invisibly returns the path to the installed JAR.

## Examples

``` r
# Show where custom JARs are stored
tools::R_user_dir("jvecfor", "data")
#> [1] "/home/runner/.local/share/R/jvecfor"

# List bundled JARs
dir(system.file("java", package = "jvecfor"), pattern = "*.jar")
#> [1] "jvecfor-1.2.0.jar"
```
