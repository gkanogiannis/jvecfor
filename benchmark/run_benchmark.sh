#!/usr/bin/env bash
# ============================================================
#  Run the jvecfor vs bluster makeSNNGraph benchmark.
#
#  Prerequisites:
#    • Java 21+ on PATH
#    • Maven 3.8+ on PATH (or set SKIP_BUILD=1 if jar is already built)
#    • R 4.5+ on PATH with internet access (packages auto-installed)
#
#  Usage:
#    cd benchmark/
#    bash run_benchmark.sh           # build jar then run R
#    SKIP_BUILD=1 bash run_benchmark.sh   # skip mvn package
# ============================================================
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JVECFOR_DIR="$(cd "$BENCHMARK_DIR/../java/jvecfor" && pwd)"
JAR_VERSION="$(grep -m1 '<version>' "$JVECFOR_DIR/pom.xml" | tr -d ' \t' | sed 's|<[^>]*>||g')"
JAR="$JVECFOR_DIR/target/jvecfor-${JAR_VERSION}.jar"

# ── Build fat jar ─────────────────────────────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo "==> Building fat jar (mvn package -q) ..."
    mvn -q package -DskipTests -f "$JVECFOR_DIR/pom.xml"
fi

if [[ ! -f "$JAR" ]]; then
    echo "ERROR: fat jar not found at $JAR" >&2
    exit 1
fi
echo "==> Using jar: $JAR"

# ── Run R benchmark ───────────────────────────────────────────────────────────
cd "$BENCHMARK_DIR"
echo "==> Running benchmark.R (this may take a long time for large n) ..."
Rscript benchmark.R

echo ""
echo "==> Done.  Results:"
echo "      benchmark_results_raw.csv"
echo "      benchmark_results_summary.csv"
echo "      benchmark_results.pdf"
