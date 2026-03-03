# Makefile for jvecfor Java project, used to build, test, and manage the project from the command line.

# Maven POM lives inside java/jvecfor/
POM := java/jvecfor/pom.xml

# Use the Maven wrapper if it exists, otherwise fall back to a system-wide mvn
MVN := $(if $(wildcard ./mvnw*),./mvnw,mvn)
SHELL := /bin/bash

# Default target executed when 'make' is run without arguments
.DEFAULT_GOAL := help

# Directory inside the R package where JARs are installed
INST_JAVA := inst/java

# JAR version is the Maven project version (kept in sync with .jvecfor_version() in R/utils.R)
JAR_VERSION := $(shell grep -m1 '<version>' java/jvecfor/pom.xml | tr -d ' \t' | sed 's|<[^>]*>||g')
JAR_NAME    := jvecfor-$(JAR_VERSION).jar

# R package version from DESCRIPTION (used to locate the built tarball)
PKG_VERSION := $(shell grep '^Version:' DESCRIPTION | awk '{print $$2}')
PKG_TARBALL := jvecfor_$(PKG_VERSION).tar.gz

# Phony targets don't represent files
.PHONY: help build package install test format format-check lint clean \
        setup-hooks test-hooks \
        r-build r-check r-bioccheck r-test r-check-all

help: ## Show this help message
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Run the full Maven build lifecycle (compile, check, test, and package)
	@echo "Building project and running all checks..."
	@$(MVN) -B verify -f $(POM)

package: ## Compile and package the application into a JAR file
	@echo "Packaging application..."
	@$(MVN) -B package -f $(POM)

install: ## Build jvecfor and install the fat JAR into inst/java/
	@echo "Building jvecfor (skipping tests)..."
	@$(MVN) -B clean package -DskipTests -f $(POM)
	@echo "Installing fat JAR into $(INST_JAVA)/"
	@mkdir -p $(INST_JAVA)
	@# Remove any stale JARs left by a previous install
	@rm -f $(INST_JAVA)/*.jar
	@# Fat (shaded) JAR — contains all dependencies + Main-Class manifest
	@cp java/jvecfor/target/$(JAR_NAME) $(INST_JAVA)/$(JAR_NAME)
	@echo "Installed JAR:"
	@ls -lh $(INST_JAVA)/

test: ## Run all the tests
	@echo "Running tests..."
	@$(MVN) -B test -f $(POM)

r-build: ## Build the R package tarball (R CMD build)
	@echo "Building R package tarball..."
	@R CMD build .
	@echo "Tarball: $(PKG_TARBALL)"

r-check: r-build ## Run R CMD check --as-cran on the built tarball
	@echo "Running R CMD check..."
	@R CMD check --as-cran $(PKG_TARBALL)

r-bioccheck: r-build ## Run BiocCheck on the built tarball
	@echo "Running BiocCheck..."
	@Rscript -e "BiocCheck::BiocCheck('$(PKG_TARBALL)', \
	    \`quit-with-status\` = TRUE, \
	    \`no-check-bioc-help\` = TRUE)"

r-test: ## Run testthat tests via devtools (no install needed)
	@echo "Running R tests..."
	@Rscript -e "devtools::test()"

r-check-all: r-build ## Run full R validation: R CMD check + BiocCheck
	@echo "Running full R validation suite..."
	@R CMD check --as-cran $(PKG_TARBALL)
	@Rscript -e "BiocCheck::BiocCheck('$(PKG_TARBALL)', \
	    \`quit-with-status\` = TRUE, \
	    \`no-check-bioc-help\` = TRUE)"
	@echo "All R checks passed."

format: ## Format Java source files
	@echo "Formatting source code..."
	@$(MVN) -B spotless:apply -f $(POM)

format-check: ## Check code formatting without applying changes
	@echo "Checking code formatting..."
	@$(MVN) -B spotless:check -f $(POM)

lint: ## Check code style
	@echo "Checking code style..."
	@$(MVN) -B checkstyle:check -f $(POM)

clean: ## Remove all build artifacts
	@echo "Cleaning project..."
	@$(MVN) -B clean -f $(POM)

setup-hooks: ## Set up pre-commit hooks
	@echo "Setting up pre-commit hooks..."
	@if ! command -v pre-commit &> /dev/null; then \
		echo "pre-commit not found. Please install it using 'pip install pre-commit'"; \
		exit 1; \
	fi
	@pre-commit install --install-hooks

test-hooks: ## Test pre-commit hooks on all files
	@echo "Testing pre-commit hooks..."
	@./scripts/test-pre-commit.sh
