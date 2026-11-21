# ============================================================================
# ARIX-Algo Makefile — convenience targets (cross-platform: Windows + Unix)
# ============================================================================
.PHONY: all build test clean format lint python-build python-test bench \
        coverage docs stats help install

BUILD_DIR      ?= build
BUILD_TYPE     ?= Release
BUILD_PY_DIR   ?= build_py
BUILD_BENCH_DIR ?= build_bench
CMAKE_ARGS     ?= -DSNEPPX_BUILD_TESTS=ON -DSNEPPX_BUILD_BENCHMARKS=ON

SHELL := $(shell command -v bash 2>/dev/null || echo cmd)

# Detect number of CPUs cross-platform
ifeq ($(OS),Windows_NT)
	NPROC := $(shell powershell -Command "(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors")
	NULL  := nul
	RM    := powershell -Command "Remove-Item -Recurse -Force"
else
	NPROC := $(shell nproc 2>/dev/null || echo 4)
	NULL  := /dev/null
	RM    := rm -rf
endif

all: build

# ---- C/C++ core ----
build:
	cmake -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j$(NPROC)

test: build
	cd $(BUILD_DIR) && ctest -C $(BUILD_TYPE) --output-on-failure --timeout 60

clean:
	$(RM) $(BUILD_DIR) $(BUILD_PY_DIR) $(BUILD_BENCH_DIR) build-coverage coverage_report

# ---- Python bindings ----
python-build:
	cmake -B $(BUILD_PY_DIR) -G Ninja -DSNEPPX_BUILD_PYTHON=ON -DSNEPPX_BUILD_TESTS=OFF -DSNEPPX_BUILD_BENCHMARKS=OFF
	cmake --build $(BUILD_PY_DIR) --target _arix_c --config $(BUILD_TYPE) -j$(NPROC)

python-test: python-build
	python -c "import sys; sys.path.insert(0, 'bindings/python'); from arix_algo import _arix_c as ax; t = ax._Tensor.zeros([2,3], ax.FLOAT32); print('Python bindings OK:', t.shape)"

# ---- Benchmarks ----
bench: bench-build bench-run

bench-build:
	cmake -B $(BUILD_BENCH_DIR) -G Ninja -DSNEPPX_BUILD_BENCHMARKS=ON -DSNEPPX_BUILD_TESTS=OFF
	cmake --build $(BUILD_BENCH_DIR) --target bench_pq_crypto --config $(BUILD_TYPE) -j$(NPROC)

bench-run:
	$(BUILD_BENCH_DIR)/tests/benchmark/$(BUILD_TYPE)/bench_pq_crypto

# ---- Formatting ----
format:
ifeq ($(OS),Windows_NT)
	powershell -Command "Get-ChildItem -Recurse -Include '*.c','*.h' | Where-Object { $$_.FullName -notmatch '\\\\build\\\\|\\\\target\\\\' } | ForEach-Object { clang-format -i -style=file $$_.FullName }"
else
	clang-format -i -style=file $$(find . -name '*.c' -o -name '*.h' | grep -v build/ | grep -v target/)
endif

lint:
ifeq ($(OS),Windows_NT)
	powershell -Command "Get-ChildItem -Recurse -Include '*.c','*.h' | Where-Object { $$_.FullName -notmatch '\\\\build\\\\|\\\\target\\\\' } | ForEach-Object { clang-format --dry-run -Werror -style=file $$_.FullName }"
else
	clang-format --dry-run -Werror -style=file $$(find . -name '*.c' -o -name '*.h' | grep -v build/ | grep -v target/)
endif

coverage:
	@scripts/coverage.sh

docs:
	pip install mkdocs mkdocs-material "mkdocstrings[python]" mike mkdocs-autorefs mkdocs-gen-files mkdocs-section-index mkdocs-literate-nav || true
	doxygen Doxyfile || true
	mkdocs build --config-file mkdocs.autodoc.yml

stats:
	@scripts/stats.sh

install: build
	cmake --install $(BUILD_DIR) --prefix $(DESTDIR)

help:
	@echo "Targets:"
	@echo "  build        — Configure and build C/C++ core (default)"
	@echo "  test         — Run test suite"
	@echo "  clean        — Remove build artifacts"
	@echo "  python-build — Build Python bindings ((_arix_c.pyd)"
	@echo "  python-test  — Build and verify Python bindings"
	@echo "  bench        — Build and run PQ crypto benchmarks"
	@echo "  format       — Format source files in-place"
	@echo "  lint         — Check formatting without modifying"
	@echo "  coverage     — Generate coverage report (Linux, GCC)"
	@echo "  docs         — Build the documentation site (mkdocs + Doxygen)"
	@echo "  stats        — Show project statistics"
	@echo "  install      — Install built artifacts"
