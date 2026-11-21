# Code Coverage

## Generating Reports

```bash
# Configure with coverage flags
cmake -B build-coverage -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_BUILD_TESTS=ON \
    -DCMAKE_C_FLAGS="--coverage -g -O0" \
    -DCMAKE_EXE_LINKER_FLAGS="--coverage"

# Build and run tests
cmake --build build-coverage -j$(nproc)
cd build-coverage && ctest --output-on-failure

# Generate HTML report
lcov --directory . --capture --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage_filtered.info
genhtml coverage_filtered.info --output-directory coverage_report
```

## Coverage Targets

- **Kernel core** (include/kernel/, kernel/): >= 90%
- **Architecture** (include/arch/, arch/): >= 80%
- **Security** (include/security/, security/): >= 95%
- **Model Zoo** (algorithms/model_zoo/): >= 70%
- **Tests**: not counted

## Current Status

Coverage reporting is not yet automated. The above commands work on Linux with GCC.

## Requirements

- **Linux** only (gcov/lcov/genhtml)
- **GCC** compiler (Clang's --coverage also works)
- **lcov** and **genhtml** installed
