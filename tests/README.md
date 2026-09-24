# Mizu Test Skeleton

This tree mirrors the public API contract in [docs/API_TEST_MATRIX.md](../docs/API_TEST_MATRIX.md).

The current goal is not to pretend the tests already exist.
The goal is to give each test class an obvious home before implementation
starts.

## Layout

- `unit/`
  - deterministic tests with fake clocks, fake backends, fake manifests, and
    temp cache roots
- `contract/`
  - C ABI, binding-shape checks, and optional local real-asset smokes that
    skip cleanly when their model files are absent
- `integration/`
  - target-machine tests for Apple and CUDA routes
- `performance/`
  - cold-versus-warm and optimizer convergence checks
- `tooling/`
  - Zig safetensors/GGUF importer fixture, boundary, and bundle tests plus
    developer-tool smoke tests that can run without model hardware
- `fixtures/`
  - shared fake backend and tiny model fixtures

## Immediate Intent

The first implementation wave should start with:

- `unit/` for lifecycle, staging, execution, routing, and optimizer semantics
- `contract/` for header compilation and opaque-handle rules

Hardware integration and performance tests come after the runtime skeleton is
real.
