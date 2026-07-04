# Mizu Safety Audit

Last reviewed: July 4, 2026

This document records the current safety posture for the repo's active runtime,
bridge, and C ABI seams. It is a review aid, not a release-readiness claim.

## Current Scope

- Covers the host-side control plane in Fortran plus the C, Objective-C, and
  CUDA bridge seams.
- Covers memory ownership, opaque-handle lifetime, and current concurrency
  limits.
- Does not claim real transformer-math correctness; placeholder execution is a
  separate concern.
- Does not claim thread-safe concurrent lifecycle mutation today.

## Current Controls

### Memory Ownership

- Host workspace memory is owned through [src/common/mod_memory.f90](../src/common/mod_memory.f90)
  and released through the runtime/workspace reset path.
- Runtime, model, and session state prefer allocatable-owned storage and use
  reset routines before slot reuse:
  - [src/runtime/mod_runtime.f90](../src/runtime/mod_runtime.f90)
  - [src/runtime/mod_session.f90](../src/runtime/mod_session.f90)
- Imported model inventory copied onto live model state is explicitly
  deallocated before replacement or reset in
  [src/c_api/mod_c_api.f90](../src/c_api/mod_c_api.f90).

### Opaque Handle Lifetime

- Public `mizu_runtime_t *`, `mizu_model_t *`, and `mizu_session_t *` handles
  are backed by active-pointer registries in
  [src/c_api/mod_c_api.f90](../src/c_api/mod_c_api.f90).
- Closed and destroyed handles are rejected by pointer identity before any box
  dereference, preventing stale-handle use-after-free on the C ABI wrapper
  path.
- Slot reuse clears the active pointer and retires the wrapper box without
  freeing it, so stale caller pointers cannot alias a later live handle after
  allocator reuse and therefore fail closed as `MIZU_STATUS_INVALID_ARGUMENT`.
- Those retired wrapper boxes now live in a bounded process-lifetime arena
  capped at `4096` retired runtime boxes, `4096` retired model boxes, and
  `4096` retired session boxes. Once a per-kind cap is reached, new
  create/open calls fail closed as `MIZU_STATUS_BUSY` instead of growing heap
  usage without bound.

### Bridge Boundaries

- Staged token, modal, and context payloads are copied into owned Fortran
  arrays before bridge launch paths hand raw pointers into C/CUDA code.
- Optional workspace buffers stay explicit as `c_ptr` inputs and default to
  `c_null_ptr` when absent, keeping null handling visible at the boundary.
- Current Apple and CUDA execution paths are placeholder backends, but their
  ownership seams are explicit rather than hidden behind borrowed global state.

### Concurrency And Data Races

- The repo currently relies on single-threaded mutation of runtime, model, and
  session lifecycle state.
- Registry state in [src/c_api/mod_c_api.f90](../src/c_api/mod_c_api.f90) is
  process-global and unsynchronized. The current safe claim is therefore:
  deterministic single-threaded lifecycle management, not general concurrent
  mutation safety.
- Any future concurrent public API claim must add explicit synchronization plus
  tests before this audit can be upgraded.

## Measured Validation

- `make check-local`
- `make check-local` currently runs:
  - `./scripts/format-local.sh --all --check`
  - `git diff --check`
  - `make test`
  - [tests/tooling/test_format_local.py](../tests/tooling/test_format_local.py)
  - [tests/tooling/test_gguf_to_mizu.py](../tests/tooling/test_gguf_to_mizu.py)
  - [tests/tooling/test_hf_safetensors_to_mizu.py](../tests/tooling/test_hf_safetensors_to_mizu.py)
- The contract and unit coverage exercised in this pass includes:
  - [tests/contract/test_handle_arena_capacity.c](../tests/contract/test_handle_arena_capacity.c)
  - [tests/contract/test_handle_lifecycle.c](../tests/contract/test_handle_lifecycle.c)
  - [tests/contract/test_modal_input_validation.c](../tests/contract/test_modal_input_validation.c)
  - [tests/contract/test_session_state_guards.c](../tests/contract/test_session_state_guards.c)
  - [tests/contract/test_struct_sizes.c](../tests/contract/test_struct_sizes.c)
  - [tests/contract/test_stage_reports.c](../tests/contract/test_stage_reports.c)
  - [tests/unit/test_runtime_workspace.f90](../tests/unit/test_runtime_workspace.f90)
  - [tests/unit/test_session_staging.f90](../tests/unit/test_session_staging.f90)

## Approved Exceptions

- Opaque-handle wrapper boxes are intentionally retained until process exit
  after close or destroy, but now only inside the bounded per-kind arena above.
  This preserves the fail-closed stale-handle contract that avoids
  use-after-free and allocator address-reuse ABA problems at the C ABI
  boundary while preventing unbounded retained-handle growth.

## Open Questions

- Sanitizer-backed C/CUDA validation is not yet part of the default repo-local
  gate.
- The public C ABI is not yet documented as safe for concurrent mutation from
  multiple threads.
