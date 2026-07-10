# Placeholder Runtime Status

This document says the quiet part out loud:

`mizu` already has a real runtime contract, but it does not yet perform real
Qwen/Gemma inference with real packed weights and real backend math.

## What Is Real Today

- the public C ABI in `include/mizu.h`
- runtime, model, and session lifecycle
- token and modal-input staging
- report generation and cache identity
- persisted optimizer evidence
- route-aware planner scaffolding
- route-aware artifact metadata
- session `park` and `resume`
- backend-owned live-context bytes
- Apple bridge seam
- CUDA bridge seam

## What Is Still Placeholder

- Apple `projector`, `prefill`, and `decode`
- CUDA `projector`, `prefill`, and `decode`
- backend-native packed-weight generation
- backend-native plan materialization
- KV/state representation as real transformer execution state
- imported target-model assets for real Qwen/Gemma runs

## What That Means In Practice

Right now the repo is best understood as:

- a real inference-runtime control plane
- a real backend-routing and cache/reporting scaffold
- a real session-lifecycle system
- a placeholder execution backend layer

It is not yet:

- a real local Qwen-3.5-9B multimodal runtime
- a real local Gemma4-21B multimodal runtime
- a `llama.cpp` replacement

## Current Target Families

The intended first real targets are still:

- `Qwen-3.5-9B` multimodal
- `Gemma4-21B` multimodal

But the repo currently exercises tiny fixtures under:

- `tests/fixtures/models/fixture_decoder_tiny`
- `tests/fixtures/models/fixture_mm_tiny`

## Shortest Path To Real Inference

1. finish family-specific importer and manifest mapping for real Qwen/Gemma assets
2. turn deterministic placeholder CUDA pack layouts into backend-native packed weights
3. materialize backend-native plan artifacts
4. replace the placeholder CUDA path with real transformer math
5. replace surrogate live-context state with a real KV/state representation
6. move Apple from placeholder bridge behavior to real Metal/ANE execution

## Why The Placeholder Path Still Matters

The current placeholder backends are not wasted work.

They are already proving:

- lifecycle correctness
- route selection
- cache and optimizer wiring
- checkpoint and restore flow
- backend bridge ownership boundaries
- deterministic public-path behavior within each placeholder implementation

One important detail:

- the real CUDA bridge and the CPU CUDA stub are both placeholder backends
- they are expected to stay internally deterministic
- they are not required to emit the same exact placeholder token sequence

That means the next real-backend work can land into a control plane that
already exists, instead of being built at the same time as the math path.
