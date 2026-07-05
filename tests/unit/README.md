# Unit Test Suites

These suites map directly to the `API-U*` sections in
[docs/API_TEST_MATRIX.md](../../docs/API_TEST_MATRIX.md).

## Planned Suites

- `test_api_abi.c`
  - `API-U001` to `API-U006`
- `test_api_model.c`
  - `API-U010` to `API-U014`
- `test_api_session.c`
  - `API-U020` to `API-U029`
- `test_api_staging.c`
  - `API-U030` to `API-U036`
- `test_api_execution.c`
  - `API-U040` to `API-U049`
- `test_api_routing.c`
  - `API-U050` to `API-U055`
- `test_api_optimizer.c`
  - `API-U060` to `API-U066`
- `test_api_cache.c`
  - `API-U070` to `API-U074`

The fake backend layer should arrive before most of these files get real test
bodies.

## Current Smoke Tests

- `test_model_manifest_loader.f90`
  - validates explicit decoder-only and multimodal manifest fixtures
  - validates malformed manifest rejection
  - validates built-in target fallback
- `test_cache_keys.f90`
  - validates deterministic cache-key generation across plan, weight, session,
    and multimodal key types
- `test_cache_store.f90`
  - validates save/load roundtrips for persisted artifact-cache presence
  - validates route-specific artifact metadata roundtrips for weight, plan,
    session, and multimodal records
  - validates planned workspace bytes survive metadata persistence
  - validates reloaded weight, plan, session, and multimodal keys report hits
- `test_plan_cache.f90`
  - validates strict plan-cache keys are required before entries can be
    recorded or looked up
  - validates route and planner-version changes miss instead of reusing stale
    plan metadata
  - validates mismatched stage/backend/route artifact metadata is rejected
  - validates save/load roundtrips and warm hydration merge persisted entries
    without replacing existing in-memory entries
- `test_weight_cache.f90`
  - validates strict weight-cache keys and backend-packed identity are required
    before packed artifacts can be reused
  - validates route and pack-format changes miss instead of reusing stale
    weight-pack metadata
  - validates save/load roundtrips and warm hydration merge persisted packed
    weight entries without replacing existing in-memory entries
- `test_session_cache.f90`
  - validates strict session-cache keys preserve parked live-context identity
    and checkpoint metadata
  - validates KV-heavy sessions receive stronger retention than small parked
    sessions
  - validates eviction chooses only inactive checkpoint-safe records while
    protecting live and uncheckpointed resident sessions
- `test_mm_cache.f90`
  - validates multimodal preprocessing reuse is content-bound and can cross
    device/projector boundaries
  - validates projector-output reuse stays bound to device, planner,
    projector revision, and embedding count
  - validates explicit invalidation reasons for content, projector, embedding,
    key, kind, and artifact changes
- `test_optimization_store.f90`
  - validates runtime-scoped winner selection based on recorded execution
    samples rather than key reuse alone
  - validates save/load roundtrips for persisted optimization evidence
  - validates stale evidence can be invalidated by candidate, plan, workload,
    and current candidate-set mismatch, and that invalidated samples are
    ignored by winner selection, stats, and persistence
- `test_backend_registry.f90`
  - validates deterministic backend-inventory aggregation independent of local
    hardware
  - validates runtime state retains the aggregated backend mask and descriptors
- `test_apple_planner.f90`
  - validates Apple planner route selection across ANE-preferred, Metal-
    preferred, and invalid-route requests
  - validates route-specific Apple weight/projector/prefill format labels and
    payload text
  - validates Apple workspace estimates are stable for the current planner
    fixtures
- `test_apple_executor.f90`
  - validates Apple projector, prefill, and decode execute through the bridge
    seam for both ANE and Metal routes
  - validates Apple live-context bytes retain route-aware lineage and semantic
    snapshot data across prefill and decode
  - validates decode rejects cross-route reuse once a decode-produced Apple
    context exists
  - validates corrupted Apple context payloads fail validation instead of
    being consumed
- `test_runtime_workspace.f90`
  - validates runtime-scoped workspace reservation keeps a reusable high-water
    mark while clearing in-use bytes after release
  - validates the workspace now owns a real reusable host scratch buffer that
    is allocated on reserve and freed on reset
  - validates host scratch buffers are aligned, high-water growth preserves
    existing bytes, and smaller reservations do not allocate again
- `test_session_staging.f90`
  - validates attached token content is copied into session staging state and
    preserved when callers later mutate or append new token buffers
  - validates copied modal bytes and content hashes are retained until clear
  - validates prefill produces a persistent live-context hash and decode
    advances it while retaining emitted tokens
  - validates a backend-owned live-context byte buffer can be stored and
    updated in session state
  - validates an offloaded backend live-context buffer makes direct decode
    invalid until the runtime restores residency, including Apple placeholder
    contexts
- `test_cuda_planner.f90`
  - validates stage-specific CUDA plan candidates for weight-pack, projector,
    prefill, and decode records
  - validates planner payload text includes materialization-relevant metadata
- `test_cuda_executor.f90`
  - validates CUDA projector execution consumes a materialized projector payload
  - validates CUDA prefill execution consumes staged tokens through a
    materialized plan payload with content-aware hashing
  - validates CUDA prefill now changes its workspace scratch when staged token
    and modal tensor contents change, even when the plan shape stays the same
  - validates CUDA prefill emits a persisted context byte buffer and CUDA
    decode consumes that buffer directly
  - validates those context buffers now carry a versioned, checksummed CUDA
    header
  - validates CUDA prefill and decode now fully populate the fixed-size CUDA
    context payload instead of leaving a partial scratch record
  - validates CUDA page-control snapshots expose owner kind, capacity,
    committed rows, free rows, epochs, logical page ids, and flags across
    prefill and decode transitions
  - validates CUDA page tensor descriptors expose storage offsets, committed
    byte spans, capacity byte spans, and row strides across prefill, decode,
    and recycled-page transitions
  - validates compact CUDA page tables rotate correctly under decode-window
    overflow and mark recycled physical slots explicitly
  - validates the fixed-size CUDA context payload now carries semantic token
    and modal digests plus explicit KV/decode-step counters
  - validates repeated CUDA decode steps advance those counters and rolling
    decode state predictably
  - validates the widened CUDA context payload now carries a compact windowed
    state image with page-like KV metadata and a recent-token ring
  - validates repeated CUDA decode steps advance that compact windowed state
    image predictably across decode continuity
  - validates the widened CUDA context payload now carries per-page slot
    payloads in addition to page metadata
  - validates repeated CUDA decode steps append emitted tokens into the
    expected page-local slot payloads
  - validates the widened CUDA context payload now carries compact key/value
    lane planes plus stable per-page digests
  - validates repeated CUDA decode steps preserve the digest of untouched pages
    while still advancing the decode-owned page digest
  - validates the widened CUDA context payload now also carries per-page tensor
    layout records for row counts, lane counts, head blocks, and generations
  - validates repeated CUDA decode steps preserve layout for untouched pages and
    advance generation only on the decode-owned page
  - validates CUDA prefill and decode now stamp an explicit imported
    pack-usage snapshot into live context state, including usage hash, byte
    total, and first/last packed tensor spans
  - validates CUDA prefill and decode now also stamp an explicit imported
    pack-dispatch snapshot into live context state for the first selected
    packed tensors, including offsets, byte spans, role codes, and layout codes
  - validates CUDA usage artifacts can now resolve compact importer-rooted
    tensor-span records from the fixture bundle and feed those sampled spans
    into executor behavior
  - validates CUDA usage artifacts now also prefer pack-owned materialized hash
    identity from `.packtiles` caches when available, and that removing the
    pack-owned reference changes the decode result for the same staged usage
    profile
  - validates those same usage artifacts now prefer a typed `.packbuffer`
    sibling with a small header and per-pack directory for pack-owned
    page/tile bytes, keep `.packpayload` as a readable fallback, preserve decode
    behavior when only raw staged bytes are rewritten under the same
    materialized identity, and change behavior when that materialized identity
    changes
  - validates compact CUDA `pack_dispatch*` entries can now collapse to
    `pack=<index>` and still restore offset, byte span, role, and layout from
    `pack_use*` plus the typed `.packbuffer` directory
  - validates a compact CUDA decode plan can now replay from derived binary
    dispatch/usage/span sidecars plus `.packbuffer` even after direct sidecar
    refs are removed from the plan payload
  - validates that same compact CUDA decode path can now replay from stage kind
    plus binary usage/dispatch/span sidecars alone, with no per-entry
    `pack_dispatch*` plan text required
  - validates those same compact CUDA decode plans now keep token identity and
    stored artifact lineage stable when typed `.packbuffer` resolution matches,
    even if equivalent artifacts differ in dispatch form or binary sidecar
    transport paths
  - validates the CUDA bridge now consumes resolved packed-entry indices as
    first-class staged execution input, while still preserving deterministic
    reference output when warm replay falls back from richer tile/page caches
    to leaner binary sidecar shapes
  - validates those same compact CUDA decode plans can still restore their
    bridge-facing dispatch from direct `pack_ref_tile_buffer=` references even
    after the `.packtiles` text index is removed
  - validates those same compact CUDA decode plans now keep token identity and
    stored artifact lineage stable even when stale textual static weight-pack
    hints are present, because warm replay derives that dependency from the
    typed `.packbuffer`
  - validates that same binary-first CUDA decode replay still preserves token
    identity after the intermediate `.packtiles` text index is removed,
    because the `.usagebuffer` sidecar now persists the resolved
    weight-pack `.packbuffer` path
  - validates that same binary-first CUDA decode replay still preserves token
    identity after the plan-local `.usagebuffer`, `.dispatchbuffer`,
    `.spanbuffer`, and `.spancache` sidecars are removed, because the
    `.execbuffer` sidecar now also persists the resolved weight-pack
    `.packbuffer` path and carries explicit v3 materialized pack hashes
  - validates CUDA bridge replay now consumes materialized pack hashes as
    first-class inputs before falling back to tile/page/sample bytes
  - keeps explicit `.usagebuffer`, `.dispatchbuffer`, and `.spanbuffer`
    fixtures as backward-compatibility coverage even though newly generated
    CUDA warm artifacts now prefer `.execbuffer` plus the typed weight-pack
    cache as the primary binary record
  - keeps explicit `.spancache` fixtures as backward-compatibility coverage
    even though newly generated CUDA warm artifacts now place their
    span/sample/page/tile warm record in `.execbuffer` first
  - keeps explicit `.tilecache` fixtures as backward-compatibility coverage
    even though newly generated CUDA warm artifacts now place their generated
    tile record in `.execbuffer` plus the typed weight-pack cache first
  - validates that same binary-first CUDA decode replay still preserves token
    identity when the plan payload is reduced to stable stage metadata only,
    because warm replay can derive compactness and sidecar locations directly
    from artifact identity
  - validates generated CUDA prefill/decode sidecars now still replay
    correctly after the transient textual pack-record expansion is removed
    from the hot artifact builder
  - validates the same hot CUDA plans now still replay correctly after the
    post-build compaction pass is removed, because they are emitted directly in
    compact stable-metadata form
  - validates CUDA projector and decode execution now match exact deterministic
    reference outputs for the current fixture path, pinned separately for the
    real CUDA bridge and the CPU stub
  - validates CUDA decode execution varies with direct context-buffer identity
  - validates CUDA decode rejects a context produced by a different decode
    artifact, even when the route stays the same
  - validates CUDA decode rejects a corrupted context payload instead of
    consuming invalid restored state
  - validates the CUDA bridge receives and stamps the runtime workspace scratch
    buffer during projector, prefill, and decode
