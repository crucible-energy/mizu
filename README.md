# Mizu

Mizu is an experimental local inference runtime for multimodal decoder models.

Current reality:

- the runtime contract, cache/reporting machinery, and session lifecycle are
  real
- backend execution is still placeholder/scaffold-level
- `mizu` does not yet run real Qwen/Gemma inference with real packed weights
  and real backend math

The project is intentionally narrow:

- Apple ANE is a priority target
- CUDA is a first-class peer target
- the public surface stays small through a C ABI
- the control plane is written primarily in Fortran
- the runtime is designed to self-optimize for speed through measured reuse

Current implementation status:

- core Fortran runtime types and lifecycle scaffolding are in place
- the public C header exists in `include/mizu.h`
- model manifests, cache keys, and session flow are implemented as scaffolds
- the loader now recognizes an optional imported `mizu_import/` bundle with
  validated tensor, modality, and projector inventories
- `tools/import/hf_safetensors_to_mizu.py` can now scan local
  HuggingFace-style safetensors model directories and emit a Mizu
  `manifest.mizu` plus `mizu_import/` asset bundle for Qwen/Gemma-shaped
  real-asset smoke testing
- `tools/import/gguf_to_mizu.py` can now scan local GGUF model assets,
  optionally pair a model GGUF with an mmproj GGUF, and emit the same bundle
  shape plus core quantized storage metadata and a GGUF tensor sidecar for
  Qwench-style real-asset smoke testing
- imported byte budgets and CUDA pack spans now use recognized source
  `storage_type` block sizes instead of always sizing tensors by staging dtype
- route-aware optimization and persisted cache metadata are implemented
- the optimization store now invalidates stale evidence by workload, candidate
  key, plan id, or current candidate set, and winner selection/persistence
  ignores retired samples so old measurements cannot keep steering reuse
- the plan cache now has strict-key in-memory, disk-backed, and warm-hydration
  paths for replaying cached plan metadata only when stage, backend, route,
  device, pack format, and shape identity match
- the weight cache now mirrors that strict reuse shape for backend-packed
  weight artifacts, including packed identity text, metadata guards, and
  disk-backed warm reuse
- the session cache now applies a 16-record KV-retention and safe-eviction
  policy at runtime for checkpointed parked live-context records, protecting
  live or uncheckpointed resident state from cache eviction
- the multimodal cache now has a module-level reuse policy that separates
  device-reusable preprocessing outputs from device/projector-bound projector
  embedding outputs, with explicit invalidation reasons for content,
  projector, embedding-shape, key, and artifact changes
- runtime backend inventory and capability-probe scaffolding exist for Apple and
  CUDA
- Apple planner scaffolding now exists for ANE and Metal, with route-specific
  plan formats, workspace estimates, and materialized Apple artifact payloads
  behind the current C API metadata path
- Apple now also has a real bridge seam through a compile-safe Objective-C
  implementation on macOS and a non-Apple stub elsewhere, so the public API
  can execute placeholder ANE and Metal projector, prefill, and decode stages
  instead of only emitting Apple planner metadata
- requested backend masks are now intersected with the runtime's detected
  backend inventory at model-open time, so impossible Apple/CUDA routes fail
  early with `MIZU_STATUS_NO_VALID_PLAN` instead of surviving into execution
- Apple session contexts now use the same live-context and park/resume
  checkpoint path as CUDA, including backend-neutral offload and restore rules
  for resident execution state
- CUDA planner scaffolding now emits route-specific plan and weight-pack records
  and materializes stub payload files under `cache_root`
- CUDA capability probing now prefers a real device bridge when available and
  falls back cleanly when it is not
- CUDA projector, prefill, and decode now run through a backend-owned CUDA
  bridge, with placeholder kernels on NVIDIA hardware and a CPU stub fallback
  when `nvcc` is unavailable
- session staging now preserves attached token content and copied modal bytes,
  along with stable content hashes that feed the current CUDA placeholder path
- live session context identity now survives prefill and advances on decode, so
  repeated CUDA decode steps depend on prior session work instead of counts
  alone
- CUDA prefill now pushes real staged token and modal buffers through the
  backend bridge instead of relying only on staged counts and hashes
- CUDA prefill now emits a persisted backend-owned live-context byte buffer
  into session state, and CUDA decode consumes and updates that buffer across
  steps
- CUDA `park` now materializes a small session-checkpoint artifact when
  `cache_root` is set, and `resume` reloads that checkpoint through the
  runtime cache layer
- parked CUDA sessions now offload the resident in-memory context buffer after
  checkpointing, so `resume` is the path that reconstructs active decode state
- CUDA live-context payloads are now versioned, self-describing, and
  checksummed, and decode validates both header and payload integrity before
  consuming restored backend state
- CUDA live-context payloads now also carry the producer artifact identity, so
  decode can reject same-route plan drift and `resume` can reject mismatched
  checkpoint state instead of silently reusing it
- CUDA live-context payloads now use a fixed-size state block with explicit
  decode-state lanes plus a compact summary word, so decode consumes
  structured backend state instead of hashing an opaque byte bag
- those CUDA live-context payloads now expose semantic state as token digest,
  modal digest, packed KV/decode-step counters, and rolling decode state, with
  unit coverage proving decode advances the structured state predictably
- CUDA live-context payloads now widen to a 128-byte windowed state image with
  page-like KV metadata, a recent-token ring, and a state-image digest, so the
  placeholder decode path can evolve against something closer to compact
  backend-owned decode state
- that CUDA live-context image now widens again to 256 bytes and carries
  explicit per-page slot payloads, so decode continuity is represented as a
  small page-backed state image instead of metadata alone
- that CUDA live-context image now widens again to 512 bytes and carries
  compact key and value lane planes plus per-page digests, so page-local decode
  state looks more like a tiny KV-style image than token slots alone
- that same 512-byte CUDA image now also carries per-page tensor-layout records
  for key rows, value rows, lane counts, head blocks, and page generations, so
  decode can preserve untouched page identity while advancing only the page it
  mutates
- that CUDA live-context image now widens again to 640 bytes and carries an
  explicit per-page control table for owner kind, usable capacity, committed
  rows, free rows, epochs, logical page ids, and flags, so decode state now
  looks more like a compact page table than layout metadata alone
- that CUDA live-context image now widens again to 768 bytes and carries an
  explicit per-page tensor descriptor table for storage offsets, committed
  byte spans, capacity byte spans, and row strides, so the compact page image
  now looks more like a tiny tensor-backed page record than a pure summary
- one narrow multimodal CUDA flow is now validated end to end through the
  public API, including session-state transitions, output readback,
  `park`/`resume`, and fresh-runtime warm reuse against persisted cache state
- imported `mizu_import/` bundle lineage is now retained on the runtime model
  state and emitted into route-specific CUDA and Apple artifact payloads, so
  weight and projector artifacts now carry real imported source-path identity
- imported tensor shapes and dtypes now also produce byte-budget estimates on
  the runtime model state, and those estimates feed backend artifact payloads
  plus weight/projector workspace hints
- CUDA model-load artifacts now go one step further and materialize a narrow
  import-driven weight-pack record with deterministic per-tensor offsets and
  packed-byte totals derived from the imported tensor inventory
- CUDA projector, prefill, and decode artifacts now explicitly depend on that
  packed layout through `pack_ref_*` metadata, and CUDA execution now reads
  that dependency back instead of treating the artifact payload as an opaque
  stage-only blob
- CUDA prefill and decode artifacts now also carry stage-specific
  `pack_use_*` records that name the exact imported tensors selected from the
  packed layout, and CUDA execution now reads those usage summaries back into
  its placeholder execution identity
- those same CUDA stage artifacts now also carry compact `pack=<index>`
  `pack_dispatch*` records for the first selected packed tensors, and the CUDA
  executor now prefers that compact record before falling back to `pack_use*`
  parsing
- those same CUDA stage artifacts now also carry compact importer-rooted
  `pack_span*` records, and the CUDA executor now resolves those bundle files,
  hashes real sampled span bytes plus source paths, and feeds that span data
  into both the real CUDA bridge and the CPU stub
- CUDA prefill and decode plan artifacts now also materialize tiny
  `.spancache` sidecars under `cache_root`, and the executor prefers those
  cached staged span records, including sampled span bytes, on warm runs
  before falling back to importer bundle reads
- those same CUDA `.spancache` sidecars now also retain compact staged
  pack-page records derived from the selected imported tensor spans, and the
  CUDA bridge now prefers those structured page words before falling back to
  raw sampled span bytes
- CUDA prefill and decode now also materialize narrow `.tilecache` payloads
  under `cache_root`, referenced by the `v4` `.spancache` sidecars, and the
  CUDA bridge now prefers those staged tile bytes before falling back to page
  words or raw sampled span bytes
- CUDA model-load artifacts now also materialize pack-owned `.packtiles`
  payloads under `cache_root`, and prefill/decode span caches now reference
  that weight-pack cache so the CUDA executor can prefer pack-owned page/tile
  records before falling back to plan-local `.tilecache` payloads or raw span
  samples
- those same CUDA `.packtiles` payloads now derive their page/tile records from
  weight-pack metadata and pack identity instead of sampled importer preview
  bytes, so warm replay is leaning more on backend-owned materialization than
  span reconstruction
- those same CUDA weight-pack caches now also materialize a dedicated typed
  binary `.packbuffer` sibling carrying a small header, per-pack directory,
  and staged page/tile bytes, with a readable `.packpayload` fallback kept
  alongside it, and the CUDA executor now hydrates that buffer directly
  instead of depending on embedded `.packtiles` previews
- when compact CUDA `pack_dispatch*` entries carry explicit `pack=` indices,
  the executor now restores offset, byte span, role, and layout from that
  typed `.packbuffer` directory before launching the bridge, so warm execution
  depends less on surrounding dispatch text
- the CUDA bridge now also receives those resolved packed-entry indices
  directly and derives staged prefill/decode execution from the resolved
  binary pack record first, with materialized pack identity ahead of the
  canonical `tile -> page -> sampled span` fallback order that keeps warm replay
  stable across different cache shapes
- those same compact CUDA plans now also normalize their stored artifact
  lineage from the resolved typed `.packbuffer` record, so warm decode stays
  stable whether a plan identifies a packed tensor by raw offset/bytes or by
  explicit `pack=` index
- CUDA projector, prefill, and decode artifacts now also carry direct
  `pack_ref_tile_buffer=` references, so warm execution can still hydrate
  pack-owned binary page/tile records even if the `.packtiles` text index is
  missing
- when those pack-owned `.packtiles` payloads are available, CUDA execution now
  prefers their materialized hash identity over raw importer-span identity, and
  the direct executor path now allocates enough artifact text capacity to carry
  longer pack-aware plan payloads without truncation
- warm CUDA replay now stays stable even if the plan-local `.spancache` and
  `.tilecache` files are removed, as long as that pack-owned `.packtiles`
  cache remains available under `cache_root`
- CUDA projector, prefill, and decode artifacts now also stamp compact
  `pack=<index>` dispatch records, so warm execution can address the
  weight-pack tile cache by packed entry identity instead of relying only on
  offset/byte matching in plan text
- those same CUDA prefill/decode artifacts now also materialize binary
  `.dispatchbuffer`, `.spanbuffer`, and `.usagebuffer` sidecars, so warm
  replay can recover selected packed-entry indices, sampled span identity, and
  `pack_use_*` summary state without depending on the removed textual
  `pack_use*`, `pack_dispatch*`, and `pack_span*` plan fragments
- compact CUDA warm artifact lineage now treats those binary sidecar paths as
  transport details and derives identity from resolved pack/span content
- generated CUDA prefill/decode plans now emit directly in their compact
  stable-metadata form, and warm replay recognizes them from the
  artifact-sidecar layout itself instead of any textual compact marker
- direct `.usagebuffer`, `.dispatchbuffer`, and `.spanbuffer` refs are no
  longer needed in plan text, because those sidecars are derived from artifact
  identity at runtime
- CUDA prefill/decode sidecars are now materialized directly from the imported
  tensor inventory and stage kind, so the hot path no longer depends on a
  transient textual `pack_use*` / `pack_dispatch*` / `pack_span*` expansion
  or a later compaction pass
- compact CUDA warm artifact lineage now also ignores `pack_use_kind=` and
  `pack_dispatch_kind=` marker text, so binary-sidecar-only plans and older
  text-plus-buffer plans converge on the same replay identity when they resolve
  to the same packed tensors
- CUDA prefill and decode now also stamp an explicit pack-usage snapshot into
  the live CUDA context payload, so backend-owned session state carries the
  selected imported tensor profile instead of hiding it only inside payload
  strings and cache keys
- that live CUDA context payload now also carries an explicit pack-dispatch
  snapshot for the first selected packed tensors, including packed offsets,
  byte spans, role codes, and layout codes that both bridge variants preserve
  through prefill and decode
- the narrow public CUDA flow now checks stable positive placeholder output plus
  warm-path reproducibility for the same multimodal staged context, while the
  unit suite still pins exact deterministic executor outputs per bridge variant
- runtime workspace reservations now back a real reusable aligned host scratch
  buffer, track arena allocations, and pass that buffer to the CUDA bridge
  during stage execution
- compact CUDA decode plans can now replay from binary usage, dispatch, and
  span sidecars together, even when textual `pack_use*` summary and
  `pack_span*` records are removed from the plan payload
- that same compact CUDA decode path can now replay from stage kind plus binary
  refs alone, with no per-entry `pack_dispatch*` text required in the plan
- the public CUDA warm path now also tolerates generated decode plans with
  per-entry `pack_use*`, `pack_dispatch*`, and `pack_span*` text stripped out,
  as long as the stable plan metadata and binary usage/dispatch/span sidecars
  remain in place
- those public CUDA binary sidecars now also persist a real effective
  `pack_use_hash` for warm replay, so generated decode plans can drop the
  textual `pack_use_*` summary fields too and still replay identically from
  stable plan metadata plus `.usagebuffer`, `.dispatchbuffer`, and
  `.spanbuffer`
- that same public CUDA warm path now also tolerates generated decode plans
  with textual `pack_span_root` and `pack_span_cache` hints removed, because
  warm replay can derive sidecar paths from the artifact path and recover span
  identity directly from the persisted binary sidecars
- that same public CUDA warm path now also tolerates generated decode plans
  with textual `pack_ref_tile_cache` hints removed, because warm replay can
  recover the pack-owned binary buffer directly from `pack_ref_tile_buffer=`
  and derive the rest of the weight-pack cache shape from artifact identity
- that same public CUDA warm path now also tolerates generated decode plans
  with textual static weight-pack hints removed, including
  `pack_ref_artifact=`, `pack_ref_hash=`, `pack_ref_bytes=`,
  `pack_ref_count=`, `weight_pack_hash=`, `weight_pack_bytes=`, and
  `weight_pack_count=`, because once `pack_ref_tile_buffer=` is present the
  warm path can derive static pack dependency from the typed `.packbuffer`
  directly instead of those text fields
- CUDA `.usagebuffer` sidecars now also persist the resolved weight-pack
  `.packbuffer` path, so compact warm replay can still recover typed pack
  records after the intermediate `.packtiles` text index is missing
- CUDA `.execbuffer` sidecars now also persist that resolved weight-pack
  `.packbuffer` path, so compact warm replay preserves static pack dependency
  and canonical pack records even after `.usagebuffer`, `.dispatchbuffer`,
  `.spanbuffer`, and `.spancache` have been removed
- CUDA `.execbuffer` sidecars now use version 3 records with an explicit
  per-entry materialized weight-pack hash lane, rather than overloading the
  span-hash fields; the executor still accepts older v1/v2 buffers as
  compatibility inputs
- CUDA bridge prefill/decode now receive those materialized hashes as
  first-class inputs and use them before falling back to tile/page/sample bytes;
  typed `.packbuffer` dependency hashing likewise prefers complete
  materialized entry identities before falling back to raw buffer bytes
- generated CUDA `prefill` and `decode` artifacts now materialize `.execbuffer`
  plus the typed weight-pack cache as the primary binary warm-path record, and
  no longer emit new plan-local `.usagebuffer`, `.dispatchbuffer`, or
  `.spanbuffer` files; the executor still supports those older sidecars as
  backward-compatibility fallbacks for manual fixtures and previously cached
  layouts
- generated CUDA `prefill` and `decode` artifacts also no longer materialize
  new `.spancache` files, because the resolved span/sample/page/tile record now
  lives in `.execbuffer`; `.spancache` remains a compatibility fallback for
  older cache layouts and explicit tests
- generated CUDA `prefill` and `decode` artifacts also no longer materialize
  new plan-local `.tilecache` files, because the generated hot-path tile record
  now lives in `.execbuffer` plus the typed weight-pack cache; `.tilecache`
  remains a compatibility fallback for older cache layouts and explicit tests
- generated CUDA model-load artifacts also no longer materialize new
  weight-pack `.packtiles` or `.packpayload` sidecars, because the generated
  weight-pack warm record now lives directly in `.packbuffer`; `.packtiles`
  and `.packpayload` remain compatibility fallbacks for older cache layouts
  and explicit tests
- generated CUDA projector artifacts now reference `pack_ref_tile_buffer=`
  directly, and no longer require `pack_ref_tile_cache=` or
  `pack_ref_tile_payload=` hints
- that same binary-first CUDA warm path now replays correctly even after the
  generated plan has no direct weight-pack buffer hint and the weight-pack
  `.packtiles` file is removed, because execution and artifact identity can
  both recover through the binary usage sidecar
- the `Makefile` now rebuilds the contract binaries when the C API Fortran
  sources change, which keeps the public-path tests from silently running stale
  executables
- `make test` now succeeds from a clean tree without relying on stray Fortran
  module files and now fails fast if any unit or contract binary fails
- Apple execution now exists as a placeholder bridge/runtime seam rather than a
  real Metal or ANE compute backend, and CUDA execution is still
  placeholder/scaffold-level rather than real transformer math

Build and test:

- `make test`

Documentation:

- [Architecture](./docs/ARCHITECTURE.md)
- [API Spec](./docs/API_SPEC.md)
- [Project Plan](./docs/PROJECT_PLAN.md)
- [Task List](./docs/TASK_LIST.md)
- [Current State](./docs/CURRENT_STATE.md)
- [Importer Layout](./docs/IMPORTER_LAYOUT.md)
- [Placeholder Runtime Status](./docs/PLACEHOLDER_RUNTIME_STATUS.md)
- [Style Guide](./STYLE_GUIDE.md)
