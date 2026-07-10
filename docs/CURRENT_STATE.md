# Mizu Current State

Last updated: 2026-07-09

## Latest Checkpoint

- current milestone: as of 2026-07-09, the repo-local QA gate is green
  through `make format-check`, `git diff --check`, `make test`, and
  `make check-debug`, with the expected optional `test_qwench_gguf_cuda_smoke`
  skip when local Qwench GGUF assets are absent
- current milestone: the current QA pass broadens contract coverage for
  stale-handle fail-closed behavior, failed-open output preservation, bounded
  C-string ingress, backend-availability and routing failures, runtime
  last-error propagation, terminal decode behavior, session eviction, and
  checkpoint-restore guards, while pre-push now auto-escalates to
  `make check-debug` for memory-sensitive runtime, bridge, and cache changes
- current milestone: CUDA weight-pack `.packbuffer` records are now version 2
  and carry per-pack `source_offset` lanes. Warm dependency hashing accepts
  both v1 and v2 buffers, folds the new offset when present, and contract
  tests now prove shared-GGUF tensors materialize distinct span hashes.
- current milestone: GGUF imports now preserve per-tensor absolute
  `source_offset` values beside the GGUF-relative `data_offset`, the loader
  threads those offsets onto imported tensor state, and CUDA span sampling /
  pack identity use source offsets plus tensor byte spans so tensors sharing a
  single GGUF file no longer collapse to one file-level identity.
- current milestone: imported tensor byte accounting is now storage-aware for
  GGUF-style source storage types. CUDA import weight packs, workspace hints,
  projector byte lineage, stage usage summaries, and generated pack buffers use
  source `storage_type` block sizes such as `q4_k`, `q5_k`, `q6_k`, `q8_0`,
  `iq2_xxs`, and `iq4_nl`, while unknown storage strings still fall back to
  the loader-facing staging `dtype`.
- current milestone: import tooling now includes a dependency-free
  GGUF smoke importer that scans local GGUF metadata/tensor headers, can pair
  the Qwen3.5 model GGUF with its mmproj GGUF, writes the same
  `manifest.mizu` plus `mizu_import/` bundle shape, preserves original GGUF
  tensor type in core `storage_type` lineage, and keeps GGUF data/source
  offsets in a sidecar inventory
- current milestone: import tooling now includes a dependency-free
  HuggingFace safetensors smoke importer that scans local model shards,
  classifies common Qwen/Gemma tensor-name patterns into Mizu tensor roles,
  writes `manifest.mizu` plus `mizu_import/`, and symlinks or copies source
  shards under safe import-relative paths
- current milestone: Phase 6 now has stale-evidence invalidation in the
  optimization store; candidates can be retired by workload, candidate key,
  plan id, or current candidate-set mismatch, and invalid evidence is ignored
  by winner selection, exploration statistics, and persistence
- current milestone: Phase 6 now has a dedicated multimodal-cache policy
  module that records reusable preprocessing artifacts separately from
  projector-output artifacts, allows preprocessing reuse across device and
  projector changes when content identity is stable, and makes modality reuse
  invalidation explicit for content, projector, embedding-shape, key, and
  artifact changes
- current milestone: Phase 6 now has a dedicated session-cache policy module
  that records parked live-context identity, scores KV retention, and evicts
  only inactive checkpoint-safe entries while protecting live or unsafe
  resident state
- current milestone: Phase 6 now also has a strict weight-cache module for
  backend-packed weight identity, in-memory packed artifact reuse, and
  disk-backed warm reuse with stage/backend/route/pack-format metadata guards
- current milestone: the strict plan cache is now disk-backed and can warm an
  existing runtime cache from persisted entries without replacing already-live
  entries, giving the self-optimization loop a concrete cold-start hydration
  seam
- current milestone: Phase 6 now has a concrete in-memory plan-cache module
  that accepts only strict plan keys, rejects stage/backend/route metadata
  mismatches, tracks plan hits, and preserves artifact metadata for winner
  replay
- current milestone: runtime workspace reservations now use a shared
  `mod_memory` aligned host allocator, preserve scratch bytes when the
  high-water arena grows, and expose allocation counts so tests can verify
  reuse paths avoid hot-path allocation
- current milestone: CUDA `.execbuffer` v3 now carries explicit per-entry
  materialized weight-pack hashes, the Fortran/C/CUDA bridge surfaces pass that
  lane directly into prefill/decode, and warm replay uses materialized identity
  before falling back to tile/page/sample bytes
- current milestone: the CUDA bridge now treats resolved binary pack records
  as the primary staged execution input for prefill and decode by consuming
  resolved `pack=` indices alongside typed `.packbuffer` / usage / dispatch /
  span sidecar state, and it now normalizes each packed entry through one
  canonical fallback order after materialized identity so warm replay stays
  stable even when different cache shapes expose different sidecar richness
- current milestone: generated CUDA weight-pack artifacts now point directly to
  typed `.packbuffer` records instead of materializing new `.packtiles` or
  `.packpayload` sidecars, and generated projector artifacts now depend on
  that direct binary pack record instead of weight-pack tile-cache or payload
  hints
- current milestone: CUDA weight-pack `.packtiles` caches now index a typed
  binary `.packbuffer` warm-path payload, generated prefill/decode plans now
  collapse compact dispatch text down to `pack=<index>` entries, and both
  sidecar generation and execution restore bridge-facing offset/bytes/role/
  layout from `pack_use*` plus that per-pack binary directory instead of
  trusting surrounding text
- current milestone: CUDA prefill/decode plans now also materialize a small
  binary `.dispatchbuffer` sidecar carrying selected packed-entry indices and
  a normalized usage hash, and warm execution now prefers that binary
  selection list over textual `pack_use*` recovery when the cached sidecars
  are present
- current milestone: CUDA prefill/decode plans now also materialize a small
  binary `.spanbuffer` sidecar carrying sampled span hashes and sampled bytes,
  and warm execution now prefers that binary span-identity sidecar over
  textual `pack_span*` recovery when the cached sidecars are present
- current milestone: CUDA prefill/decode plans now also materialize a small
  binary `.usagebuffer` sidecar carrying `pack_use_*` count/byte/offset/hash
  summary, and warm execution now prefers that binary usage-summary sidecar
  over textual `pack_use*` summary recovery when the cached sidecars are
  present
- current milestone: compact CUDA warm artifact lineage now treats binary
  sidecar paths as transport details and derives identity from resolved
  pack/span content instead, so warm replay stays stable when equivalent plans
  differ only in sidecar path text
- current milestone: compact CUDA warm artifact lineage now also ignores
  `pack_use_kind=` and `pack_dispatch_kind=` marker text, so older
  text-plus-buffer plans and newer binary-sidecar-only plans converge on the
  same replay identity when they resolve to the same packed tensors
- current milestone: the loader now understands a first concrete imported
  asset-bundle shape through `mizu_import/`, with validated tensor, modality,
  and projector inventories layered on top of the logical root manifest
- the importer contract is now documented, fixture-backed, and enforced by the
  loader before the runtime accepts imported assets
- imported tensor and projector lineage is now retained on `model_state`, so
  materialized backend artifacts can carry real imported source-path identity
  instead of only generic stage/shape metadata
- imported tensor shapes and dtypes now also fold into runtime byte-budget
  estimates for model and projector assets, and those estimates now feed
  materialized artifact payloads plus backend workspace hints
- CUDA model-load artifact materialization now uses the imported tensor
  inventory directly to emit a narrow weight-pack layout with deterministic
  packed offsets and packed-byte totals for non-projector tensors
- that CUDA weight-pack layout now also feeds projector, prefill, and decode
  cache identity plus artifact payload dependencies, so those stage artifacts
  rotate when the packed model layout changes and the CUDA executor now reads
  the declared pack dependency back into its placeholder execution identity
- CUDA prefill and decode artifact materialization now also emits
  stage-specific `pack_use_*` records with exact imported tensor names, roles,
  packed offsets, and byte spans for the tensors each stage declares it uses,
  and those usage records now feed plan identity plus CUDA placeholder
  execution identity
- those same CUDA stage artifacts now also emit compact numeric
  `pack_dispatch*` records for the first selected packed tensors, and the CUDA
  executor now prefers that compact record before falling back to verbose
  `pack_use*` parsing
- those same CUDA stage artifacts now also emit compact importer-rooted
  `pack_span*` records for the first selected packed tensors, and the CUDA
  executor now resolves those bundle files, hashes real sampled span bytes plus
  source paths, and feeds that span data into both the real CUDA bridge and
  the CPU stub
- CUDA prefill and decode plan artifacts now also materialize tiny
  `.spancache` sidecars under `cache_root`, and the CUDA executor now prefers
  those cached staged span records, including sampled span bytes, on warm runs
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
  and the staged page/tile bytes themselves, with a readable `.packpayload`
  fallback kept alongside it, and the CUDA executor now hydrates that buffer
  directly instead of depending on embedded `.packtiles` previews
- when compact CUDA `pack_dispatch*` entries carry explicit `pack=` indices,
  the executor now restores offset, byte span, role, and layout from that
  typed `.packbuffer` directory before launching the bridge, so warm execution
  depends less on surrounding dispatch text
- those same compact CUDA plans now also normalize their stored artifact
  lineage from the resolved typed `.packbuffer` record, so warm decode stays
  stable whether a plan identifies a packed tensor by raw offset/bytes or by
  explicit `pack=` index
- the CUDA bridge now also receives those resolved packed-entry indices
  directly from the executor and derives staged prefill/decode execution from
  the resolved binary pack record first, with payload text demoted to salt
  instead of primary execution identity
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
  weight-pack tile cache by packed entry identity instead of relying on
  offset/byte matching in plan text
- those same CUDA prefill/decode artifacts now also materialize binary
  `.dispatchbuffer`, `.spanbuffer`, and `.usagebuffer` sidecars, so warm
  replay can recover selected packed-entry indices, sampled span identity, and
  `pack_use_*` summary state without depending on the removed textual
  `pack_use*`, `pack_dispatch*`, and `pack_span*` plan fragments
- compact CUDA decode can now replay from stage kind plus binary sidecars
  alone, with no per-entry `pack_dispatch*` text required in the plan payload
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
- the public CUDA warm contract path now also replays correctly after a
  generated decode plan drops its per-entry `pack_use*`, `pack_dispatch*`, and
  `pack_span*` text, as long as the binary usage/dispatch/span sidecars and
  stable static plan metadata remain available
- those public CUDA binary sidecars now also persist a real effective
  `pack_use_hash`, so the warm contract path can drop textual `pack_use_*`
  summary fields as well and still replay identically from stable static plan
  metadata plus `.usagebuffer`, `.dispatchbuffer`, and `.spanbuffer`
- that same public CUDA warm contract path now also replays correctly after a
  generated decode plan drops the textual `pack_span_root` and
  `pack_span_cache` hints, because the warm path can derive those sidecar
  locations from artifact identity and use the persisted binary span records
  directly
- that same public CUDA warm contract path now also replays correctly after a
  generated decode plan drops the textual `pack_ref_tile_cache` hint, because
  the warm path can recover the pack-owned binary buffer from
  `pack_ref_tile_buffer=` and derive the rest of the weight-pack cache shape
  from artifact identity
- that same public CUDA warm contract path now also replays correctly after a
  generated decode plan drops the textual static weight-pack hints
  `pack_ref_artifact=`, `pack_ref_hash=`, `pack_ref_bytes=`,
  `pack_ref_count=`, `weight_pack_hash=`, `weight_pack_bytes=`, and
  `weight_pack_count=`, because once `pack_ref_tile_buffer=` is present the
  warm path can derive static pack dependency from the typed `.packbuffer`
  directly instead of those text fields
- CUDA `.usagebuffer` sidecars now also persist the resolved weight-pack
  `.packbuffer` path, so compact warm replay can recover typed pack records
  even when the intermediate `.packtiles` text index is gone
- CUDA `.execbuffer` sidecars now also persist that resolved weight-pack
  `.packbuffer` path, so compact warm replay preserves static pack dependency
  and canonical pack records even after `.usagebuffer`, `.dispatchbuffer`,
  `.spanbuffer`, and `.spancache` are gone
- CUDA `.execbuffer` sidecars now write version 3 records with an explicit
  per-entry materialized weight-pack hash lane, while the executor continues to
  accept v1/v2 buffers as compatibility inputs
- CUDA bridge prefill/decode now consume those materialized pack hashes as
  first-class bridge inputs before falling back to tile/page/sample bytes, and
  typed `.packbuffer` dependency hashing now prefers complete materialized
  entry identity before falling back to raw buffer-byte hashing
- generated CUDA `prefill` and `decode` artifacts now rely on `.execbuffer`
  plus the typed weight-pack cache as the primary plan-local binary warm-path
  record, and no longer emit new `.usagebuffer`, `.dispatchbuffer`, or
  `.spanbuffer` sidecars; the executor still supports those older buffers as
  compatibility fallbacks for manual fixtures and older cache layouts
- generated CUDA `prefill` and `decode` artifacts also no longer emit new
  `.spancache` files, because the resolved span/sample/page/tile record now
  lives in `.execbuffer`; `.spancache` remains supported as a compatibility
  fallback for older warm-cache layouts
- generated CUDA `prefill` and `decode` artifacts also no longer emit new
  plan-local `.tilecache` files, because the generated hot-path tile record
  now lives in `.execbuffer` plus the typed weight-pack cache; `.tilecache`
  remains supported as a compatibility fallback for older warm-cache layouts
- generated CUDA model-load artifacts also no longer emit new weight-pack
  `.packtiles` or `.packpayload` sidecars, because the generated weight-pack
  warm record now lives directly in `.packbuffer`; `.packtiles` and
  `.packpayload` remain supported as compatibility fallbacks for older
  warm-cache layouts
- generated CUDA projector artifacts now reference `pack_ref_tile_buffer=`
  directly, and no longer require `pack_ref_tile_cache=` or
  `pack_ref_tile_payload=` hints
- that same binary-first CUDA warm contract path now also replays correctly
  after the plan has no direct weight-pack buffer hint and the weight-pack
  `.packtiles` file has been removed, because execution and artifact identity
  can both recover through the binary usage sidecar
- CUDA prefill and decode now also stamp an explicit pack-usage snapshot into
  the live CUDA context payload, so backend-owned session state carries the
  selected imported tensor profile instead of hiding it only inside artifact
  text and cache identity
- that same live CUDA context payload now also carries an explicit
  pack-dispatch snapshot for the first selected packed tensors, including
  packed offsets, byte spans, role codes, and layout codes, so bridge-owned
  session state preserves a small structural view of pack consumption
- real-asset smoke: `~/.qwench/models/qwen3.5-9b-instruct-q4_k_m.gguf` plus
  `mmproj-Qwen_Qwen3.5-9B-f16.gguf` imports into 761 tensor records with
  projector present; `gemma-4-26B-A4B-it-UD-IQ2_M.gguf` imports into 658 tensor
  records and this local file does not expose projector tensors
- real-asset smoke: when Qwench GGUF assets are present, the contract suite now
  imports them into `/tmp`, opens the generated bundles through the CUDA route,
  drives Qwen projector/prefill/decode placeholder execution, and verifies
  quantized storage markers such as `q4_k` and `iq2_xxs` survive into CUDA
  weight artifacts while mmproj tensors stay out of the decoder weight pack
- immediate next target: finish the family-specific Qwen/Gemma import-mapping
  gaps the current inventories still expose, then replace placeholder CUDA
  packed-weight and execution materialization with backend-native artifacts and
  math without relaxing the current contract coverage

## Roadmap Status

- repository bootstrap and core contracts are effectively complete
- the Fortran control plane is well past scaffolding:
  - runtime, model, and session lifecycle exist
  - park and resume are wired
  - workspace reuse exists
  - optimization evidence and cache identity are persisted
- public C ABI QA is now broad enough to lock fail-closed behavior for stale
  handles, failed opens, routing and availability failures, session guard
  paths, and last-error propagation through the contract suite
- CUDA is the most advanced backend:
  - capability probing exists
  - planner and bridge seams exist
  - projector, prefill, and decode all execute through placeholder CUDA paths
  - backend-owned session state survives prefill, decode, park, and resume
- the cache and self-optimization layers are real runtime surfaces, including
  strict plan, weight, session, and multimodal caches plus persisted winner
  evidence, but backend-native packed weights and backend-native executable
  plan artifacts are still ahead of us
- Apple is now beyond pure scaffolding:
  - capability probing uses the Apple bridge
  - planner parity exists for ANE and Metal
  - projector, prefill, and decode run through placeholder Apple execution
  - Apple live contexts use the same park/resume checkpoint path as CUDA
  - model-open and stage routing now respect detected backend availability
  - hardware validation is still the biggest Apple gap
- model import and target-asset mapping are still only partially done, but
  there are now concrete safetensors and GGUF smoke-import paths for local
  Qwen/Gemma-shaped asset directories and Qwench-style GGUF caches
- the importer/output-layout contract is documented and enforced, but target
  family tensor mapping is still mostly ahead of us

In short:

- the control plane and runtime contract are in good shape
- the CUDA backend is credible as a bring-up path
- real inference math, real packed weights, and Apple hardware validation are
  still major milestones ahead

## What Exists

### Core Contracts

- the project has a Fortran-first source tree under `src/`
- the public C ABI exists in `include/mizu.h`
- runtime, model, and session handles are implemented as opaque C-facing boxes
- status codes, type enums, and execution report records are defined in source

### Runtime Skeleton

- runtime create and destroy are wired through the C bridge
- model open and close are wired through the C bridge
- session open, close, park, resume, prefill, decode, and output read are wired
- manifest loading and validation are implemented
- target fallback manifests exist for the current Qwen and Gemma targets
- the loader now recognizes an optional `mizu_import/` bundle under model root
  and can apply imported tensor, modality, and projector inventories on top of
  the logical manifest
- imported bundles now validate inventory shape, asset-path safety, and
  referenced file existence before the runtime accepts them
- runtime create now records a detected backend inventory for Apple and CUDA
- model open now intersects the requested backend mask with that detected
  runtime inventory and fails early with `MIZU_STATUS_NO_VALID_PLAN` when no
  requested backend is actually available
- session staging now retains attached token values and copied modal-byte inputs
  long enough for stage execution, along with stable content hashes
- live session context identity now survives prefill and advances after decode,
  so later decode steps can depend on prior staged and emitted content
- live sessions now also retain a persisted backend-owned context byte buffer
  for the active route, starting with CUDA and now including Apple placeholder
  contexts
- `park` and `resume` now materialize and reload a small session-checkpoint
  payload when `cache_root` is configured and route-specific context state
  exists
- parked sessions with backend-owned live contexts now offload that resident
  context buffer after the checkpoint is safely materialized, and `resume`
  restores it before the session becomes live again
- CUDA live-context payloads now include a small versioned, checksummed
  header, and both `resume` and CUDA decode validate header plus payload
  integrity before treating restored bytes as usable backend state
- CUDA live-context payloads now also retain producer-stage and producer-plan
  lineage, so decode rejects incompatible plan drift and resume rejects
  mismatched checkpoint reloads for offloaded CUDA sessions
- CUDA live-context payloads now use a fixed-size state block with explicit
  state lanes and a compact summary word, so decode consumes structured
  backend state instead of folding arbitrary context bytes into one seed
- those structured CUDA state lanes now have stable semantics: token digest,
  modal digest, packed KV/decode-step counters, and a rolling decode-state
  word
- the runtime can now read that semantic CUDA state back through a Fortran-side
  extractor, which keeps tests and future planner logic from treating the
  payload as an opaque blob
- the CUDA live-context payload now reserves 128 bytes instead of 64 and uses
  the extra space for a small windowed state image
- that windowed state image carries:
  - page-like KV metadata for a few compact logical decode pages
  - a recent-token ring
  - a state-image digest over the compact window
- the CUDA live-context payload now reserves 256 bytes instead of 128 so it
  can carry compact per-page slot contents in addition to page metadata
- those slot payloads now give each tracked decode page a small explicit token
  image, which makes the placeholder runtime state look more like compact
  backend decode state than a metadata-only summary
- the CUDA live-context payload now reserves 512 bytes instead of 256 so it
  can widen those page images into compact key and value lane planes, along
  with per-page lane digests
- the runtime can now read those compact KV-style lane planes back through a
  Fortran-side extractor, which makes the widened decode image inspectable in
  tests instead of opaque
- the remaining tail of that 512-byte payload now carries explicit per-page
  tensor-layout records:
  - key row count
  - key lane count
  - value row count
  - value lane count
  - derived head block
  - page generation
- the runtime can now read those page-layout records back through a second
  Fortran-side extractor, which makes the compact page image look more like a
  small tensor record than a bag of lanes
- the CUDA live-context payload now reserves 640 bytes instead of 512 so it
  can carry a compact per-page control table after the layout block
- those page-control records now capture:
  - owner kind
  - usable capacity
  - committed rows
  - free rows
  - page epoch
  - recycle epoch
  - logical page id
  - page flags
- the runtime can now read those page-control records back through a third
  Fortran-side extractor, which makes compact decode state inspectable as a
  small page table rather than just a lane image plus shape metadata
- the CUDA executor tests now drive the compact page table through window
  overflow, which validates logical page rotation and recycled-slot marking
  instead of only cold page allocation and append behavior
- the CUDA live-context payload now reserves 768 bytes instead of 640 so it
  can carry a compact per-page tensor descriptor block after the control table
- those tensor descriptor records now capture:
  - key storage offset
  - key committed byte span
  - key capacity byte span
  - key row stride
  - value storage offset
  - value committed byte span
  - value capacity byte span
  - value row stride
- the runtime can now read those tensor descriptor records back through a
  fourth Fortran-side extractor, which makes the compact page image inspectable
  as a small tensor-backed page record rather than only a lane image plus page
  metadata
- the public CUDA contract path now validates one narrow multimodal flow end
  to end:
  - open runtime, model, and session
  - attach staged tokens and one modal payload
  - run projector, prefill, and decode through the public API
  - read back output tokens through `mizu_session_read_output`
  - park and resume session state through a persisted checkpoint artifact
  - reopen a fresh runtime and confirm warm cache reuse plus token
    reproducibility for the same multimodal staged context
- the current CUDA unit suite still locks exact deterministic reference outputs
  for the executor path, with separate pinned expectations for the real CUDA
  bridge and the CPU CUDA stub:
  - projector embedding count on the executor fixture
  - decode-token sequence across repeated executor steps
  - alternate-context decode token divergence
- the public CUDA contract suite now asserts stable positive decode output plus
  warm-path reproducibility for the narrow multimodal fixture instead of
  pinning a single build-specific public API token

### Self-Optimization

- stage selection uses route-neutral optimization identities
- ANE, Metal, and CUDA candidates can be explored under one shared workload key
- exploration is bounded by `exploration_budget`
- repeated work can reuse the measured winner
- optimization evidence is persisted to disk through `optimization_store_v1.txt`

### Build and Backend Scaffolding

- a top-level `Makefile` now builds and runs the current test set through
  `make test`
- backend scaffolding now exists under:
  - `src/backends/apple/`
  - `src/backends/cuda/`
- initial capability probes exist for:
  - Apple Metal through the Apple bridge seam on macOS
  - Apple ANE through the Apple bridge seam on macOS
  - CUDA via a real CUDA device bridge, with `nvidia-smi` and override fallback
- Apple planner scaffolding now exists for:
  - model-load weight-pack records
  - projector plan records
  - prefill plan records
  - decode plan records
- route-specific Apple artifact payloads are now materialized through the C API
  metadata path for both ANE and Metal candidates
- Apple bridge basics now exist through:
  - `src/backends/apple/apple_bridge.h`
  - `src/backends/apple/apple_bridge.m`
  - `src/backends/apple/apple_bridge_stub.c`
  - `src/backends/apple/mod_apple_bridge.f90`
- Apple placeholder execution now exists for:
  - projector
  - prefill
  - decode
- Apple execution now supports a no-`cache_root` virtual-payload path, so
  low-ceremony runtimes can still execute Apple placeholder stages without
  forcing artifact materialization to disk
- Apple live-context bytes now carry route-aware lineage and are accepted by
  the shared session checkpoint path, so parked Apple sessions can be restored
  through the same runtime/cache machinery as CUDA
- CUDA planner scaffolding exists for:
  - model load weight-pack records
  - projector plan records
  - prefill plan records
  - decode plan records
- CUDA-selected artifacts can now materialize stub payload files under the
  configured `cache_root`
- CUDA-selected projector, prefill, and decode stages now execute through a
  backend-owned CUDA bridge that launches minimal real kernels on NVIDIA
  hardware
- CUDA weight, projector, prefill, and decode artifact payloads now retain
  imported pack dependency lineage, and prefill/decode now also retain
  stage-specific `pack_use_*` usage records that name the exact imported
  tensors selected from the packed layout
- CUDA prefill and decode artifact payloads now also retain compact numeric
  `pack_dispatch*` records for the first selected packed tensors, and the CUDA
  executor now prefers those records when reconstructing its pack-dispatch
  working set
- CUDA prefill and decode artifact payloads now also retain compact
  importer-rooted `pack_span*` records for the first selected packed tensors,
  and the CUDA executor now resolves those bundle files directly so bridge
  execution depends on sampled imported tensor spans instead of only artifact
  text and numeric dispatch identity
- those same CUDA prefill and decode plans now also materialize tiny
  `.spancache` sidecars under `cache_root`, and the CUDA executor prefers
  those persisted staged span records, including sampled span bytes, on warm
  runs before falling back to importer bundle reads
- those same CUDA `.spancache` sidecars now also retain compact staged
  pack-page records derived from the selected imported tensor spans, and the
  CUDA bridge now prefers those structured page words before falling back to
  raw sampled span bytes
- CUDA prefill and decode now also materialize narrow `.tilecache` payloads
  under `cache_root`, referenced by the `v4` `.spancache` sidecars, and the
  CUDA bridge now prefers those staged tile bytes before falling back to page
  words or raw sampled span bytes
- CUDA live-context payloads now also retain an explicit pack-usage snapshot
  with usage hash, byte total, and first/last packed tensor spans, so session
  state carries selected imported tensor profile data directly
- CUDA live-context payloads now also retain an explicit pack-dispatch snapshot
  for up to four selected packed tensors, including packed offsets, byte spans,
  role codes, and layout codes, and both the real CUDA bridge and the CPU stub
  now preserve that snapshot through prefill and decode
- CUDA placeholder projector and prefill execution now incorporate staged-input
  content hashes instead of relying on counts alone
- CUDA prefill now also receives copied staged token buffers and modal byte
  buffers through the backend bridge, so placeholder execution can depend on
  actual staged tensor content
- CUDA prefill now emits a persisted live-context byte buffer into session
  state
- CUDA placeholder decode execution now consumes that persisted byte buffer
  and updates it across steps instead of depending on a tiny surrogate record
- CUDA decode now advances explicit KV-token and decode-step counters inside
  that persisted context payload, and the compact summary word retains the
  last emitted token plus stop reason
- CUDA decode now also advances the page-like KV window and recent-token ring
  inside the widened state image, so the placeholder backend state has a
  compact but more realistic notion of decode continuity
- CUDA decode now also appends emitted tokens into explicit per-page slot
  payloads inside that state image, so page continuity is represented through
  actual compact payload contents instead of only page fill counters
- CUDA decode now also writes compact value-lane payloads and stable per-page
  lane digests, so unchanged pages retain their own identity while the overall
  state image digest still advances across decode
- CUDA decode now also preserves layout metadata for untouched pages and
  advances the generation counter only on the decode-owned page, which makes
  checkpointed state more honest about what changed
- runtime workspace reservations now allocate a reusable aligned host scratch
  buffer instead of tracking bytes alone, preserving prior scratch contents
  across high-water growth and tracking allocation count for no-allocation
  assertions
- CUDA projector, prefill, and decode now receive that runtime workspace buffer
  through the backend bridge and stamp stage-local scratch data into it
- the build now falls back to a CPU CUDA bridge stub when `nvcc` is not present,
  so non-CUDA environments can still build and run the current tests
- `make test` now passes from a clean tree without requiring stray root-level
  `.mod` files from earlier compiler runs
- the contract test binaries now depend on the C API source list in the
  `Makefile`, so `mod_c_api.f90` edits do not leave stale public-path test
  executables behind

### Cache and Artifact Identity

- deterministic cache keys exist for:
  - weights
  - plans
  - sessions
  - multimodal projector inputs
- runtime-scoped in-memory cache presence tracking exists
- persisted artifact presence tracking exists through `artifact_cache_v1.txt`
- persisted artifact records now include backend-specific metadata:
  - backend family
  - execution route
  - stage kind
  - materialization flag
  - payload byte count
  - planned workspace byte count
  - artifact format label
  - payload fingerprint
  - future payload path
- session-store metadata now persists alongside weight, plan, and multimodal
  metadata, so parked-session checkpoint artifacts can be reloaded across
  runtime create/destroy boundaries

### Tests That Pass

- `test_model_manifest_loader`
- `test_cache_keys`
- `test_cache_store`
- `test_optimization_store`
- `test_stage_reports`
- `test_backend_availability`
- `test_backend_registry`
- `test_apple_planner`
- `test_apple_executor`
- `test_runtime_workspace`
- `test_session_staging`
- `test_cuda_planner`
- `test_cuda_executor`
- `test_cuda_artifacts`

## What Is Still Stubbed

- no real Apple Metal or ANE compute backend exists yet
- the Apple bridge currently executes deterministic placeholder projector,
  prefill, and decode paths rather than real transformer math
- the Apple planner is still heuristic/scaffold-level rather than
  hardware-validated
- no real ANE executor exists yet
- no real Metal executor exists yet
- CUDA planner records are still scaffold-level and do not launch kernels
- model load does not build actual packed weights
- plan selection does not materialize backend-native plan payloads
- CUDA projector, prefill, and decode use real but placeholder kernels; they do
  not execute transformer math yet
- decode still does not consume a persisted tensor/KV buffer from prior
  execution state; it now uses a persisted backend-owned context byte buffer
  plus placeholder kernel logic
- Go bindings do not exist yet

## Important Honesty Notes

- persisted artifact metadata is real metadata, not fake cache hits
- route-specific artifact descriptors are persisted even when payload bytes are
  `0`
- `payload_path` currently points at the future artifact location that a real
  backend-owned pack or plan record would use
- `is_materialized = .false.` means the runtime knows the artifact identity and
  route, but does not yet claim to have built the payload
- CUDA-selected artifacts are the first exception: they now write stub planner
  payload files to the persisted artifact location and mark those records as
  materialized
- parked CUDA sessions are now the second exception: they write a small
  checkpoint payload keyed by the live context hash and configuration so
  `resume` can reload backend-owned session state through the runtime cache
- once that checkpoint is materialized, the parked session drops its resident
  CUDA context bytes in memory and relies on `resume` to reconstruct them
- CUDA projector, prefill, and decode now consume those materialized payloads
  through a backend-owned CUDA bridge and launch tiny placeholder kernels to
  prove the runtime-to-device seam
- persisted artifact metadata now carries planned workspace bytes from the
  stage planner, and the runtime now keeps a reusable high-water-mark workspace
  arena around stage execution, backed by a reusable host scratch buffer
- staged token and modal content are now preserved inside session state, but
  only projector, prefill, and the derived live decode context currently fold
  that content identity into the CUDA placeholder execution path
- CUDA prefill now reads real staged token and modal buffers through the bridge,
  but it still uses them for placeholder seed generation rather than real
  transformer activations
- the persisted CUDA context buffer is still a fixed-capacity backend-owned
  surrogate, not a real KV-cache or transformer activation buffer
- the parked-session checkpoint currently persists that same surrogate buffer,
  not a full backend KV-cache image
- decode now explicitly requires a resident CUDA live-context buffer for CUDA
  sessions; parked/offloaded sessions have to come back through `resume`
- CUDA live-context bytes are no longer just unstructured payloads; they now
  carry format/version/kind markers so stale or corrupted context can fail fast
- the structured CUDA context payload is still a compact surrogate for backend
  decode state; it is semantically readable now, but it is not yet a real
  KV-cache image or tensor-backed decode-state record
- the new windowed state image is still intentionally tiny and summary-heavy;
  it behaves more like a compact rehearsal for backend decode state than a
  real device-resident KV layout
- the widened key/value lane planes are still compact synthetic state, not real
  transformer KV tensors or backend-native cache pages
- the per-page lane digests are page-identity aids for the runtime and tests,
  not real backend checksums over device-resident tensor tiles
- the new page-layout records are still synthetic descriptors, not real tensor
  strides, allocator metadata, or backend-owned page tables
- the end-to-end multimodal CUDA contract coverage proves lifecycle and cache
  reuse behavior, but it still validates placeholder execution paths rather
  than true model outputs
- the new reference-output checks lock current placeholder behavior
  intentionally; when the backend math becomes more real, these expectations
  should evolve with the implementation instead of being treated as model truth
- the Apple planner currently uses route-specific heuristic workspace estimates
  and format labels; those choices are contract-shaping scaffolds, not measured
  hardware truths yet
- Apple ANE detection is still conservative and scaffold-level; it currently
  relies on an explicit environment override instead of validated hardware
  probing

## Most Useful Next Steps

1. Define the Apple bridge boundary and ownership/error contracts.
2. Replace the compact CUDA key/value lane image and synthetic page-layout
   records with a more realistic tensor-backed page record or backend-owned
   KV-state payload.
3. Start the thin Go binding once the C ABI settles a bit more.
