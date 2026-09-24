# Implementation Log

## 2026-09-24: Replace Python model importers and developer-tool tests

### User Experience Findings

- Model-import commands depended on Python even though the runtime toolchain is
  moving to Zig. This made setup less predictable and left import behavior
  split across two implementations.
- The previous contract-test fixtures generated large model-like payloads to
  test small metadata paths. A compact generated GGUF fixture covers the same
  contract without the unnecessary disk and runtime cost.
- Python developer-tool tests were a hidden prerequisite of `make test`, so
  the repo could not validate its local hooks and formatting behavior in a
  Zig-only tool environment.
- A Python packbuffer inspector embedded in the real-model CUDA smoke had been
  moved to C during the first pass, leaving one replacement outside Zig.
- Import-path edge cases needed explicit coverage: shard-index escapes,
  symlink traversal, output replacement, malformed headers, and metadata
  newline injection.
- Checking safetensors tensor ranges against every earlier tensor made a large
  model header quadratic to validate.
- The CUDA contract smokes reused fixed `/tmp` directories and deleted prior
  contents before each run, risking unrelated local artifacts.
- The generic text normalizer still classified `.py` files after all Python
  importers and test harnesses were removed, leaving a stale source-language
  affordance in the tooling.
- Running the required pre-push check outside a Git push exposed a Bash 3.2
  `set -u` failure when its push-ref array was empty.
- The user-requested local Aletheia/Citadel harvest could not be read because
  both configured corpus-status adapters failed with `native_request_failed`;
  the provided share page did not expose the conversation body.
- The repository's debug-validation target wrote `build-debug/` outputs that
  were not ignored, so a normal QA run appeared as untracked source work.

### Engineering Decisions

- Put GGUF and safetensors parsing, validation, and bundle rendering in one
  shared Zig module; keep small format-specific command wrappers to retain the
  existing CLI contracts.
- Preserve bundle schema, generated metadata, output determinism, and safety
  checks. The Zig fixtures assert representative legacy output fields and
  exercise hostile paths; a repeat-run comparison checks deterministic output.
- Retain the former importer coverage for indexed two-shard Qwen safetensors,
  Gemma safetensors, Gemma GGUF, and absolute GGUF source offsets using small
  generated fixtures.
- Replace the four Python developer-tool harnesses with a single Zig test
  driver covering formatter staging, hook installation, pre-push rules, and a
  local bare-remote push.
- Keep packbuffer inspection in that Zig test driver and invoke its recursive
  scanner from the C API smoke; fixture tests cover distinct and identical
  spans without requiring large model files.
- Keep Playwright out of this scope: the migrated tools and hook workflows are
  command-line behavior and have no browser-facing end-to-end surface.
- Keep Mizu's implementation scope on the assigned importer and devtool
  migrations; the [Aletheia edge-deployment thread](https://chatgpt.com/share/6a7c4c50-d09c-83ea-bc4e-99f2ec70b1fa)
  is a separate scope input whose body was unavailable from both local
  Citadel status adapters and the share page. No edge-deployment product
  requirements were inferred from its title.
- Include untracked Zig sources in formatter targets so newly added code is
  checked before it is committed.
- Remove `.py` from the normalizer's supported source-file suffixes after the
  Python files and runtime invocations are removed.
- Compute sensitive-path escalation while reading each push-ref line instead
  of buffering refs in an array. This keeps the pre-push check valid both in a
  real hook invocation and when run directly with no ref input.
- Ignore the dedicated `build-debug/` directory used by `make check-debug`,
  matching the existing ignored `build/` artifact policy.
- Preserve the safetensors overlap guard with a sort-and-scan pass, reducing
  range validation from quadratic comparisons to `O(n log n)` while retaining
  deterministic failure for overlapping tensors.
- Give each CUDA smoke a unique `mkdtemp` root and print its path for inspection;
  this removes destructive cleanup of a fixed shared temp path.
- Add a source-owned, `internal_only` sibling evidence packet for open issue
  [#33](https://github.com/crucible-energy/mizu/issues/33). Its posture is
  limited to what the repository documents and it grants no campaign or
  release authority.

### Validation

- The latest `make tool-tests` passed all four importer unit tests, the
  importer and packbuffer fixture integration tests, and developer-tool tests.
- `bash scripts/mizu-pre-push-check.sh` passed after the no-ref-array fix; it
  ran the full default suite, formatting and whitespace checks, and shell
  syntax checks.
- `make check-debug` passed with Fortran bounds checks enabled, including the
  Zig importers, both Zig tooling suites, and all contract tests.
- `make format-check` passed after the final source, script, and documentation
  edits.
- The qwench smoke compiled and skipped because Qwench model assets were not
  available locally.
- The Zig devtools fixture covers an empty-input invocation of the pre-push
  script, preventing the Bash 3.2 unbound-array regression.
- The CUDA integration fixture uses a unique per-run temporary directory and
  keeps it for diagnosis; old fixed shared test output is left untouched.
- The Mizu evidence packet passed Jazz Hands' `--validate-file` contract
  checker in an isolated temporary validation root; the dirty Jazz Hands
  checkout was left untouched.

### Known Limitations

- Validation uses small generated fixtures rather than downloading full
  Qwen/Gemma model weights; the real-model smoke test remains dependent on
  locally available model assets.
- GGUF tensor-type support remains bounded to the IDs supported by the prior
  importer. New upstream tensor types require an explicit parser update.
- The packet for issue [#33](https://github.com/crucible-energy/mizu/issues/33)
  still requires the Jazz Hands reconciliation workflow and portfolio-owner
  review. It does not add a campaign role or authorize a release.
- Citadel corpus status failed with `native_request_failed` in both the local
  and Aletheia adapters while resolving the Aletheia-harvested shared thread.
  The public share page exposed its title, “Aletheia Edge Deployment
  Strategy,” but no conversation body. No repo-scope conclusions have been
  inferred from that title; revisit the scope update when the retained harvest
  is readable.
- This migration removes Python from Mizu source and test execution. It does
  not certify that other Crucible Energy repositories have completed their
  separate language migrations.
