# AGENTS.md

mizu / Crucible Contribution Guidance (Agents + Humans)

Last reviewed: July 3, 2026

The GitHub repo description for `crucible-energy/mizu` is:
"Work in progress inference layer optimized for small agent systems."

Until that changes, work here is judged on whether it makes that inference
layer more truthful, reviewable, validated, and execution-ready without
overstating backend completeness.

This file is minimally adapted from `../subsequent/AGENTS.md` for this project
at `/Users/sam/git/crucible/mizu`.

## Working Context

- Repository path: `/Users/sam/git/crucible/mizu`.
- Read `README.md` first for current repo framing.
- Read `SAM.md` when collaboration style, ownership context, or ambition
  management materially affects the work.
- Read `LYNN.md` when Lynn's product ownership, final-decision context,
  collaboration style, or systems-method context materially affects the work.
- Read `SCOTT.md` when Scott's review or collaboration expectations materially
  affect the work.
- `SAM.md` is an intentional sibling copy from `../shrinkray/SAM.md`.
- `LYNN.md` is an intentional sibling copy from
  `../gonzo-post-merge-wrapup/LYNN.md`.
- `SCOTT.md` is an intentional sibling copy from `../shrinkray/SCOTT.md`.
- `AGENTS.md` is intentionally adapted, not an exact copy, because repo mission,
  path, and trust boundaries differ.

## Repository Mission

- Treat this repo as a work-in-progress local inference runtime with real
  cache, planner, session, importer, and C-ABI surfaces, but still-incomplete
  backend execution.
- Favor guidance, structure, and artifacts that improve truthfulness,
  reviewability, provenance, and future extensibility.
- Preserve provenance for externally derived knowledge. Record source, date,
  and whether material is quoted, summarized, synthesized, inferred, or
  original when that distinction matters.
- Respect copyright, confidentiality, and licensing boundaries. Do not import
  private, proprietary, or non-redistributable material unless explicit
  authorization and handling rules are documented.
- Separate current state, planned direction, implementation scaffolding, and open
  questions. Do not blur speculation into canonical repo posture.
- Prefer smaller, composable, cross-linkable units over an undifferentiated
  omnibus doc set.

## Repo Non-Negotiables

- This repo is code-and-docs first. Fortran runtime code, bridge code, the C
  ABI, importer tooling, tests, and current-state docs are all authority-bearing
  review surfaces. Add new structure only when there is a clear local need.
- The primary artifacts are runtime code, cache/planner/session machinery,
  importer tooling, tests, manifests, and docs that keep current state and
  constraints trustworthy.
- Preserve the core loop:
  1. confirm the current repo state and stated scope
  2. make the smallest coherent change that improves the repo honestly
  3. verify the touched surfaces
  4. record remaining gaps, uncertainty, and next constraints
- Do not collapse the repo into a scratchpad, generic notes dump, or
  placeholder-heavy scaffold with no clear review value.
- Do not silently upgrade a draft note, speculative direction, or copied
  sibling pattern into canonical repo policy without making the change explicit.
- No new CI, GitHub Actions, scheduled automation, branch protection, or other
  enforcement surfaces without explicit approval from Sam.
- When useful work is complete and validated, commit and push it unless
  explicitly asked to hold it back or unless doing so would publish a known
  broken build or failed required checks. Prefer frequent small commits over
  large uncommitted deltas.
- Use `SAM_GH_TOKEN` from `~/.zshrc` for GitHub API access when needed.
  Never print, inspect, persist, commit, or copy the token value. Prefer
  secret-free API readiness checks.
- Treat Codex Review, CodeRabbit, human reviewer, and other automated review
  findings with the same discipline: address each actionable finding or record
  exactly why it is being skipped.
- When addressing GitHub PR review feedback, reply directly to each actionable
  finding comment with the fix or the reason it is being skipped.
- If an actionable inline review thread is addressed and validated, resolve it
  in the same pass to reduce reviewer noise. This is required for Codex review
  threads after a direct reply.
- If a finding only exists in a review summary or outside-diff note with no
  inline reply surface, leave a PR comment that names the finding and records
  the disposition.
- Do not add noise replies to non-actionable informational bot comments such as
  rate-limit notices, review summaries with no requested change, or status-only
  updates.
- Time-sensitive claims about repo scope, external systems, platform behavior,
  provider pricing, benchmarks, laws, or best practices need a concrete date
  when the date materially affects meaning.
- Keep Markdown free of machine-specific absolute filesystem paths except where
  the path is intentionally part of repo-local operating guidance, such as the
  repository path above or a user-specified local reference.
- Preserve sibling-repo alignment where it helps, but adapt for this repo's
  actual state rather than cargo-culting structure.

## Memory And Concurrency Safety

- Since this repo's active runtime and bridge surfaces are primarily Fortran
  with C, Objective-C, and CUDA seams rather than Rust-managed or
  Python-managed runtime surfaces, these safety properties must be enforced
  explicitly in design, implementation, review, and validation.
- The following are non-negotiable release gates in this repo:
  - Memory safety
  - Use-after-free prevention
  - Double-free prevention
  - Dangling pointer prevention
  - Data race prevention
- Apply those constraints here with the simplest enforceable local posture:
  - prefer allocatable/value ownership over Fortran pointer aliasing and
    lifetime ambiguity
  - treat Fortran `pointer`/`target` lifetimes, C interop buffers, and bridge
    handles as exceptional surfaces that need explicit owner and cleanup rules
  - keep C, Objective-C, and CUDA bridge ownership, buffer sizes, and lifetime
    boundaries explicit and single-owner
  - avoid hidden shared mutable state; if concurrency is introduced, the
    synchronization story must be explicit and reviewable
- Until an explicit reviewed design says otherwise, the Mizu runtime contract
  is explicit-state, fail-closed, and deterministic before it is clever.
- Any exception to those rules requires:
  - explicit justification in the diff
  - a focused validation story for the affected safety property
  - an update to the repo-local safety audit so the exception is reviewable
- `make test` is part of the repo's required local validation path, and
  targeted `build/tests/test_*` binaries plus tooling tests are appropriate
  during iteration before rerunning the full suite.
- `make check-debug` is the repo-local deeper safety pass for the same suite
  under Fortran runtime checks (`-fcheck=all -fbacktrace`) in an isolated
  debug build directory. Use it when touching bridge, lifetime, handle, or
  memory-sensitive surfaces.

## Repo Priorities

- Prefer plain, durable formats first: Markdown for prose and reviewable policy;
  JSON/YAML/TOML/CSV only when structured metadata or tool contracts actually
  need them.
- Design repo additions for retrieval and review: stable headings, clear
  scope, explicit status, provenance notes, and low duplication.
- Favor quality, density, and trust over volume.
- Pair consequential claims with source references, observed evidence, or clear
  reasoning whenever feasible.
- Treat repo scaffolds, schemas, tooling, and automation surfaces as separate
  capability layers until they actually exist.
- Prefer deterministic local tooling over hidden prompt-only behavior.
- Pair repo-level claims with executable checks whenever feasible.

## Guidance Discipline

- Prefer 2-3 focused guidance modules for any reusable procedure. Good future
  splits for this repo may be:
  - repo framing and scope
  - source/provenance posture
  - validation and release workflow
- If a workflow becomes longer than a compact guidance module can reliably
  carry, split it into a focused companion doc instead of bloating `README.md`
  or `AGENTS.md`.
- Favor curated, human-reviewed procedures over self-generated boilerplate.
- Require deterministic checks wherever possible:
  - `git diff --check`
  - referenced-path and command sanity checks
  - schema or metadata validation when structured files exist
  - link or docs validation when those surfaces exist
  - provenance notes for consequential claims
- Assume some guidance will hurt clarity if it becomes too long, too generic,
  or too comprehensive. Pressure-test it.

## Anti-Patterns

- Do not invent product scope, implementation status, or operational maturity
- Do not add placeholder automation or ceremonial structure with no current
  proof path
- Do not import copyrighted, confidential, proprietary, or private material
  without explicit authorization
- Do not mix raw notes, active policy, speculative ideas, and future plans
  without labels
- Do not publish readiness claims without measured evidence, review evidence,
  or clearly labeled rationale

## Documentation And Claim Discipline

- Use labels such as `Current`, `Planned`, `Observed`, `Measured`, `Inferred`,
  `Hypothesis`, and `Open Question` when the distinction matters.
- For repo documentation, prefer this information shape:
  - purpose and scope
  - current repo state
  - source and provenance
  - constraints and non-negotiables
  - concrete next steps
  - open questions and unsupported cases
- Keep source summary, synthesis, recommendation, and implementation strategy
  separate when that improves clarity.
- If a recommendation is reversible, partial, risky, or conditional, say so
  explicitly.
- If a capability, workflow, or enforcement surface does not yet exist, do not
  write as though support is complete.
- Keep terminology stable:
  - `mizu` for the repo/project name
  - `branch-first` for the git workflow used here
  - `Current`, `Observed`, `Measured`, `Planned`, and `Open Question` for claim
    and lifecycle status

## Branch-First Policy

- All work must happen on a feature branch. Direct commits to `main` are
  prohibited except for explicitly approved automated maintenance.
- Good branch names document function, value, or domain meaning:
  `docs/agent-guidance`, `fix/cache-persistence`, `feat/<clear-domain>`,
  `fix/<specific-issue>`.
- Bad branch names: `tmp`, `misc`, `final-final`, `fix`.
- Every branch must pass the repo's relevant local checks before opening a pull
  request. The default local gate here is:
  - install the repo-local hooks once with `make hooks`
  - run `make format-check`
  - run `git diff --check`
  - run `make test`
  - during narrow iteration, prefer the smallest relevant `build/tests/test_*`
    binary or tooling test before rerunning the full suite
  - when touching memory-sensitive runtime or bridge code, also run
    `make check-debug`
- The repo-local hooks are part of normal development, not optional hygiene:
  - pre-commit runs `./scripts/format-local.sh --staged --write --restage`
    and `git diff --cached --check`
  - pre-push runs `bash scripts/mizu-pre-push-check.sh`
  - pre-push always runs the default local gate and automatically escalates to
    `make check-debug` when the pushed range touches memory-sensitive runtime,
    bridge, cache, or related build surfaces
  - before claiming a branch is ready, ensure those same checks pass when run
    directly via `make format-check`, `git diff --check`, and `make test`
  - `make check-debug` remains available as a direct manual command when you
    want the heavier safety pass before the hook would require it
- After useful validated work on a feature branch, push the branch promptly so
  remote state stays aligned with local progress. Default to publishing each
  validated useful change rather than batching unrelated work locally.
- Pull requests should be small, reviewable, and focused on a single concern.
- Do not open draft pull requests in this repo. If a pull request should
  exist, open it ready for review after the relevant automated validation has
  passed.
- Be selective about pull request creation. Do not open a pull request merely
  because a branch has changes; open one when the branch has reached a
  meaningful reviewable checkpoint with sufficient automated coverage for the
  claimed behavior.
- Whenever giving a status update about a pull request, include the full PR URL
  in addition to any shorthand such as `#1`.
- Squash-merge into `main` when the branch is ready, review findings are
  resolved, and required checks have passed.
- Address actionable review findings directly in the branch that received them.
- Clear merge conflicts in that same branch before asking reviewers to reason
  about stale diffs.
- Before claiming a pull request is clean, re-query unresolved review threads
  and actionable comments on that PR rather than assuming earlier passes were
  complete.
- Treat outdated but still-open review threads as requiring explicit
  disposition. Do not leave them hanging just because newer commits moved the
  diff.
  When a finding is fixed or otherwise fully dispositioned:
  - reply directly on the review thread with the concrete resolution
  - include the commit or validation evidence that closes the finding
  - resolve the thread to reduce reviewer noise once the disposition is final

## Working Style

- Keep changes focused and reviewable. Avoid opportunistic repo reshaping
  unless the task is explicitly a repo-organization pass.
- Match sibling-repo conventions when they help, but prefer truthful local fit
  over superficial consistency.
- Prefer modular docs over one expanding omnibus file.
- Read the nearest governing surfaces before editing:
  - `README.md` for repo framing
  - `AGENTS.md` for operating rules
  - `SAM.md`, `LYNN.md`, and `SCOTT.md` when collaboration context matters
- If the repo grows, add focused docs for:
  - repo framing and scope
  - provenance and licensing posture
  - validation and release workflow
  - integration boundaries
    rather than overloading root files.
- Use names that describe function, state, or responsibility. Do not add repo
  or brand words to variables, type names, helper functions, or internal
  identifiers unless the boundary is intentionally public or
  integration-bearing.
- Useful outputs here include:
  - repo-framing docs
  - focused specs
  - manifests
  - provenance notes
  - validation reports
  - small reviewed scaffolds
