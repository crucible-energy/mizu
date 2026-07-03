# AGENTS.md

subsequent / Crucible Contribution Guidance (Agents + Humans)

Last reviewed: June 20, 2026

The GitHub repo description for `crucible-energy/subsequent` is currently
unset.

Until that changes, work here is judged on whether it keeps the repo bootstrap
truthful, reviewable, provenance-aware, and easy to extend without locking the
project into an invented mission too early.

This file is minimally adapted from the latest complete sibling bootstrap
guidance copied from `../nichthub/AGENTS.md` for this project at
`/Users/sam/git/crucible/subsequent`.

## Working Context

- Repository path: `/Users/sam/git/crucible/subsequent`.
- Read `README.md` first for current repo framing.
- Read `SAM.md` when collaboration style, ownership context, or ambition
  management materially affects the work.
- Read `LYNN.md` when Lynn's product ownership, final-decision context,
  collaboration style, or systems-method context materially affects the work.
- Read `SCOTT.md` when Scott's review or collaboration expectations materially
  affect the work.
- `SAM.md` is an intentional sibling copy from `../nichthub/SAM.md`.
- `LYNN.md` is an intentional sibling copy from `../nichthub/LYNN.md`.
- `SCOTT.md` is an intentional sibling copy from `../nichthub/SCOTT.md`.
- `AGENTS.md` is intentionally adapted, not an exact copy, because repo mission,
  path, and trust boundaries differ.

## Repository Bootstrap Mission

- Treat this repo as an intentionally minimal bootstrap surface until a real
  product, code, or corpus direction is explicitly established.
- Favor guidance, structure, and artifacts that improve truthfulness,
  reviewability, provenance, and future extensibility.
- Preserve provenance for externally derived knowledge. Record source, date,
  and whether material is quoted, summarized, synthesized, inferred, or
  original when that distinction matters.
- Respect copyright, confidentiality, and licensing boundaries. Do not import
  private, proprietary, or non-redistributable material unless explicit
  authorization and handling rules are documented.
- Separate current state, planned direction, bootstrap scaffolding, and open
  questions. Do not blur speculation into canonical repo posture.
- Prefer smaller, composable, cross-linkable units over an undifferentiated
  omnibus doc set.

## Repo Non-Negotiables

- This repo is Markdown-first until a stronger local contract exists. Markdown
  is the canonical review surface for bootstrap guidance, working policy,
  initial specs, and repo-shape decisions. Add structured data or code only
  when there is a clear local need.
- The primary bootstrap artifacts are root guidance files, repo-framing docs,
  lightweight manifests, and small validation helpers that keep those artifacts
  trustworthy.
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
- The initial June 14, 2026 guidance bootstrap is the intentional exception to
  the branch-first rule: copy/adapt these root guidance files, commit them on
  `main`, then follow the Branch-First Policy below for all future work.
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

- Since this repo's implementation surface is Zig and TypeScript rather than
  Rust or Python, memory and concurrency safety must be explicit repo policy,
  not assumed language magic.
- The following are non-negotiable in this repo:
  - memory safety
  - use-after-free prevention
  - double-free prevention
  - dangling pointer prevention
  - data race prevention
- Apply those constraints here with the simplest enforceable local posture:
  - no manual heap allocation or free in repo Zig code
  - no raw pointer casts, pointer-int casts, or const-discarding casts in repo
    Zig code
  - no thread or atomic shared-state primitives in repo Zig code
  - no `SharedArrayBuffer` or `Atomics` surfaces in repo JS or TS code
- Until an explicit reviewed design says otherwise, the Subsequent runtime
  contract is single-threaded, value-oriented, and heap-avoidant.
- Any exception to those rules requires:
  - explicit justification in the diff
  - a focused validation story for the affected safety property
  - an update to the repo-local safety audit so the exception is reviewable
- `bun run safety:audit` is part of the repo's required local validation path,
  and its emitted report is the source of truth for checked-file counts and
  scan scope.

## Bootstrap Priorities

- Prefer plain, durable formats first: Markdown for prose and reviewable policy;
  JSON/YAML/TOML/CSV only when structured metadata or tool contracts actually
  need them.
- Design bootstrap additions for retrieval and review: stable headings, clear
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
- For bootstrap documentation, prefer this information shape:
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
  - `subsequent` for the repo/project name
  - `bootstrap` for the initial root-guidance setup
  - `branch-first` for the post-bootstrap git workflow
  - `Current`, `Observed`, `Measured`, `Planned`, and `Open Question` for claim
    and lifecycle status

## Branch-First Policy

- All future work after the initial guidance bootstrap must happen on a feature
  branch. Direct commits to `main` are prohibited except for explicitly
  approved automated maintenance.
- Good branch names document function, value, or domain meaning:
  `docs/repo-bootstrap`, `chore/root-guidance`, `feat/<clear-domain>`,
  `fix/<specific-issue>`.
- Bad branch names: `tmp`, `misc`, `final-final`, `fix`.
- Every branch must pass the repo's relevant local checks before opening a pull
  request. This repo now has a stronger repo-native gate:
  - run `bun install` after clone so `postinstall` sets `core.hooksPath` to
    `.githooks`
  - if needed, repair that explicitly with `bun run hooks:install`
  - rely on the repo-local pre-commit hook to apply deterministic formatting
    before every commit using the shared sibling website `biome.json` plus
    `zig fmt`
  - rely on the repo-local pre-push hook, which runs
    `bash scripts/subsequent-pre-push-check.sh`, before every push
  - run `bun run check` and `git diff --check` before opening or updating a PR
- After useful validated work on a feature branch, push the branch promptly so
  remote state stays aligned with local progress. Default to publishing each
  validated useful change rather than batching unrelated work locally.
- Pull requests should be small, reviewable, and focused on a single concern.
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
