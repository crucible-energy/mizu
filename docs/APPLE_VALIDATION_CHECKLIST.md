# Apple Validation Checklist

This checklist is for the first hardware pass on a real Apple machine.

Current scope:

- verify that the Apple bridge compiles and links on macOS
- verify that the placeholder Apple execution path runs end to end
- verify that the current ANE-versus-Metal routing surface behaves honestly
- capture enough system context to debug failures without guesswork

What a pass means:

- the current Apple bridge seam is healthy on macOS
- the runtime can execute placeholder Apple `projector`, `prefill`, and
  `decode` paths through the public API
- the repo is ready for the next Apple slice

What a pass does not mean:

- real Metal kernels exist
- real ANE execution exists
- real transformer math is running on Apple hardware yet

## Preflight

Run these first and save the output:

```bash
git rev-parse HEAD
git status --short
sw_vers
uname -a
clang --version
xcode-select -p
system_profiler SPHardwareDataType
system_profiler SPDisplaysDataType
sysctl hw.memsize
sysctl hw.optional.arm64
```

Expected:

- `git status --short` is empty
- the checkout is on the expected commit or later
- `xcode-select -p` resolves cleanly
- `hw.optional.arm64` is `1` on Apple Silicon

## Clean Environment

Make sure no forced backend overrides are already set:

```bash
unset MIZU_FORCE_APPLE_ANE_AVAILABLE
unset MIZU_FORCE_APPLE_METAL_AVAILABLE
unset MIZU_FORCE_CUDA_AVAILABLE
env | rg '^MIZU_FORCE_' || true
```

Expected:

- no `MIZU_FORCE_*` lines remain

## Clean Build

```bash
make clean && make test
```

Expected:

- build succeeds
- all unit tests pass
- all contract tests pass
- there are no Objective-C compile failures around
  `src/backends/apple/apple_bridge.m`
- there are no linker failures around `Foundation` or `Metal`

Important note:

- the current suite includes both natural Apple codepaths and a few tests that
  explicitly force Apple route availability for deterministic coverage
- that is expected for this milestone

## Focused Apple Checks

Run these individually even if `make test` already passed, and save the output:

```bash
./build/tests/test_apple_executor
./build/tests/test_stage_reports
./build/tests/test_backend_availability
```

Expected:

- `test_apple_executor: PASS`
- `test_stage_reports: PASS`
- `test_backend_availability: PASS`

What each one tells us:

- `test_apple_executor`
  - the Apple bridge seam runs both `ANE` and `Metal` placeholder paths
  - Apple live-context bytes are valid across `prefill` and `decode`
  - cross-route misuse is rejected after decode state exists
- `test_stage_reports`
  - public API Apple stages emit route-honest reports
  - Apple projector/prefill paths participate in cache and optimizer reporting
  - persisted Apple artifact metadata is written as expected
- `test_backend_availability`
  - model open now fails early when the requested backend family is not
    available on the runtime
  - the runtime surfaces a useful error string for that case

## Native Probe Sanity Check

This round does not yet have a dedicated public utility that prints detected
runtime inventory directly.

So for now, native Apple probe validation is indirect:

- `apple_bridge.m` must compile and link on the machine
- the machine info above must confirm Apple hardware context
- `make test` and the focused Apple checks must pass

If something looks suspicious, capture it rather than guessing.

## Failure Capture

If anything fails, please send back:

- the exact command that failed
- the full terminal output
- whether any `MIZU_FORCE_*` env vars were set at the time
- the current commit from `git rev-parse HEAD`
- the hardware and OS output from the preflight section
- whether the failure is:
  - compile-time
  - link-time
  - runtime crash
  - test assertion failure

If a test assertion fails, include:

- the test name
- the failing assertion text
- whether the failure reproduces on rerun

## Nice-To-Have Extra Notes

If convenient, also note:

- whether the machine is Apple Silicon Mac mini or another Apple system
- whether Xcode command-line tools were already installed
- whether `Metal` framework link time felt normal or flaky
- whether any warnings stood out even when the tests passed

## Report Template

Sam can send feedback back in this shape:

- machine:
- macOS version:
- commit:
- clean tree:
- `make clean && make test`:
- `test_apple_executor`:
- `test_stage_reports`:
- `test_backend_availability`:
- preflight anomalies:
- compile/link anomalies:
- runtime anomalies:
- logs attached:
