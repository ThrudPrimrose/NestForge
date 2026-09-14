# Phase 3: Offload

prev: [2 Define Scopes](2-define-scopes.md) · next: [4 Optimize Kernels](4-optimize-kernels.md)

Phase 3 gives each kernel a device and places the copies that choice needs. Without a GPU target
every kernel stays on the CPU and the program is unchanged.

With a GPU target the default runs DaCe's `OffloadToAccelerator`. Every kernel runs on the GPU and
the data it touches moves to device memory: inputs are copied down, outputs are copied back, and an
output the kernel overwrites in full is not copied down. An agent can supply its own kernel-to-device
schedule instead.

`define_scopes` may run again after offloading to lower maps still left, and it keeps the placement.
A placement that needs different kernel boundaries restarts from the phase 2 program; placements
also feed the [placement analysis](feedback.md).

| | |
|---|---|
| default | CPU only, or `OffloadToAccelerator` with a GPU target |
| session | `offload()` returns each kernel's device under a fresh id, and the copies |
| code | `nestforge/phases/offload.py` |
| planned | agent-provided schedules |
