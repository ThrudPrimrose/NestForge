# Phase 3: Offload

prev: [2 Define Scopes](2-define-scopes.md) · next: [4 Optimize Kernels](4-optimize-kernels.md)

Offloading runs after phase 2 defines the scopes, on the program whose kernels are `ExternalCall` nodes. It
assigns each kernel a device and places the host/device copies that follow from that choice.
Without a GPU target every kernel stays on the CPU and the program is not touched.

With a GPU target, the default is DaCe's `OffloadToAccelerator` pass. It schedules every host-level
library node, so every kernel, on the GPU, moves the data those kernels touch to device memory, and
inserts a copy where data changes location. An input is copied down, an output is copied back, and
an output the kernel overwrites in full is not copied down. An agent may instead provide the
kernel-to-device schedule. A placement that needs different kernel boundaries returns the program to
[phase 2](2-define-scopes.md); placements also go to the [analysis agent](feedback.md).

`define_scopes` can run again after offloading and lowers any parallel top-level map still left. It
does not undo the placement, so a scope change that moves an offloaded kernel starts again from the
phase 2 program.

| | |
|---|---|
| default | CPU only, or `OffloadToAccelerator` with a GPU target |
| session | `offload()` returns each kernel's device under a fresh id, and the copies |
| code | `nestforge/phases/offload.py` |
| status | default implemented; agent schedules planned |
