# Phase 3: Offload

prev: [2 Define Scopes](2-define-scopes.md) · next: [4 Optimize Kernels](4-optimize-kernels.md)

Offloading runs after phase 2 defines the scopes, on the program whose kernels are `ExternalCall` nodes. It
assigns each kernel a device and places the host/device copies that follow from that choice.
Without a GPU target every kernel stays on the CPU.

With a GPU target, the default is DaCe's `OffloadToAccelerator` pass, which schedules library nodes
and inserts copies where data changes location. An agent may instead provide the kernel-to-device
schedule. A placement that needs different kernel boundaries returns the program to
[phase 2](2-define-scopes.md); placements also go to the [analysis agent](feedback.md).

| | |
|---|---|
| default | CPU only, or `OffloadToAccelerator` with a GPU target |
| code | `nestforge/phases/offload.py` |
| status | planned |
