# Feedback

[Overview](../../README.md) · re-enters: [1 Shape Kernels](1-shape-kernels.md)

Runtime data and analysis can show that an earlier choice was suboptimal. The analysis agent reads
two inputs and, when it finds a problem, requests changes from phase 1. The deterministic default
stops after one pass.

- **Placement** from [3 Offload](3-offload.md): which kernels run where and how much data crosses
  between devices. Frequent copies between neighboring kernels suggest fusing them or changing their
  scopes.
- **Runtimes** from [5 Sweep Configurations](5-sweep-configurations.md): per-kernel and whole-program
  times. A fused kernel that runs no faster than its parts is a candidate to split again.
- **Request Changes** to [1 Shape Kernels](1-shape-kernels.md): fusion or fission moves, after which
  the later phases run again for the kernels that changed.

Every round re-validates against the NumPy oracle, so feedback changes speed, not results.

| | |
|---|---|
| default | none (single pass) |
| helper | `run_feedback_loop(sdfg, measure, apply_move)` |
| code | `nestforge/phases/feedback.py` |
| open | which placement and runtime signals trigger a request, and when a round stops |
