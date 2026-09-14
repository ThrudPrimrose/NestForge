# Feedback

[Overview](../../README.md) · re-enters: [1 Shape Kernels](1-shape-kernels.md),
[2 Define Scopes](2-define-scopes.md)

Measurements can show that an earlier choice was a poor one. Two analysis agents read them and
request changes; the deterministic default runs one pass and stops.

- **Runtime analysis** reads per-kernel and whole-program times from
  [5 Sweep Configurations](5-sweep-configurations.md). A fused kernel that runs no faster than its
  parts is a candidate to split, so this agent requests phase 1 moves.
- **Placement analysis** reads where kernels run and how much data crosses devices from
  [3 Offload](3-offload.md). Frequent copies between neighboring kernels suggest different scopes, so
  this agent sends the program back to phase 2.

The later phases rerun for the kernels that changed, and every round revalidates against the NumPy
oracle.

| | |
|---|---|
| default | none (single pass) |
| helper | `run_feedback_loop(sdfg, measure, apply_move)` |
| code | `nestforge/phases/feedback.py` |
| open | which signals trigger a request, and when a round stops |
