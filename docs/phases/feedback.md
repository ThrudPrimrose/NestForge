# Feedback edges

[Overview](../../README.md) · re-enters: [1 Inter-kernel schedule](1-inter-kernel.md)

Two edges carry measurements back to phase 1. An analysis agent reads them and requests fusion or
fission moves; the deterministic default stops after one pass.

- **(e) after offloading.** The placement and its copy volume. Frequent host/device copies between
  neighboring kernels suggest fusing them or changing their scopes.
- **(g) after codegen variants.** Per-kernel and whole-program times next to each kernel's symbolic
  work, depth and OI. A fused kernel whose speedup trails its gain in OI is a fission candidate.

Every round re-validates against the NumPy oracle, so feedback changes speed, not results.

| | |
|---|---|
| default | none (single pass) |
| helper | `run_feedback_loop(sdfg, measure, apply_move)` |
| code | `nestforge/phases/feedback.py` |
| open | what exactly (e) and (g) report, and when a round should stop |
