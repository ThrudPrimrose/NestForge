# Emitter

[../README.md](../README.md) · related: [1 Inter-kernel schedule](phases/1-inter-kernel.md) ·
[3 Kernel optimization](phases/3-kernel-opt.md)

The emitter turns an extracted SDFG into NumPy source: `nestforge/ir/emit_numpy.py` (control flow,
copies, nested SDFGs), `nestforge/ir/emit_libnode.py` (BLAS/reduce/FFT library nodes), and
`nestforge/ir/emit_yaml.py` (the argument manifest). Phase 3 renders each kernel this way before it
optimizes or hands the kernel to an agent.

## Contract

- **C-style allocation.** The kernel allocates nothing. Inputs, outputs, `__return` and scratch
  transients are caller-pre-allocated buffer parameters written in place. Only true scalars are
  Python locals.
- **Sizable buffers only.** Every buffer shape must be a static function of the kernel's
  size-symbols. A shape that reads array data (a CSR span) is refused rather than emitted:
  `sizable` (`emit_numpy.py`) walks the expression tree for a `Subscript`/`Indexed` head or any
  atom named in `sdfg.arrays`, and `reject_unsizable_scratch` raises on the first dimension that
  fails it.
- **Read-only emission.** Emission never mutates the caller's SDFG. Widening, inlining and
  `replace` run on a deep copy.
- **Semantics-preserving.** Bit-exact vs NumPy wherever floating-point associativity allows.
- **Signature/manifest parity.** `emit_yaml.array_names` and the NumPy signature are the same
  positional list, or a native call passes mismatched pointers.

## Invariants

- **Access rendering.** A `(name, subset)` becomes a Python string by one rule:
  `scalar_local -> bare name`, otherwise `name[index_str(subset)]` (`access`). `copy_side` and
  `reshape_side` render the two sides of a data copy with their own squeeze policy: same-rank
  copies squeeze length-1 axes so a `(N,1)` buffer and a `(1,N)` view meet at the same shape,
  rank-changing copies keep the reshaping side's subset explicit instead.
- **Copy direction.** A memlet's `subset` indexes its `data` field, a DaCe invariant; on an
  in-place copy (`A[i] = A[j]`) both endpoints share a name, so `copy_direction` resolves the
  source by testing `data` against the edge's source first, matching how DaCe itself breaks the
  tie.
- **Nested SDFG inlining.** `emit_nested_sdfg` aliases each connector to the outer buffer it binds
  and reconciles the two descriptors (`reconcile_connector_descriptor`) rather than overwriting the
  inner shape outright, so an offset multi-dim connector cannot silently collapse to a shorter
  index.
- **Range direction.** `range_stop` picks `end + 1` for an ascending map/loop and `end - 1` for a
  descending one, since DaCe's range end is inclusive in both directions.
- **Symbol substitution.** `symbol_mapping_lines` binds a nested SDFG's symbol mapping through
  temporaries whenever a target also appears on some right-hand side, so a swap like `{i: j, j: i}`
  does not clobber.
- **Conditional order.** `emit_conditional` emits branches in stored order and refuses a
  non-final unconditional branch, matching DaCe codegen's own rule that the first matching branch
  wins.

`emit_region`, `state_body`, `map_lines`, `emit_loop`, the `LIBNODE_EMITTERS` registry and
`normalize_casts` are the stable core; changes there ripple through every emitted kernel.
