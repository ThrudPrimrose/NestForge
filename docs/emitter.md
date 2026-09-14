# Emitter

[Overview](../README.md) · related: [1 Shape Kernels](phases/1-shape-kernels.md),
[4 Optimize Kernels](phases/4-optimize-kernels.md)

The emitter turns an extracted SDFG into NumPy source, which serves as the kernel's correctness oracle
and as the translator's input. In `nestforge/ir/`, `emit_numpy.py` handles control flow, copies and
nested SDFGs, `emit_libnode.py` handles library nodes (BLAS, reduce, FFT), and `emit_yaml.py` writes
the argument manifest.

## Contract

- **Caller allocates.** Inputs, outputs, `__return` and scratch transients are buffer parameters
  written in place; only true scalars are locals.
- **Sizable buffers.** Every buffer shape is a static function of size symbols. A shape that reads
  array data, such as a CSR span, is refused (`reject_unsizable_scratch`).
- **Read-only.** Emission works on a deep copy and never mutates the caller's SDFG.
- **Exact.** Bit-exact against NumPy wherever floating-point associativity allows.
- **Signature parity.** `emit_yaml.array_names` and the NumPy signature list the same arguments in the
  same order.

## Invariants

- `access` renders a scalar local bare and anything else as `name[index]`. `copy_side` and
  `reshape_side` squeeze length-1 axes for same-rank copies and keep the reshaping subset explicit
  otherwise.
- `copy_direction` resolves the source of an in-place copy (`A[i] = A[j]`) the way DaCe does, by
  testing `data` against the edge's source.
- `emit_nested_sdfg` aliases connectors to outer buffers and reconciles their descriptors
  (`reconcile_connector_descriptor`), so an offset multi-dimensional connector keeps its rank.
- `range_stop` treats DaCe range ends as inclusive in both directions.
- `symbol_mapping_lines` binds through temporaries, so a swap such as `{i: j, j: i}` is safe.
- `emit_conditional` keeps branch order and refuses a non-final unconditional branch, as DaCe codegen
  does.

Changes to `emit_region`, `state_body`, `map_lines`, `emit_loop`, `LIBNODE_EMITTERS` or
`normalize_casts` reach every emitted kernel.
