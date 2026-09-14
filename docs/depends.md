# Kernel Dependencies

[../README.md](../README.md) · code: `nestforge/ir/depends.py`

`kernel_dependencies(sdfg)` answers one question for every `ExternalCall` argument: which producers
can reach it. It reads the lowered SDFG and never changes it.

```
extcall_1: T <- extcall_0.T, N <- program
extcall_0: A <- extcall_1.A [carried: for_39] | program, N <- program
exit: A <- extcall_1.A | program
```

## Producers

- `program`: a non-transient container or free symbol not written before the read.
- `extcall_i.arg`: an `ExternalCall` output. The argument comes from the connector (`_out_<arg>`),
  not the data name, so phase 3's `A_gpu` renames do not show.
- `host:<state>`: any other writer (tasklet, nested SDFG, other library node).

A whole `AccessNode -> AccessNode` copy forwards its source's producers, so offload copies stay
invisible.

## Rules

- Whole containers only. A read reads all of it; a write replaces all of it.
- `if` without `else`: branch results plus the incoming reach. With `else`: branch results only.
- Loop: fixpoint. A reach that crossed the back edge is tagged `[carried: <loop>]`. After the loop
  the incoming reach stays unless `loop_provably_at_least_one_iteration` proves one trip.
- `break` joins the loop exit, `continue` joins the back edge, `return` joins the SDFG exit.
- Interstate `k = v`: `k` takes the producers of every name `v` reads, and `via` records the text.
- Kernel symbols come from the manifest (`input_args` minus `array_args`).
- Refused with `UnsupportedProgram`: `Reference` containers, an `ExternalCall` inside a nested
  SDFG, an `ExternalCall` without a manifest.

## Output

`KernelGraph.lines()` prints one line per kernel in program order, then `exit:`. `to_json()` is
sorted and byte-stable across runs. `consumers_of(kernel)` lists the argument edges a kernel feeds.
