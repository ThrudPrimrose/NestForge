# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whole-symbol veclib matching (:func:`nestforge.toolchain.serves_op`) vs. the substring test it replaced.

The old ``candidate in nm_output`` test credited a library for elementals it does not serve whenever one
symbol name is a PREFIX of another (``tanh`` starts with ``tan``, ``log10``/``log1p``/``log2`` all start
with ``log``, ``exp2``/``exp10``/``expm1`` all start with ``exp``). Every case below is a REAL glibc/SLEEF/
SVML symbol pair, not a hypothetical.
"""
import shutil
import subprocess

import pytest

from nestforge.build.toolchain import VECTOR_LIBS, nm_symbol_names, packed_ops_called, serves_op, veclib_library_path

GCC_MISSING = shutil.which("gcc") is None


@pytest.mark.parametrize("names,veclib,op", [
    (frozenset({"_ZGVdN4v_tanh"}), "libmvec", "tan"),
    (frozenset({"_ZGVdN4v_atan_u35"}), "sleef", "atan"),
    (frozenset({"_ZGVdN4v_log10"}), "libmvec", "log"),
    (frozenset({"_ZGVdN4v_exp2"}), "libmvec", "exp"),
    (frozenset({"__svml_sincos2"}), "svml", "sin"),
])
def test_serves_op_rejects_a_prefix_match(names, veclib, op):
    """``tanh``/``atan_u35``/``log10``/``exp2``/``sincos2`` all start with the probed op's stem; a
    substring test credited each of these to a library that does not export the op at all."""
    assert serves_op(names, veclib, op) is False


@pytest.mark.parametrize("suffix", ["", "2", "4", "8"])
def test_serves_op_accepts_the_real_symbol_and_every_svml_lane_count(suffix):
    """The op itself, and SVML's lane-count suffix (digits only), must still match -- the fix must not
    overcorrect into rejecting genuine exports."""
    assert serves_op(frozenset({"_ZGVdN4v_sin"}), "libmvec", "sin") is True
    assert serves_op(frozenset({f"__svml_sin{suffix}"}), "svml", "sin") is True


def test_serves_op_matches_the_binary_op_double_v_mangling():
    """Two-argument elementals mangle the extra operand as an extra ``v`` (``_ZGVdN4vv_pow``, not
    ``_ZGVdN4v_pow``); missing the double-v would read a real pow call as unserved."""
    assert serves_op(frozenset({"_ZGVdN4vv_pow"}), "libmvec", "pow") is True


def test_nm_symbol_names_strips_the_glibc_version_suffix():
    """Real ``nm -D --defined-only`` on this box's libmvec prints ``_ZGVdN4v_sin@@GLIBC_2.22``; a
    whole-name match against the unstripped string would never equal the bare candidate ``_ZGVdN4v_sin``."""
    path = veclib_library_path(VECTOR_LIBS["libmvec"], "gcc")
    assert path is not None, "libmvec not found by gcc (glibc libmvec ships in libc6)"
    assert "_ZGVdN4v_sin" in nm_symbol_names(path, dynamic_only=True)


_TANH_LOG10_EXP2 = ("#include <math.h>\n"
                    "void k(double *restrict a, const double *restrict b, int n) {\n"
                    "  for (int i = 0; i < n; ++i) a[i] = tanh(b[i]) + log10(b[i]) + exp2(b[i]);\n"
                    "}\n")
_SIN = ("#include <math.h>\n"
        "void k(double *restrict a, const double *restrict b, int n) {\n"
        "  for (int i = 0; i < n; ++i) a[i] = sin(b[i]);\n"
        "}\n")


def compile_to_object(tmp_path, stem: str, source: str):
    # the ".c" suffix is load-bearing: an extensionless path reads to gcc as a LINKER input, so "-c"
    # silently no-ops (exit 0, no object written) instead of compiling anything.
    src = tmp_path / f"{stem}.c"
    src.write_text(source)
    obj = tmp_path / f"{stem}.o"
    done = subprocess.run(["gcc", "-O3", "-march=native", "-ffast-math", "-c",
                           str(src), "-o", str(obj)],
                          capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return obj


def test_packed_ops_called_no_longer_credits_tan_log_exp_to_a_tanh_log10_exp2_caller(tmp_path):
    """The exact regression: this kernel calls tanh, log10 and exp2 -- none of ``VECLIB_PROBE_OPS``
    (sin/cos/pow/log/exp/tan/atan) -- but the old substring test read it as calling tan+log+exp because
    each probed stem prefixes the real symbol gcc emits."""
    assert not GCC_MISSING, "gcc not on PATH (setup_apt.sh installs it)"
    obj = compile_to_object(tmp_path, "tle", _TANH_LOG10_EXP2)
    assert packed_ops_called("libmvec", str(obj)) == ()


def test_packed_ops_called_still_finds_a_real_sin_call(tmp_path):
    """Positive control for the test above: without it, a broken probe that always returns empty would
    also pass the negative case for the wrong reason."""
    assert not GCC_MISSING, "gcc not on PATH (setup_apt.sh installs it)"
    obj = compile_to_object(tmp_path, "s", _SIN)
    assert packed_ops_called("libmvec", str(obj)) == ("sin", )
