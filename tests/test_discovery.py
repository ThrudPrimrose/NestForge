# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compiler + BLAS backend discovery return well-formed link flags."""
from nestforge.build.arena import BlasBackend, discover_blas_libraries, discover_compilers


def test_discover_compilers_returns_existing_paths():
    import os
    compilers = discover_compilers()
    assert compilers, "expected at least one of gcc/clang on PATH"
    for name, path in compilers.items():
        assert os.path.exists(path)


def test_discover_blas_backends_are_link_flags():
    backends = discover_blas_libraries()
    assert isinstance(backends, dict)
    for name, backend in backends.items():
        assert isinstance(backend, BlasBackend)
        assert backend.link_flags and all(f.startswith(("-l", "-L")) for f in backend.link_flags)
        assert any(f.startswith("-l") for f in backend.link_flags)
