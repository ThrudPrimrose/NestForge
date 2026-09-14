# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""BLAS backend discovery returns well-formed link flags."""
from nestforge.build.arena import BlasBackend, discover_blas_libraries


def test_discover_blas_backends_are_link_flags():
    backends = discover_blas_libraries()
    assert isinstance(backends, dict)
    for name, backend in backends.items():
        assert isinstance(backend, BlasBackend)
        assert backend.link_flags and all(f.startswith(("-l", "-L")) for f in backend.link_flags)
        assert any(f.startswith("-l") for f in backend.link_flags)
