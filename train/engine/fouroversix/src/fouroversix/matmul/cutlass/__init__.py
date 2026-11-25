# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/matmul/cutlass/__init__.py
# Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/matmul/cutlass/__init__.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.

from .backend import CUTLASSMatmulBackend

__all__ = ["CUTLASSMatmulBackend"]
