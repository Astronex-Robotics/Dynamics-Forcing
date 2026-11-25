# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/weight_conversions/__init__.py
# Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/weight_conversions/__init__.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.

import warnings

try:
    from .conversions import WeightConversions
    from .gpt_oss import FourOverSixGptOssDeserialize, GptOssWeightConverter
except ImportError:
    warnings.warn("Install transformers>=5.0 to use weight conversions", stacklevel=2)

    WeightConversions = None
    FourOverSixGptOssDeserialize = None
    GptOssWeightConverter = None

__all__ = ["FourOverSixGptOssDeserialize", "GptOssWeightConverter", "WeightConversions"]
