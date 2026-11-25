<!--
Long-WAM file attribution; existing upstream notices are retained below.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Provenance: Original Long-WAM implementation.
License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
End Long-WAM attribution.
-->

# Bundled video runtime notices

The repository includes video runtime source under this directory. The relevant
copyright and attribution statements are in the redistribution notice below, the individual
source headers and adjacent license sidecars. The Apache-2.0 text is included in
[LICENSE](../../LICENSE). The files below retain their component-specific terms.

| Component | Included material | License |
| --- | --- | --- |
| [LongLive 2.0](https://github.com/NVlabs/LongLive) | `engine/`; robot additions and wrappers come from our author branch | [Apache-2.0](../../LICENSE), except nested components below |
| [Wan](https://github.com/Wan-Video/Wan2.2) | Wan model, VAE, text encoder and utilities included through LongLive 2.0 | [Apache-2.0](../../LICENSE); Alibaba notices retained |
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | LongLive training, model and pipeline components | [Apache-2.0](../../LICENSE) |
| [Stable Video Infinity](https://github.com/vita-epfl/Stable-Video-Infinity) | LongLive error buffer | [Apache-2.0](../../LICENSE) |
| [FramePack](https://github.com/lllyasviel/FramePack) | LongLive memory helpers | [Apache-2.0](../../LICENSE) |
| [Diffusers](https://github.com/huggingface/diffusers) | Wan solver implementations | [Apache-2.0](../../LICENSE) |
| [Transformers](https://github.com/huggingface/transformers) | Wan T5 implementation | [Apache-2.0](../../LICENSE) |
| [qwen-vl-utils](https://github.com/kq-chen/qwen-vl-utils) | Wan vision-language utilities | [Apache-2.0](../../LICENSE) |
| [FourOverSix](https://github.com/mit-han-lab/fouroversix) | Optional stage-1 FP4 implementation | [MIT](../../train/engine/fouroversix/LICENSE.md); Jack Cook and original per-file notices retained |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Adapted FourOverSix CUDA headers and build helpers | [BSD-3-Clause](FlashAttention-BSD-3-Clause.txt); Tri Dao |
| [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) | FourOverSix Hadamard transform | [BSD-3-Clause](fast-hadamard-transform-BSD-3-Clause.txt); Tri Dao |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | FourOverSix FP4 GEMM examples and CUDA copy helper | [BSD-3-Clause](CUTLASS-BSD-3-Clause.txt) |
| [DALI](https://github.com/NVIDIA/DALI) | FourOverSix static-switch helper | [Apache-2.0](../../LICENSE) |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | FourOverSix quantization utility | [Apache-2.0](../../LICENSE) |
| [wheel](https://github.com/pypa/wheel) | Archive handling snippet in FourOverSix build setup | [MIT](wheel-MIT.txt) |

The robot-video inference adapter and configurations include code adapted from
the LongLive 2.0 robot-video demo and author integration snapshot. Source
identifiers and modification notices are recorded in the corresponding files.
The bundled FourOverSix source includes its own license and preserves the
notices on incorporated portions.

The attribution for the engine dependency list is in [its sidecar](train-engine-requirements.txt.license).

## Redistribution notice

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Provenance: Original Long-WAM release documentation.

Long-WAM

Original Long-WAM code and NVIDIA-authored modifications are distributed under
the Apache License, Version 2.0. Copyright statements for NVIDIA modifications
do not claim ownership of underlying third-party code.

Third-party files and incorporated portions retain their original licenses and
copyright notices. See config/licenses/THIRD_PARTY_NOTICES.md and per-file headers (or adjacent
.license files for formats that cannot safely contain comments). Where a file
combines Apache-2.0 changes with MIT/BSD code, the SPDX AND expression records
the combined obligations; it is not an alternative-license choice.

This bundled video runtime includes work from LongLive 2.0, Wan, Self-Forcing,
FourOverSix, FlashAttention, fast-hadamard-transform, CUTLASS, Transformer
Engine, DALI, FramePack, Stable Video Infinity, Diffusers, Transformers,
qwen-vl-utils and wheel. Original notices are retained.

This code license does not grant rights to external datasets, model weights,
simulator assets, trademarks or hosted services. Obtain them under their own
terms. This notice does not modify any third-party license.
