# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VisualGen TRTLLM backend for shared SkipSoftmax attention."""

from .....attention_backend.sparse.skip_softmax import SkipSoftmaxParams
from ...trtllm import TrtllmAttention


class SkipSoftmaxTrtllmAttention(TrtllmAttention):
    """Bind VisualGen metadata handling to the core TRTLLM SkipSoftmax path."""

    def __init__(self, *, sparse_params: SkipSoftmaxParams | None = None, **kwargs) -> None:
        if not isinstance(sparse_params, SkipSoftmaxParams):
            raise TypeError("SkipSoftmaxTrtllmAttention requires SkipSoftmaxParams")
        super().__init__(sparse_params=sparse_params, **kwargs)


__all__ = ["SkipSoftmaxTrtllmAttention"]
