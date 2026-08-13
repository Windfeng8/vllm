"""Pure-text compatibility adapter for the Ascend Qwen3.5 worker patch.

The target vLLM-Ascend patch routes Qwen3.5 full-attention layers through an
M-RoPE fused kernel.  This adapter exposes text-only 3D positions so the
dummy-weight validation can exercise that path without a vision encoder.
"""

import torch

from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM
from vllm.model_executor.models.interfaces import SupportsMRoPE


class Qwen35MoeTextCompat(Qwen3_5MoeForCausalLM, SupportsMRoPE):
    """Qwen3.5-MoE text model with text-only M-RoPE positions."""

    def get_mrope_input_positions(self, input_tokens, mm_features):
        del mm_features
        seq_len = len(input_tokens)
        text_positions = torch.arange(seq_len, dtype=torch.long)
        positions = text_positions.unsqueeze(0).repeat(3, 1)
        return positions, 0
