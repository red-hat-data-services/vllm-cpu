# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM model for EncoderClassifier (nn.TransformerEncoder-based classifier).

Loads weights from the HF checkpoint (nn.TransformerEncoder format) and
splits the combined in_proj_weight into QKVParallelLinear shards.
"""

import math
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.encoder_only_attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import (SupportsCrossEncoding,
                                                   SupportsQuant)
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learned parameters)."""

    def __init__(self,
                 hidden_size: int,
                 dropout: float = 0.0,
                 max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() *
            (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, hidden_size]

    def forward(self, x: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[positions])


class EncoderClassifierSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_attention_heads,
            total_num_kv_heads=num_attention_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        q_size = self.num_heads * self.head_dim
        self.q_size = q_size
        self.kv_size = q_size

        self.attn = EncoderOnlyAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class EncoderClassifierLayer(nn.Module):
    """Mirrors nn.TransformerEncoderLayer (pre-norm=False)."""

    def __init__(self,
                 config,
                 cache_config=None,
                 quant_config=None,
                 prefix=""):
        super().__init__()
        self.self_attn = EncoderClassifierSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.linear1 = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear1",
        )
        self.linear2 = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear2",
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(hidden_states)
        hidden_states = self.norm1(hidden_states + attn_output)
        ffn, _ = self.linear1(hidden_states)
        ffn = nn.functional.relu(ffn)
        ffn, _ = self.linear2(ffn)
        hidden_states = self.norm2(hidden_states + ffn)
        return hidden_states


class EncoderClassifierEncoder(nn.Module):

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layers = nn.ModuleList([
            EncoderClassifierLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            ) for i in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ClassifierHead(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int, dtype=None):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, num_labels, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dense.weight.dtype)
        x = self.dense(x)
        x = x * torch.sigmoid(x)  # swish
        return self.out_proj(x)


@default_pooling_type(seq_pooling_type="MEAN")
class EncoderClassifierForSequenceClassification(nn.Module,
                                                 SupportsCrossEncoding,
                                                 SupportsQuant):

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.embedding = VocabParallelEmbedding(config.vocab_size,
                                                config.hidden_size)
        self.pos_encoder = PositionalEncoding(
            config.hidden_size,
            getattr(config, "pos_dropout", 0.0),
            config.max_position_embeddings,
        )
        self.encoder = EncoderClassifierEncoder(config,
                                                vllm_config,
                                                prefix=f"{prefix}.encoder")
        self.classifier = ClassifierHead(
            config.hidden_size,
            config.num_labels,
            dtype=vllm_config.model_config.dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            classifier=self.classifier,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        hidden_states = self.pos_encoder(inputs_embeds, positions)
        return self.encoder(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        other_weights: list[tuple[str, torch.Tensor]] = []

        for name, loaded_weight in weights:
            if "pos_encoder.pe" in name:
                continue

            if ".self_attn.in_proj_weight" in name:
                qkv_name = name.replace("in_proj_weight", "qkv_proj.weight")
                if qkv_name in params_dict:
                    param = params_dict[qkv_name]
                    q, k, v = loaded_weight.chunk(3, dim=0)
                    param.weight_loader(param, q, "q")
                    param.weight_loader(param, k, "k")
                    param.weight_loader(param, v, "v")
                    loaded_params.add(qkv_name)
                continue
            if ".self_attn.in_proj_bias" in name:
                qkv_name = name.replace("in_proj_bias", "qkv_proj.bias")
                if qkv_name in params_dict:
                    param = params_dict[qkv_name]
                    q, k, v = loaded_weight.chunk(3, dim=0)
                    param.weight_loader(param, q, "q")
                    param.weight_loader(param, k, "k")
                    param.weight_loader(param, v, "v")
                    loaded_params.add(qkv_name)
                continue

            other_weights.append((name, loaded_weight))

        loader = AutoWeightsLoader(self)
        loaded_params.update(loader.load_weights(other_weights))
        return loaded_params
