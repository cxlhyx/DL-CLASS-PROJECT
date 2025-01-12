# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py
import math
import os
import sys

from enhence import enhance_score

sys.path.append(os.path.split(sys.path[0])[0])

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat


def generate_relative_position_encoding(max_length, embedding_dim):

    # 创建一个零矩阵来存储相对位置编码
    rel_pos_encoding = torch.zeros((max_length, max_length, embedding_dim))

    # 计算相对位置编码
    for i in range(max_length):
        for j in range(max_length):
            # 使用正弦和余弦函数生成编码
            rel_pos_encoding[i, j] = torch.tensor([math.sin(i * (2 ** (2 * k // embedding_dim)))for k in range(embedding_dim // 2)] +
            [math.cos(j * (2 ** (2 * k // embedding_dim)))for k in range(embedding_dim // 2, embedding_dim)])

    return rel_pos_encoding

def generate_relative_positions_matrix(length, max_relative_position):
    """Generate matrix of relative positions between inputs."""
    range_vec = torch.arange(length)
    range_mat = range_vec.repeat(length, 1)
    distance_mat = range_mat - range_mat.t()
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat

def relative_position_embeddings(length, depth, max_relative_position):
    """Generate relative position embedding."""
    relative_positions_matrix = generate_relative_positions_matrix(
        length, max_relative_position)
    vocab_size = max_relative_position * 2 + 1
    embeddings_table = torch.zeros([vocab_size, depth])
    for pos in range(vocab_size):
        for i in range(depth // 2):
            embeddings_table[pos, 2 * i] = torch.sin(pos / torch.pow(10000, 2 * i / depth))
            embeddings_table[pos, 2 * i + 1] = torch.cos(pos / torch.pow(10000, 2 * i / depth))
    embeddings_table = embeddings_table.to(torch.float32)
    flat_relative_positions_matrix = relative_positions_matrix.view(-1)
    one_hot_relative_positions_matrix = F.one_hot(flat_relative_positions_matrix, num_classes=vocab_size).float()
    embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
    embeddings = embeddings.view([length, length, depth])
    return embeddings


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_first_frame: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_first_frame=use_first_frame,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, exemplar_latent=None, timestep=None, reshape_exemplar=False, return_dict: bool = True, exemplar_encoder_hidden_states=None):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        if isinstance(encoder_hidden_states, dict):
            pass
        else:
            encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)
        if exemplar_encoder_hidden_states is not None:
            exemplar_encoder_hidden_states = repeat(exemplar_encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        if reshape_exemplar:
            exemplar_latent = rearrange(exemplar_latent, "b c f h w -> (b f) c h w")
            batch_exp, channel_exp, height_exp, weight_exp = exemplar_latent.shape
            residual_exemplar_latent = exemplar_latent

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if reshape_exemplar:
            exemplar_latent = self.norm(exemplar_latent)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)
        if reshape_exemplar:
            if not self.use_linear_projection:
                exemplar_latent = self.proj_in(exemplar_latent)
                inner_dim = exemplar_latent.shape[1]
                exemplar_latent = exemplar_latent.permute(0, 2, 3, 1).reshape(batch_exp, height_exp * weight_exp, inner_dim)
            else:
                inner_dim = exemplar_latent.shape[1]
                exemplar_latent = exemplar_latent.permute(0, 2, 3, 1).reshape(batch_exp, height_exp * weight_exp, inner_dim)
                exemplar_latent = self.proj_in(exemplar_latent)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                exemplar_latent=exemplar_latent
            )
            if reshape_exemplar:
                exemplar_latent = block(
                    exemplar_latent,
                    encoder_hidden_states=encoder_hidden_states if exemplar_encoder_hidden_states is None else exemplar_encoder_hidden_states,
                    timestep=timestep,
                    video_length=video_length,
                    exemplar_latent=None
                )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_first_frame: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        # print(only_cross_attention)
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.use_first_frame = use_first_frame

        # SC-Attn
        if use_first_frame:
            self.attn1 = SparseCausalAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
            # print(cross_attention_dim)
        else:
            # self.attn1 = CrossAttention(
            #     query_dim=dim,
            #     heads=num_attention_heads,
            #     dim_head=attention_head_dim,
            #     dropout=dropout,
            #     bias=attention_bias,
            #     cross_attention_dim=None,
            #     upcast_attention=upcast_attention,
            # )
            self.attn1 = EnhenceCrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=None,
                upcast_attention=upcast_attention,
            )

        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = EnhenceCrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None, exemplar_latent=None, exemplar_timestep=None):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if (exemplar_latent is not None):
            exemplar_latent = (
                self.norm1(exemplar_latent, exemplar_timestep) if self.use_ada_layer_norm else self.norm1(exemplar_latent)
                )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            if self.use_first_frame:
                hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length, exemplar_latent=exemplar_latent) + hidden_states
            else:
                hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )
        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class EnhenceCrossAttention(CrossAttention):
    def _get_enhance_scores(
            self,
            # attn: CrossAttention,
            query: torch.Tensor,
            key: torch.Tensor,
            head_dim: int,
            text_seq_length: int,
    ) -> torch.Tensor:
        self.num_frames = 15
        spatial_dim = int((query.shape[2] - text_seq_length) / self.num_frames)

        query_image = rearrange(
            query[:, :, text_seq_length:],
            "B N (T S) C -> (B S) N T C",
            N=self.heads,
            T=self.num_frames,
            S=spatial_dim,
            C=head_dim,
        )
        key_image = rearrange(
            key[:, :, text_seq_length:],
            "B N (T S) C -> (B S) N T C",
            N=self.heads,
            T=self.num_frames,
            S=spatial_dim,
            C=head_dim,
        )
        return enhance_score(query_image, key_image, head_dim, self.num_frames)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, exemplar_latent=None) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Apply RoPE if needed
        # if image_rotary_emb is not None:
        #     from diffusers.models.embeddings import apply_rotary_emb
        #
        #     query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
        #     if not attn.is_cross_attention:
        #         key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # ========== Enhance-A-Video ==========

        enhance_scores = self._get_enhance_scores(query, key, head_dim, text_seq_length)
        # ========== Enhance-A-Video ==========

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        # ========== Enhance-A-Video ==========

        hidden_states = hidden_states * enhance_scores
        # ========== Enhance-A-Video ==========

        return hidden_states


class SparseCausalAttention(CrossAttention):
    def _attention_with_prob(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        rel_pos_encoding = generate_relative_position_encoding(10, self.dim_head)
        query += rel_pos_encoding
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states, attention_scores

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, exemplar_latent=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class SparseCausalAttention(CrossAttention):
    # def __init__(self, query_dim, heads, dim_head, dropout, bias, cross_attention_dim=None, upcast_attention=False):
    #     super().__init__()
    #     self.query_dim = query_dim
    #     self.heads = heads
    #     self.dim_head = dim_head
    #     self.scale = dim_head ** -0.5
    #     self.to_q = nn.Linear(query_dim, heads * dim_head, bias=bias)
    #     self.to_k = nn.Linear(cross_attention_dim or query_dim, heads * dim_head, bias=bias)
    #     self.to_v = nn.Linear(cross_attention_dim or query_dim, heads * dim_head, bias=bias)
    #     self.to_out = nn.Sequential(
    #         nn.Linear(heads * dim_head, query_dim),
    #         nn.Dropout(dropout)
    #     )
    #     self.upcast_attention = upcast_attention
    #     self.group_norm = nn.GroupNorm(num_groups=32, num_channels=query_dim, eps=1e-6, affine=True)

#     def reshape_heads_to_batch_dim(self, tensor):
#         batch_size, seq_len, _ = tensor.shape
#         tensor = tensor.view(batch_size, seq_len, self.heads, self.dim_head)
#         tensor = tensor.permute(0, 2, 1, 3).contiguous()
#         tensor = tensor.view(batch_size * self.heads, seq_len, self.dim_head)
#         return tensor

#     def reshape_batch_dim_to_heads(self, tensor):
#         batch_size, seq_len, _ = tensor.shape
#         tensor = tensor.view(batch_size // self.heads, self.heads, seq_len, self.dim_head)
#         tensor = tensor.permute(0, 2, 1, 3).contiguous()
#         tensor = tensor.view(batch_size // self.heads, seq_len, self.heads * self.dim_head)
#         return tensor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, exemplar_latent=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = self.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # Generate relative position embeddings
        rel_pos = relative_position_embeddings(sequence_length, self.dim_head, 4)
        rel_pos = rel_pos.to(query.device)

        # Add relative position embeddings to query
        query = query + rel_pos

        # Compute attention scores
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Compute attention probabilities
        attention_probs = attention_scores.softmax(dim=-1)

        # Compute hidden states
        hidden_states = torch.bmm(attention_probs, value)

        # Reshape hidden states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # Linear projection and dropout
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states