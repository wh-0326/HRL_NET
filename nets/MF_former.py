import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple, Optional, List, Tuple

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return tuple.__getitem__(self, key)


def _get_attention_node_data(fixed):
    return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# ==============================================================================
# 步骤 1: 创建支持 KV Caching 的自定义 Transformer 解码器层
# ==============================================================================

class CachedTransformerDecoderLayer(nn.Module):
    """
    一个支持键值缓存的 Transformer 解码器层。
    它的 forward 方法接受并返回一个 past_key_value 元组。
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(CachedTransformerDecoderLayer, self).__init__()
        # 使用 batch_first=False 因为我们处理的序列形状是 (S, N, E)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 标准化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            tgt (Tensor): 当前时间步的输入, shape `(1, N, E)`
            memory (Tensor): 编码器的输出, shape `(S, N, E)`
            past_key_value (Tuple): 上一时间步的 (key, value) 缓存, shape `(L-1, N, E)`

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: 输出和更新后的 (key, value) 缓存, shape `(L, N, E)`
        """
        # --- 自注意力模块 (带缓存) ---
        query = tgt

        if past_key_value is not None:
            # past_key_value[0] is key, past_key_value[1] is value
            # 将当前 key/value 与缓存拼接
            key = torch.cat([past_key_value[0], tgt], dim=0)
            value = torch.cat([past_key_value[1], tgt], dim=0)
        else:  # 第一步，没有缓存
            key = value = tgt

        current_key_value = (key, value)

        # 自注意力。Query 是当前步，Key/Value 是历史所有步
        # 由于 Query 长度为1，它只能关注历史，不需要掩码
        tgt2, _ = self.self_attn(query, key, value)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # --- 交叉注意力模块 ---
        # Encoder-Decoder attention (对 memory 的 attention)
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # --- 前馈网络 ---
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, current_key_value


# ==============================================================================
# 步骤 2: 创建支持 KV Caching 的自定义 Transformer 解码器
# ==============================================================================

class CachedTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(CachedTransformerDecoder, self).__init__()
        # 创建多个解码器层
        self.layers = nn.ModuleList([
            CachedTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        output = tgt
        new_kv_cache = []

        if kv_cache is None:
            kv_cache = [None] * self.num_layers

        for i, layer in enumerate(self.layers):
            output, new_cache = layer(output, memory, past_key_value=kv_cache[i])
            new_kv_cache.append(new_cache)

        return output, new_kv_cache


# ==============================================================================
# 步骤 3: 将所有模块整合到最终的 AttentionModel
# ==============================================================================

class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 n_decode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 dy=False):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.is_MCLP = problem.NAME == 'MCLP'
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        node_dim = 5
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        self.pos_encoder = PositionalEncoding(embedding_dim)

        # 使用我们自定义的、支持缓存的解码器
        self.transformer_decoder = CachedTransformerDecoder(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            num_layers=n_decode_layers
        )

        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.first_placeholder = nn.Parameter(torch.Tensor(embedding_dim))
        self.first_placeholder.data.uniform_(-1, 1)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        if self.checkpoint_encoder and self.training:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost = -self.problem.get_total_num(input, pi)
        ll = _log_p.sum(dim=-1)

        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _init_embed(self, input):
        return self.init_embed(input['facilities'])

    def _precompute(self, embeddings):
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed),
            self._make_heads(glimpse_val_fixed),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _inner(self, input, embeddings):
        outputs = []
        sequences = []
        batch_size = embeddings.size(0)

        state = self.problem.make_state(input)
        fixed = self._precompute(embeddings)
        memory = fixed.node_embeddings.permute(1, 0, 2)

        # --- 高效解码循环 ---
        kv_cache = None
        decoder_input = self.first_placeholder.view(1, 1, -1).expand(1, batch_size, -1)

        for t in range(state.p):
            # 对当前步(长度为1)的输入添加位置编码
            decoder_input_with_pos = decoder_input + self.pos_encoder.pe[t:t + 1]

            # 调用解码器，传入并接收更新后的缓存
            decoder_output, kv_cache = self.transformer_decoder(
                tgt=decoder_input_with_pos,
                memory=memory,
                kv_cache=kv_cache
            )

            # 使用解码器输出作为 Query 进行 Pointer
            query = decoder_output.transpose(0, 1)

            glimpse_K, glimpse_V, logit_K = _get_attention_node_data(fixed)
            mask = state.get_mask()
            log_p, _ = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
            logp_selected, selected_idx = self._select_node(log_p.exp().squeeze(1), mask.squeeze(1))

            state = state.update(selected_idx)

            # 准备下一步的输入: 仅使用新选出节点的嵌入
            decoder_input = fixed.node_embeddings.gather(
                1,
                selected_idx.view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
            ).permute(1, 0, 2)

            outputs.append(logp_selected)
            sequences.append(selected_idx)

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), "Decode greedy: infeasible action"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"

        logp = probs.gather(1, selected.unsqueeze(-1)).squeeze(-1).log()
        # 使用clamp防止log(0)导致-inf，继而引发NaN梯度
        logp = torch.clamp(logp, min=-100)
        return logp, selected

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        return sample_many(
            lambda input: self._inner(*input),
            lambda input, pi: -self.problem.get_total_num(input[0], pi),
            (input, self.embedder(self._init_embed(input))[0]),
            batch_rep, iter_rep
        )

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )