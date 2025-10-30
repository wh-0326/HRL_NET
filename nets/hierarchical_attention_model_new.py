import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Optional, List, Tuple, NamedTuple
from nets.graph_encoder import GraphAttentionEncoder
from utils.functions import sample_many


class AttentionModelFixed(NamedTuple):
    """固定的注意力上下文，支持高效索引"""
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
                glimpse_key=self.glimpse_key[:, key],
                glimpse_val=self.glimpse_val[:, key],
                logit_key=self.logit_key[key]
            )
        return tuple.__getitem__(self, key)


class CachedTransformerDecoderLayer(nn.Module):
    """支持KV缓存的Transformer解码器层"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        前向传播，支持KV缓存
        Returns: (output, updated_kv_cache)
        """
        query = tgt

        if past_key_value is not None:
            # 拼接历史KV
            key = torch.cat([past_key_value[0], tgt], dim=1)
            value = torch.cat([past_key_value[1], tgt], dim=1)
        else:
            key = value = tgt

        current_key_value = (key, value)

        # 自注意力
        tgt2, _ = self.self_attn(query, key, value)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 交叉注意力
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, current_key_value


class CachedTransformerDecoder(nn.Module):
    """支持KV缓存的Transformer解码器"""

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CachedTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        output = tgt
        new_kv_cache = []

        if kv_cache is None:
            kv_cache = [None] * self.num_layers

        for i, layer in enumerate(self.layers):
            output, new_cache = layer(output, memory, past_key_value=kv_cache[i])
            new_kv_cache.append(new_cache)

        return output, new_kv_cache


class ImprovedHierarchicalAttentionModel(nn.Module):
    """
    改进的分层强化学习模型
    - 集成KV缓存加速
    - 改进的上下层协调机制
    - 支持Actor-Critic架构
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 n_decode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='instance',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 max_facilities_per_step=5,
                 use_critic=False,
                 dy=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.n_decode_layers = n_decode_layers
        self.decode_type = None
        self.temp = 1.0
        self.max_facilities_per_step = max_facilities_per_step
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.problem = problem
        self.n_heads = n_heads
        self.use_critic = use_critic

        # 节点嵌入
        node_dim = 2  # MCLP facilities只有(x, y)坐标
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        # 共享编码器
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # ===== 改进的上层网络 =====
        self.option_actor = ImprovedOptionActor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            max_facilities=max_facilities_per_step
        )

        # ===== 改进的下层网络（使用KV缓存） =====
        self.action_decoder = CachedTransformerDecoder(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            num_layers=n_decode_layers
        )

        # 投影层
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # 查询初始化
        self.first_query = nn.Parameter(torch.Tensor(embedding_dim))
        self.first_query.data.uniform_(-1, 1)

        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=100)

        # 上下文融合
        self.context_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # ===== Critic网络（可选） =====
        if use_critic:
            self.value_head = ValueNetwork(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim
            )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """前向传播"""
        embeddings, graph_embedding = self.embedder(self._init_embed(input))

        # 执行分层决策（使用KV缓存加速）
        action_log_p, pi, option_log_p, option_decisions = self._hierarchical_inner_cached(
            input, embeddings, graph_embedding
        )

        # 计算成本
        cost = -self.problem.get_total_num(input, pi)

        # 计算对数似然
        action_ll = action_log_p.sum(dim=-1)
        option_ll = option_log_p.sum(dim=-1) if option_log_p is not None else torch.zeros_like(action_ll)

        # 如果使用Critic，计算价值估计
        value = None
        if self.use_critic and hasattr(self, 'value_head'):
            value = self.value_head(graph_embedding)

        if return_pi:
            return cost, action_ll, option_ll, pi, option_decisions, value
        return cost, action_ll, option_ll

    def _init_embed(self, input):
        return self.init_embed(input['facilities'])

    def _precompute(self, embeddings):
        """预计算固定的注意力数据"""
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

    def _hierarchical_inner_cached(self, input, embeddings, graph_embedding):
        """使用KV缓存的分层决策"""
        batch_size = embeddings.size(0)
        state = self.problem.make_state(input)
        fixed = self._precompute(embeddings)
        total_p = int(state.p)

        action_outputs = []
        action_sequences = []
        option_outputs = []
        option_sequences = []

        selected_mask = torch.zeros(batch_size, embeddings.size(1), dtype=torch.bool, device=embeddings.device)

        # KV缓存初始化
        kv_cache = None
        step = 0

        while not state.all_finished() and step < total_p:
            remaining_facilities = total_p - step

            # ===== 上层决策 =====
            context = self._build_context(
                graph_embedding, embeddings, selected_mask, step, total_p
            )

            option_logits = self.option_actor(
                embeddings, graph_embedding, context, remaining_facilities
            )

            # 限制选择范围
            if remaining_facilities < self.max_facilities_per_step:
                valid_logits = option_logits[:, :remaining_facilities]
            else:
                valid_logits = option_logits

            option_probs = F.softmax(valid_logits / self.temp, dim=-1)

            if self.decode_type == "greedy":
                _, option_selected = option_probs.max(1)
            elif self.decode_type == "sampling":
                option_selected = option_probs.multinomial(1).squeeze(1)
            else:
                option_selected = torch.zeros(batch_size, dtype=torch.long, device=option_probs.device)

            facilities_to_select = option_selected + 1

            option_log_p = torch.log(option_probs.gather(1, option_selected.unsqueeze(1)).squeeze(1))
            option_outputs.append(option_log_p)
            option_sequences.append(facilities_to_select)

            # ===== 下层决策（使用KV缓存） =====
            kv_cache = None  # 每个上层步单独的下层历史
            action_log_p, selected_facilities, state, kv_cache = self._select_facilities_cached(
                fixed, state, facilities_to_select, selected_mask,
                graph_embedding, step, kv_cache
            )

                        # 使用环境状态更新已选掩码，避免占位索引歧义
            selected_mask = state.get_mask()

            action_outputs.extend(action_log_p)
            action_sequences.extend(selected_facilities)

            step += facilities_to_select.max().item()

        # 整理输出
        while len(action_sequences) < total_p:
            action_outputs.append(torch.zeros(batch_size, device=embeddings.device))
            action_sequences.append(torch.zeros(batch_size, dtype=torch.long, device=embeddings.device))

        action_log_p = torch.stack(action_outputs[:total_p], 1)
        pi = torch.stack(action_sequences[:total_p], 1)

        if option_outputs:
            option_log_p = torch.stack(option_outputs, 1)
            option_decisions = torch.stack(option_sequences, 1)
        else:
            option_log_p = None
            option_decisions = None

        return action_log_p, pi, option_log_p, option_decisions

    def _select_facilities_cached(self, fixed, state, facilities_to_select, selected_mask,
                                  graph_embedding, global_step, kv_cache):
        """使用KV缓存的设施选择"""
        batch_size = fixed.node_embeddings.size(0)
        device = fixed.node_embeddings.device

        action_log_p = []
        selected_facilities = []

        # 初始查询
        num_to_select = facilities_to_select.float().unsqueeze(1)  # [B,1] 按样本传入
        task_embedding = self.context_fusion(torch.cat([
            graph_embedding,
            self.first_query.unsqueeze(0).expand(batch_size, -1),
            num_to_select
        ], dim=-1))

        decoder_input = task_embedding.unsqueeze(1)
        memory = fixed.node_embeddings

        for i in range(self.max_facilities_per_step):
            active_mask = i < facilities_to_select
            if not active_mask.any():
                break

            # 添加位置编码
            pos_encoded_input = decoder_input + self.pos_encoder.pe[:, i:i+1, :]  # 使用步号i的位置编码

            # 使用KV缓存的解码器
            decoder_output, kv_cache = self.action_decoder(
                tgt=pos_encoded_input,
                memory=memory,
                kv_cache=kv_cache
            )

            # 计算注意力分数
            query = decoder_output
            glimpse_K = fixed.glimpse_key
            glimpse_V = fixed.glimpse_val
            logit_K = fixed.logit_key

            mask = state.get_mask()
            logits = self._compute_logits(query, glimpse_K, glimpse_V, logit_K, mask)

            # 选择节点
            log_p = F.log_softmax(logits / self.temp, dim=-1).squeeze(1)
            probs = log_p.exp()

            if self.decode_type == "greedy":
                _, selected = probs.max(1)
            elif self.decode_type == "sampling":
                selected = probs.multinomial(1).squeeze(1)
            else:
                selected = torch.zeros(batch_size, dtype=torch.long, device=device)

            final_log_p = torch.zeros(batch_size, device=device)
            final_selected = torch.zeros(batch_size, dtype=torch.long, device=device)

            final_log_p[active_mask] = log_p[active_mask].gather(1, selected[active_mask].unsqueeze(1)).squeeze(1)
            final_selected[active_mask] = selected[active_mask]

            if active_mask.any():
                state = state.update(final_selected, active_mask)

            # 准备下一步输入
            if i < self.max_facilities_per_step - 1:
                selected_node_embeddings = fixed.node_embeddings.gather(
                    1, final_selected.view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
                )
                decoder_input = torch.where(
                    active_mask.view(-1, 1, 1),
                    selected_node_embeddings,
                    decoder_input
                )

            action_log_p.append(final_log_p)
            selected_facilities.append(final_selected)

        return action_log_p, selected_facilities, state, kv_cache

    def _build_context(self, graph_embedding, node_embeddings, selected_mask, step, total_p):
        """构建上下文信息"""
        batch_size = graph_embedding.size(0)
        device = graph_embedding.device

        step_ratio = step / total_p if total_p > 0 else 0.0
        remaining_ratio = (total_p - step) / total_p if total_p > 0 else 1.0
        selected_ratio = selected_mask.float().mean(dim=1, keepdim=True)

        step_tensor = torch.full((batch_size, 1), step_ratio, device=device)
        remaining_tensor = torch.full((batch_size, 1), remaining_ratio, device=device)

        # 使用已选择设施的嵌入的平均作为解决方案表示（更有信息量）
        if selected_mask.any():
            # selected_mask: True=已选；node_embeddings: [B, N, E]
            sel = selected_mask.float()  # 1=已选
            denom = torch.clamp(sel.sum(dim=1, keepdim=True), min=1.0)
            solution_embedding = (node_embeddings * sel.unsqueeze(-1)).sum(dim=1) / denom
        else:
            solution_embedding = torch.zeros_like(graph_embedding)

        context = torch.cat([
            graph_embedding,
            solution_embedding,
            step_tensor,
            remaining_tensor,
            selected_ratio
        ], dim=-1)

        return context

    def _compute_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """计算注意力logits"""
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner and mask is not None:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )

        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        if self.mask_logits and mask is not None:
            logits[mask] = -math.inf

        return logits

    def _make_heads(self, v, num_steps=None):
        """创建多头注意力的头"""
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )


class ImprovedOptionActor(nn.Module):
    """改进的上层Actor网络"""

    def __init__(self, embedding_dim, hidden_dim, n_heads, max_facilities):
        super().__init__()

        context_dim = embedding_dim * 2 + 3

        # 改进的上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 图注意力
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_facilities)
        )

    def forward(self, node_embeddings, graph_embedding, context, remaining_facilities):
        # 编码上下文
        context_encoded = self.context_encoder(context)

        # 图注意力
        query = graph_embedding.unsqueeze(1)
        attended_graph, _ = self.graph_attention(query, node_embeddings, node_embeddings)
        attended_graph = attended_graph.squeeze(1)

        # 组合特征
        combined_features = torch.cat([attended_graph, context_encoded], dim=-1)

        # 输出策略logits
        option_logits = self.policy_head(combined_features)

        return option_logits


class ValueNetwork(nn.Module):
    """价值网络（Critic）"""

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_embedding):
        return self.value_net(graph_embedding).squeeze(-1)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


def set_decode_type(model, decode_type):
    """设置解码类型"""
    if hasattr(model, 'module'):
        model = model.module
    model.set_decode_type(decode_type)
