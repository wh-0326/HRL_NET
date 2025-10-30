# import torch
# from torch import nn
# from torch.nn import functional as F
# import math
# from typing import Optional, List, Tuple
#
# from nets.graph_encoder import GraphAttentionEncoder
# from nets.attention_model import AttentionModel, AttentionModelFixed, _get_attention_node_data
# from utils.functions import sample_many
#
#
# class HierarchicalAttentionModel(nn.Module):
#     """
#     改进的分层强化学习模型
#     - 使用AttentionModel作为下层网络基础
#     - 优化上下层信息传递
#     - 简化实现，提高性能
#     """
#
#     def __init__(self,
#                  embedding_dim,
#                  hidden_dim,
#                  problem,
#                  n_encode_layers=2,
#                  n_decode_layers=2,
#                  tanh_clipping=10.,
#                  mask_inner=True,
#                  mask_logits=True,
#                  normalization='batch',
#                  n_heads=8,
#                  checkpoint_encoder=False,
#                  shrink_size=None,
#                  max_facilities_per_step=5,
#                  dy=False):
#         super(HierarchicalAttentionModel, self).__init__()
#
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.n_encode_layers = n_encode_layers
#         self.n_decode_layers = n_decode_layers
#         self.decode_type = None
#         self.temp = 1.0
#         self.max_facilities_per_step = max_facilities_per_step
#
#         self.tanh_clipping = tanh_clipping
#         self.mask_inner = mask_inner
#         self.mask_logits = mask_logits
#         self.problem = problem
#         self.n_heads = n_heads
#
#         # 节点嵌入 - 适配MCLP问题的2维坐标数据
#         node_dim = 2  # MCLP facilities只有(x, y)坐标
#         self.init_embed = nn.Linear(node_dim, embedding_dim)
#
#         # 共享的图编码器
#         self.embedder = GraphAttentionEncoder(
#             n_heads=n_heads,
#             embed_dim=embedding_dim,
#             n_layers=self.n_encode_layers,
#             normalization=normalization
#         )
#
#         # ===== 上层网络：预算分配 =====
#         self.option_network = OptionNetwork(
#             embedding_dim=embedding_dim,
#             hidden_dim=hidden_dim,
#             n_heads=n_heads,
#             max_facilities=max_facilities_per_step
#         )
#
#         # ===== 下层网络：基于AttentionModel的设施选择 =====
#         # 使用简化的Transformer解码器
#         self.action_decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=embedding_dim,
#                 nhead=n_heads,
#                 dim_feedforward=hidden_dim * 4,
#                 dropout=0.1,
#                 batch_first=True
#             ),
#             num_layers=n_decode_layers
#         )
#
#         # 投影层（来自AttentionModel）
#         self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
#         self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
#
#         # 初始化查询向量
#         self.first_query = nn.Parameter(torch.Tensor(embedding_dim))
#         self.first_query.data.uniform_(-1, 1)
#
#         # 位置编码
#         self.pos_encoder = PositionalEncoding(embedding_dim, max_len=100)
#
#         # 上下文融合层
#         self.context_fusion = nn.Sequential(
#             nn.Linear(embedding_dim * 2 + 1, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, embedding_dim)
#         )
#
#     def set_decode_type(self, decode_type, temp=None):
#         self.decode_type = decode_type
#         if temp is not None:
#             self.temp = temp
#
#     def forward(self, input, return_pi=False):
#         """改进的前向传播"""
#         embeddings, graph_embedding = self.embedder(self._init_embed(input))
#
#         # 执行分层决策
#         action_log_p, pi, option_log_p, option_decisions = self._hierarchical_inner(
#             input, embeddings, graph_embedding
#         )
#
#         # 计算成本
#         cost = -self.problem.get_total_num(input, pi)
#
#         # 计算对数似然
#         action_ll = action_log_p.sum(dim=-1)
#         option_ll = option_log_p.sum(dim=-1) if option_log_p is not None else torch.zeros_like(action_ll)
#
#         if return_pi:
#             return cost, action_ll, option_ll, pi, option_decisions
#         return cost, action_ll, option_ll
#
#     def _init_embed(self, input):
#         return self.init_embed(input['facilities'])
#
#     def _precompute(self, embeddings):
#         """预计算固定的注意力数据（与AttentionModel相同）"""
#         graph_embed = embeddings.mean(1)
#         fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
#
#         glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
#             self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)
#
#         fixed_attention_node_data = (
#             self._make_heads(glimpse_key_fixed),
#             self._make_heads(glimpse_val_fixed),
#             logit_key_fixed.contiguous()
#         )
#         return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
#
#     def _hierarchical_inner(self, input, embeddings, graph_embedding):
#         """改进的分层决策核心逻辑"""
#         batch_size = embeddings.size(0)
#         state = self.problem.make_state(input)
#         fixed = self._precompute(embeddings)
#         total_p = int(state.p)
#
#         action_outputs = []
#         action_sequences = []
#         option_outputs = []
#         option_sequences = []
#
#         # 用于跟踪已选择的设施
#         selected_mask = torch.zeros(batch_size, embeddings.size(1), dtype=torch.bool, device=embeddings.device)
#
#         step = 0
#         while not state.all_finished() and step < total_p:
#             remaining_facilities = total_p - step
#
#             # ===== 上层决策：选择设施数量 =====
#             # 构建更丰富的上下文 - 简化为基于已选择设施数量的表示
#             if step > 0:
#                 # 使用已选择设施的平均坐标作为解决方案嵌入的代理
#                 num_selected = selected_mask.sum(dim=1, keepdim=True).float()
#                 solution_embedding = graph_embedding * (num_selected / embeddings.size(1))
#             else:
#                 solution_embedding = torch.zeros_like(graph_embedding)
#
#             context = self._get_enhanced_context(
#                 graph_embedding, solution_embedding, selected_mask, step, total_p
#             )
#
#             # 获取选项logits
#             option_logits = self.option_network(embeddings, graph_embedding, context, remaining_facilities)
#
#             # 选择设施数量
#             if remaining_facilities < self.max_facilities_per_step:
#                 # 限制选择范围
#                 valid_logits = option_logits[:, :remaining_facilities]
#             else:
#                 valid_logits = option_logits
#
#             # 计算选项概率和选择
#             option_probs = F.softmax(valid_logits / self.temp, dim=-1)
#             if self.decode_type == "greedy":
#                 _, option_selected = option_probs.max(1)
#             elif self.decode_type == "sampling":
#                 option_selected = option_probs.multinomial(1).squeeze(1)
#             else:
#                 option_selected = torch.zeros(batch_size, dtype=torch.long, device=option_probs.device)
#
#             facilities_to_select = option_selected + 1  # 1-indexed
#
#             # 记录选项决策
#             option_log_p = torch.log(option_probs.gather(1, option_selected.unsqueeze(1)).squeeze(1))
#             option_outputs.append(option_log_p)
#             option_sequences.append(facilities_to_select)
#
#             # ===== 下层决策：选择具体设施 =====
#             action_log_p, selected_facilities, state = self._select_facilities_improved(
#                 fixed, state, facilities_to_select, selected_mask, graph_embedding, step
#             )
#
#             # 更新已选择的设施掩码
#             for i, indices in enumerate(selected_facilities):
#                 for idx in indices:
#                     if idx > 0:  # 有效的选择
#                         selected_mask[i, idx] = True
#
#             action_outputs.extend(action_log_p)
#             action_sequences.extend(selected_facilities)
#
#             # 更新步数
#             step += facilities_to_select.max().item()
#
#         # 整理输出
#         # Pad sequences to fixed length
#         while len(action_sequences) < total_p:
#             action_outputs.append(torch.zeros(batch_size, device=embeddings.device))
#             action_sequences.append(torch.zeros(batch_size, dtype=torch.long, device=embeddings.device))
#
#         action_log_p = torch.stack(action_outputs[:total_p], 1)
#         pi = torch.stack(action_sequences[:total_p], 1)
#
#         if option_outputs:
#             option_log_p = torch.stack(option_outputs, 1)
#             option_decisions = torch.stack(option_sequences, 1)
#         else:
#             option_log_p = None
#             option_decisions = None
#
#         return action_log_p, pi, option_log_p, option_decisions
#
#     def _get_enhanced_context(self, graph_embedding, solution_embedding, selected_mask, step, total_p):
#         """构建增强的上下文信息"""
#         batch_size = graph_embedding.size(0)
#         device = graph_embedding.device
#
#         # 基础信息
#         step_ratio = step / total_p if total_p > 0 else 0.0
#         remaining_ratio = (total_p - step) / total_p if total_p > 0 else 1.0
#
#         # 已选择设施的比例
#         selected_ratio = selected_mask.float().mean(dim=1, keepdim=True)
#
#         # 确保所有张量的形状正确
#         step_tensor = torch.full((batch_size, 1), step_ratio, device=device)
#         remaining_tensor = torch.full((batch_size, 1), remaining_ratio, device=device)
#
#         # 调试信息
#         # print(f"Debug - graph_embedding.shape: {graph_embedding.shape}")
#         # print(f"Debug - solution_embedding.shape: {solution_embedding.shape}")
#         # print(f"Debug - step_tensor.shape: {step_tensor.shape}")
#         # print(f"Debug - remaining_tensor.shape: {remaining_tensor.shape}")
#         # print(f"Debug - selected_ratio.shape: {selected_ratio.shape}")
#
#         context = torch.cat([
#             graph_embedding,
#             solution_embedding,
#             step_tensor,
#             remaining_tensor,
#             selected_ratio
#         ], dim=-1)
#
#         return context
#
#     def _select_facilities_improved(self, fixed, state, facilities_to_select, selected_mask,
#                                     graph_embedding, global_step):
#         """改进的设施选择方法"""
#         batch_size = fixed.node_embeddings.size(0)
#         device = fixed.node_embeddings.device
#
#         action_log_p = []
#         selected_facilities = []
#
#         # 为每个批次准备解码器输入
#         # 初始查询：结合全局信息和当前任务
#         num_to_select = facilities_to_select.float().mean()
#         task_embedding = self.context_fusion(torch.cat([
#             graph_embedding,
#             self.first_query.unsqueeze(0).expand(batch_size, -1),
#             num_to_select.view(1, 1).expand(batch_size, 1)
#         ], dim=-1))
#
#         decoder_input = task_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]
#
#         # 准备memory（编码器输出）
#         memory = fixed.node_embeddings  # [batch_size, num_nodes, embedding_dim]
#
#         for i in range(self.max_facilities_per_step):
#             # 检查哪些批次还需要选择
#             active_mask = i < facilities_to_select
#             if not active_mask.any():
#                 break
#
#             # 添加位置编码
#             pos_encoded_input = decoder_input + self.pos_encoder(torch.zeros(1, 1, 1, device=device)).squeeze(0)
#
#             # 解码器前向传播
#             decoder_output = self.action_decoder(
#                 tgt=pos_encoded_input,
#                 memory=memory
#             )
#
#             # 使用解码器输出计算注意力分数
#             query = decoder_output
#             glimpse_K, glimpse_V, logit_K = _get_attention_node_data(fixed)
#
#             # 获取当前掩码
#             mask = state.get_mask()
#
#             # 计算logits
#             logits = self._compute_logits(query, glimpse_K, glimpse_V, logit_K, mask)
#
#             # 选择节点
#             log_p = F.log_softmax(logits / self.temp, dim=-1).squeeze(1)
#             probs = log_p.exp()
#
#             if self.decode_type == "greedy":
#                 _, selected = probs.max(1)
#             elif self.decode_type == "sampling":
#                 selected = probs.multinomial(1).squeeze(1)
#             else:
#                 selected = torch.zeros(batch_size, dtype=torch.long, device=device)
#
#             # 只更新active的批次
#             final_log_p = torch.zeros(batch_size, device=device)
#             final_selected = torch.zeros(batch_size, dtype=torch.long, device=device)
#
#             final_log_p[active_mask] = log_p[active_mask].gather(1, selected[active_mask].unsqueeze(1)).squeeze(1)
#             final_selected[active_mask] = selected[active_mask]
#
#             # 更新状态（只更新active的批次）
#             if active_mask.any():
#                 state = state.update(final_selected, active_mask)
#
#             # 准备下一步的输入
#             if i < self.max_facilities_per_step - 1:
#                 selected_node_embeddings = fixed.node_embeddings.gather(
#                     1, final_selected.view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
#                 )
#                 # 只更新active批次的decoder输入
#                 decoder_input = torch.where(
#                     active_mask.view(-1, 1, 1),
#                     selected_node_embeddings,
#                     decoder_input
#                 )
#
#             action_log_p.append(final_log_p)
#             selected_facilities.append(final_selected)
#
#         return action_log_p, selected_facilities, state
#
#     def _compute_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
#         """计算注意力logits（基于AttentionModel）"""
#         batch_size, num_steps, embed_dim = query.size()
#         key_size = val_size = embed_dim // self.n_heads
#
#         # 重塑query用于多头注意力
#         glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
#
#         # 计算兼容性分数
#         compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
#
#         if self.mask_inner and mask is not None:
#             compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
#
#         # 计算注意力头
#         heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)
#
#         # 投影得到最终的context
#         glimpse = self.project_out(
#             heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
#         )
#
#         # 计算最终的logits
#         logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))
#
#         # 应用tanh裁剪
#         if self.tanh_clipping > 0:
#             logits = torch.tanh(logits) * self.tanh_clipping
#
#         # 应用掩码
#         if self.mask_logits and mask is not None:
#             logits[mask] = -math.inf
#
#         return logits
#
#     def _make_heads(self, v, num_steps=None):
#         """创建多头注意力的头（与AttentionModel相同）"""
#         assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
#         return (
#             v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
#             .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
#             .permute(3, 0, 1, 2, 4)
#         )
#
#     def sample_many(self, input, batch_rep=1, iter_rep=1):
#         """采样多个解"""
#         self.eval()
#         embeddings, graph_embedding = self.embedder(self._init_embed(input))
#         prepared_input = (input, embeddings, graph_embedding)
#
#         return sample_many(
#             lambda model_input: self._hierarchical_inner(*model_input)[:2],
#             lambda model_input, pi: -self.problem.get_total_num(model_input[0], pi),
#             prepared_input,
#             batch_rep,
#             iter_rep
#         )
#
#
# class OptionNetwork(nn.Module):
#     """改进的上层网络：预算分配决策"""
#
#     def __init__(self, embedding_dim, hidden_dim, n_heads, max_facilities):
#         super(OptionNetwork, self).__init__()
#
#         # 上下文编码器
#         context_dim = embedding_dim * 2 + 3  # graph_emb + solution_emb + 3 ratios
#         self.context_encoder = nn.Sequential(
#             nn.Linear(context_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU()
#         )
#
#         # 图注意力层
#         self.graph_attention = nn.MultiheadAttention(
#             embed_dim=embedding_dim,
#             num_heads=n_heads,
#             dropout=0.1,
#             batch_first=True
#         )
#
#         # 输出层
#         self.output_layer = nn.Sequential(
#             nn.Linear(embedding_dim + hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, max_facilities)
#         )
#
#     def forward(self, node_embeddings, graph_embedding, context, remaining_facilities):
#         """
#         前向传播
#         Args:
#             node_embeddings: [batch_size, n_nodes, embedding_dim]
#             graph_embedding: [batch_size, embedding_dim]
#             context: [batch_size, context_dim]
#             remaining_facilities: int
#         """
#         # 编码上下文
#         context_encoded = self.context_encoder(context)
#
#         # 图注意力
#         query = graph_embedding.unsqueeze(1)
#         attended_graph, _ = self.graph_attention(query, node_embeddings, node_embeddings)
#         attended_graph = attended_graph.squeeze(1)
#
#         # 组合特征
#         combined_features = torch.cat([attended_graph, context_encoded], dim=-1)
#
#         # 输出选项logits
#         option_logits = self.output_layer(combined_features)
#
#         return option_logits
#
#
# class PositionalEncoding(nn.Module):
#     """位置编码"""
#
#     def __init__(self, d_model, max_len=100):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
#
#     def forward(self, x):
#         # x: [batch_size, seq_len, d_model]
#         return self.pe[:, :x.size(1)]
#
#
# def set_decode_type(model, decode_type):
#     """设置解码类型"""
#     if hasattr(model, 'module'):
#         model = model.module
#     model.set_decode_type(decode_type)
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Optional, List, Tuple

from nets.graph_encoder import GraphAttentionEncoder
from nets.attention_model import AttentionModel, AttentionModelFixed, _get_attention_node_data
from utils.functions import sample_many


class HierarchicalAttentionModel(nn.Module):
    """
    改进的分层强化学习模型
    - 使用AttentionModel作为下层网络基础
    - 优化上下层信息传递
    - 简化实现，提高性能
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
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 max_facilities_per_step=5,
                 dy=False):
        super(HierarchicalAttentionModel, self).__init__()

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

        # 节点嵌入 - 适配MCLP问题的2维坐标数据
        node_dim = 2  # MCLP facilities只有(x, y)坐标
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        # 共享的图编码器
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # ===== 上层网络：预算分配 =====
        self.option_network = OptionNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            max_facilities=max_facilities_per_step
        )

        # ===== 下层网络：基于AttentionModel的设施选择 =====
        # 使用简化的Transformer解码器
        self.action_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=n_decode_layers
        )

        # 投影层（来自AttentionModel）
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # 初始化查询向量
        self.first_query = nn.Parameter(torch.Tensor(embedding_dim))
        self.first_query.data.uniform_(-1, 1)

        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=100)

        # 上下文融合层
        self.context_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """改进的前向传播"""
        embeddings, graph_embedding = self.embedder(self._init_embed(input))

        # 执行分层决策
        action_log_p, pi, option_log_p, option_decisions = self._hierarchical_inner(
            input, embeddings, graph_embedding
        )

        # 计算成本
        cost = -self.problem.get_total_num(input, pi)

        # 计算对数似然
        action_ll = action_log_p.sum(dim=-1)
        option_ll = option_log_p.sum(dim=-1) if option_log_p is not None else torch.zeros_like(action_ll)

        if return_pi:
            return cost, action_ll, option_ll, pi, option_decisions
        return cost, action_ll, option_ll

    def _init_embed(self, input):
        return self.init_embed(input['facilities'])

    def _precompute(self, embeddings):
        """预计算固定的注意力数据（与AttentionModel相同）"""
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

    def _hierarchical_inner(self, input, embeddings, graph_embedding):
        """改进的分层决策核心逻辑"""
        batch_size = embeddings.size(0)
        state = self.problem.make_state(input)
        fixed = self._precompute(embeddings)
        total_p = int(state.p)

        action_outputs = []
        action_sequences = []
        option_outputs = []
        option_sequences = []

        # 用于跟踪已选择的设施
        selected_mask = torch.zeros(batch_size, embeddings.size(1), dtype=torch.bool, device=embeddings.device)

        step = 0
        while not state.all_finished() and step < total_p:
            remaining_facilities = total_p - step

            # ===== 上层决策：选择设施数量 =====
            # 构建更丰富的上下文 - 简化为基于已选择设施数量的表示
            if step > 0:
                # 使用已选择设施的平均坐标作为解决方案嵌入的代理
                num_selected = selected_mask.sum(dim=1, keepdim=True).float()
                solution_embedding = graph_embedding * (num_selected / embeddings.size(1))
            else:
                solution_embedding = torch.zeros_like(graph_embedding)

            context = self._get_enhanced_context(
                graph_embedding, solution_embedding, selected_mask, step, total_p
            )

            # 获取选项logits
            option_logits = self.option_network(embeddings, graph_embedding, context, remaining_facilities)

            # 选择设施数量
            if remaining_facilities < self.max_facilities_per_step:
                # 限制选择范围
                valid_logits = option_logits[:, :remaining_facilities]
            else:
                valid_logits = option_logits

            # 计算选项概率和选择
            option_probs = F.softmax(valid_logits / self.temp, dim=-1)
            if self.decode_type == "greedy":
                _, option_selected = option_probs.max(1)
            elif self.decode_type == "sampling":
                option_selected = option_probs.multinomial(1).squeeze(1)
            else:
                option_selected = torch.zeros(batch_size, dtype=torch.long, device=option_probs.device)

            facilities_to_select = option_selected + 1  # 1-indexed

            # 记录选项决策
            option_log_p = torch.log(option_probs.gather(1, option_selected.unsqueeze(1)).squeeze(1))
            option_outputs.append(option_log_p)
            option_sequences.append(facilities_to_select)

            # ===== 下层决策：选择具体设施 =====
            action_log_p, selected_facilities, state = self._select_facilities_improved(
                fixed, state, facilities_to_select, selected_mask, graph_embedding, step
            )

            # 更新已选择的设施掩码
            for i, indices in enumerate(selected_facilities):
                for idx in indices:
                    if idx > 0:  # 有效的选择
                        selected_mask[i, idx] = True

            action_outputs.extend(action_log_p)
            action_sequences.extend(selected_facilities)

            # 更新步数
            step += facilities_to_select.max().item()

        # 整理输出
        # Pad sequences to fixed length
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

    def _get_enhanced_context(self, graph_embedding, solution_embedding, selected_mask, step, total_p):
        """构建增强的上下文信息"""
        batch_size = graph_embedding.size(0)
        device = graph_embedding.device

        # 基础信息
        step_ratio = step / total_p if total_p > 0 else 0.0
        remaining_ratio = (total_p - step) / total_p if total_p > 0 else 1.0

        # 已选择设施的比例
        selected_ratio = selected_mask.float().mean(dim=1, keepdim=True)

        # 确保所有张量的形状正确
        step_tensor = torch.full((batch_size, 1), step_ratio, device=device)
        remaining_tensor = torch.full((batch_size, 1), remaining_ratio, device=device)

        # 调试信息
        # print(f"Debug - graph_embedding.shape: {graph_embedding.shape}")
        # print(f"Debug - solution_embedding.shape: {solution_embedding.shape}")
        # print(f"Debug - step_tensor.shape: {step_tensor.shape}")
        # print(f"Debug - remaining_tensor.shape: {remaining_tensor.shape}")
        # print(f"Debug - selected_ratio.shape: {selected_ratio.shape}")

        context = torch.cat([
            graph_embedding,
            solution_embedding,
            step_tensor,
            remaining_tensor,
            selected_ratio
        ], dim=-1)

        return context

    def _select_facilities_improved(self, fixed, state, facilities_to_select, selected_mask,
                                    graph_embedding, global_step):
        """改进的设施选择方法"""
        batch_size = fixed.node_embeddings.size(0)
        device = fixed.node_embeddings.device

        action_log_p = []
        selected_facilities = []

        # 为每个批次准备解码器输入
        # 初始查询：结合全局信息和当前任务
        num_to_select = facilities_to_select.float().mean()
        task_embedding = self.context_fusion(torch.cat([
            graph_embedding,
            self.first_query.unsqueeze(0).expand(batch_size, -1),
            num_to_select.view(1, 1).expand(batch_size, 1)
        ], dim=-1))

        decoder_input = task_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # 准备memory（编码器输出）
        memory = fixed.node_embeddings  # [batch_size, num_nodes, embedding_dim]

        for i in range(self.max_facilities_per_step):
            # 检查哪些批次还需要选择
            active_mask = i < facilities_to_select
            if not active_mask.any():
                break

            # 添加位置编码
            pos_encoded_input = decoder_input + self.pos_encoder(torch.zeros(1, 1, 1, device=device)).squeeze(0)

            # 解码器前向传播
            decoder_output = self.action_decoder(
                tgt=pos_encoded_input,
                memory=memory
            )

            # 使用解码器输出计算注意力分数
            query = decoder_output
            glimpse_K, glimpse_V, logit_K = _get_attention_node_data(fixed)

            # 获取当前掩码
            mask = state.get_mask()

            # 计算logits
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

            # 只更新active的批次
            final_log_p = torch.zeros(batch_size, device=device)
            final_selected = torch.zeros(batch_size, dtype=torch.long, device=device)

            final_log_p[active_mask] = log_p[active_mask].gather(1, selected[active_mask].unsqueeze(1)).squeeze(1)
            final_selected[active_mask] = selected[active_mask]

            # 更新状态（只更新active的批次）
            if active_mask.any():
                state = state.update(final_selected, active_mask)

            # 准备下一步的输入
            if i < self.max_facilities_per_step - 1:
                selected_node_embeddings = fixed.node_embeddings.gather(
                    1, final_selected.view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
                )
                # 只更新active批次的decoder输入
                decoder_input = torch.where(
                    active_mask.view(-1, 1, 1),
                    selected_node_embeddings,
                    decoder_input
                )

            action_log_p.append(final_log_p)
            selected_facilities.append(final_selected)

        return action_log_p, selected_facilities, state

    def _compute_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """计算注意力logits（基于AttentionModel）"""
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # 重塑query用于多头注意力
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # 计算兼容性分数
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner and mask is not None:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # 计算注意力头
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # 投影得到最终的context
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )

        # 计算最终的logits
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

        # 应用tanh裁剪
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        # 应用掩码
        if self.mask_logits and mask is not None:
            logits[mask] = -math.inf

        return logits

    def _make_heads(self, v, num_steps=None):
        """创建多头注意力的头（与AttentionModel相同）"""
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """采样多个解"""
        self.eval()
        embeddings, graph_embedding = self.embedder(self._init_embed(input))
        prepared_input = (input, embeddings, graph_embedding)

        return sample_many(
            lambda model_input: self._hierarchical_inner(*model_input)[:2],
            lambda model_input, pi: -self.problem.get_total_num(model_input[0], pi),
            prepared_input,
            batch_rep,
            iter_rep
        )


class OptionNetwork(nn.Module):
    """改进的上层网络：预算分配决策"""

    def __init__(self, embedding_dim, hidden_dim, n_heads, max_facilities):
        super(OptionNetwork, self).__init__()

        # 上下文编码器
        context_dim = embedding_dim * 2 + 3  # graph_emb + solution_emb + 3 ratios
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 图注意力层
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_facilities)
        )

    def forward(self, node_embeddings, graph_embedding, context, remaining_facilities):
        """
        前向传播
        Args:
            node_embeddings: [batch_size, n_nodes, embedding_dim]
            graph_embedding: [batch_size, embedding_dim]
            context: [batch_size, context_dim]
            remaining_facilities: int
        """
        # 编码上下文
        context_encoded = self.context_encoder(context)

        # 图注意力
        query = graph_embedding.unsqueeze(1)
        attended_graph, _ = self.graph_attention(query, node_embeddings, node_embeddings)
        attended_graph = attended_graph.squeeze(1)

        # 组合特征
        combined_features = torch.cat([attended_graph, context_encoded], dim=-1)

        # 输出选项logits
        option_logits = self.output_layer(combined_features)

        return option_logits


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.pe[:, :x.size(1)]


def set_decode_type(model, decode_type):
    """设置解码类型"""
    if hasattr(model, 'module'):
        model = model.module
    model.set_decode_type(decode_type)