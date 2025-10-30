# # hierarchical_reinforce_baselines.py
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from scipy.stats import ttest_rel
# import copy
# import numpy as np
# from utils import get_inner_model
# from nets.reinforce_baselines import Baseline
#
#
# class HierarchicalBaseline(Baseline):
#     """Base class for hierarchical baselines"""
#
#     def wrap_dataset(self, dataset):
#         return dataset
#
#     def unwrap_batch(self, batch):
#         return batch, None
#
#     def eval(self, x, c, option_c=None):
#         """
#         Evaluate baseline
#         Args:
#             x: Input data
#             c: Action cost
#             option_c: Option cost (optional)
#         """
#         raise NotImplementedError("Override this method")
#
#     def get_learnable_parameters(self):
#         return []
#
#     def epoch_callback(self, model, epoch):
#         pass
#
#     def state_dict(self):
#         return {}
#
#     def load_state_dict(self, state_dict):
#         pass
#
#
# class HierarchicalExponentialBaseline(HierarchicalBaseline):
#     """
#     改进的指数移动平均基线
#     - 支持分别跟踪action和option的基线
#     - 使用自适应的beta值
#     - 更好的初始化策略
#     """
#
#     def __init__(self, beta=0.9, option_beta=0.9, warmup_epochs=10, adaptive_beta=True):
#         super(HierarchicalBaseline, self).__init__()
#         self.beta = beta
#         self.option_beta = option_beta
#         self.warmup_epochs = warmup_epochs
#         self.adaptive_beta = adaptive_beta
#
#         self.v = None
#         self.option_v = None
#         self.n_epochs = 0
#         self.n_updates = 0
#
#         # 用于warmup期间的历史记录
#         self.cost_history = []
#         self.option_cost_history = []
#
#         # 用于自适应beta
#         self.cost_variance = None
#         self.option_cost_variance = None
#
#     def eval(self, x, c, option_c=None):
#         """评估基线值"""
#         self.n_epochs += 1
#         self.n_updates += 1
#
#         # Action baseline
#         if self.n_epochs <= self.warmup_epochs:
#             # Warmup期间使用历史平均
#             self.cost_history.append(c.mean().item())
#             if len(self.cost_history) > 200:  # 保持合理的历史长度
#                 self.cost_history.pop(0)
#             v = torch.tensor(np.mean(self.cost_history), device=c.device)
#         else:
#             # 使用指数移动平均
#             if self.v is None:
#                 v = c.mean()
#             else:
#                 # 自适应beta：当方差较大时使用较小的beta（更快适应）
#                 if self.adaptive_beta and self.cost_variance is not None:
#                     adaptive_beta = self.beta * (1 - min(0.5, self.cost_variance / (abs(self.v) + 1e-6)))
#                 else:
#                     adaptive_beta = self.beta
#
#                 v = adaptive_beta * self.v + (1. - adaptive_beta) * c.mean()
#
#             # 更新方差估计
#             if self.cost_variance is None:
#                 self.cost_variance = 0.0
#             else:
#                 self.cost_variance = 0.9 * self.cost_variance + 0.1 * ((c.mean() - v) ** 2).item()
#
#         self.v = v.detach()
#
#         # Option baseline
#         if option_c is not None:
#             if self.n_epochs <= self.warmup_epochs:
#                 self.option_cost_history.append(option_c.mean().item())
#                 if len(self.option_cost_history) > 200:
#                     self.option_cost_history.pop(0)
#                 option_v = torch.tensor(
#                     np.mean(self.option_cost_history) if self.option_cost_history else option_c.mean().item(),
#                     device=option_c.device
#                 )
#             else:
#                 if self.option_v is None:
#                     option_v = option_c.mean()
#                 else:
#                     # 自适应beta
#                     if self.adaptive_beta and self.option_cost_variance is not None:
#                         adaptive_option_beta = self.option_beta * (
#                                     1 - min(0.5, self.option_cost_variance / (abs(self.option_v) + 1e-6)))
#                     else:
#                         adaptive_option_beta = self.option_beta
#
#                     option_v = adaptive_option_beta * self.option_v + (1. - adaptive_option_beta) * option_c.mean()
#
#                 # 更新方差估计
#                 if self.option_cost_variance is None:
#                     self.option_cost_variance = 0.0
#                 else:
#                     self.option_cost_variance = 0.9 * self.option_cost_variance + 0.1 * (
#                                 (option_c.mean() - option_v) ** 2).item()
#
#             self.option_v = option_v.detach()
#         else:
#             option_v = torch.zeros_like(v)
#
#         return (self.v, self.option_v if self.option_v is not None else option_v), 0
#
#     def state_dict(self):
#         return {
#             'v': self.v,
#             'option_v': self.option_v,
#             'n_epochs': self.n_epochs,
#             'n_updates': self.n_updates,
#             'cost_history': self.cost_history,
#             'option_cost_history': self.option_cost_history,
#             'cost_variance': self.cost_variance,
#             'option_cost_variance': self.option_cost_variance
#         }
#
#     def load_state_dict(self, state_dict):
#         self.v = state_dict.get('v', None)
#         self.option_v = state_dict.get('option_v', None)
#         self.n_epochs = state_dict.get('n_epochs', 0)
#         self.n_updates = state_dict.get('n_updates', 0)
#         self.cost_history = state_dict.get('cost_history', [])
#         self.option_cost_history = state_dict.get('option_cost_history', [])
#         self.cost_variance = state_dict.get('cost_variance', None)
#         self.option_cost_variance = state_dict.get('option_cost_variance', None)
#
#
# class HierarchicalRolloutBaseline(HierarchicalBaseline):
#     """
#     改进的Rollout基线
#     - 更积极的更新策略
#     - 支持动态调整更新阈值
#     - 更好的初始化
#     """
#
#     def __init__(self, model, problem, opts, epoch=0):
#         super(HierarchicalBaseline, self).__init__()
#         self.problem = problem
#         self.opts = opts
#
#         # 改进的参数设置
#         self.warmup_epochs = getattr(opts, 'bl_warmup_epochs', 10)
#         self.min_update_interval = 1  # 最小更新间隔
#         self.max_update_interval = getattr(opts, 'bl_update_interval', 5)
#         self.adaptive_threshold = getattr(opts, 'adaptive_threshold', True)
#         self.improvement_threshold = 0.02  # 2%的改进阈值
#
#         # 跟踪性能历史
#         self.performance_history = []
#         self.update_history = []
#
#         self._update_model(model, epoch)
#
#     def _update_model(self, model, epoch, dataset=None):
#         """更新基线模型"""
#         self.model = copy.deepcopy(model)
#
#         # 验证数据集
#         if dataset is not None:
#             if len(dataset) != self.opts.val_size:
#                 print("Warning: not using saved baseline dataset since val_size does not match")
#                 dataset = None
#             elif (dataset[0]["users"]).size(0) != self.opts.n_users:
#                 print("Warning: not using saved baseline dataset since graph_size does not match")
#                 dataset = None
#
#         if dataset is None:
#             # 创建验证数据集
#             self.dataset = self.problem.make_dataset(
#                 n_users=self.opts.n_users,
#                 n_facilities=self.opts.n_facilities,
#                 num_samples=self.opts.val_size,
#                 filename=getattr(self.opts, 'val_dataset', None),
#                 p=self.opts.p,
#                 r=self.opts.r,
#                 distribution=self.opts.data_distribution
#             )
#         else:
#             self.dataset = dataset
#
#         print(f"Evaluating baseline model on validation dataset (epoch {epoch})")
#         self.bl_vals = self._hierarchical_rollout(self.model, self.dataset, self.opts).cpu().numpy()
#         self.mean = self.bl_vals.mean()
#         self.std = self.bl_vals.std()
#         self.epoch = epoch
#
#         # 记录性能
#         self.performance_history.append(self.mean)
#         self.update_history.append(epoch)
#
#         print(f"Baseline updated: mean={self.mean:.4f}, std={self.std:.4f}")
#
#     def _hierarchical_rollout(self, model, dataset, opts):
#         """执行分层模型的rollout评估"""
#         from nets.hierarchical_attention_model import set_decode_type
#
#         set_decode_type(model, "greedy")
#         model.eval()
#
#         def eval_model_batch(batch):
#             with torch.no_grad():
#                 cost, action_ll, option_ll = model(move_to(batch, opts.device))
#             return cost.data.cpu()
#
#         from torch.utils.data import DataLoader
#         from tqdm import tqdm
#         from utils import move_to
#
#         costs = []
#         for batch in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size),
#                           disable=opts.no_progress_bar, desc="Rollout evaluation"):
#             costs.append(eval_model_batch(batch))
#
#         return torch.cat(costs, 0)
#
#     def wrap_dataset(self, dataset):
#         """包装数据集以包含基线值"""
#         print("Computing baseline values for dataset...")
#         baseline_vals = self._hierarchical_rollout(self.model, dataset, self.opts).view(-1, 1)
#         return HierarchicalBaselineDataset(dataset, baseline_vals)
#
#     def unwrap_batch(self, batch):
#         """解包批次数据"""
#         return batch['data'], batch['baseline'].view(-1)
#
#     def eval(self, x, c, option_c=None):
#         """评估基线值"""
#         with torch.no_grad():
#             # 使用当前基线模型进行评估
#             cost, _, _ = self.model(x)
#
#         # 返回基线值和0损失（rollout基线不需要训练）
#         return (cost, torch.zeros_like(cost)), 0
#
#     def epoch_callback(self, model, epoch):
#         """
#         改进的基线更新策略
#         - 动态调整更新间隔
#         - 使用自适应阈值
#         - 考虑性能趋势
#         """
#         # 动态更新间隔
#         if epoch < self.warmup_epochs:
#             update_interval = self.min_update_interval
#         else:
#             # 根据最近的改进情况调整更新间隔
#             if len(self.performance_history) >= 3:
#                 recent_improvements = [
#                     self.performance_history[i - 1] - self.performance_history[i]
#                     for i in range(max(1, len(self.performance_history) - 3), len(self.performance_history))
#                 ]
#                 avg_improvement = np.mean(recent_improvements)
#
#                 if avg_improvement > self.improvement_threshold:
#                     # 快速改进时频繁更新
#                     update_interval = self.min_update_interval
#                 else:
#                     # 改进缓慢时减少更新频率
#                     update_interval = min(
#                         self.max_update_interval,
#                         self.min_update_interval + (epoch - self.warmup_epochs) // 10
#                     )
#             else:
#                 update_interval = self.min_update_interval
#
#         # 检查是否应该更新
#         if (epoch - self.epoch) < update_interval:
#             return
#
#         print(f"\nEvaluating candidate model at epoch {epoch}")
#         candidate_vals = self._hierarchical_rollout(model, self.dataset, self.opts).cpu().numpy()
#         candidate_mean = candidate_vals.mean()
#         candidate_std = candidate_vals.std()
#
#         improvement = self.mean - candidate_mean
#         improvement_ratio = improvement / (abs(self.mean) + 1e-6)
#
#         print(f"Candidate: mean={candidate_mean:.4f}, std={candidate_std:.4f}")
#         print(f"Current baseline: mean={self.mean:.4f}, std={self.std:.4f}")
#         print(f"Improvement: {improvement:.4f} ({improvement_ratio * 100:.2f}%)")
#
#         # 决定是否更新
#         should_update = False
#
#         if improvement > 0:  # 有改进
#             # 统计显著性检验
#             t, p = ttest_rel(candidate_vals, self.bl_vals)
#             p_val = p / 2  # 单侧检验
#
#             # 自适应阈值
#             if self.adaptive_threshold:
#                 # 大的改进使用更宽松的p值阈值
#                 if improvement_ratio > 0.05:  # 5%以上的改进
#                     alpha_threshold = 0.2
#                 elif improvement_ratio > 0.02:  # 2%以上的改进
#                     alpha_threshold = 0.1
#                 else:
#                     alpha_threshold = self.opts.bl_alpha
#             else:
#                 alpha_threshold = self.opts.bl_alpha
#
#             print(f"Statistical test: t={t:.4f}, p-value={p_val:.4f}, threshold={alpha_threshold:.4f}")
#
#             # 更新条件
#             if p_val < alpha_threshold:
#                 should_update = True
#                 print("Statistically significant improvement detected!")
#             elif epoch < self.warmup_epochs:
#                 should_update = True
#                 print("Updating during warmup period")
#             elif improvement_ratio > self.improvement_threshold:
#                 should_update = True
#                 print(f"Large improvement ({improvement_ratio * 100:.2f}%) detected!")
#
#         if should_update:
#             print(">>> Updating baseline model <<<")
#             self._update_model(model, epoch)
#         else:
#             print("No update performed")
#
#     def state_dict(self):
#         return {
#             'model': self.model,
#             'dataset': self.dataset,
#             'epoch': self.epoch,
#             'mean': self.mean,
#             'std': self.std,
#             'performance_history': self.performance_history,
#             'update_history': self.update_history
#         }
#
#     def load_state_dict(self, state_dict):
#         # 加载模型
#         load_model = copy.deepcopy(self.model)
#         get_inner_model(load_model).load_state_dict(
#             get_inner_model(state_dict['model']).state_dict()
#         )
#
#         # 恢复状态
#         self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])
#         if 'mean' in state_dict:
#             self.mean = state_dict['mean']
#         if 'std' in state_dict:
#             self.std = state_dict['std']
#         if 'performance_history' in state_dict:
#             self.performance_history = state_dict['performance_history']
#         if 'update_history' in state_dict:
#             self.update_history = state_dict['update_history']
#
#
# class HierarchicalCriticBaseline(HierarchicalBaseline):
#     """
#     改进的Critic基线
#     - 分别为action和option使用不同的critic网络
#     - 支持权重共享
#     - 改进的网络架构
#     """
#
#     def __init__(self, action_critic, option_critic=None, shared_encoder=None):
#         super(HierarchicalBaseline, self).__init__()
#         self.action_critic = action_critic
#         self.option_critic = option_critic
#         self.shared_encoder = shared_encoder
#
#         # 损失权重
#         self.action_weight = 1.0
#         self.option_weight = 0.5  # Option损失的权重可以小一些
#
#     def eval(self, x, c, option_c=None):
#         """评估critic值"""
#         # 如果有共享编码器，先编码
#         if self.shared_encoder is not None:
#             encoded = self.shared_encoder(x)
#             action_input = encoded
#             option_input = encoded
#         else:
#             action_input = x
#             option_input = x
#
#         # Action critic
#         action_v = self.action_critic(action_input)
#         action_loss = F.mse_loss(action_v, c.detach())
#
#         # Option critic
#         if option_c is not None and self.option_critic is not None:
#             option_v = self.option_critic(option_input)
#             option_loss = F.mse_loss(option_v, option_c.detach())
#             total_loss = self.action_weight * action_loss + self.option_weight * option_loss
#         else:
#             option_v = torch.zeros_like(action_v)
#             total_loss = action_loss
#
#         return (action_v.detach(), option_v.detach()), total_loss
#
#     def get_learnable_parameters(self):
#         """获取可学习参数"""
#         params = list(self.action_critic.parameters())
#         if self.option_critic is not None:
#             params.extend(list(self.option_critic.parameters()))
#         if self.shared_encoder is not None:
#             params.extend(list(self.shared_encoder.parameters()))
#         return params
#
#     def state_dict(self):
#         state = {'action_critic': self.action_critic.state_dict()}
#         if self.option_critic is not None:
#             state['option_critic'] = self.option_critic.state_dict()
#         if self.shared_encoder is not None:
#             state['shared_encoder'] = self.shared_encoder.state_dict()
#         return state
#
#     def load_state_dict(self, state_dict):
#         self.action_critic.load_state_dict(state_dict['action_critic'])
#         if self.option_critic is not None and 'option_critic' in state_dict:
#             self.option_critic.load_state_dict(state_dict['option_critic'])
#         if self.shared_encoder is not None and 'shared_encoder' in state_dict:
#             self.shared_encoder.load_state_dict(state_dict['shared_encoder'])
#
#
# class HierarchicalBaselineDataset(Dataset):
#     """分层基线数据集包装器"""
#
#     def __init__(self, dataset=None, baseline=None):
#         super(HierarchicalBaselineDataset, self).__init__()
#         self.dataset = dataset
#         self.baseline = baseline
#         assert len(self.dataset) == len(self.baseline), \
#             f"Dataset and baseline size mismatch: {len(self.dataset)} vs {len(self.baseline)}"
#
#     def __getitem__(self, item):
#         return {
#             'data': self.dataset[item],
#             'baseline': self.baseline[item]
#         }
#
#     def __len__(self):
#         return len(self.dataset)
#
#
# class WarmupBaseline(HierarchicalBaseline):
#     """分层模型的Warmup基线"""
#
#     def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
#         super(HierarchicalBaseline, self).__init__()
#
#         self.baseline = baseline
#         assert n_epochs > 0, "n_epochs to warmup must be positive"
#         self.warmup_baseline = HierarchicalExponentialBaseline(warmup_exp_beta, warmup_exp_beta)
#         self.alpha = 0
#         self.n_epochs = n_epochs
#
#     def wrap_dataset(self, dataset):
#         if self.alpha > 0:
#             return self.baseline.wrap_dataset(dataset)
#         return self.warmup_baseline.wrap_dataset(dataset)
#
#     def unwrap_batch(self, batch):
#         if self.alpha > 0:
#             return self.baseline.unwrap_batch(batch)
#         return self.warmup_baseline.unwrap_batch(batch)
#
#     def eval(self, x, c, option_c=None):
#         if self.alpha == 1:
#             return self.baseline.eval(x, c, option_c)
#         if self.alpha == 0:
#             return self.warmup_baseline.eval(x, c, option_c)
#         v, l = self.baseline.eval(x, c, option_c)
#         vw, lw = self.warmup_baseline.eval(x, c, option_c)
#         # Return convex combination of baseline and of loss
#         return (self.alpha * v[0] + (1 - self.alpha) * vw[0],
#                 self.alpha * v[1] + (1 - self.alpha) * vw[1]), self.alpha * l + (1 - self.alpha) * lw
#
#     def epoch_callback(self, model, epoch):
#         # Need to call epoch callback of inner model (also after first epoch if we have not used it)
#         self.baseline.epoch_callback(model, epoch)
#         self.alpha = (epoch + 1) / float(self.n_epochs)
#         if epoch < self.n_epochs:
#             print("Set warmup alpha = {}".format(self.alpha))
#
#     def state_dict(self):
#         # Checkpointing within warmup stage makes no sense, only save inner baseline
#         return self.baseline.state_dict()
#
#     def load_state_dict(self, state_dict):
#         # Checkpointing within warmup stage makes no sense, only load inner baseline
#         self.baseline.load_state_dict(state_dict)
#
#     def get_learnable_parameters(self):
#         return self.baseline.get_learnable_parameters()

# hierarchical_reinforce_baselines.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
import numpy as np
from utils import get_inner_model
from nets.reinforce_baselines import Baseline


class HierarchicalBaseline(Baseline):
    """Base class for hierarchical baselines"""

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c, option_c=None):
        """
        Evaluate baseline
        Args:
            x: Input data
            c: Action cost
            option_c: Option cost (optional)
        """
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class HierarchicalExponentialBaseline(HierarchicalBaseline):
    """
    改进的指数移动平均基线
    - 支持分别跟踪action和option的基线
    - 使用自适应的beta值
    - 更好的初始化策略
    """

    def __init__(self, beta=0.9, option_beta=0.9, warmup_epochs=10, adaptive_beta=True):
        super(HierarchicalBaseline, self).__init__()
        self.beta = beta
        self.option_beta = option_beta
        self.warmup_epochs = warmup_epochs
        self.adaptive_beta = adaptive_beta

        self.v = None
        self.option_v = None
        self.n_epochs = 0
        self.n_updates = 0

        # 用于warmup期间的历史记录
        self.cost_history = []
        self.option_cost_history = []

        # 用于自适应beta
        self.cost_variance = None
        self.option_cost_variance = None

    def eval(self, x, c, option_c=None):
        """评估基线值"""
        self.n_epochs += 1
        self.n_updates += 1

        # Action baseline
        if self.n_epochs <= self.warmup_epochs:
            # Warmup期间使用历史平均
            self.cost_history.append(c.mean().item())
            if len(self.cost_history) > 200:  # 保持合理的历史长度
                self.cost_history.pop(0)
            v = torch.tensor(np.mean(self.cost_history), device=c.device)
        else:
            # 使用指数移动平均
            if self.v is None:
                v = c.mean()
            else:
                # 自适应beta：当方差较大时使用较小的beta（更快适应）
                if self.adaptive_beta and self.cost_variance is not None:
                    adaptive_beta = self.beta * (1 - min(0.5, self.cost_variance / (abs(self.v) + 1e-6)))
                else:
                    adaptive_beta = self.beta

                v = adaptive_beta * self.v + (1. - adaptive_beta) * c.mean()

            # 更新方差估计
            if self.cost_variance is None:
                self.cost_variance = 0.0
            else:
                self.cost_variance = 0.9 * self.cost_variance + 0.1 * ((c.mean() - v) ** 2).item()

        self.v = v.detach()

        # Option baseline
        if option_c is not None:
            if self.n_epochs <= self.warmup_epochs:
                self.option_cost_history.append(option_c.mean().item())
                if len(self.option_cost_history) > 200:
                    self.option_cost_history.pop(0)
                option_v = torch.tensor(
                    np.mean(self.option_cost_history) if self.option_cost_history else option_c.mean().item(),
                    device=option_c.device
                )
            else:
                if self.option_v is None:
                    option_v = option_c.mean()
                else:
                    # 自适应beta
                    if self.adaptive_beta and self.option_cost_variance is not None:
                        adaptive_option_beta = self.option_beta * (
                                    1 - min(0.5, self.option_cost_variance / (abs(self.option_v) + 1e-6)))
                    else:
                        adaptive_option_beta = self.option_beta

                    option_v = adaptive_option_beta * self.option_v + (1. - adaptive_option_beta) * option_c.mean()

                # 更新方差估计
                if self.option_cost_variance is None:
                    self.option_cost_variance = 0.0
                else:
                    self.option_cost_variance = 0.9 * self.option_cost_variance + 0.1 * (
                                (option_c.mean() - option_v) ** 2).item()

            self.option_v = option_v.detach()
        else:
            option_v = torch.zeros_like(v)

        return (self.v, self.option_v if self.option_v is not None else option_v), 0

    def state_dict(self):
        return {
            'v': self.v,
            'option_v': self.option_v,
            'n_epochs': self.n_epochs,
            'n_updates': self.n_updates,
            'cost_history': self.cost_history,
            'option_cost_history': self.option_cost_history,
            'cost_variance': self.cost_variance,
            'option_cost_variance': self.option_cost_variance
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict.get('v', None)
        self.option_v = state_dict.get('option_v', None)
        self.n_epochs = state_dict.get('n_epochs', 0)
        self.n_updates = state_dict.get('n_updates', 0)
        self.cost_history = state_dict.get('cost_history', [])
        self.option_cost_history = state_dict.get('option_cost_history', [])
        self.cost_variance = state_dict.get('cost_variance', None)
        self.option_cost_variance = state_dict.get('option_cost_variance', None)


class HierarchicalRolloutBaseline(HierarchicalBaseline):
    """
    改进的Rollout基线
    - 更积极的更新策略
    - 支持动态调整更新阈值
    - 更好的初始化
    """

    def __init__(self, model, problem, opts, epoch=0):
        super(HierarchicalBaseline, self).__init__()
        self.problem = problem
        self.opts = opts

        # 改进的参数设置
        self.warmup_epochs = getattr(opts, 'bl_warmup_epochs', 10)
        self.min_update_interval = 1  # 最小更新间隔
        self.max_update_interval = getattr(opts, 'bl_update_interval', 5)
        self.adaptive_threshold = getattr(opts, 'adaptive_threshold', True)
        self.improvement_threshold = 0.02  # 2%的改进阈值

        # 跟踪性能历史
        self.performance_history = []
        self.update_history = []

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        """更新基线模型"""
        self.model = copy.deepcopy(model)

        # 验证数据集
        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0]["users"]).size(0) != self.opts.n_users:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            # 创建验证数据集
            self.dataset = self.problem.make_dataset(
                n_users=self.opts.n_users, n_facilities=self.opts.n_facilities, num_samples=self.opts.val_size,
                filename='data/MCLP_1000_30_normal_Normalization.pkl',
                p=self.opts.p, r=self.opts.r, distribution=self.opts.data_distribution)
            print(f"p = {self.opts.p}")
        else:
            self.dataset = dataset

        print(f"Evaluating baseline model on validation dataset (epoch {epoch})")
        self.bl_vals = self._hierarchical_rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.std = self.bl_vals.std()
        self.epoch = epoch

        # 记录性能
        self.performance_history.append(self.mean)
        self.update_history.append(epoch)

        print(f"Baseline updated: mean={self.mean:.4f}, std={self.std:.4f}")

    def _hierarchical_rollout(self, model, dataset, opts):
        """执行分层模型的rollout评估"""
        from nets.hierarchical_attention_model import set_decode_type

        set_decode_type(model, "greedy")
        model.eval()

        def eval_model_batch(batch):
            with torch.no_grad():
                cost, action_ll, option_ll = model(move_to(batch, opts.device))
            return cost.data.cpu()

        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from utils import move_to

        costs = []
        for batch in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size),
                          disable=opts.no_progress_bar, desc="Rollout evaluation"):
            costs.append(eval_model_batch(batch))

        return torch.cat(costs, 0)

    def wrap_dataset(self, dataset):
        """包装数据集以包含基线值"""
        print("Computing baseline values for dataset...")
        baseline_vals = self._hierarchical_rollout(self.model, dataset, self.opts).view(-1, 1)
        return HierarchicalBaselineDataset(dataset, baseline_vals)

    def unwrap_batch(self, batch):
        """解包批次数据"""
        return batch['data'], batch['baseline'].view(-1)

    def eval(self, x, c, option_c=None):
        """评估基线值"""
        with torch.no_grad():
            # 使用当前基线模型进行评估
            cost, _, _ = self.model(x)

        # 返回基线值和0损失（rollout基线不需要训练）
        return (cost, torch.zeros_like(cost)), 0

    def epoch_callback(self, model, epoch):
        """
        改进的基线更新策略
        - 动态调整更新间隔
        - 使用自适应阈值
        - 考虑性能趋势
        """
        # 动态更新间隔
        if epoch < self.warmup_epochs:
            update_interval = self.min_update_interval
        else:
            # 根据最近的改进情况调整更新间隔
            if len(self.performance_history) >= 3:
                recent_improvements = [
                    self.performance_history[i - 1] - self.performance_history[i]
                    for i in range(max(1, len(self.performance_history) - 3), len(self.performance_history))
                ]
                avg_improvement = np.mean(recent_improvements)

                if avg_improvement > self.improvement_threshold:
                    # 快速改进时频繁更新
                    update_interval = self.min_update_interval
                else:
                    # 改进缓慢时减少更新频率
                    update_interval = min(
                        self.max_update_interval,
                        self.min_update_interval + (epoch - self.warmup_epochs) // 10
                    )
            else:
                update_interval = self.min_update_interval

        # 检查是否应该更新
        if (epoch - self.epoch) < update_interval:
            return

        print(f"\nEvaluating candidate model at epoch {epoch}")
        candidate_vals = self._hierarchical_rollout(model, self.dataset, self.opts).cpu().numpy()
        candidate_mean = candidate_vals.mean()
        candidate_std = candidate_vals.std()

        improvement = self.mean - candidate_mean
        improvement_ratio = improvement / (abs(self.mean) + 1e-6)

        print(f"Candidate: mean={candidate_mean:.4f}, std={candidate_std:.4f}")
        print(f"Current baseline: mean={self.mean:.4f}, std={self.std:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement_ratio * 100:.2f}%)")

        # 决定是否更新
        should_update = False

        if improvement > 0:  # 有改进
            # 统计显著性检验
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2  # 单侧检验

            # 自适应阈值
            if self.adaptive_threshold:
                # 大的改进使用更宽松的p值阈值
                if improvement_ratio > 0.05:  # 5%以上的改进
                    alpha_threshold = 0.2
                elif improvement_ratio > 0.02:  # 2%以上的改进
                    alpha_threshold = 0.1
                else:
                    alpha_threshold = self.opts.bl_alpha
            else:
                alpha_threshold = self.opts.bl_alpha

            print(f"Statistical test: t={t:.4f}, p-value={p_val:.4f}, threshold={alpha_threshold:.4f}")

            # 更新条件
            if p_val < alpha_threshold:
                should_update = True
                print("Statistically significant improvement detected!")
            elif epoch < self.warmup_epochs:
                should_update = True
                print("Updating during warmup period")
            elif improvement_ratio > self.improvement_threshold:
                should_update = True
                print(f"Large improvement ({improvement_ratio * 100:.2f}%) detected!")

        if should_update:
            print(">>> Updating baseline model <<<")
            self._update_model(model, epoch)
        else:
            print("No update performed")

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch,
            'mean': self.mean,
            'std': self.std,
            'performance_history': self.performance_history,
            'update_history': self.update_history
        }

    def load_state_dict(self, state_dict):
        # 加载模型
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(
            get_inner_model(state_dict['model']).state_dict()
        )

        # 恢复状态
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])
        if 'mean' in state_dict:
            self.mean = state_dict['mean']
        if 'std' in state_dict:
            self.std = state_dict['std']
        if 'performance_history' in state_dict:
            self.performance_history = state_dict['performance_history']
        if 'update_history' in state_dict:
            self.update_history = state_dict['update_history']


class HierarchicalCriticBaseline(HierarchicalBaseline):
    """
    改进的Critic基线
    - 分别为action和option使用不同的critic网络
    - 支持权重共享
    - 改进的网络架构
    """

    def __init__(self, action_critic, option_critic=None, shared_encoder=None):
        super(HierarchicalBaseline, self).__init__()
        self.action_critic = action_critic
        self.option_critic = option_critic
        self.shared_encoder = shared_encoder

        # 损失权重
        self.action_weight = 1.0
        self.option_weight = 0.5  # Option损失的权重可以小一些

    def eval(self, x, c, option_c=None):
        """评估critic值"""
        # 如果有共享编码器，先编码
        if self.shared_encoder is not None:
            encoded = self.shared_encoder(x)
            action_input = encoded
            option_input = encoded
        else:
            action_input = x
            option_input = x

        # Action critic
        action_v = self.action_critic(action_input)
        action_loss = F.mse_loss(action_v, c.detach())

        # Option critic
        if option_c is not None and self.option_critic is not None:
            option_v = self.option_critic(option_input)
            option_loss = F.mse_loss(option_v, option_c.detach())
            total_loss = self.action_weight * action_loss + self.option_weight * option_loss
        else:
            option_v = torch.zeros_like(action_v)
            total_loss = action_loss

        return (action_v.detach(), option_v.detach()), total_loss

    def get_learnable_parameters(self):
        """获取可学习参数"""
        params = list(self.action_critic.parameters())
        if self.option_critic is not None:
            params.extend(list(self.option_critic.parameters()))
        if self.shared_encoder is not None:
            params.extend(list(self.shared_encoder.parameters()))
        return params

    def state_dict(self):
        state = {'action_critic': self.action_critic.state_dict()}
        if self.option_critic is not None:
            state['option_critic'] = self.option_critic.state_dict()
        if self.shared_encoder is not None:
            state['shared_encoder'] = self.shared_encoder.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.action_critic.load_state_dict(state_dict['action_critic'])
        if self.option_critic is not None and 'option_critic' in state_dict:
            self.option_critic.load_state_dict(state_dict['option_critic'])
        if self.shared_encoder is not None and 'shared_encoder' in state_dict:
            self.shared_encoder.load_state_dict(state_dict['shared_encoder'])


class HierarchicalBaselineDataset(Dataset):
    """分层基线数据集包装器"""

    def __init__(self, dataset=None, baseline=None):
        super(HierarchicalBaselineDataset, self).__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert len(self.dataset) == len(self.baseline), \
            f"Dataset and baseline size mismatch: {len(self.dataset)} vs {len(self.baseline)}"

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)


class WarmupBaseline(HierarchicalBaseline):
    """分层模型的Warmup基线"""

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        super(HierarchicalBaseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = HierarchicalExponentialBaseline(warmup_exp_beta, warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c, option_c=None):
        if self.alpha == 1:
            return self.baseline.eval(x, c, option_c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c, option_c)
        v, l = self.baseline.eval(x, c, option_c)
        vw, lw = self.warmup_baseline.eval(x, c, option_c)
        # Return convex combination of baseline and of loss
        return (self.alpha * v[0] + (1 - self.alpha) * vw[0],
                self.alpha * v[1] + (1 - self.alpha) * vw[1]), self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)

    def get_learnable_parameters(self):
        return self.baseline.get_learnable_parameters()