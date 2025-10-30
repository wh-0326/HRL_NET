import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import ttest_rel
import copy
import numpy as np
from utils import get_inner_model, move_to
from nets.reinforce_baselines import Baseline
from tqdm import tqdm


class ImprovedHierarchicalRolloutBaseline(Baseline):
    """
    改进的Rollout基线
    - 更激进的更新策略（每次有改进就更新）
    - 自适应阈值
    - 更好的性能跟踪
    """

    def __init__(self, model, problem, opts, epoch=0):
        super().__init__()
        self.problem = problem
        self.opts = opts

        # 改进的参数设置
        self.warmup_epochs = getattr(opts, 'bl_warmup_epochs', 5)
        self.improvement_threshold = 0.01  # 1%改进就更新
        self.force_update_interval = 10  # 每10个epoch强制更新一次

        # 性能跟踪
        self.performance_history = []
        self.update_history = []
        self.best_mean = float('inf')
        self.epochs_since_update = 0

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        """更新基线模型"""
        self.model = copy.deepcopy(model)

        # 验证数据集
        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: dataset size mismatch")
                dataset = None
            elif dataset[0]["users"].size(0) != self.opts.n_users:
                print("Warning: graph size mismatch")
                dataset = None

        if dataset is None:
            self.dataset = self.problem.make_dataset(
                n_users=self.opts.n_users,
                n_facilities=self.opts.n_facilities,
                num_samples=self.opts.val_size,
                filename=getattr(self.opts, 'val_dataset', None),
                p=self.opts.p,
                r=self.opts.r,
                distribution=self.opts.data_distribution
            )
        else:
            self.dataset = dataset

        print(f"Evaluating baseline model on validation dataset (epoch {epoch})")
        self.bl_vals = self._hierarchical_rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.std = self.bl_vals.std()
        self.epoch = epoch

        # 更新最佳性能
        if self.mean < self.best_mean:
            self.best_mean = self.mean

        # 记录性能
        self.performance_history.append(self.mean)
        self.update_history.append(epoch)
        self.epochs_since_update = 0

        print(f"Baseline updated: mean={self.mean:.4f}, std={self.std:.4f}, best={self.best_mean:.4f}")

    def _hierarchical_rollout(self, model, dataset, opts):
        """执行分层模型的rollout评估"""
        from nets.hierarchical_attention_model import set_decode_type

        set_decode_type(model, "greedy")
        model.eval()

        def eval_model_batch(batch):
            with torch.no_grad():
                cost, _, _ = model(move_to(batch, opts.device))
            return cost.data.cpu()

        costs = []
        for batch in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size),
                          disable=opts.no_progress_bar, desc="Rollout evaluation"):
            costs.append(eval_model_batch(batch))

        return torch.cat(costs, 0)

    def epoch_callback(self, model, epoch):
        """
        改进的更新策略：
        1. 任何改进都触发更新（更激进）
        2. 长时间没更新时强制更新
        3. Warmup期间总是更新
        """
        self.epochs_since_update += 1

        # Warmup期间总是更新
        if epoch < self.warmup_epochs:
            print(f"Warmup period: updating baseline at epoch {epoch}")
            self._update_model(model, epoch)
            return

        # 强制更新检查
        if self.epochs_since_update >= self.force_update_interval:
            print(f"Force update after {self.force_update_interval} epochs without update")
            self._update_model(model, epoch)
            return

        # 评估候选模型
        print(f"\nEvaluating candidate model at epoch {epoch}")
        candidate_vals = self._hierarchical_rollout(model, self.dataset, self.opts).cpu().numpy()
        candidate_mean = candidate_vals.mean()
        candidate_std = candidate_vals.std()

        improvement = self.mean - candidate_mean
        improvement_ratio = improvement / (abs(self.mean) + 1e-6)

        print(f"Candidate: mean={candidate_mean:.4f}, std={candidate_std:.4f}")
        print(f"Current baseline: mean={self.mean:.4f}, std={self.std:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement_ratio * 100:.2f}%)")

        # 更激进的更新策略：任何改进都更新
        if improvement > 0:
            # 统计显著性检验
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2  # 单侧检验

            print(f"Statistical test: t={t:.4f}, p-value={p_val:.4f}")

            # 更宽松的更新条件
            if p_val < 0.3 or improvement_ratio > self.improvement_threshold:
                print(">>> Updating baseline model (improvement detected) <<<")
                self._update_model(model, epoch)
            else:
                print("No significant improvement, keeping current baseline")
        else:
            print("No improvement, keeping current baseline")

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
            cost, _, _ = self.model(x)
        return (cost, torch.zeros_like(cost)), 0

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch,
            'mean': self.mean,
            'std': self.std,
            'best_mean': self.best_mean,
            'performance_history': self.performance_history,
            'update_history': self.update_history,
            'epochs_since_update': self.epochs_since_update
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(
            get_inner_model(state_dict['model']).state_dict()
        )
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])
        self.mean = state_dict.get('mean', self.mean)
        self.std = state_dict.get('std', self.std)
        self.best_mean = state_dict.get('best_mean', self.best_mean)
        self.performance_history = state_dict.get('performance_history', [])
        self.update_history = state_dict.get('update_history', [])
        self.epochs_since_update = state_dict.get('epochs_since_update', 0)