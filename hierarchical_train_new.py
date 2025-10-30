import torch
from torch.utils.data import DataLoader
from nets.hierarchical_attention_model_v2 import set_decode_type
from utils import move_to, get_inner_model
from utils.log_utils import log_values
from tqdm import tqdm
import time
import os
import math
import numpy as np
import torch.nn.functional as F


def validate(model, dataset, opts):
    """验证分层模型"""
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    """执行分层模型的rollout"""
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, _ = model(move_to(bat, opts.device))
            return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size),
                        disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def compute_entropy(probs):
    """计算策略熵用于正则化"""
    return -(probs * probs.log()).sum(dim=-1).mean()


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    """改进的分层模型训练epoch"""
    print("Start hierarchical train epoch {}, lr={} for run {}".format(
        epoch, optimizer.param_groups[0]['lr'], opts.run_name))

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # 改进的课程学习策略
    if hasattr(opts, 'curriculum') and opts.curriculum:
        # 使用sigmoid曲线实现更平滑的难度增长
        progress = min(1.0, epoch / (opts.n_epochs * 0.6))
        sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))

        min_p = max(3, opts.p // 3)
        curriculum_p = int(min_p + (opts.p - min_p) * sigmoid_progress)

        # 温度调度：早期使用更高的温度鼓励探索
        temperature = max(0.5, 2.0 - 1.5 * sigmoid_progress)
        model.temp = temperature

        print(f"Curriculum: p={curriculum_p}, temp={temperature:.2f}, progress={progress:.2f}")
    else:
        curriculum_p = opts.p
        temperature = 1.0

    # 创建训练数据集
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            n_users=opts.n_users,
            n_facilities=opts.n_facilities,
            num_samples=opts.epoch_size,
            filename=opts.train_dataset if hasattr(opts, 'train_dataset') else None,
            p=curriculum_p,
            r=opts.r,
            distribution=opts.data_distribution
        )
    )
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=opts.batch_size,
        num_workers=4,
        pin_memory=True if opts.use_cuda else False
    )

    # 设置模型为训练模式
    model.train()
    set_decode_type(model, "sampling", temp=temperature)

    # 训练统计
    epoch_stats = {
        'costs': [],
        'action_losses': [],
        'option_losses': [],
        'value_losses': [],
        'grad_norms': []
    }

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        stats = train_batch_v2(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        # 收集统计信息
        for key in epoch_stats:
            if key in stats:
                epoch_stats[key].append(stats[key])

        step += 1

    # 打印epoch统计
    epoch_duration = time.time() - start_time
    print(f"Finished epoch {epoch}, took {time.strftime('%H:%M:%S', time.gmtime(epoch_duration))}")
    print(f"Average cost: {np.mean(epoch_stats['costs']):.4f}")
    print(f"Average action loss: {np.mean(epoch_stats['action_losses']):.4f}")
    print(f"Average option loss: {np.mean(epoch_stats['option_losses']):.4f}")

    # 保存检查点
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if opts.use_cuda else None,
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # 验证
    avg_val_cost = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_val_cost, step)

    # 基线回调
    baseline.epoch_callback(model, epoch)

    # 学习率调度
    lr_scheduler.step()


def train_batch_v2(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts):
    """改进的批次训练函数"""
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # 前向传播（支持Actor-Critic）
    if hasattr(model, 'use_critic') and model.use_critic:
        cost, action_ll, option_ll, pi, option_decisions, value = model(x, return_pi=True)
    else:
        cost, action_ll, option_ll = model(x)
        value = None
        pi = None
        option_decisions = None

    # 评估基线
    if bl_val is None:
        (action_bl_val, option_bl_val), bl_loss = baseline.eval(x, cost, option_ll)
    else:
        action_bl_val = bl_val
        option_bl_val = torch.zeros_like(bl_val)
        bl_loss = 0

    # 计算优势
    if value is not None:
        # 使用Critic的值估计
        action_advantage = (cost - value).detach()
        option_advantage = action_advantage  # 可以使用相同的优势
    else:
        # 使用基线
        action_advantage = (cost - action_bl_val).detach()
        option_advantage = (cost - option_bl_val).detach()

    # 标准化优势（提高训练稳定性）
    if opts.normalize_advantage:
        action_advantage = (action_advantage - action_advantage.mean()) / (action_advantage.std() + 1e-8)
        option_advantage = (option_advantage - option_advantage.mean()) / (option_advantage.std() + 1e-8)

    # 计算策略损失
    action_reinforce_loss = (action_advantage * action_ll).mean()
    option_reinforce_loss = (option_advantage * option_ll).mean()

    # 熵正则化（鼓励探索）
    entropy_loss = 0
    if hasattr(opts, 'entropy_coef') and opts.entropy_coef > 0:
        # 这里需要获取实际的概率分布来计算熵
        # 简化版本：使用对数概率的负值作为熵的代理
        action_entropy = -action_ll.mean()
        option_entropy = -option_ll.mean()
        entropy_loss = -opts.entropy_coef * (action_entropy + option_entropy)

    # 组合损失
    option_weight = getattr(opts, 'option_loss_weight', 0.5)
    reinforce_loss = action_reinforce_loss + option_weight * option_reinforce_loss

    # 如果使用Critic，添加价值损失
    if value is not None:
        value_loss = F.mse_loss(value, cost.detach())
        value_loss_coef = getattr(opts, 'value_loss_coef', 0.5)
        total_loss = reinforce_loss + value_loss_coef * value_loss + entropy_loss + bl_loss
    else:
        value_loss = torch.tensor(0.0)
        total_loss = reinforce_loss + entropy_loss + bl_loss

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()

    # 梯度裁剪
    grad_norms, grad_norms_clipped = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)

    # 优化步骤
    optimizer.step()

    # 记录日志
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   action_ll, reinforce_loss, bl_loss, tb_logger, opts)

        # 额外的日志
        if not opts.no_tensorboard:
            tb_logger.log_value('option_loss', option_reinforce_loss.item(), step)
            tb_logger.log_value('value_loss', value_loss.item() if value is not None else 0, step)
            tb_logger.log_value('entropy_loss',
                                entropy_loss if isinstance(entropy_loss, float) else entropy_loss.item(), step)
            tb_logger.log_value('advantage_mean', action_advantage.mean().item(), step)
            tb_logger.log_value('advantage_std', action_advantage.std().item(), step)

    # 返回统计信息
    return {
        'costs': cost.mean().item(),
        'action_losses': action_reinforce_loss.item(),
        'option_losses': option_reinforce_loss.item(),
        'value_losses': value_loss.item() if value is not None else 0,
        'grad_norms': grad_norms[0] if grad_norms else 0
    }


class PPOTrainer:
    """PPO训练器（可选的高级训练方法）"""

    def __init__(self, model, optimizer, opts):
        self.model = model
        self.optimizer = optimizer
        self.opts = opts

        # PPO参数
        self.clip_epsilon = getattr(opts, 'ppo_clip_epsilon', 0.2)
        self.ppo_epochs = getattr(opts, 'ppo_epochs', 4)
        self.value_loss_coef = getattr(opts, 'value_loss_coef', 0.5)
        self.entropy_coef = getattr(opts, 'entropy_coef', 0.01)
        self.max_grad_norm = getattr(opts, 'max_grad_norm', 0.5)

    def train_ppo_epoch(self, states, actions, old_log_probs, rewards, values, returns, advantages):
        """执行PPO更新"""
        total_loss = 0

        for _ in range(self.ppo_epochs):
            # 重新计算当前策略下的对数概率
            _, new_log_probs, _, _, _, new_values = self.model(states, return_pi=True)

            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 裁剪的目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            # 策略损失
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = F.mse_loss(new_values, returns)

            # 熵损失
            entropy = -(new_log_probs.exp() * new_log_probs).sum(dim=-1).mean()

            # 总损失
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.ppo_epochs


def train_with_ppo(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    """使用PPO训练（高级选项）"""
    ppo_trainer = PPOTrainer(model, optimizer, opts)

    # 收集轨迹
    trajectories = collect_trajectories(model, problem, opts)

    # 计算优势和回报
    advantages, returns = compute_advantages_and_returns(trajectories, baseline, opts)

    # PPO更新
    loss = ppo_trainer.train_ppo_epoch(
        trajectories['states'],
        trajectories['actions'],
        trajectories['log_probs'],
        trajectories['rewards'],
        trajectories['values'],
        returns,
        advantages
    )

    print(f"PPO epoch {epoch} loss: {loss:.4f}")

    return loss


def collect_trajectories(model, problem, opts, num_steps=2048):
    """收集训练轨迹用于PPO或其他高级算法"""
    model.eval()
    set_decode_type(model, "sampling")

    trajectories = {
        'states': [],
        'actions': [],
        'log_probs': [],
        'rewards': [],
        'values': [],
        'dones': [],
        'option_actions': [],
        'option_log_probs': []
    }

    # 生成批量数据
    dataset = problem.make_dataset(
        n_users=opts.n_users,
        n_facilities=opts.n_facilities,
        num_samples=min(num_steps, opts.batch_size * 4),
        p=opts.p,
        r=opts.r,
        distribution=opts.data_distribution
    )

    dataloader = DataLoader(dataset, batch_size=opts.batch_size)

    with torch.no_grad():
        for batch in dataloader:
            batch = move_to(batch, opts.device)

            # 前向传播获取所有需要的信息
            if hasattr(model, 'use_critic') and model.use_critic:
                cost, action_ll, option_ll, pi, option_decisions, value = model(batch, return_pi=True)
            else:
                cost, action_ll, option_ll, pi, option_decisions, _ = model(batch, return_pi=True)
                value = torch.zeros_like(cost)

            # 收集轨迹
            trajectories['states'].append(batch)
            trajectories['actions'].append(pi)
            trajectories['log_probs'].append(action_ll)
            trajectories['rewards'].append(-cost)  # 负成本作为奖励
            trajectories['values'].append(value)
            trajectories['dones'].append(torch.zeros_like(cost))  # MCLP是单步环境

            if option_ll is not None:
                trajectories['option_actions'].append(option_decisions)
                trajectories['option_log_probs'].append(option_ll)

    # 合并所有批次
    for key in trajectories:
        if len(trajectories[key]) > 0:
            if key == 'states':
                # 状态是字典，需要特殊处理
                continue
            else:
                trajectories[key] = torch.cat(trajectories[key], dim=0)

    return trajectories


def compute_advantages_and_returns(trajectories, gamma=0.99, gae_lambda=0.95):
    """
    计算GAE优势和折扣回报
    使用Generalized Advantage Estimation (GAE)
    """
    rewards = trajectories['rewards']
    values = trajectories['values']
    dones = trajectories['dones']

    num_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # 计算GAE
    gae = 0
    next_value = 0  # 终止状态的值为0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        # TD误差
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE累积
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    # 标准化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def train_with_ppo(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    """使用PPO训练（高级选项）"""
    print(f"PPO Training Epoch {epoch}")

    # 创建PPO训练器
    ppo_trainer = PPOTrainer(model, optimizer, opts)

    # 收集轨迹
    print("Collecting trajectories...")
    trajectories = collect_trajectories(model, problem, opts, num_steps=opts.epoch_size // 10)

    # 计算优势和回报
    print("Computing advantages...")
    advantages, returns = compute_advantages_and_returns(
        trajectories,
        gamma=opts.gamma,
        gae_lambda=opts.gae_lambda
    )

    # PPO更新
    print("PPO updates...")
    total_loss = 0
    num_updates = 0

    # 创建数据集用于PPO更新
    ppo_dataset = PPODataset(trajectories, advantages, returns)
    ppo_dataloader = DataLoader(
        ppo_dataset,
        batch_size=opts.ppo_batch_size,
        shuffle=True
    )

    for ppo_epoch in range(opts.ppo_epochs):
        epoch_loss = 0

        for batch_idx, ppo_batch in enumerate(ppo_dataloader):
            loss = ppo_trainer.update_step(
                ppo_batch['states'],
                ppo_batch['actions'],
                ppo_batch['old_log_probs'],
                ppo_batch['advantages'],
                ppo_batch['returns'],
                ppo_batch.get('option_actions'),
                ppo_batch.get('option_log_probs')
            )

            epoch_loss += loss
            num_updates += 1

            if tb_logger and batch_idx % 10 == 0:
                step = epoch * opts.ppo_epochs * len(ppo_dataloader) + ppo_epoch * len(ppo_dataloader) + batch_idx
                tb_logger.log_value('ppo_loss', loss, step)

        avg_epoch_loss = epoch_loss / len(ppo_dataloader)
        print(f"  PPO Epoch {ppo_epoch}: Loss = {avg_epoch_loss:.4f}")
        total_loss += epoch_loss

    avg_loss = total_loss / num_updates if num_updates > 0 else 0
    print(f"PPO epoch {epoch} average loss: {avg_loss:.4f}")

    # 验证
    avg_val_cost = validate(model, val_dataset, opts)

    if tb_logger:
        tb_logger.log_value('val_avg_reward', avg_val_cost, epoch)

    # 更新基线
    baseline.epoch_callback(model, epoch)

    # 学习率调度
    lr_scheduler.step()

    return avg_loss


class PPODataset(Dataset):
    """PPO训练数据集"""

    def __init__(self, trajectories, advantages, returns):
        self.trajectories = trajectories
        self.advantages = advantages
        self.returns = returns
        self.length = len(advantages)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {
            'advantages': self.advantages[idx],
            'returns': self.returns[idx]
        }

        # 添加轨迹数据
        for key in ['actions', 'old_log_probs', 'option_actions', 'option_log_probs']:
            if key in self.trajectories and self.trajectories[key] is not None:
                if len(self.trajectories[key]) > 0:
                    item[key] = self.trajectories[key][idx]

        # 状态需要特殊处理（因为是字典）
        if 'states' in self.trajectories:
            # 简化处理：返回索引，在训练时重新获取
            item['state_idx'] = idx

        return item


class PPOTrainer:
    """改进的PPO训练器"""

    def __init__(self, model, optimizer, opts):
        self.model = model
        self.optimizer = optimizer
        self.opts = opts

        # PPO参数
        self.clip_epsilon = getattr(opts, 'ppo_clip_epsilon', 0.2)
        self.value_loss_coef = getattr(opts, 'value_loss_coef', 0.5)
        self.entropy_coef = getattr(opts, 'entropy_coef', 0.01)
        self.max_grad_norm = getattr(opts, 'max_grad_norm', 0.5)

        # 用于跟踪
        self.update_steps = 0

    def update_step(self, states, actions, old_log_probs, advantages, returns,
                    option_actions=None, option_log_probs=None):
        """执行单步PPO更新"""

        # 设置模型为训练模式
        self.model.train()

        # 重新计算当前策略下的对数概率和值
        if hasattr(self.model, 'use_critic') and self.model.use_critic:
            _, new_action_log_probs, new_option_log_probs, _, _, values = self.model(states, return_pi=True)
        else:
            _, new_action_log_probs, new_option_log_probs = self.model(states)
            values = torch.zeros_like(returns)

        # 计算action的比率
        action_ratio = torch.exp(new_action_log_probs.sum(dim=-1) - old_log_probs.sum(dim=-1))

        # 计算裁剪的策略损失
        surr1 = action_ratio * advantages
        surr2 = torch.clamp(action_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        action_policy_loss = -torch.min(surr1, surr2).mean()

        # Option策略损失（如果有）
        option_policy_loss = 0
        if option_log_probs is not None and new_option_log_probs is not None:
            option_ratio = torch.exp(new_option_log_probs.sum(dim=-1) - option_log_probs.sum(dim=-1))
            option_surr1 = option_ratio * advantages
            option_surr2 = torch.clamp(option_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            option_policy_loss = -torch.min(option_surr1, option_surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(values, returns) if values is not None else 0

        # 熵损失（鼓励探索）
        action_entropy = -(new_action_log_probs.exp() * new_action_log_probs).sum(dim=-1).mean()
        option_entropy = 0
        if new_option_log_probs is not None:
            option_entropy = -(new_option_log_probs.exp() * new_option_log_probs).sum(dim=-1).mean()

        entropy_loss = self.entropy_coef * (action_entropy + option_entropy)

        # 总损失
        total_loss = (
                action_policy_loss +
                0.5 * option_policy_loss +  # Option损失权重
                self.value_loss_coef * value_loss -
                entropy_loss
        )

        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.update_steps += 1

        # 记录有用的统计信息
        with torch.no_grad():
            # 计算KL散度用于监控
            action_kl = (old_log_probs.sum(dim=-1) - new_action_log_probs.sum(dim=-1)).mean()

            # 计算裁剪比例
            clipped = ((action_ratio - 1.0).abs() > self.clip_epsilon).float().mean()

        # 打印调试信息（每100步）
        if self.update_steps % 100 == 0:
            print(f"  Step {self.update_steps}: Loss={total_loss:.4f}, "
                  f"Policy={action_policy_loss:.4f}, Value={value_loss:.4f}, "
                  f"KL={action_kl:.4f}, Clipped={clipped:.2%}")

        return total_loss.item()