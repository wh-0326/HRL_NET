#!/usr/bin/env python

import os
import json
import pprint as pp
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from hierarchical_options import get_hierarchical_options
from hierarchical_train import train_epoch, validate
from nets.hierarchical_reinforce_baselines import (
    HierarchicalRolloutBaseline, 
    HierarchicalExponentialBaseline,
    HierarchicalCriticBaseline
)
from nets.hierarchical_attention_model import HierarchicalAttentionModel
from utils import torch_load_cpu, get_inner_model
# 直接导入MCLP问题
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ReCovNet-master'))
try:
    from problems.MCLP.problem_MCLP import MCLP
    def load_problem(name):
        if name == 'MCLP':
            return MCLP()
        else:
            raise ValueError(f"Unknown problem: {name}")
except ImportError:
    # 如果找不到问题定义，创建一个简化版本
    print("Warning: Could not import MCLP problem, using mock version")
    class MockMCLP:
        NAME = 'MCLP'
        
        @staticmethod
        def get_total_num(dataset, pi):
            return torch.rand(pi.size(0))
        
        @staticmethod
        def make_dataset(*args, **kwargs):
            # 创建模拟数据集
            class MockDataset:
                def __init__(self, n_users=50, n_facilities=20, num_samples=100, p=10, r=0.2):
                    self.data = []
                    for i in range(num_samples):
                        self.data.append({
                            'users': torch.rand(n_users, 2),
                            'facilities': torch.rand(n_facilities, 5),
                            'demand': torch.rand(n_users, 1),
                            'p': torch.tensor([p]),
                            'r': torch.tensor([r])
                        })
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            return MockDataset(*args, **kwargs)
        
        @staticmethod
        def make_state(input):
            class MockState:
                def __init__(self, p):
                    self.p = p
                    self.i = 0
                    
                def all_finished(self):
                    return self.i >= self.p
                    
                def get_mask(self):
                    batch_size = 1 if isinstance(self.p, int) else len(self.p)
                    n_facilities = 100  # 默认设施数量
                    return torch.zeros(batch_size, 1, n_facilities, dtype=torch.bool)
                    
                def update(self, selected):
                    new_state = MockState(self.p)
                    new_state.i = self.i + 1
                    return new_state
                    
            p_val = input['p'][0] if torch.is_tensor(input['p']) else input['p']
            return MockState(p_val)
    
    def load_problem(name):
        if name == 'MCLP':
            return MockMCLP()
        else:
            raise ValueError(f"Unknown problem: {name}")


def run(opts):
    # 打印运行参数
    pp.pprint(vars(opts))

    # 配置tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(
            opts.log_dir, 
            "{}_{}".format(opts.problem, opts.n_users), 
            "hierarchical_" + opts.run_name
        ))

    os.makedirs(opts.save_dir, exist_ok=True)
    
    # 保存参数配置
    with open(os.path.join(opts.save_dir, "hierarchical_args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # 设置设备
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # 加载问题
    problem = load_problem(opts.problem)

    # 加载模型参数
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading hierarchical data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # 初始化分层模型
    print("Initializing Hierarchical Attention Model...")
    model = HierarchicalAttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        n_decode_layers=getattr(opts, 'n_decode_layers', 2),
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        max_facilities_per_step=getattr(opts, 'max_facilities_per_step', 5),
        dy=False
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 加载预训练模型参数
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # 初始化分层基线
    baseline_type = getattr(opts, 'hierarchical_baseline', 'rollout')
    if baseline_type == 'exponential':
        baseline = HierarchicalExponentialBaseline(
            beta=opts.exp_beta,
            option_beta=getattr(opts, 'option_exp_beta', opts.exp_beta)
        )
    elif baseline_type == 'critic':
        # 这里需要定义critic网络，暂时使用rollout
        baseline = HierarchicalRolloutBaseline(model, problem, opts)
    else:  # rollout
        baseline = HierarchicalRolloutBaseline(model, problem, opts)

    # 加载基线状态
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # 初始化优化器
    optimizer_params = [{'params': model.parameters(), 'lr': opts.lr_model}]
    
    # 添加基线参数（如果有的话）
    baseline_params = baseline.get_learnable_parameters()
    if len(baseline_params) > 0:
        optimizer_params.append({'params': baseline_params, 'lr': opts.lr_critic})

    optimizer = optim.Adam(optimizer_params)

    # 加载优化器状态
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # 初始化学习率调度器
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # 创建验证数据集
    val_dataset = problem.make_dataset(
        n_users=opts.n_users, 
        n_facilities=opts.n_facilities, 
        num_samples=opts.val_size,
        filename='data/MCLP_1000_20_tezheng_Normalization.pkl', 
        p=opts.p,
        r=opts.r, 
        distribution=opts.data_distribution
    )

    # 处理恢复训练
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming hierarchical training after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # 添加分层相关的超参数
    if not hasattr(opts, 'option_loss_weight'):
        opts.option_loss_weight = 0.5  # 选项损失的权重

    # 开始训练或评估
    if opts.eval_only:
        print("Evaluating hierarchical model only...")
        validate(model, val_dataset, opts)
    else:
        print("Starting hierarchical training...")
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    # 获取分层强化学习的选项
    opts = get_hierarchical_options()
    run(opts) 
#!/usr/bin/env python

# import json
# import pprint as pp
# import torch
# import torch.optim as optim
# from tensorboard_logger import Logger as TbLogger

# from hierarchical_options import get_hierarchical_options
# from hierarchical_train import train_epoch, validate
# from nets.hierarchical_reinforce_baselines import (
#     HierarchicalRolloutBaseline, 
#     HierarchicalExponentialBaseline,
#     HierarchicalCriticBaseline,
#     ActionCritic,
#     OptionCritic
# )
# from nets.hierarchical_attention_model import HierarchicalAttentionModel
# from utils import torch_load_cpu, get_inner_model
# # 直接导入MCLP问题
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ReCovNet-master'))
# try:
#     from problems.MCLP.problem_MCLP import MCLP
#     def load_problem(name):
#         if name == 'MCLP':
#             return MCLP()
#         else:
#             raise ValueError(f"Unknown problem: {name}")
# except ImportError:
#     # ... (Mock problem definition remains the same) ...
#     pass


# def run(opts):
#     pp.pprint(vars(opts))

#     tb_logger = None
#     if not opts.no_tensorboard:
#         tb_logger = TbLogger(os.path.join(
#             opts.log_dir, 
#             "{}_{}".format(opts.problem, opts.n_users), 
#             "hierarchical_" + opts.run_name
#         ))

#     os.makedirs(opts.save_dir, exist_ok=True)
    
#     with open(os.path.join(opts.save_dir, "hierarchical_args.json"), 'w') as f:
#         json.dump(vars(opts), f, indent=True)

#     opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
#     problem = load_problem(opts.problem)

#     load_data = {}
#     assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
#     load_path = opts.load_path if opts.load_path is not None else opts.resume
#     if load_path is not None:
#         print('  [*] Loading hierarchical data from {}'.format(load_path))
#         load_data = torch_load_cpu(load_path)

#     print("Initializing Hierarchical Attention Model...")
#     model = HierarchicalAttentionModel(
#         opts.embedding_dim,
#         opts.hidden_dim,
#         problem,
#         n_encode_layers=opts.n_encode_layers,
#         n_decode_layers=getattr(opts, 'n_decode_layers', 2),
#         mask_inner=True,
#         mask_logits=True,
#         normalization=opts.normalization,
#         tanh_clipping=opts.tanh_clipping,
#         checkpoint_encoder=opts.checkpoint_encoder,
#         shrink_size=opts.shrink_size,
#         max_facilities_per_step=getattr(opts, 'max_facilities_per_step', 5),
#         dy=False
#     ).to(opts.device)

#     if opts.use_cuda and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model)

#     model_ = get_inner_model(model)
#     model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

#     # --- MODIFICATION START: Initialize baseline and optimizer based on type ---
    
#     baseline_type = opts.hierarchical_baseline
#     if baseline_type == 'exponential':
#         baseline = HierarchicalExponentialBaseline(
#             beta=opts.exp_beta,
#             option_beta=getattr(opts, 'option_exp_beta', opts.exp_beta)
#         )
#     elif baseline_type == 'critic':
#         shared_encoder = get_inner_model(model).embedder
#         main_model = get_inner_model(model)
        
#         action_critic = ActionCritic(
#             encoder=shared_encoder,
#             embedding_dim=opts.embedding_dim,
#             hidden_dim=opts.critic_hidden_dim,
#             n_process_blocks=0
#         ).to(opts.device)
        
#         option_critic = OptionCritic(
#             encoder=shared_encoder,
#             embedding_dim=opts.embedding_dim,
#             hidden_dim=opts.critic_hidden_dim,
#             n_process_blocks=0
#         ).to(opts.device)
        
#         # 复制主模型的init_embed权重到评论家网络
#         # 确保权重和偏置都被正确复制
#         with torch.no_grad():
#             action_critic.init_embed.weight.copy_(main_model.init_embed.weight)
#             action_critic.init_embed.bias.copy_(main_model.init_embed.bias)
#             option_critic.init_embed.weight.copy_(main_model.init_embed.weight)
#             option_critic.init_embed.bias.copy_(main_model.init_embed.bias)
        
#         baseline = HierarchicalCriticBaseline(action_critic, option_critic)
#     else:  # 'rollout'
#         baseline = HierarchicalRolloutBaseline(model, problem, opts)

#     if 'baseline' in load_data:
#         baseline.load_state_dict(load_data['baseline'])

#     print("Setting up optimizer with different learning rates...")
#     optimizer_params = []
    
#     optimizer_params.append({
#         'params': get_inner_model(model).option_network.parameters(),
#         'lr': opts.lr_option
#     })
    
#     action_params = [param for name, param in get_inner_model(model).named_parameters() 
#                      if not name.startswith('option_network.')]
#     optimizer_params.append({'params': action_params, 'lr': opts.lr_model})
    
#     baseline_params = baseline.get_learnable_parameters()
#     if len(baseline_params) > 0:
#         optimizer_params.append({'params': baseline_params, 'lr': opts.lr_critic})

#     optimizer = optim.Adam(optimizer_params)
    
#     # --- MODIFICATION END ---

#     if 'optimizer' in load_data:
#         optimizer.load_state_dict(load_data['optimizer'])
#         for state in optimizer.state.values():
#             for k, v in state.items():
#                 if torch.is_tensor(v):
#                     state[k] = v.to(opts.device)

#     lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

#     val_dataset = problem.make_dataset(
#         n_users=opts.n_users, 
#         n_facilities=opts.n_facilities, 
#         num_samples=opts.val_size,
#         filename='data/MCLP_1000_20_tezheng_Normalization.pkl', 
#         p=opts.p,
#         r=opts.r, 
#         distribution=opts.data_distribution
#     )

#     if opts.resume:
#         epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
#         torch.set_rng_state(load_data['rng_state'])
#         if opts.use_cuda:
#             torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
#         baseline.epoch_callback(model, epoch_resume)
#         print("Resuming hierarchical training after {}".format(epoch_resume))
#         opts.epoch_start = epoch_resume + 1

#     if opts.eval_only:
#         print("Evaluating hierarchical model only...")
#         validate(model, val_dataset, opts)
#     else:
#         print("Starting hierarchical training...")
#         for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
#             train_epoch(
#                 model,
#                 optimizer,
#                 baseline,
#                 lr_scheduler,
#                 epoch,
#                 val_dataset,
#                 problem,
#                 tb_logger,
#                 opts
#             )


# if __name__ == "__main__":
#     opts = get_hierarchical_options()
#     run(opts)