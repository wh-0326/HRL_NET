# state_MCLP.py (修正版)

import torch
from typing import NamedTuple, Optional
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateMCLP(NamedTuple):
    # 固定输入
    users: torch.Tensor
    facilities: torch.Tensor
    p: torch.Tensor
    radius: torch.Tensor
    dist: torch.Tensor

    # 实例ID
    ids: torch.Tensor

    # 动态状态
    prev_a: torch.Tensor
    visited_: torch.Tensor  # 标记已选择的设施 (facility)
    mask_cover: torch.Tensor  # 标记已覆盖的用户 (user) - 注意：原代码的命名可能易混淆
    dynamic: torch.Tensor
    solution: torch.Tensor  # 【修正】预分配固定大小张量
    cover_num: torch.Tensor
    i: torch.Tensor  # 跟踪每个样本已选择的设施数量

    @property
    def visited(self):
        # 这是一个辅助属性，将长整型掩码转换为布尔型
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            # 注意: 此处 n 的大小应为设施数量
            return mask_long2bool(self.visited_, n=self.facilities.size(1))

    @staticmethod
    def initialize(data, visited_dtype=torch.bool):
        users = data["users"]
        facilities = data["facilities"]
        p = data['p'][0]
        radius = data['r'][0]
        batch_size, n_users, _ = users.size()
        _, n_facilities, _ = facilities.size()

        dist = (facilities[:, :, None, :2] - users[:, None, :, :]).norm(p=2, dim=-1)

        # 【修正】预分配 solution 张量，用-1等特殊值填充
        solution = torch.full((batch_size, p), -1, dtype=torch.long, device=facilities.device)

        return StateMCLP(
            users=users,
            facilities=facilities,
            p=p,
            radius=radius,
            dist=dist,
            ids=torch.arange(batch_size, dtype=torch.int64, device=facilities.device)[:, None],
            visited_=(
                torch.zeros(batch_size, 1, n_facilities, dtype=torch.bool, device=users.device)
            ),
            mask_cover=(
                torch.zeros(batch_size, 1, n_users, dtype=torch.bool, device=users.device)
            ),
            dynamic=torch.ones(batch_size, 1, n_facilities, dtype=torch.float, device=facilities.device),
            solution=solution,
            cover_num=torch.zeros(batch_size, 1, dtype=torch.long, device=facilities.device),
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=facilities.device),
            i=torch.zeros(batch_size, 1, dtype=torch.long, device=facilities.device)
        )

    def get_cover_num(self, facility_solution: torch.Tensor):
        """
        【重写】根据给定的设施方案计算覆盖的用户数。
        这个版本兼容我们预分配的、含有-1填充值的solution张量。
        """
        batch_size, n_users, _ = self.users.size()

        # 创建一个掩码，标记哪些是有效的已选设施 (不等于-1)
        valid_facilities_mask = facility_solution != -1

        # 初始化一个全False的用户覆盖掩码
        is_user_covered = torch.zeros(batch_size, n_users, dtype=torch.bool, device=self.users.device)

        # 遍历p个可能的位置
        for i in range(facility_solution.size(1)):
            # 获取当前步骤选择的设施，以及哪些样本在这一步有有效选择
            step_facilities = facility_solution[:, i]
            step_active_mask = valid_facilities_mask[:, i]

            # 如果当前步骤中至少有一个样本做出了有效选择
            if step_active_mask.any():
                # self.dist shape: [B, N_fac, N_user]
                # 获取有效选择的设施到所有用户的距离
                # f_u_dist_active shape: [num_active, N_user]
                f_u_dist_active = self.dist[step_active_mask].gather(
                    1, step_facilities[step_active_mask].view(-1, 1, 1).expand(-1, -1, n_users)
                ).squeeze(1)

                # 判断这些距离是否在半径内，得到覆盖关系
                step_cover_active = (f_u_dist_active < self.radius)

                # 使用 "logical or" 更新用户总覆盖掩码
                # 只更新那些在当前步做出了有效选择的样本
                is_user_covered[step_active_mask] |= step_cover_active

        # 计算总覆盖数 (每个样本覆盖了多少用户)
        cover_num = is_user_covered.long().sum(dim=-1)
        return cover_num.unsqueeze(-1)

    def update(self, selected: torch.Tensor, active_mask: torch.Tensor):
        """
        【最终重构版】带有调试信息的、最稳健的update函数
        """
        # --- 调试信息: 打印输入形状 ---
        # print("\n--- Entering state.update ---")
        # print(f"Input 'selected' shape: {selected.shape}")
        # print(f"Input 'active_mask' shape: {active_mask.shape}")
        # print(f"Current step 'self.i' (sum): {self.i.sum().item()}")

        # 克隆所有将被修改的状态张量，确保操作的安全性
        new_prev_a = self.prev_a.clone()
        new_visited_ = self.visited_.clone()
        new_solution = self.solution.clone()
        new_i = self.i.clone()

        # 仅当有样本需要更新时才执行操作
        if active_mask.any():
            # --- 核心逻辑：将布尔掩码转换为整数索引 ---
            batch_idx_active = torch.where(active_mask)[0]

            # 提取出仅针对 active 样本的数据
            selected_active = selected[active_mask]
            step_indices_active = self.i[active_mask].squeeze(-1)

            # --- 调试信息: 打印索引信息 ---
            # print(f"Number of active samples: {len(batch_idx_active)}")
            # print(f"Active batch indices (sample): {batch_idx_active[:5]}")
            # print(f"Selections for active samples (sample): {selected_active[:5]}")
            # print(f"Step indices for active samples (sample): {step_indices_active[:5]}")

            # 1. 更新 prev_a
            new_prev_a[batch_idx_active] = selected_active.unsqueeze(-1)

            # 2. 更新 visited_ (已选设施掩码)
            new_visited_[batch_idx_active, 0, selected_active] = True

            # 3. 更新 solution (解序列)
            new_solution[batch_idx_active, step_indices_active] = selected_active

            # 4. 更新 i (步数计数器)
            new_i[batch_idx_active] += 1

        # 5. 基于更新后的完整解，重新计算覆盖数 (逻辑参考原始版本)
        new_cover_num = self.get_cover_num(new_solution)

        # print("--- Exiting state.update ---\n")

        return self._replace(
            prev_a=new_prev_a,
            visited_=new_visited_,
            solution=new_solution,
            cover_num=new_cover_num,
            i=new_i
        )

    def all_finished(self):
        # 当所有样本的步数都达到 p 时，整个批次完成
        return (self.i >= self.p).all()

    def get_finished(self):
        # 【新增】返回每个样本是否完成的布尔掩码
        return self.i >= self.p

    def get_mask(self):
        # 返回已选设施的掩码，用于防止重复选择
        # 确保返回的是布尔型
        if self.visited_.dtype == torch.bool:
            return self.visited_.clone()
        else:
            return mask_long2bool(self.visited_, n=self.facilities.size(1))