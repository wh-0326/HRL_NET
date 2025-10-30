from PIL.PngImagePlugin import is_cid
from torch.utils.data import Dataset
import torch
import os
import pickle
import math
import  numpy as np
from problems.MCLP.state_MCLP import StateMCLP



class MCLP(object):
    NAME = 'MCLP'
    @staticmethod
    def get_total_num(dataset, pi):
        w1 = 0.6
        w2 = 0.4
        da = 1000 / 38686.093359901104
        db = 2000 / 38686.093359901104
        users = dataset['users']
        facilities = dataset['facilities']
        demand = dataset['demand']  # 假设形状为 [batch_size, n_user, 1]
        radius = dataset['r'][0]
        batch_size, n_user, _ = users.size()
        _, n_facilities, _ = facilities.size()
        _, p = pi.size()

        # 计算距离
        # dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)
        dist = (facilities[:, :, None, :2] - users[:, None, :, :]).norm(p=2, dim=-1)

        # 获取选中的设施与用户之间的距离
        facility_tensor = pi.unsqueeze(-1).expand(-1, -1, n_user)
        f_u_dist_tensor = dist.gather(1, facility_tensor)

        # 计算覆盖情况
        mask = f_u_dist_tensor < radius

        def F(dij, da, db):
            result = torch.zeros_like(dij)
            mask1 = dij <= da
            mask2 = (dij > da) & (dij <= db)
            mask3 = dij > db

            result[mask1] = 1.0
            if (db - da) > 1e-8:  # 避免除以零
                angle = (math.pi / (db - da)) * (dij[mask2] - (da + db) / 2) + math.pi / 2
                result[mask2] = 0.5 * torch.cos(angle) + 0.5
            result[mask3] = 0.0
            return result

        F_dij = F(f_u_dist_tensor, da, db)


        w_j = demand.squeeze(-1)

        # coverage_sum = (F_dij * mask).max(dim=1)[0]
        # print(f"coverage_sum: {coverage_sum},coverage_sum.min: {coverage_sum.min()}, coverage_sum.max: {coverage_sum.max()}")
        #
        # term1_sum = (w_j * (F_dij * mask).max(dim=1)[0]).sum(dim=1)  # [batch_size]
        #
        # term2 = (f_u_dist_tensor * mask.float()).sum(dim=(1, 2))  # [batch_size]
        #
        # # 5. 组合目标函数
        # weighted_covernum = w1 * term1_sum - w2 * term2
        #
        # return weighted_covernum

        # 1. 计算每个用户的最大收益值，以及提供该值的设施索引
        F_masked = F_dij * mask.float()  # [batch, p, n_user]
        best_F_values, best_facility_indices = F_masked.max(dim=1)  # [batch, n_user]

        # 2. 计算收益项 (您的 term1_sum 逻辑已经等价于此，可以保持)
        term1_sum = (w_j * best_F_values).sum(dim=1)  # [batch_size]

        # 3. 根据找到的最佳设施索引，提取对应的距离
        # f_u_dist_tensor 的形状是 [batch, p, n_user]
        # 我们需要从 p 这个维度上，根据 best_facility_indices 来选择距离
        best_distances = f_u_dist_tensor.gather(1, best_facility_indices.unsqueeze(1)).squeeze(1)  # [batch, n_user]

        # 4. 对于未被覆盖的用户，其距离成本应为0
        # best_F_values > 0 可以作为用户是否被覆盖的判断依据
        covered_mask = (best_F_values > 0).float()
        term2_sum = (best_distances * covered_mask).sum(dim=1)  # [batch_size]

        # 5. 组合目标函数
        weighted_covernum = w1 * term1_sum - w2 * term2_sum

        return weighted_covernum


    @staticmethod
    def make_dataset(*args, **kwargs):
        return MCLPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMCLP.initialize(*args, **kwargs)


class MCLPDataset(Dataset):
    def __init__(self, filename=None, n_users=50, n_facilities=20, num_samples=5000, offset=0, p=8, r=0.2, distribution=None):
        super(MCLPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [row for row in (data[offset:offset + num_samples])]
                p = self.data[0]['p']
                r = self.data[0]['r']
        else:
            # Sample points randomly in [0, 1] square
            self.data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                              facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                              demand=torch.FloatTensor(n_users, 1).uniform_(1, 10),
                              p=p,
                              r=r)
                         for i in range(num_samples)]

        self.size = len(self.data)
        self.p = p
        self.r = r

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]