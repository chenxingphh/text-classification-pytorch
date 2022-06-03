'''
@Author: Haihui Pan
@Date: 2022-6-2
@Desc: 固定随机种子用于复现结果
'''
import torch
import random
import numpy as np


def seed_torch(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
