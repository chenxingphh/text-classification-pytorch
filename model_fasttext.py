'''
@Author: Haihui Pan
@Date: 2021-12-15
@Desc: FastText的实现
@Ref: Bag of Tricks for Efficient Text Classification-2016
'''
import torch
from torch import nn
import numpy as np


class FastText(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, padding_idx, dropout=0.2, embedding_weight=None):
        super(FastText, self).__init__()
        # 词向量矩阵
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, padding_idx=padding_idx)
        # 使用预训练词向量初始化
        if embedding_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_weight)))

        self.pred = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(embed_dim, num_class)
                                  )

    def forward(self, x):
        # x: batch,input_len

        # x: batch,input_len,dim
        x = self.embedding(x)

        # 预测结果
        logit = self.pred(x)

        return logit
