'''
@Author: Haihui Pan
@Date: 2021-12-15
@Desc: RCNN模型实现
@Ref: Recurrent Convolutional Neural Networks for Text Classification-AAAI2015
'''
import torch
from torch import nn
import numpy as np


class RCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, input_len, num_class, padding_idx, dropout=0.2, embedding_weight=None):
        super(RCNN, self).__init__()

        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
        # 使用预训练词向量初始化
        if embedding_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_weight)))

        # 输入文本长度
        self.input_len = input_len
        # 词向量维度
        self.embed_dim = embed_dim
        # RNN神经元数目，双向则*2
        self.hidden_dim = embed_dim // 4
        # Bi-GRU层
        self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        # 用于对拼接的结果进行线性变换
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim * 2 + embed_dim, self.hidden_dim),
                                nn.Tanh()
                                )

        # 输出层
        self.pred = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(self.hidden_dim, num_class)
                                  )

    def forward(self, x, mask):
        # x: batch, seq_len
        # mask: batch, seq_len

        # x: batch, seq_len, embedding_dim
        x = self.embedding(x)

        # x: batch, seq_len, hidden_dim*2
        gru_x, _ = self.gru(x)

        # 拼接gru_x,x
        # x: batch, seq_len, 2*hidden_dim+embedding_dim
        x = torch.concat([x, gru_x], dim=-1)

        # 进行线性变换+tanh
        # x: batch, seq_len, hidden_dim
        x = self.fc(x)

        # --------------
        # 最大池化操作
        # --------------
        # mask: batch, seq_len, 1
        mask = mask.unsqueeze(dim=-1)

        # x: batch, seq_len, hidden_dim
        x = x.masked_fill(mask == 0, -float('inf')).max(dim=1)[0]

        logit = self.pred(x)
        return logit


def model_forward_test():
    batch, input_len, embedding_dim = 10, 200, 128
    num_class = 3
    x = torch.ones((10, 200), dtype=torch.int)
    model = RCNN(12000, embedding_dim, input_len, num_class, padding_idx=0)
    model(x, x)


if __name__ == '__main__':
    model_forward_test()
