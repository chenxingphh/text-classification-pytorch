'''
@Author: Haihui Pan
@Date: 2021-12-15
@Desc: Bi-LSTM的实现
'''
import torch
from torch import nn
import numpy as np
import time


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, input_len, num_class, padding_idx, dropout=0.2, embedding_weight=None):
        super(BiLSTM, self).__init__()

        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
        # 使用预训练词向量初始化
        if embedding_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_weight)))

        # 输入文本长度
        self.input_len = input_len
        # 词向量维度
        self.embed_dim = embed_dim
        # LSTM神经元数目，双向则*2
        self.hidden_dim = embed_dim // 4
        # Bi-GRU层
        self.gru = nn.LSTM(embed_dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        # 全连接
        self.pred = nn.Sequential(nn.Linear(self.hidden_dim * 2, embed_dim),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(embed_dim, num_class)
                                  )

    def forward(self, x, mask):
        # x: batch, seq_len
        # mask: batch, seq_len

        # x: batch, seq_len, embedding_dim
        x = self.embedding(x)

        # x: batch, seq_len, hidden_dim*2
        x, _ = self.gru(x)

        # mask: batch, seq_len, 1
        mask = mask.unsqueeze(dim=-1)

        # 最大池化操作
        # x: batch, seq_len, hidden_dim*2
        x = x.masked_fill(mask == 0, -float('inf')).max(dim=1)[0]

        # 拼接第一个和最后一个输出
        # x: batch, seq_len, hidden_dim*4
        # x = torch.cat([x[:, 0, :], x[:, -1, :]], dim=-1)

        logit = self.pred(x)
        return logit


def model_forward_test():
    start_time = time.time()

    batch, input_len, embedding_dim = 10, 200, 128
    num_class = 3
    x = torch.ones((batch, 200), dtype=torch.int)
    mask = torch.ones((batch, 200), dtype=torch.int)
    model = BiLSTM(12000, embedding_dim, input_len, num_class, padding_idx=0)
    model(x, mask)

    end_time = time.time()

    # 1w个样本：8.86s
    print('time cost:', end_time - start_time)


if __name__ == '__main__':
    model_forward_test()
