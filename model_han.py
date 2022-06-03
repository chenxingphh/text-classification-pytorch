'''
@Author: Haihui Pan
@Date:2021-12-17
@Desc: 单层HAN模型
@Ref: Hierarchical Attention Networks for Document Classification - NAACL-2016
'''
import torch
from torch import nn
import numpy as np
import time


class WordAttNet(nn.Module):
    '''对句子进行Attention'''

    def __init__(self, vocab_size, embed_dim, max_sent_len, padding_idx, embedding_weight=None):
        super(WordAttNet, self).__init__()

        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
        # 使用预训练词向量初始化
        if embedding_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_weight)))

        # 单个句子的最大长度
        self.max_sent_len = max_sent_len
        # Bi-GRU神经元数目，双向则*2
        self.hidden_dim = embed_dim // 2
        # Bi-GRU层
        self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        # 用于对gru的结果进行线性变换的操作
        self.W = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)

        # 用于对齐的上下文向量
        self.context_vec = nn.Parameter(torch.rand(2 * self.hidden_dim, 1), requires_grad=True)

    def forward(self, x, word_mask=None):
        # x: batch, max_sent_len
        # mask: batch, max_sent_len

        # x: batch,max_sent_len, dim
        x = self.embedding(x)

        # x: batch,max_sent_len, hidden_dim*2
        x, _ = self.gru(x)

        # 注意力对齐  x:batch,hidden_dim*2
        x = self.word_attention(x, self.context_vec, word_mask)

        return x

    def word_attention(self, x, context_vec, mask=None):
        # x: batch,max_sent_len,dim
        # context_vec: dim,1
        # mask: batch, max_sent

        # 对x进行一次非线性变换 x: batch,max_sent_len,dim
        u = torch.tanh(self.W(x))
        # u = x

        # u与context_vec进行矩阵相乘，相当于点积, att_score:batch,max_sent_len,1
        att_score = u.matmul(context_vec)

        if mask != None:
            # mask: batch,max_sent, 1
            mask = torch.unsqueeze(mask, dim=-1)
            # 将mask=0的位置使用-1e9来进行替换
            att_score = att_score.masked_fill(mask == 0, -1e9)

        # 进行softmax操作, att_score:batch,max_sent_len,1
        att_score = torch.softmax(att_score, dim=1)

        # 对应未知的注意力权重乘上对应的向量,x: batch,max_sent_len,dim
        x = x * att_score

        # 将同一个句子的向量进行累加, x:batch,dim
        x = torch.sum(x, dim=1)

        return x


class HAN(nn.Module):
    '''Hierarchical Attention Network(只进行单层级的注意力对齐)'''

    def __init__(self, vocab_size, embed_dim, max_sent_len,  num_class, padding_idx,dropout=0.2, embedding_weight=None):
        super(HAN, self).__init__()

        # word级别的attention
        self.word_attn = WordAttNet(vocab_size, embed_dim, max_sent_len, padding_idx, embedding_weight=embedding_weight)

        # 输出层
        self.pred = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(embed_dim, num_class)
                                  )

    def forward(self, x, word_mask=None):
        # x: batch, seq_len
        # word_mask: batch, seq_len

        # x: batch,dim
        x = self.word_attn(x, word_mask)

        # logit: batch, num_class
        logit = self.pred(x)

        return logit


def model_forward_test():
    start_time = time.time()

    batch, input_len, embedding_dim = 10, 200, 128
    num_class = 3
    x = torch.ones((batch, 200), dtype=torch.int)
    mask = torch.ones((batch, 200), dtype=torch.int)
    model = HAN(1230, embedding_dim, input_len, padding_idx=0, num_class=num_class)
    model(x, mask)

    end_time = time.time()

    # 1w个样本：8.86s
    print('time cost:', end_time - start_time)


if __name__ == '__main__':
    model_forward_test()
