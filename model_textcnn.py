'''
@Author: Haihui Pan
@Date: 2021-12-02
@Date: TextCNN的搭建
@Ref: Convolutional Neural Networks for Sentence Classification- EMNLP 2014
'''
import torch
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, input_len, num_class, padding_idx, dropout=0.2, embedding_weight=None):
        super(TextCNN, self).__init__()
        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # 使用预训练词向量初始化
        if embedding_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_weight)))

        # 输入文本长度
        self.input_len = input_len
        # 词向量维度
        self.embed_dim = embed_dim
        # 卷积核数目
        self.out_channels = 2

        # 使用不同大小的卷积核对输入文本进行卷积（2,3,4共6个卷积核）
        self.c1 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(2, self.embed_dim),
                            stride=1)
        self.c2 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, self.embed_dim),
                            stride=2)
        self.c3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(4, self.embed_dim),
                            stride=2)

        self.conv_list = nn.ModuleList([self.c1, self.c2, self.c3])

        # 激活函数
        self.activation = torch.nn.ReLU()

        # 池化层
        self.pooling_len = input_len // 2
        self.avg_pool = torch.nn.AdaptiveMaxPool2d((self.pooling_len, 1))

        # 拉平层
        self.flat = nn.Flatten()

        # 输出层
        self.pred = nn.Sequential(nn.Linear(self.out_channels * len(self.conv_list) * self.pooling_len, embed_dim),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(embed_dim, num_class)
                                  )

    def forward(self, text):
        # text: batch,input_len

        # x: batch,input_len, dim
        x = self.embedding(text)

        # x: batch,1,input_len,dim
        x = torch.unsqueeze(x, dim=1)

        # 进行卷积
        conv_result = []
        for conv in self.conv_list:
            # x_: batch, kernel_num, (input_len - size) / stride + 1, 1
            x_ = conv(x)
            x_ = self.activation(x_)

            # x_: batch, out_channel, self.avg_len, 1
            x_ = self.avg_pool(x_)

            # x_: batch, out_channel, self.avg_len
            x_ = torch.squeeze(x_, dim=-1)
            conv_result.append(x_)

        # x_combine: batch, out_channel*len(conv_result), self.avg_len
        x_combine = torch.cat(conv_result, dim=1)

        # x_combine: batch,out_channel*3*self.avg_len
        x_combine = self.flat(x_combine)
        logit = self.pred(x_combine)

        return logit
