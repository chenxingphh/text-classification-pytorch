'''
@Author: Haihui Pan
@Date: 2021-12-15
@Desc: DPCNN的实现
@Ref: Deep Pyramid Convolutional Neural Networks for Text Categorization -ACL 2017
'''
import torch
from torch import nn
import numpy as np


class DPCNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, input_len, num_class, padding_idx, dropout=0.2, embedding_weight=None):
        super(DPCNN, self).__init__()
        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # 使用预训练词向量初始化
        if embedding_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_weight)))

        self.out_channels = 4

        # region embedding
        self.region_conv = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(1, embed_dim),
                                     stride=1, padding=(0, 0))  # (垂直，水平)
        # conv block
        self.conv_1 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3, 1),
                                stride=1, padding=(1, 0))
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3, 1),
                                stride=1, padding=(1, 0))

        # resnet block
        self.res_conv_1 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3, 1),
                                    stride=1, padding=(1, 0))
        self.res_conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3, 1),
                                    stride=1, padding=(1, 0))

        # 激活函数
        self.activation = torch.nn.ReLU()

        # 拉平层
        self.flat = nn.Flatten()

        # 输出层
        self.pred = nn.Sequential(nn.Linear(self.out_channels * (input_len // 4), embed_dim),
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

        # region embedding
        x = self.region_conv(x)
        x = self.activation(x)

        # max pooling
        x = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=2)(x)

        # resnet-block
        x_ = self.conv_1(x)
        x_ = self.activation(x_)
        x_ = self.conv_2(x_)
        x_ = self.activation(x_)
        x = x + x_

        # resnet-block
        x_ = self.res_conv_1(x)
        x_ = self.activation(x_)
        x_ = self.res_conv_2(x_)
        x_ = self.activation(x_)
        x = x + x_

        # 最大池化
        x = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=2)(x)

        # 拉平
        x = self.flat(x)

        logit = self.pred(x)
        return logit
