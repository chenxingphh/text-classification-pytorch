'''
@Author: Haihui Pan
@Date: 2021-12-17
@Desc: Transformer-Encoder的实现
@Ref: Attention is all you need-Neurips2018
'''
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class FFN(nn.Module):
    '''FFN(包含残差&Layer Norm)'''

    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()

        # FFN模块
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model)
        )

        # layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: batch, max_len, d_model
        residual = x

        # FFN前馈计算
        x = self.ffn(x)

        # 残差&Layer Norm
        x = x + residual
        x = self.norm(x)

        return x


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        """

        :param d_model:
        :param n_head:
        :param scale: 是否scale输出
        """
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head

        # 输入的维度是d_model,输出的是三倍的
        self.qkv_linear = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        # 缩放因子
        self.scale = math.sqrt(d_model // n_head)

    def forward(self, x, mask=None):
        """

        :param x: bsz x max_len x d_model
        :param mask: bsz x max_len
        :return:
        """
        # 获取输入size
        batch_size, max_len, d_model = x.size()

        # 将输入的x进行线性变换
        x = self.qkv_linear(x)

        # 将x切分为q,k,v
        q, k, v = torch.chunk(x, 3, dim=-1)

        # 将q,k,v按照给定的head再进行划分
        # 将q转换为: (batch, head, input_len, dim)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        # 将k先转换为(batch, head, input_len, dim)再进行转置(batch, head, dim, input_len)
        # K^T
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)

        # 将v转换为: batch, head, input_len, dim
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        # 进行QK: (batch, head, input_len, input_len)
        attn = torch.matmul(q, k)
        attn = attn / self.scale

        # 进行mask填充: mask为0的地方使用'-inf'进行填充，经过softmax之后会趋近于0
        if mask != None:
            attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        # 进行softmax操作
        attn = F.softmax(attn, dim=-1)
        # 对attn进行dropout操作
        attn = self.dropout_layer(attn)

        # (batch, head, input_len, dim)
        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head

        # 先将head,input_len轴进行交换，在合并最后的dim
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)

        # 进行一层fc连接
        v = self.fc(v)

        return v


class PositionalEncoding(nn.Module):
    '''绝对位置编码'''

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # shape:(max_len,1) value: [[0],[1]...,[max_len-1]]
        position = torch.arange(max_len).unsqueeze(dim=1)

        '''
        PE(pos,2i)=sin(pos/10000^(2i/d_model))
        PE(pos,2i+1)=cos(pos/10000^(2i/d_model))

        1、pos/10000^(2i/d_model)=pos*10000^-(2i/d_model)
        2、10000^-(2i/d_model)=exp[log(10000^-(2i/d_model))]=-(2i/d_model)*log(10000)=-2i*log(10000)/d_model
        3、2i就是给定维度上的偶数列 
           2i=torch.arange(0, d_model, 2)
        '''
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Position Embedding(第一个维度是用于shape不一致时的board casting)
        pe = torch.zeros(1, max_len, d_model)

        # 一个位置上的偶数维度
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # 每一个位置上的奇数维度
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # 将pe保存为常量，使得模型在进行参数更新时不会更新pe
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: shape [batch_size,seq_len,embedding_dim]
        """
        x = x + self.pe  # 采用broadcasting进行相加
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, max_sent_len, padding_idx, num_class, d_model, n_head, dropout=0.1,
                 embedding_weight=None):
        super(TransformerEncoder, self).__init__()

        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
        # 使用预训练词向量初始化
        if embedding_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_weight)))

        # 单个句子的最大长度
        self.max_sent_len = max_sent_len

        # 位置编码
        self.position_enc = PositionalEncoding(d_model=d_model, max_len=max_sent_len)

        # 自注意力维度
        self.d_model = d_model
        # 自注意力机制
        self.atten = MultiHeadAttn(d_model, n_head, dropout)

        self.ffn = FFN(d_model, d_model * 2)

        self.norm1 = nn.LayerNorm(d_model)

        # 输出层
        self.pred = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model, num_class)
                                  )

    def forward(self, x, mask=None):
        # x: batch, input_len
        # mask: batch, input_len

        # x: batch,input_len, dim
        x = self.embedding(x)

        # x: batch,input_len, dim
        x = self.position_enc(x)

        # 多头自注意力机制
        # x: batch,max_sent_len, d_model
        residual = x
        x = self.atten(x, mask)
        x = x + residual
        x = self.norm1(x)

        # FFN
        x = self.ffn(x)

        # --------------
        # 最大池化操作
        # --------------
        # mask: batch, seq_len, 1
        mask = mask.unsqueeze(dim=-1)

        # x: batch, hidden_dim
        x = x.masked_fill(mask == 0, -float('inf')).max(dim=1)[0]

        # 输出层
        logits = self.pred(x)

        return logits


def model_forward_test():
    vocab_size, embed_dim, max_sent_len, padding_idx, num_class, d_model, n_head = 2000, 32, 100, 0, 3, 32, 2
    model = TransformerEncoder(vocab_size, embed_dim, max_sent_len, padding_idx, num_class, d_model, n_head,
                               dropout=0.1, )

    x = torch.ones((10, max_sent_len), dtype=torch.int)
    mask = torch.ones((10, max_sent_len), dtype=torch.int)
    model(x, mask)


if __name__ == '__main__':
    model_forward_test()
