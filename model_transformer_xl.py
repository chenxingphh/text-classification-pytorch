'''
@Author: Haihui Pan
@Date: 2022-3-14
@Desc: Transformer-XL的Encoder
@Ref: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context 2019ACL
'''
import torch
from torch import nn
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random


class RelativePositionalEmbedding(nn.Module):
    '''Transformer-XL中的相对位置编码'''

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0

        # 将权重设置为常量 (input_len+1, dim )
        weights = self.get_embedding(
            init_size + 1,  # 相对位置的范围
            embedding_dim,
            padding_idx,
        )

        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """返回从[-num_embeddings,num_embeddings]范围的位置编码，(先前的就只有[0,max_len])
        """

        half_dim = embedding_dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)

        # pos*embedding
        emb = torch.arange(-num_embeddings // 2, num_embeddings // 2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)

        # embedding_dim设置为奇数时，多出的一个维度设置为0
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        # 由于nn.embedding里面不允许存在负的index,因此只能对返回的num_embedding进行偏移了！)
        self.origin_shift = num_embeddings // 2 + 1  # 代表的是中心点的距离！！！
        return emb

    def forward(self, x):
        # x: batch, seq_len

        bsz, seq_len = x.size()

        # 最长位置
        max_pos = self.padding_idx + seq_len  # 通常padding_idx为0,因此就等同于

        # 最长位置>原始偏移
        if max_pos > self.origin_shift:  # self.original=601
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,  # 因为有左右偏移的，所以要*2
                self.embedding_dim,
                self.padding_idx, )

            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        # [583-618]
        positions = torch.arange(-seq_len, seq_len).to(x.device).long() + self.origin_shift  # 2*seq_len

        # 选择[-seq_len, seq_len]+self.origin_shift对应idx上的embedding
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        # qkv线性变换矩阵
        self.qkv_linear = nn.Linear(d_model, d_model * 3, bias=False)
        # 多头数目
        self.n_head = n_head
        # head dim
        self.head_dim = d_model // n_head
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)

        # 相对位置编码(1200相当于特别长的距离)
        self.pos_embed = RelativePositionalEmbedding(d_model // n_head, 0, 1200)

        # r_r_bias就是v, r_w_bias就是u
        self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))

        # 添加transformer中的W_k,R矩阵
        self.w_kr = nn.Parameter(nn.init.xavier_normal_(torch.zeros(d_model // n_head, d_model // n_head)))

    def forward(self, x, mask=None):
        # x: batch, seq_len, d_model
        # mask: batch, seq_len

        batch_size, max_len, d_model = x.size()

        # mask: (batch,max_len) (相对位置编码)
        # pos_embed: (2*max_len, d_model//head)
        pos_embed = self.pos_embed(mask)  # l x head_dim

        # q,k,v: batch, n_head, seq_len, dim
        qkv = self.qkv_linear(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        rw_head_q = q + self.r_r_bias[:, None]
        # 相当于将k进行转置，再跟rw_head_q进行相乘（就相差了一个转换矩阵w_kr）
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])
        rel_pos_embed = pos_embed.matmul(self.w_kr)
        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, rel_pos_embed)[None, :, None]
        B_ = torch.einsum('bnqd,ld->bnql', q, rel_pos_embed)
        BD = B_ + D_
        BD = self._shift(BD)
        attn = AC + BD

        # 除于缩放因子
        attn = attn / self.scale

        # 进行mask填充
        if mask != None:
            attn = attn.masked_fill(mask[:, None, None, :].eq(0), 1e-9)

        # 归一化
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)
        return v

    def _shift(self, BD):
        """
        transformer-XL中提出的计算相对位置的Tip: 即先计算每个token的任意相对位置，再通过shift来得到目标token的相对位置
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        转换为
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch, n_head, seq_len, 2seq_len
        :return: batch, n_head, seq_len, seq_len
        """

        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, feedforward_dim, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout)

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        # ----------------
        # 自注意力机制
        # ----------------
        residual = x
        x = self.self_attn(x, mask)
        # 残差 & layer norm归一化
        x = x + residual
        x = self.norm1(x)

        # -----------------
        # FFN
        # -----------------
        residual = x
        x = self.ffn(x)
        # 残差 & layer norm归一化
        x = residual + x
        x = self.norm2(x)

        return x


class TransformerXLEncoder(nn.Module):
    '''Transformer-XL的Encoder'''

    def __init__(self, vocab_size, embed_dim, max_sent_len, padding_idx, num_class, d_model, n_head, dropout=0.1,
                 embedding_weight=None):
        super(TransformerXLEncoder, self).__init__()

        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
        # 使用预训练词向量初始化
        if embedding_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_weight)))

        # 单个句子的最大长度
        self.max_sent_len = max_sent_len

        # 自注意力维度
        self.d_model = d_model

        # 自注意力机制
        self.attn = TransformerLayer(d_model, n_head, feedforward_dim=d_model, dropout=0.01)

        # 输出层
        self.pred = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model, num_class)
                                  )

    def forward(self, x, mask=None):
        # x: bacth, input_len
        # mask: batch, input_len

        # x: batch, input_len, dim
        x = self.embedding(x)

        # -----------------------------
        # 相对位置编码下的自注意力机制
        # -----------------------------
        x = self.attn(x, mask)

        # --------------
        # 最大池化操作
        # --------------
        # mask: batch, seq_len, 1
        mask = mask.unsqueeze(dim=-1)

        # x: batch, hidden_dim
        x = x.masked_fill(mask == 0, -float('inf')).max(dim=1)[0]

        logit = self.pred(x)

        return logit


def model_forward_test():
    vocab_size, embed_dim, max_sent_len, padding_idx, num_class, d_model, n_head = 2000, 32, 100, 0, 3, 32, 2
    model = TransformerXLEncoder(vocab_size, embed_dim, max_sent_len, padding_idx, num_class, d_model, n_head,
                                 dropout=0.1, )

    x = torch.ones((10, max_sent_len), dtype=torch.int)
    mask = torch.ones((10, max_sent_len), dtype=torch.int)
    model(x, mask)


if __name__ == '__main__':
    model_forward_test()
