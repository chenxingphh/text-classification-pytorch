'''
@Author: Haihui Pan
@Date: 2022-6-3
@Desc: 对IMDB数据集进行预处理
'''

from torchtext.data.utils import get_tokenizer
import torch
from torch.utils.data import Dataset, TensorDataset
import jieba
import re
from tqdm import tqdm
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from torchtext.vocab import GloVe
from torchtext.datasets import IMDB


class IMDB_Dataset(Dataset):
    '''用于非预训练模型'''

    def __init__(self, mode, mask=False, dim=None):
        # 获取分词器
        self.tokenizer = get_tokenizer('basic_english')
        # 'train', 'test', 'dev'
        self.mode = mode
        # 是否需要返回输入word的mask
        self.mask = mask
        # 标签-数值映射
        self.label_dict = {'neg': 0, 'pos': 1}
        # 句子最大长度
        self.MAX_LEN = 300
        # 获取原始文本数据
        self.text_list, self.label_list = self.load_raw_data(mode)
        # 预训练词向量维度
        self.dim = dim
        # 文本词汇表
        self.train_text_vocab = self.build_vocab()
        # 文本词汇映射
        self.train_text_vocab_dict = self.train_text_vocab.get_stoi()
        # <pad>,<unk>的idx
        self.pad_idx = self.train_text_vocab_dict.get('<pad>')
        self.unk_idx = self.train_text_vocab_dict.get('<unk>')

    def load_raw_data(self, mode):
        '''
        加载原始文本数据集,只完成分词的部分
        :param mode:
        :return:
        '''
        if mode not in ['train', 'test']:
            raise ValueError('The value of mode can only be: train or test!')
        # 加载原始数据
        data_list = list(IMDB(split=mode))

        text_list, label_list = [], []
        for label, text in tqdm(data_list):
            # 对于文本从反向进行截取
            text_list.append(self.tokenizer(text)[-self.MAX_LEN:])
            label_list.append(self.label_dict[label])

        # 获取平均句子长度
        # s1_len_list = [len(s1) for s1 in text_list]
        # print('s1_len:', np.percentile(s1_len_list, q=(50, 75, 90, 99, 100)))
        # s1_len: [ 202.  329.  530. 1049. 2752.]

        return text_list, label_list

    def __len__(self):
        # 获取数据集数目
        return len(self.label_list)

    def build_vocab(self):
        '''
        创建词汇表
        :return:
        '''
        # 只能使用训练集的文本来创建词汇表
        if self.mode == 'train':
            text_list = self.text_list
        else:
            text_list, _ = self.load_raw_data(mode='train')

        # -----------------
        # 创建文本词汇表
        # -----------------
        # 转为迭代器
        iter_text = iter(text_list)
        # 创建词汇表
        train_text_vocab = build_vocab_from_iterator(iter_text, specials=["<unk>", "<pad>"])
        # 设置<unk>,用于oov的情况
        train_text_vocab.set_default_index(train_text_vocab["<unk>"])

        print('Finish building text vocab...')
        return train_text_vocab

    def get_pretrain_embedding(self):
        '''获取预训练的embedding'''

        if self.dim not in [50, 100, 200, 300]:
            raise ValueError('The value of dim can only be: 50, 100, 200 or 300!')

        # 获取从0-n对应的词汇
        word_vocb = [0] * len(self.train_text_vocab_dict)
        for k, v in self.train_text_vocab_dict.items():
            word_vocb[v] = k

        # 加载词向量
        glove = GloVe(name='6B', dim=self.dim)
        # glove = GloVe(name='840B', dim=self.dim)

        # 词向量
        vectors = []
        for i in range(len(word_vocb)):
            word = word_vocb[i]
            if word in glove.stoi.keys():
                vectors.append(glove.get_vecs_by_tokens(word).numpy())
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, self.dim))

        print('Finish loading pretrain embedding...')
        return vectors

    def __getitem__(self, idx):
        '''
        对文本进行数字化的过程
        :param idx:
        :return:
        '''
        text, label = self.text_list[idx], self.label_list[idx]

        # 文本数字化
        text_token = self.train_text_vocab(text)

        # text的mask
        text_mask = [1] * len(text_token)

        # padding填充
        text_padding = [self.pad_idx] * (self.MAX_LEN - len(text_token))
        text_token.extend(text_padding)

        text_mask_padding = [0] * (self.MAX_LEN - len(text_mask))
        text_mask.extend(text_mask_padding)

        # 长度截断
        text_token = text_token[:self.MAX_LEN]
        text_mask = text_mask[:self.MAX_LEN]

        # 转tensor
        text_token = torch.tensor(text_token)
        text_mask = torch.tensor(text_mask)
        label = torch.tensor(label)

        if self.mask:
            return text_token, text_mask, label
        else:
            return text_token, label


if __name__ == '__main__':
    train_data = IMDB_Dataset(mode='test', mask=True, dim=50)
    text_token, text_mask, label = train_data.__getitem__(9)
    print(text_token, text_mask, label)
    train_data.get_pretrain_embedding()
    print(len(train_data.label_dict))
