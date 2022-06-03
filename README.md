# text-classification-pytorch
基于Pytorch实现大多数常见的文本分类模型

## 实现的分类模型
* FastText: Bag of Tricks for Efficient Text Classification
* DPCNN: Deep Pyramid Convolutional Neural Networks for Text Categorization 
* TextCNN: Convolutional Neural Networks for Sentence Classification
* RCNN: Recurrent Convolutional Neural Networks for Text Classification 
* HAN: Hierarchical Attention Networks for Document Classification 
* Bi-GRU: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
* Bi-LSTM: Long Short-term Memory
* Transformer-Encoder: Attention is all you need
* Transformer-XL-Encoder: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context 

## 数据集
IMDB是一个电影评论的二分类（pos, neg）数据集，专门用于情绪分析。IMDB评级< 5则表示负面评论，而评级>=7的情绪得分为1。IMDB的训练集数目为25000，测试集数目
为25000，没有额外提供验证集。


