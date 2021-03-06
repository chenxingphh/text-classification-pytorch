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
    IMDB是一个电影评论的二分类（pos, neg）数据集，专门用于情绪分析。
    IMDB的训练集数目为25000，测试集数目为25000，没有额外提供验证集。

## 超参数
    seed: 512
    epoch: 20
    optimizer: Adam
    warm_up_proportion: 0.1
    lr: 0.0005
    batch_size: 128
    pretrain_embedding: Glove-6B
    
 ## 模型效果
 * 对于非预训练模型，这里使用了不同维度的Glove-6B预训练词向量，维度包括50, 100, 200
 
| Model     | Test(dim=50)   | Test(dim=100)| Test(dim=200)|
| ----------- | ----------- |-----------|-----------|
| FastText   |  0.8809       |   0.8830     |   0.8836        |     
| DPCNN   |     0.8648    |   0.8722     |    0.8798     |   
| TextCNN  |  0.8680       |     0.8689   |   0.8808        | 
| RCNN   |     0.8716    |     0.8864   |       0.9006    | 
|  HAN  |  0.8852      |  0.8904      |    0.8973       | 
|  Bi-GRU  |    0.8535     |    0.8715    |    0.8873       | 
|  Bi-LSTM  |    0.8611    |    0.8708    |    0.8871     | 
|  Transformer-Encoder  |  0.8650       |   0.8670     |      0.8774     | 
|  Transformer-XL-Encoder  |   0.8751      |  0.8844      |  0.8791         | 

## License
[MIT LICENSE](LICENSE)
