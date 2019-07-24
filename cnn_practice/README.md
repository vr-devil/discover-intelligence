# 建模日志
## 阶段一：使用 Keras 高级 API 创建模型
- 第一个模型简单起见，能让模型能正常运行就成；因此网络结构为：输入层、一个卷积层、一个MaxPooling层、一个 Dropout 层、一个稠密层、输出层；卷积层使用 ReLU 做为激活函数，稠密连接层和输出层使用 Sigmoid 作为激活函数，最后使用 CrossEntropy 做为损失函数，使用随机梯度下降进行模型的训练方法；在该模型下准确率为89.42%。参见模型代码 [96ec43d - keras_model.py](https://github.com/kai-zhong/discover-intelligence/blob/96ec43d97740f59f81405845e80f6aebd1700bd2/cnn_practice/keras_model.py)
- 在上一个简单模型基础上调整参数很难超过90%的准确度，所以把网络结构做了调整，使用了两个卷积层，两个稠密层并将激活函数改为ReLU，改变网络结构的同时将参数规模扩大后，达到了90.97%的精确度，算是前进了一小步。参见模型代码 [8239e08 - keras_model.py](https://github.com/kai-zhong/discover-intelligence/blob/8239e08077dba39ef9da7c30b40974cbd8a50e9a/cnn_practice/keras_model.py)
