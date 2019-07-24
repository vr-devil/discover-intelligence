# 简介
**探索人工智能** 是一个学习项目，围绕项目主题了解什么是智能，有哪些人工智能的应用，以及如何实现人工智能。该项目记录学习过程中的想法，寻找到的资料，动手实践的成果。

# 学习路径
- [写给正在填报志愿并对CS/AI感兴趣的考生们-2019](https://zhuanlan.zhihu.com/p/68474477)
- 《Deep Learning》[Chapter 5 - Machine Learning Basics](http://www.deeplearningbook.org/contents/ml.html) / 第5章-机器学习基础 
- 《Deep Learning》[Chapter 9 - Convolutional Networks](http://www.deeplearningbook.org/contents/convnets.html) / 第9章-卷积网络，不适合初学者看，不过其中一节讲述了视觉皮层的结构和功能比较有趣。
- 《Neural Networks and Deep Learning》[Chapter 6 - Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html) / 第6章-深度学习，通过讲解卷积神经网络学习深度学习，适合初学者学习。
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 2012年这篇论文引发了深度学习的热潮，值得了解下。
- 《Neural Networks and Deep Learning》[Learning with gradient descent](http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent) / 学习梯度下降，简洁易懂。
- 《Neural Networks and Deep Learning》[Chapter 2 - How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html) / 第2章-向后传播算法，训练神经网络的基本算法，必须得了解。
- 《Neural Networks and Deep Learning》[Chapter 3 - Improving the way neural networks learn](http://neuralnetworksanddeeplearning.com/chap3.html) / 第3章-改善神经网络学习的方法，在该章节中了解交叉熵损失函数，正则化的概念以及一些可用的方法包括权重衰减(L2 正则化)、L1 正则化、Dropout。
- [Keras](https://www.tensorflow.org/guide/keras) TensorFlow官网的Keras入门指南；Keras是一个用于构建和训练深度学习模型的高级API；学习Keras为接下来动手实践做准备。
- [机器学习科研的十年](https://zhuanlan.zhihu.com/p/74249758)，对要走机器学习之路的同学应该能带来一些启示。
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)，了解了下 Fashion-MNIST 数据集，在卷积网络的练习中使用。

# 练习
## 卷积网络练习
练习分3个阶段：
- 第一阶段是通过 Keras 的高级 API 搭建网络模型，不断地调整网络结构和参数，直到在 Fashion-MNIST 的测试集取得92%的识别准确度后进入第二阶段；
- 第二阶段是将第一阶段的模型使用 TensorFlow 的低级 API 实现；
- 第三阶段是将第一阶段的模型使用纯 Python 实现。

Keras 模型 - [keras_model.py](https://github.com/kai-zhong/discover-intelligence/blob/master/cnn_practice/keras_model.py)， 当前模型在测试集的准确度为 0.8942。

# 应用
- [机甲大师 RoboMaster S1](https://www.dji.com/cn/robomaster-s1?site=brandsite&from=homepage) - 大疆机器人， 6 类人工智能编程模块。
- [Runway ML](https://runwayml.com/) - Machine learning for creators；一个工具类产品，有多种机器学习的模型，比如图像生成，动作捕捉等，用于设计创造类的工作。

# 问题 & 想法
- 人工智能的核心构件有哪些？[#2](https://github.com/kai-zhong/discover-intelligence/issues/2)
- 梯度降下算法中求最小值的想法[#3](https://github.com/kai-zhong/discover-intelligence/issues/3)
