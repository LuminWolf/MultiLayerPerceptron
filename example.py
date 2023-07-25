import numpy as np

from MultiLayerPerceptron import nn


def example():
    feature_size = 2  # 特征数量
    label_size = 1  # 标签数量
    batch_size = 10  # batch的大小
    dataset_size = 100  # 数据集大小

    model_1 = nn.TwoLayerNet(feature_size, 3, label_size)

    main_data = nn.DataSet(np.random.randn(dataset_size, feature_size+label_size))
    data_train, data_val = main_data.split()
    model_1.learn(data_train, data_val, batch_size, 5)


if __name__ == '__main__':
    example()
