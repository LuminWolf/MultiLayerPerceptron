import csv

import numpy as np

from . import layer


class NetWork:
    def __init__(self, feature_size, hidden_size, label_size, layers):
        # 生成层
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.layers = layers
        self.loss_layer = None
        self.params = []
        self.grads = []
        self.load_params()
        self.loss_layer = layer.Mse()

    def load_params(self):
        # 将所有的权重整理到列表中
        for lay in self.layers:
            self.params += lay.params
            self.grads += lay.grads

    def predict(self, x):
        for lay in self.layers:
            x = lay.forward(x)
        return x

    def forward(self, x, label):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, label)
        return loss

    def backward(self):
        dout = self.loss_layer.backward()
        for lay in reversed(self.layers):
            dout = lay.backward(dout)
        return dout

    def iteration(self, feature, label):
        loss = self.forward(feature, label)
        self.backward()
        optimizer = SGD()
        optimizer.update(self.params, self.grads)
        return loss

    def epoch(self, dataset, batchsize):
        data_loader = DataLoader(dataset, batchsize)
        data_loader.split()
        loss = None
        for i in range(len(data_loader.mini_batches)):
            feature, label = data_loader.load_batch(self.feature_size, self.label_size, i)
            loss = self.iteration(feature, label)
        print('Train Loss:', loss)

    def learn(self, data_train, data_val, batch_size, epoch_num):
        test_loader = DataLoader(data_val, batch_size)
        test_feature, test_label = test_loader.load(self.feature_size, self.label_size)
        for i in range(epoch_num):
            print('Epoch', i+1)
            self.epoch(data_train, batch_size)
            print('Val Loss:', self.forward(test_feature, test_label))
            print('')


class TwoLayerNet(NetWork):
    def __init__(self, feature_size, hidden_size, label_size):
        w1 = np.random.randn(feature_size, hidden_size)
        b1 = np.random.randn(hidden_size)
        w2 = np.random.randn(hidden_size, label_size)
        b2 = np.random.randn(label_size)
        layers = [
            layer.Affine(w1, b1),
            layer.ReLU(),
            layer.Affine(w2, b2)
        ]
        super().__init__(feature_size, hidden_size, label_size, layers)
        self.layers = layers


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class DataSet:
    def __init__(self, data):
        # A group of examples
        self.data = data

    def load_csv(self, filepath):
        self.data = np.loadtxt(filepath, delimiter=',')

    def export_csv(self, filepath):
        np.savetxt(filepath, self.data, delimiter=',')

    def split(self):
        train_percentage = 0.9
        val_percentage = 0.1
        data_indices = np.random.permutation(self.data.shape[0])
        train_number = int((len(data_indices) * train_percentage))
        val_number = int((len(data_indices) * val_percentage))
        train = self.data[data_indices[:train_number], :]
        val = self.data[data_indices[train_number:val_number+train_number], :]
        return DataSet(train), DataSet(val)


class DataLoader:
    def __init__(self, dataset, batchsize):
        self.dataset = dataset
        self.batch_size = batchsize
        self.mini_batches = None

    def split(self):
        np.random.shuffle(self.dataset.data)
        last_batch_size = len(self.dataset.data) % self.batch_size
        num_batches = int(np.ceil(len(self.dataset.data) / self.batch_size))
        if last_batch_size:
            last_batch = self.dataset.data[-last_batch_size:]
            mini_batches = np.split(self.dataset.data[:-last_batch_size], num_batches-1)
            mini_batches.append(last_batch)
            self.mini_batches = mini_batches
        else:
            self.mini_batches = np.split(self.dataset.data, num_batches)

    def load_batch(self, feature_size, label_size, iter_num):
        feature = self.mini_batches[iter_num][:, :feature_size]
        label = self.mini_batches[iter_num][:, feature_size:feature_size+label_size]
        return feature, label

    def load(self, feature_size, label_size):
        feature = self.dataset.data[:, :feature_size]
        label = self.dataset.data[:, feature_size:feature_size + label_size]
        return feature, label


def save_model(model, params_filepath):
    with open(params_filepath, 'w') as f:
        for i in range(0, len(model.params), 2):
            weight = model.params[i]
            bias = model.params[i+1]
            layer_data = np.vstack((weight, bias))
            with open(params_filepath, 'a') as f_1:
                np.savetxt(f_1, layer_data, delimiter=',')


def load(model, params_filepath):
    with open(params_filepath, 'r') as f:
        cr = csv.reader(f)
        raw_data = []
        for row in cr:
            raw_data.append(row)
    # load layer shapes first
    layers_shape = []
    for i in range(len(model.params)):
        layers_shape.append(np.shape(model.params[i]))
    # load layers
    layers = []
    row_index = 0
    for i in range(0, len(layers_shape), 2):
        shape = layers_shape[i]
        lay = []
        for j in range(row_index, row_index+shape[0]+1):
            lay.append(raw_data[j])
        layers.append(np.array(lay, dtype=np.float32))
        row_index += shape[0]+1
    # load params
    params = []
    for i in range(len(layers)):
        weight = layers[i][:-1]
        bias = layers[i][-1]
        params.append(weight)
        params.append(bias)
    model.params = params


def example():
    feature_size = 2  # 特征数量
    label_size = 1  # 标签数量
    batch_size = 10  # batch的大小
    dataset_size = 100  # 数据集大小

    model_1 = TwoLayerNet(feature_size, 3, label_size)

    main_data = DataSet(np.random.randn(dataset_size, feature_size+label_size))
    data_train, data_val = main_data.split()
    model_1.learn(data_train, data_val, batch_size, 5)


if __name__ == '__main__':
    example()
