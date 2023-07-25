import numpy as np


class Layer:
    def __init__(self):
        self.params = []
        self.grads = []


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = np.maximum(0, x)
        self.out = out
        return out

    def backward(self, dout):
        grad = self.out[self.out > 0] = 1
        dx = dout * grad
        return dx


class Affine(Layer):
    def __init__(self, w, b):
        super().__init__()
        self.params = [w, b]
        self.grads = [np.zeros_like(w), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        w, b = self.params
        out = np.dot(x, w) + b
        self.x = x
        return out

    def backward(self, dout):
        w, b = self.params
        dx = np.dot(dout, w.T)
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dw
        self.grads[1][...] = db
        return dx


class Mse(Layer):
    def __init__(self):
        super().__init__()
        self.predicted = None
        self.label = None

    def forward(self, predicted, label):
        self.predicted = predicted
        self.label = label
        out = (np.sum(np.square(label - predicted)) / np.size(label))
        return out

    def backward(self):
        n = self.label.shape[0]
        grad = (2 / n) * (self.predicted - self.label)
        return grad


class BatchNormalization(Layer):
    def __init__(self):
        super().__init__()
        gama = None
        beta = None
        miub = None
        sigmab = None
        self.params.append([gama, beta])
        self.out = None

    def forward(self, x):
        out = x
        self.out = out
        return out

    def backward(self):
        pass