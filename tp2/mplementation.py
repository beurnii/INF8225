import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the fashion-mnist pre-shuffled train data and test data
fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (test_images, test_labels) = fashion_mnist.load_data()
print("x_train shape:", X_train.shape, "y_train shape:", Y_train.shape)
plt.imshow(X_train[1])

L = 2
K = 10
# X_train = []
# Y_train = []
lr = 0.001

W = []
Z = [None] * L + 2
a = [None] * L + 2
delta = [None] * L + 2

W.append(np.random.normal(0, 0.01, (738, 300)))
for i in range(L-1):
    W.append(np.random.normal(0, 0.01, (301, 300)))
W.append(np.random.normal(0, 0.01, (300, K)))


def softmax(z):
    exp = np.exp(z - z.max(0))
    return np.array(exp / exp.sum(0))


def softMaxInverse(z):
    sm = softmax(z)
    return sm * (1-sm)


def ReLU(z):
    return np.maximum(z, 0)


def ReLUReverse(z):
    return (z >= 0).astype(int)


def main():
    for x in range(len(X_train)):
        #forward propagation
        a[0] = X_train[x]
        for i in range(L):
            Z[i + 1] = a[i] @ W[i]
            a[i + 1] = np.append(ReLU(Z[i+1]), 1)

        Z[-1] = a[L] @ W[L]
        a[-1] = softmax(Z[-1])

        #back propagation
        delta[-1] = softMaxInverse(Z[-1]) * (Y_train[x] - a[-1])
        for i in range(L, 0, -1):
            delta[i] = ReLUReverse(Z[i]) * np.sum(W[i+1] * delta[i+1], axis=0)

        for i in range(len(W)):
            W[i] = W[i] + lr * a[i] @ delta[i+1]


