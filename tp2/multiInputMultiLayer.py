import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the fashion-mnist pre-shuffled train data and test data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# print("x_train shape:", X_train_sq.shape, "y_train shape:", Y_train.shape)
# plt.imshow(X_train_sq[1])
# plt.show()

L = 2
K = 10


X_train = np.reshape(train_images, (train_images.shape[0], 784))
X_test = np.reshape(test_images, (test_images.shape[0], 784))

Y_train = np.zeros((train_labels.shape[0], len(np.unique(train_labels))))
Y_train[np.arange(Y_train.shape[0]), train_labels] = 1

Y_test = np.zeros((test_labels.shape[0], len(np.unique(test_labels))))
Y_test[np.arange(Y_test.shape[0]), test_labels] = 1


lr = 0.001
nb_epochs = 20
W = []
losses_test = []

W.append(np.random.normal(0, 0.01, (784, 300)))
for i in range(L-1):
    W.append(np.random.normal(0, 0.01, (301, 300)))
W.append(np.random.normal(0, 0.01, (301, K)))


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


def get_loss(y, y_pred):
    return np.nanmean(-y * np.log(y_pred))


def main():
    minibatch_size = len(X_train) // 20
    for epoch in range(nb_epochs):
        for x in range(0, X_train.shape[0], minibatch_size):

            print(x)
            Z = [None] * (L + 2)
            a = [None] * (L + 2)
            delta = [None] * (L + 2)

            #forward propagation
            a[0] = X_train[x:x+minibatch_size]
            for i in range(L):
                Z[i + 1] = a[i] @ W[i]
                a[i + 1] = np.append(ReLU(Z[i+1]), np.ones((Z[i+1].shape[0], 1), dtype=int), axis=1)

            Z[-1] = a[L] @ W[L]
            a[-1] = softmax(Z[-1])

            #back propagation
            delta[-1] = softMaxInverse(Z[-1]) * (Y_train[x:x+minibatch_size] - a[-1])
            for i in range(L, 0, -1):
                m = (delta[i+1] @ W[i].T)
                delta[i] = ReLUReverse(Z[i]) * m[:,:-1]


            for i in range(len(W)):
                o = np.matmul(a[i][:,:,np.newaxis], delta[i+1][:,np.newaxis,:])
                W[i] = W[i] + lr * np.mean(o, axis=0)

        #test
        Z_test = [None] * (L + 2)
        a_test = [None] * (L + 2)

        a_test[0] = X_test
        for i in range(L):
            Z_test[i + 1] = a_test[i] @ W[i]
            q = ReLU(Z_test[i+1])
            f = np.ones((Z_test[i+1].shape[0], 1), dtype=int)
            a_test[i + 1] = np.append(ReLU(Z_test[i+1]), np.ones((Z_test[i+1].shape[0], 1), dtype=int), axis=1)

        Z_test[-1] = a_test[L] @ W[L]
        a_test[-1] = softmax(Z_test[-1])

        losses_test.append(get_loss(Y_test, a_test[-1]))


main()
plt.plot(losses_test)
plt.show()

