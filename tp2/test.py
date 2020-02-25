import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


class NeuralNet:

    def __init__(self, X, y, hidden_layers, neurons_by_layers, learning_rate, epochs):
        # Random seed
        rand_state = np.random.RandomState(42)
        self.X = X
        self.y = y
        self.hidden_layers = hidden_layers
        self.neurons_by_layer = neurons_by_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.layers = []
        self.parameters = {}

        input_layer = self.X[0].shape[0]  # Number of attributes
        output_layer = np.unique(self.y).size  # Number of classes

        self.layers = [input_layer]
        for l in range(int(hidden_layers)):
            self.layers.append(neurons_by_layers)
        self.layers.append(output_layer)

        # xavier initialization
        for i in range(1, len(self.layers)):
            bound = np.sqrt(6. / (self.layers[i - 1] + self.layers[i]))
            self.parameters['W' + str(i)] = rand_state.uniform(-bound, bound, (self.layers[i - 1], self.layers[i]))
            self.parameters['B' + str(i)] = rand_state.uniform(-bound, bound, self.layers[i])

    def one_hot_labels(self, y):
        classes = list(set(y))
        n_samples = y.shape[0]
        n_classes = len(classes)

        classes = np.asarray(classes)
        sorted_class = np.sort(classes)

        # binariser les labels
        Y = np.zeros((n_samples, n_classes))
        indices = np.searchsorted(sorted_class, y)
        Y[np.arange(n_samples), indices] = 1

        return Y

    def softmax(self, X):
        tmp = X - X.max(axis=1)[:, np.newaxis]
        np.exp(tmp, out=X)
        X /= X.sum(axis=1)[:, np.newaxis]

        return X

    def relu(self, x):
        np.clip(x, 0, np.finfo(x.dtype).max, out=x)
        return x

    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def log_loss(self, one_hot, predictions):
        return -np.multiply(one_hot, np.log(predictions)).sum() / predictions.shape[0]

    def forward_propagation(self):
        self.parameters['H' + str(0)] = self.X

        for i in range(1, len(self.layers)):
            self.parameters['A' + str(i)] = np.add(
                np.dot(self.parameters['H' + str(i - 1)], self.parameters['W' + str(i)]),
                self.parameters['B' + str(i)])

            if i != len(self.layers) - 1:
                # Couche cachée utilise fonction d'activation ReLU
                self.parameters['H' + str(i)] = self.relu(self.parameters['A' + str(i)])
            else:
                # Couche finale utilise fonction d'activation Softmax
                self.parameters['H' + str(i)] = self.softmax(self.parameters['A' + str(i)])

        return self.parameters['H' + str(len(self.layers) - 1)]

    def back_propagation(self, y_one_hot):
        m = self.X.shape[0]

        self.parameters['dA' + str(len(self.layers) - 1)] = (1 / m) * (
                self.parameters['H' + str(len(self.layers) - 1)] - y_one_hot) * ((self.parameters['H' + str(len(self.layers) - 1)]) * (1 -self.parameters['H' + str(len(self.layers) - 1)]))
        self.parameters['dW' + str(len(self.layers) - 1)] = np.dot(
            np.transpose(self.parameters['H' + str(len(self.layers) - 2)]),
            self.parameters['dA' + str(len(self.layers) - 1)])
        self.parameters['dB' + str(len(self.layers) - 1)] = self.parameters['dA' + str(len(self.layers) - 1)].sum()

        for i in range(len(self.layers) - 2, 0, -1):
            self.parameters['dA' + str(i)] = (1 / m) * (np.dot(self.parameters['dA' + str(i + 1)], np.transpose(self.parameters['W' + str(i + 1)])) * (self.relu_derivative(self.parameters['A' + str(i)])))
            self.parameters['dW' + str(i)] = np.dot(np.transpose(self.parameters['H' + str(i - 1)]), self.parameters['dA' + str(i)])self.parameters['dB' + str(i)] = self.parameters['dA' + str(i)].sum()

    def prep_vars(self):
        params, grads = [], []

        for j in "W", "B":
            for i in range(1, len(self.layers)):
                params.append(self.parameters[j + str(i)])
                grads.append(self.parameters['d' + j + str(i)])

        params = np.asarray(params)
        grads = np.asarray(grads)

        return params, grads

    def params_unpack(self, params):
        j = 0
        for i in range(1, len(self.layers)):
            self.parameters['W' + str(i)] = params[j]
            j += 1

        for i in range(1, len(self.layers)):
            self.parameters['B' + str(i)] = params[j]
            j += 1

    def fit(self):
        """
        Fonction principale qui entraîne le modèle a travers plusieurs epochs.
        Retourne un réseau entraîné pouvant prédire des données de test
        """
        losses = []
        learning_curve = []
        class_dict = dict()
        y_one_hot = self.one_hot_labels(self.y)
        for j in range(self.epochs):
            # y_pred -> predictions/probabilités des label par relu
            y_pred = self.forward_propagation()

            loss = self.log_loss(y_one_hot, y_pred)
            losses.append(loss)
            print(str(j) + ": " + str(loss))

            self.back_propagation(y_one_hot)

            # Passer les paramètres dans Adam pour ajuster le modèle
            params, grads = self.prep_vars()
            # learning_rate_init = self.learning_rate
            # optimizer = AdamOptimizer(params, learning_rate_init)
            # params = optimizer.update_params(grads)
            self.params_unpack(params)

            if j == 0:
                for i in range(len(y_one_hot)):
                    class_dict[str(y_one_hot[i])] = self.y[i]

        # Making pyplots
        print("cost_function 0, -1", losses[0], losses[-1])
        print("learning_curve 0, -1", learning_curve[0], learning_curve[-1])

        learning_curve = pd.DataFrame(learning_curve)
        learning_curve.to_csv("learning_curve_blackbox21.csv", header=False, index=False)
        print("learning_curve", learning_curve)

        plt.plot(learning_curve)
        plt.title("Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()

        cost_function = pd.DataFrame(losses)

        plt.plot(cost_function)
        plt.title("Logistic Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

        return self.parameters, class_dict


((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], -1) / 255.0
testX = testX.reshape(testX.shape[0], -1) / 255.0
nn = NeuralNet(trainX, trainY, 2, 300, 0.001, 100)
parameters, class_dict = nn.fit()
