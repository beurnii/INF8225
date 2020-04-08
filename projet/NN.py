import numpy as np
import random

class NN:
    def __init__(self, input_nodes, hidden_layers, output_nodes):
        self.W = []
        if len(hidden_layers):
            self.W.append(np.random.normal(0, 0.1, (input_nodes+1, hidden_layers[0])))
            for i in range(1, len(hidden_layers)):
                self.W.append(np.random.normal(0, 0.1, (hidden_layers[i-1]+1, hidden_layers[i])))
            self.W.append(np.random.normal(0, 0.1, (hidden_layers[-1]+1, output_nodes)))
        else:
            self.W.append(np.random.normal(0, 0.1, (input_nodes+1, output_nodes)))

    def hidden_activation(self, Z):
        return np.maximum(Z, 0)

    def softmax_activation(self, Z):
        exp = np.exp(Z - Z.max())
        return np.array(exp / exp.sum())

    def predict(self, inputs):
        a = np.append(np.array(inputs), 1)
        for i in range(len(self.W)-1):
            Z = np.squeeze(a @ self.W[i])
            a = np.append(self.hidden_activation(Z), 1)
        Z = np.squeeze(a @ self.W[-1])
        a = self.softmax_activation(Z)
        return np.argmax(a)

    def copy(self):
        pass

    def crossover(self):
        pass

    def mutate(self, rate):
        newW = []
        for m in self.W:
            newM = np.copy(m)
            newW.append(newM)
            for w in np.nditer(newM, op_flags=['readwrite']):
                if random.random() < rate:
                    w[...] = random.random()
        return newW

