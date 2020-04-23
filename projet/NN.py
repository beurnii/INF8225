import numpy as np
import random
import copy


class NN:
    def __init__(self, w):
        self.W = w

    @classmethod
    def from_params(cls, input_nodes, hidden_layers, output_nodes):  # NN.from_params(3,[3,4,5],1)
        w = []
        if len(hidden_layers):
            w.append(np.random.normal(0, 0.1, (input_nodes + 1, hidden_layers[0])))
            for i in range(1, len(hidden_layers)):
                w.append(np.random.normal(0, 0.1, (hidden_layers[i - 1] + 1, hidden_layers[i])))
            w.append(np.random.normal(0, 0.1, (hidden_layers[-1] + 1, output_nodes)))
        else:
            w.append(np.random.normal(0, 0.1, (input_nodes + 1, output_nodes)))
        return cls(w)

    @classmethod
    def from_weights(cls, w):
        return cls(w)

    @classmethod
    def crossover(cls, dna_1, dna_2):
        new_w = []
        for m in dna_1:
            new_m = np.array(dna_1.shape)
            for i in m:
                for j in i:
                    new_m[j][i] = dna_1[i][j] if random.random > 0.5 else dna_2[i][j]
            new_w.append(new_m)
        return new_w

    @classmethod
    def mutate(cls, dna, rate):
        _w = []
        for m in dna:
            new_m = np.copy(m)
            for w in np.nditer(new_m, op_flags=['readwrite']):
                if random.random() < rate:
                    w[...] = random.random()
            _w.append(new_m)
        return _w

    def hidden_activation(self, Z):
        return np.maximum(Z, 0)

    def softmax_activation(self, Z):
        exp = np.exp(Z - Z.max())
        return np.array(exp / exp.sum())

    def predict(self, inputs):
        a = np.append(np.array(inputs), 1)
        for i in range(len(self.W) - 1):
            Z = np.squeeze(a @ self.W[i])
            a = np.append(self.hidden_activation(Z), 1)
        Z = np.squeeze(a @ self.W[-1])
        a = self.softmax_activation(Z)
        return np.argmax(a)

    def get_weights_copy(self):
        _w = []
        for w in self.W:
            _w.append(np.copy(w))
        return _w


