import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = [np.random.randn(layers[i], layers[i - 1]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(layers[i], 1) for i in range(1, self.num_layers)]

    def forward(self, x):
        a = x
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = sigmoid(z)
        return a

    def backward(self, x, y):
        activations = [x]
        zs = []
        a = x
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        delta = (activations[-1] - y) * sigmoid(zs[-1]) * (1 - sigmoid(zs[-1]))
        grad_w = [np.dot(delta, activations[-2].T)]
        grad_b = [np.sum(delta, axis=1, keepdims=True)]

        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid(z) * (1 - sigmoid(z))
            delta = np.dot(self.weights[-i + 1].T, delta) * sp
            grad_w.append(np.dot(delta, activations[-i - 1].T))
            grad_b.append(np.sum(delta, axis=1, keepdims=True))

        grad_w.reverse()
        grad_b.reverse()
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b, learning_rate):
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            for j in range(X.shape[0]):
                x = X[j].reshape(-1, 1)
                y_true = y[j]
                grad_w, grad_b = self.backward(x, y_true)
                self.update_parameters(grad_w, grad_b, learning_rate)

    def predict(self, X):
        y_pred = np.zeros((X.shape[0],))
        for i in range(X.shape[0]):
            x = X[i].reshape(-1, 1)
            y_pred[i] = self.forward(x)
        return y_pred
