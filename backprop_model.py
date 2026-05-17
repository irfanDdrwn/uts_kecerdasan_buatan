import numpy as np

class BackpropNN:
    def __init__(self, layer_sizes, lr=0.01):
        self.lr = lr
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.val_loss_history = []

        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_d(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        self.zs = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ W + b
            self.zs.append(z)

            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = z

            self.activations.append(a)

        return self.activations[-1]

    def backward(self, y):
        m = y.shape[0]
        y = y.reshape(-1, 1)

        delta = (self.activations[-1] - y) / m

        for i in reversed(range(len(self.weights))):
            dW = self.activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

            if i > 0:
                delta = delta @ self.weights[i].T * self.relu_d(self.zs[i - 1])

    def train(self, X, y, X_val, y_val, epochs=100, batch_size=32):
        for epoch in range(epochs):
            idx = np.random.permutation(X.shape[0])

            X_sh = X[idx]
            y_sh = y[idx]

            for start in range(0, X.shape[0], batch_size):
                Xb = X_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]

                self.forward(Xb)
                self.backward(yb)

            pred = self.forward(X).flatten()
            loss = np.mean((pred - y) ** 2)

            vpred = self.forward(X_val).flatten()
            vloss = np.mean((vpred - y_val) ** 2)

            self.loss_history.append(float(loss))
            self.val_loss_history.append(float(vloss))

    def predict(self, X):
        return self.forward(X).flatten()