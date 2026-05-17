class BackpropNN:
    def __init__(self, layer_sizes=None, lr=0.01):
        self.lr = lr
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.val_loss_history = []
        if layer_sizes:
            for i in range(len(layer_sizes) - 1):
                W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
                b = np.zeros((1, layer_sizes[i+1]))
                self.weights.append(W)
                self.biases.append(b)

    def relu(self, z): return np.maximum(0, z)

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ W + b
            self.zs.append(z)
            a = self.relu(z) if i < len(self.weights) - 1 else z
            self.activations.append(a)
        return self.activations[-1]

    def predict(self, X):
        return self.forward(X).flatten()