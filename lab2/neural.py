import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        # Инициализация весов скрытого слоя
        self.Win = np.zeros((1 + inputSize, hiddenSizes))
        self.Win[0, :] = np.random.randint(0, 3, size=(hiddenSizes))
        self.Win[1:, :] = np.random.randint(-1, 2, size=(inputSize, hiddenSizes))
        
        # Инициализация весов выходного слоя
        self.Wout = np.random.randint(0, 2, size=(1 + hiddenSizes, outputSize)).astype(np.float64)

    def predict(self, Xp):
        # Прямой проход через скрытый слой
        hidden_predict = np.where((np.dot(Xp, self.Win[1:, :]) + self.Win[0, :]) >= 0.0, 1, -1).astype(np.float64)
        # Прямой проход через выходной слой
        out = np.where((np.dot(hidden_predict, self.Wout[1:, :]) + self.Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict

    def train(self, X, y, n_iter=5, eta=0.01):
        for epoch in range(n_iter):
            # Перемешиваем данные для стохастического обучения
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for xi, target in zip(X_shuffled, y_shuffled):
                pr, hidden = self.predict(xi)
                error = target - pr
                # Обновление весов выходного слоя
                self.Wout[1:] += (eta * error * hidden[:, np.newaxis])
                self.Wout[0] += eta * error
        return self