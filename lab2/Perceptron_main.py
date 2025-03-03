import pandas as pd
import numpy as np
from neural import Perceptron

# Загрузка данных
df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

# Используем все 4 признака и 3 класса
X = df.iloc[:, [0, 1, 2, 3]].values
y = df.iloc[:, 4].values

# Преобразуем метки в числовой формат
unique_labels = np.unique(y)
y = np.array([np.where(unique_labels == label)[0][0] for label in y])

# Параметры сети
inputSize = X.shape[1]  # Количество входных сигналов (4 признака)
hiddenSizes = 10        # Количество нейронов скрытого слоя
outputSize = len(unique_labels)  # Количество классов (3)

# Создаем и обучаем нейронную сеть
NN = Perceptron(inputSize, hiddenSizes, outputSize)
NN.train(X, y, n_iter=100, eta=0.01)

# Проверка точности
out, _ = NN.predict(X)
predictions = np.argmax(out, axis=1)
accuracy = np.mean(predictions == y)
print(f"Точность: {accuracy:.2f}")