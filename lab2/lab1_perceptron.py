# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загружаем и подготавливаем данные
df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values

inputSize = X.shape[1]  # Количество входных сигналов
hiddenSizes = 10  # Количество нейронов скрытого слоя
outputSize = 1 if len(y.shape) else y.shape[1]  # Количество выходных сигналов

# Создаем матрицу весов скрытого слоя
Win = np.zeros((1 + inputSize, hiddenSizes))
Win[0, :] = np.random.randint(0, 3, size=(hiddenSizes))  # Пороги w0
Win[1:, :] = np.random.randint(-1, 2, size=(inputSize, hiddenSizes))  # Веса

# Инициализируем веса выходного слоя
Wout = np.random.randint(0, 2, size=(1 + hiddenSizes, outputSize)).astype(np.float64)

# Функция ReLU
def relu(x):
    return np.maximum(0, x)

# Производная ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Функция прямого прохода (предсказания)
def predict(Xp):
    # Входы скрытого слоя
    hidden_input = np.dot(Xp, Win[1:, :]) + Win[0, :]
    # Выходы скрытого слоя с ReLU
    hidden_output = relu(hidden_input)
    # Выходы выходного слоя
    out = np.where((np.dot(hidden_output, Wout[1:, :]) + Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_output

# Обучение
n_iter = 100
eta = 0.01
for i in range(n_iter):
    for xi, target in zip(X, y):
        pr, hidden = predict(xi)
        error = target - pr
        # Обновление весов выходного слоя
        Wout[1:] += (eta * error * hidden).reshape(-1, 1)
        Wout[0] += eta * error

    # Проверка ошибки на всей выборке
    y_all = df.iloc[:, 4].values
    y_all = np.where(y_all == "Iris-setosa", 1, -1)
    X_all = df.iloc[:, [0, 2]].values
    pr, _ = predict(X_all)
    errSum = sum(np.abs(pr - y_all.reshape(-1, 1))) / 2
    if errSum == 0:
        print("Обучение закончено досрочно")
        break

# Проверка точности
y_all = df.iloc[:, 4].values
y_all = np.where(y_all == "Iris-setosa", 1, -1)
X_all = df.iloc[:, [0, 2]].values
pr, _ = predict(X_all)
accuracy = 1 - sum(np.abs(pr - y_all.reshape(-1, 1))) / (2 * len(y_all))
print(f"Точность: {accuracy:.2f}")