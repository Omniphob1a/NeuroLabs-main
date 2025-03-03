# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import MLPptorch, MLPptorchReLU
import torch
import torch.nn as nn

# функция обучения
def train(net, x, y, num_iter=5000):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    lossFn = nn.MSELoss()
    for i in range(num_iter):
        pred = net(x)
        loss = lossFn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % (num_iter//10) == 0:
            print(f'Итерация {i}: Ошибка {loss.item():.4f}')
    return loss.item()

# Загрузка данных
df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

# Подготовка данных
X = df.iloc[0:100, 0:3].values
y = df.iloc[0:100, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor':3}).values.reshape(-1,1)
Y = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y[:,i-1] = np.where(y == i, 1, 0).reshape(1,-1)

X_test = df.iloc[100:150, 0:3].values
y = df.iloc[100:150, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor':3}).values.reshape(-1,1)
Y_test = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y_test[:,i-1] = np.where(y == i, 1, 0).reshape(1,-1)

# Инициализация сетей
inputSize = X.shape[1]
hiddenSizes = [50, 30, 20, 56]  # Три скрытых слоя
outputSize = Y.shape[1]

# Создаем две модели для сравнения
net_sigmoid = MLPptorch(inputSize, hiddenSizes, outputSize)
net_relu = MLPptorchReLU(inputSize, hiddenSizes, outputSize)

# Обучение Sigmoid сети
print("Обучение сети с Sigmoid:")
loss_sigmoid = train(net_sigmoid, 
                    torch.from_numpy(X.astype(np.float32)),
                    torch.from_numpy(Y.astype(np.float32)))

# Обучение ReLU сети
print("\nОбучение сети с ReLU:")
loss_relu = train(net_relu,
                 torch.from_numpy(X.astype(np.float32)),
                 torch.from_numpy(Y.astype(np.float32)))

# Вывод сравнения
print(f"\nСравнение конечных ошибок:")
print(f"Sigmoid: {loss_sigmoid:.4f}, ReLU: {loss_relu:.4f}")

# Тестирование Sigmoid сети
pred = net_sigmoid.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y_test))
print(f"Ошибка на тестовых данных (Sigmoid): {err}")

# Тестирование ReLU сети
pred = net_relu.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y_test))
print(f"Ошибка на тестовых данных (ReLU): {err}")