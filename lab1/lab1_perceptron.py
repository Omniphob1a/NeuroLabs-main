# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np

# Загружаем и подготавливаем данные
df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values

inputSize = X.shape[1]
hiddenSizes = 3
outputSize = 1

# Инициализация весов (оставляем как было)
Win = np.zeros((1+inputSize,hiddenSizes)) 
Win[0,:] = np.random.randint(0, 3, size = (hiddenSizes)) 
Win[1:,:] = np.random.randint(-1, 2, size = (inputSize,hiddenSizes)) 
Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)

# Функция предсказания (без изменений)
def predict(Xp):
    hidden_predict = np.where((np.dot(Xp, Win[1:,:]) + Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
    out = np.where((np.dot(hidden_predict, Wout[1:,:]) + Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_predict

# Параметры обучения с добавлением условий остановки
eta = 0.01
max_epochs = 1000
prev_error = None
stable_count = 0

for epoch in range(max_epochs):
    # Обучение без изменений
    for xi, target, j in zip(X, y, range(X.shape[0])):
        pr, hidden = predict(xi) 
        Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
        Wout[0] += eta * (target - pr)
    
    # Проверка условий остановки
    pr, _ = predict(X)
    current_error = np.sum(pr != y.reshape(-1, 1))
    
    # Остановка при сходимости
    if current_error == 0:
        print(f"Обучение завершено: сходимость на эпохе {epoch+1}")
        break
    
    # Остановка при зацикливании
    if prev_error == current_error:
        stable_count += 1
        if stable_count > 5:
            print(f"Обучение остановлено: зацикливание на эпохе {epoch+1}")
            break
    else:
        stable_count = 0
        prev_error = current_error
else:
    print("Достигнуто максимальное число эпох")

# Оценка (без изменений)
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
pr, hidden = predict(X)
print(f"Всего ошибок после обучения: {sum(pr != y.reshape(-1, 1))}")