import torch
import torch.nn as nn

# Класс для задания 1: несколько скрытых слоев с Sigmoid
class MLPptorch(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size):
        super().__init__()
        layers = []
        sizes = [in_size] + hidden_sizes
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(sizes[-1], out_size))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Класс для задания 2: несколько скрытых слоев с ReLU
class MLPptorchReLU(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size):
        super().__init__()
        layers = []
        sizes = [in_size] + hidden_sizes
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-1], out_size))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)