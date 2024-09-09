import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Архитектура сети: сверточные слои для анализа доски,
        # полносвязанные слои для оценки позиций и выбора ходов

        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(8 * 8 * 64, 512)
        self.fc2 = nn.Linear(512, 4096)  # Выходной слой с количеством нейронов, равным количеству возможных ходов

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)  # Изменяем размерность с помощью reshape
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
