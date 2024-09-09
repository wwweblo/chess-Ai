В проекте используется несколько подходов к регуляризации нейронных сетей для предотвращения переобучения и улучшения обобщающих способностей модели. Вот основные понятия из регуляризации и примеры их применения:

### 1. **Dropout**
Dropout — это метод регуляризации, при котором случайные нейроны выключаются во время обучения, что помогает предотвратить переобучение.

**Пример в коде:**
```python
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.5)  # Dropout с вероятностью 0.5
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)  # Применение dropout
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 2. **Batch Normalization**
Batch Normalization — это метод, который нормализует входы для каждого слоя, что способствует улучшению обучения и может служить косвенной формой регуляризации.

**Пример в коде:**
```python
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Применение Batch Normalization
        return x
```

### 3. **Weight Decay**
Weight Decay (или L2-регуляризация) — это метод, который добавляет штраф за большие веса к функции потерь, что помогает предотвратить переобучение.

**Пример в коде:**
```python
import torch.optim as optim

# Оптимизатор с weight decay
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
```

Эти методы вместе способствуют более стабильному обучению модели и помогают улучшить её способность к обобщению на новых данных.