import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ChessNet
from data_loader import create_dataset, load_lichess_data
import os
from datetime import datetime

# Гиперпараметры
learning_rate = 0.001
batch_size = 32
num_epochs = 10
max_games = 5000  # Ограничение на количество загружаемых партий, можно изменить

# Папка для сохранения модели
model_dir = 'data/model'
os.makedirs(model_dir, exist_ok=True)

# Загрузка данных
data_dir = 'data/pgn'
games = load_lichess_data(data_dir, max_games=max_games)
dataset = create_dataset(games)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Модель, функция потерь, оптимизатор
model = ChessNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Scheduler для изменения learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Устройство (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Цикл обучения
for epoch in range(num_epochs):
    model.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Убедитесь, что размерность входных данных соответствует ожиданиям модели
        inputs = inputs.permute(0, 3, 1, 2)  # Перемещение оси канала перед высотой и шириной

        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Логирование потерь
        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # Логируем каждые 100 мини-батчей
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # Обновление learning rate
    scheduler.step()

    # Сохранение модели после каждой эпохи с текущей датой в имени файла
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_filename = f'{model_dir}/checkpoint_epoch_{epoch + 1}_{date_str}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved as {model_filename}')

print('Обучение завершено')