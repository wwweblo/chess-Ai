В проекте используются следующие понятия из дифференцирования:

### 1. **Градиент**:
   Градиент — это вектор, который указывает направление наибольшего увеличения функции. В контексте обучения нейронных сетей, градиент функции потерь относительно весов модели помогает обновить веса в сторону минимизации функции потерь.

Пример из кода:
```python
loss.backward()  # Вычисление градиента
```
Этот вызов вычисляет градиенты функции потерь по всем параметрам нейросети с использованием метода обратного распространения ошибки (backpropagation).

### 2. **Обновление весов (Шаг оптимизации)**:
   После вычисления градиентов, веса модели обновляются в направлении, противоположном градиенту, чтобы минимизировать функцию потерь. Это шаг аналогичен использованию производной для нахождения минимума функции.

Пример из кода:
```python
optimizer.step()  # Обновление весов с учетом градиента
```

### 3. **Стохастический градиентный спуск (SGD)**:
   Это метод оптимизации, при котором обновление весов происходит на основе малой случайной выборки (мини-батчей) данных. Используется для ускорения сходимости.

Пример:
```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```
Здесь `Adam` — это улучшенный вариант градиентного спуска, который использует моменты для сглаживания шага оптимизации.

Эти понятия лежат в основе процесса обучения нейронной сети, где целью является минимизация функции потерь через итеративное обновление параметров.