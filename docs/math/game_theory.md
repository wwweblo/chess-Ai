В проекте используются несколько ключевых понятий из теории игр, которые применяются для решения задач принятия решений и оптимизации в игре в шахматы. Вот основные из них:

### 1. **Алгоритм Минимакс**
Минимакс используется для принятия решений в условиях игры с нулевой суммой, где один игрок выигрывает, если другой проигрывает. Алгоритм ищет оптимальный ход, предполагая, что противник всегда будет выбирать наихудший для вас вариант.

Пример в коде:
```python
def minimax(self, board, depth, maximizing_player):
    if depth == 0 or board.is_game_over():
        return self.evaluate_position(board), None

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            eval, _ = self.minimax(board, depth - 1, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            eval, _ = self.minimax(board, depth - 1, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move
```

### 2. **Оценка позиции (Utility Function)**
Каждая позиция на шахматной доске получает численную оценку, которая описывает, насколько она выгодна для игрока. Эта оценка моделируется нейронной сетью, и ее значение используется для определения наилучшего хода.

Пример:
```python
def evaluate_position(self, board):
    fen = board.fen()
    tensor = fen_to_tensor(fen).unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2)

    with torch.no_grad():
        outputs = self.model(tensor)
        score = torch.max(outputs, 1)[0].item()

    return score
```

### 3. **Игра с нулевой суммой**
В шахматах выгода одного игрока эквивалентна потере другого игрока. В алгоритме Минимакс это выражается в том, что один игрок максимизирует свою выгоду, а другой — минимизирует.

### 4. **Альфа-бета отсечение**
Этот метод позволяет избежать излишних вычислений в алгоритме Минимакс, обрезая те ветви дерева возможных ходов, которые заведомо хуже ранее найденных вариантов.