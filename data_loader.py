import torch
import chess.pgn
import os
import numpy as np

def load_lichess_data(data_dir, max_games):
    """Загружает все PGN-файлы из указанной директории и преобразует их в объекты chess.pgn.Game.
    
    Args:
        data_dir: Путь к директории с PGN-файлами.
        max_games: Максимальное количество загружаемых партий (для ограничения размера выборки).

    Returns:
        Список объектов chess.pgn.Game.
    """
    all_games = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pgn'):
            with open(os.path.join(data_dir, filename)) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    print(f'file {data_dir}/{filename} was read')
                    if game is None:
                        break
                    all_games.append(game)
                    if max_games and len(all_games) >= max_games:
                        return all_games
    print(f'Browsed {len(all_games)} games')
    return all_games

def preprocess_game(game):
    """Преобразует партию в последовательность FEN-строк, представляющих состояния доски.
    
    Args:
        game: Объект chess.pgn.Game.

    Returns:
        Список FEN-строк.
    """
    board = game.board()
    fens = []
    for move in game.mainline_moves():
        fens.append(board.fen())
        board.push(move)
    return fens

def fen_to_tensor(fen):
    """Преобразует FEN-строку в тензор с one-hot encoding для каждой клетки.

    Args:
        fen: FEN-строка, представляющая позицию на шахматной доске.

    Returns:
        Tensor размера (8, 8, 13), где 13 - количество возможных значений на клетке
        (12 фигур + цвет на ходу).
    """
    board = chess.Board(fen)
    tensor = torch.zeros((8, 8, 13), dtype=torch.float32)

    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row, col = divmod(square, 8)
            tensor[row, col, piece_map[piece.symbol()]] = 1

    # Добавление дополнительного канала для цвета на ходу
    tensor[:, :, 12] = 1 if board.turn == chess.WHITE else 0

    return tensor

def create_dataset(games):
    """Создает датасет PyTorch из списка партий.

    Args:
        games: Список объектов chess.pgn.Game.

    Returns:
        PyTorch датасет с тензорами состояний доски и метками ходов.
    """
    tensors = []
    labels = []

    for game in games:
        fens = preprocess_game(game)
        moves = [move.uci() for move in game.mainline_moves()]  # Ходы в формате UCI
        for fen, move in zip(fens, moves):
            tensor = fen_to_tensor(fen)
            tensors.append(tensor)
            label = move_to_label(move)  # Преобразуем ход в метку
            labels.append(label)

    # Преобразование списков тензоров и меток в PyTorch-формат
    tensors = torch.stack(tensors)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(tensors, labels)
    return dataset

def move_to_label(move):
    """Преобразует ход в формате UCI в индекс, который можно использовать в качестве метки.

    Args:
        move: Строка в формате UCI, представляющая шахматный ход.

    Returns:
        Целочисленная метка для обучения.
    """
    # В простейшем случае можно просто закодировать UCI-ходы в уникальные индексы
    # Здесь нужен более сложный подход для правильной кодировки ходов (например, индексирование ходов).
    
    # Пример простой кодировки, где a1-a8 = 0-63, b1-b8 = 64-127 и т.д.
    # Это простой пример. В реальной задаче, вероятно, понадобится более сложная функция кодировки.
    from_square = chess.SQUARE_NAMES.index(move[:2])
    to_square = chess.SQUARE_NAMES.index(move[2:4])
    return from_square * 64 + to_square  # Пример простого индекса

