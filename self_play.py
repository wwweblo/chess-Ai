import chess
import chess.pgn
import torch
from chess_ai import ChessAI
from model import ChessNet

# Загрузка модели
model_path = 'data/model/checkpoint_epoch_10_2024-09-08_21-16-15.pth'
model = ChessNet()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Инициализация шахматного AI
ai = ChessAI(model, depth=2)


def self_play():
    """Функция, в которой бот играет сам с собой и выводит игру в формате PGN."""
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    while not board.is_game_over():
        print(board)  # Вывод текущей позиции
        best_move = ai.get_best_move(board)
        print(f"Лучший ход: {best_move}")

        board.push(best_move)

        # Запись хода в PGN
        node = node.add_variation(best_move)

    print("Игра завершена!")
    print(f"Результат: {board.result()}")

    # Возвращаем игру в формате PGN
    return game


if __name__ == "__main__":
    pgn_game = self_play()
    print(f"\nPGN игры:\n{pgn_game}")
