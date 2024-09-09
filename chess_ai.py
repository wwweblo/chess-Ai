import chess
import torch
import concurrent.futures
from model import ChessNet
from data_loader import fen_to_tensor


class ChessAI:
    def __init__(self, model, depth=2):
        """Инициализация класса ChessAI.

        Args:
            model: Обученная нейросеть.
            depth: Глубина поиска для определения лучшего хода.
        """
        self.model = model
        self.depth = depth
        self.transposition_table = {}

    def evaluate_position(self, board):
        """Оценка позиции с помощью нейросети.

        Args:
            board: Текущая позиция (объект chess.Board).

        Returns:
            Оценка позиции.
        """
        fen = board.fen()
        tensor = fen_to_tensor(fen).unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            outputs = self.model(tensor)
            score = torch.max(outputs, 1)[0].item()

        return score

    def order_moves(self, board):
        """Упорядочивает ходы для альфа-бета отсечения, начиная с потенциально сильных ходов."""
        capture_moves = []
        other_moves = []

        for move in board.legal_moves:
            if board.is_capture(move):
                capture_moves.append(move)
            else:
                other_moves.append(move)

        return capture_moves + other_moves

    def alphabeta(self, board, depth, alpha, beta, maximizing_player):
        """Альфа-бета отсечение с использованием транспозиционных таблиц."""
        board_key = board.fen()

        # Использование таблицы транспозиций
        if board_key in self.transposition_table and self.transposition_table[board_key][1] >= depth:
            return self.transposition_table[board_key][0], None

        if depth == 0 or board.is_game_over():
            score = self.evaluate_position(board)
            self.transposition_table[board_key] = (score, depth)
            return score, None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            ordered_moves = self.order_moves(board)
            for move in ordered_moves:
                board.push(move)
                eval, _ = self.alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Обрезка
            self.transposition_table[board_key] = (max_eval, depth)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            ordered_moves = self.order_moves(board)
            for move in ordered_moves:
                board.push(move)
                eval, _ = self.alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Обрезка
            self.transposition_table[board_key] = (min_eval, depth)
            return min_eval, best_move

    def minimax_parallel(self, board, depth, maximizing_player):
        """Параллельная версия алгоритма Минимакс с использованием потоков."""
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for move in board.legal_moves:
                board.push(move)
                futures[move] = executor.submit(self.alphabeta, board.copy(), depth - 1, float('-inf'), float('inf'),
                                                not maximizing_player)
                board.pop()

            results = {move: future.result() for move, future in futures.items()}

        if maximizing_player:
            best_move = max(results, key=lambda x: results[x][0])
        else:
            best_move = min(results, key=lambda x: results[x][0])

        return results[best_move][0], best_move

    def get_best_move(self, board):
        """Получает лучший ход для текущей позиции с использованием альфа-бета отсечения."""
        _, best_move = self.alphabeta(board, self.depth, float('-inf'), float('inf'), board.turn == chess.WHITE)
        return best_move
