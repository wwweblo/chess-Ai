В проекте были использованы следующие понятия из **многопоточности**:

1. **Параллелизация задач**: Использование нескольких потоков для одновременной обработки нескольких задач, таких как оценка возможных ходов в шахматной партии. Это позволяет сократить время поиска лучшего хода.

    **Пример**:
    ```python
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_move = {executor.submit(self.evaluate_move, board, move): move for move in board.legal_moves}
        for future in concurrent.futures.as_completed(future_to_move):
            move = future_to_move[future]
            try:
                score = future.result()
                # Логика выбора лучшего хода
            except Exception as e:
                print(f"Ошибка при оценке хода {move}: {e}")
    ```

2. **Асинхронные задачи**: Асинхронное выполнение задач позволяет не блокировать основной поток программы во время долгих вычислений (например, во время оценки шахматных ходов), что улучшает отзывчивость программы.

    **Пример**:
    ```python
    async def get_best_move_async(self, board):
        loop = asyncio.get_event_loop()
        best_move = await loop.run_in_executor(None, self.get_best_move, board)
        return best_move
    ```

3. **ThreadPoolExecutor**: Использование пула потоков (ThreadPoolExecutor) для того, чтобы выполнить сразу несколько задач, каждая из которых оценивает ход в шахматной позиции.

Эти примеры помогают оптимизировать поиск ходов за счет многопоточности, что значительно ускоряет игру шахматного бота.