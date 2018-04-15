"""Module containing all Snake game specific code."""

import numpy as np

_DIRS = [
    (1, 0),  # Right
    (0, -1),  # Up
    (-1, 0),  # Left
    (0, 1),  # Down
]


class Game(object):
    def __init__(self, dims=(20, 20), snake_size=5):
        """Constructor.

        Args:
            dims: A tuple of the dimensions of the game board
            snake_size: The initial size of the snake
        """
        self._dims = dims
        self.snake = self._generate_initial_snake(size=snake_size)
        self.dir = (0, -1)
        self.apple = self._generate_apple()

    def _generate_initial_snake(self, size=5):
        # TODO(EugenHotaj): snake should generate in random configurations.
        snake = []
        x = np.random.randint(0, self._dims[0] - size)
        y = int(self._dims[1] / 2)
        snake.append((x, y))
        for i in range(1, size):
            x = x + 1
            y = y
            snake.append((x, y))
        return snake

    def step(self):
        """Executes one step of the game."""
        new_head = tuple(np.add(self.snake[-1], self.dir) % self._dims)
        if new_head in self.snake:
            return True

        self.snake = self.snake[1:]
        self.snake.append(new_head)

        if new_head == self.apple:
            # Append the new snake piece outside the game screen. It will be
            # correctly placed on the next call to #update().
            self.apple = self._generate_apple()
            self.snake = [(-100, -100)] + self.snake

        return False

    def _generate_apple(self):
        board = np.zeros(self._dims)
        for x, y in self.snake:
            board[x, y] = 1
        empty = np.where(board == 0)
        x = np.random.choice(empty[0])
        y = np.random.choice(empty[0])
        return x, y


