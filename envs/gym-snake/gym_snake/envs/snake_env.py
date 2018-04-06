"""TODO."""

import sys

import gym
from gym import logger
from gym import spaces
import numpy as np
import pygame

_DIRS = [
    (1, 0),  # Right
    (0, -1),  # Up
    (-1, 0),  # Left
    (0, 1),  # Down
]


class Game(object):
    def __init__(self, dims=(20, 20), snake_size=5):
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


_WHITE = (255, 255, 255)
_RED = (255, 0, 0)
_GREEN = (0, 255, 0)


class SnakeEnv(gym.Env):
    """An environment which runs the game of Snake."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._screen = None
        self._game = None
        self._done = False

        self._dims = (20, 20)
        self._px_size = 16

        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self):
        self._game = Game(dims=self._dims)
        self._done = False

    def step(self, action):
        """Advances the environment by 1 step.

        Args:
            action: A discreete integer taking values in {0, 1, 2} which
                correspond to the following:
                    0 = Turn right
                    1 = Do nothing
                    2 = Turn left
        """
        if self._done:
            logger.warn(('The environment has reached a terminal state and '
                         'should be reset (by calling env.reset()'))
            return

        if not self._game:
            return

        if not self.action_space.contains(action):
            action = 1
        action -= 1
        new_direction = (_DIRS.index(self._game.dir) + action) % len(_DIRS)
        self._game.dir = _DIRS[new_direction]

        self._done = self._game.step()
        return (None,
                len(self._game.snake),
                self._done,
                {})

    def render(self, mode='human', close=False):
        """Only human mode is currently supported."""
        if not self._screen and not close:
            pygame.init()
            self._screen = pygame.display.set_mode(np.multiply(self._dims,
                                                               self._px_size))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                close = True

        if close:
            self._screen = None
            pygame.quit()
            return

        # Rendering code.
        self._screen.fill(_WHITE)
        for x, y in self._game.snake:
            self._screen.fill(
                _RED,
                (x * self._px_size,
                 y * self._px_size, self._px_size, self._px_size))
        self._screen.fill(
            _GREEN,
            (self._game.apple[0] * self._px_size,
             self._game.apple[1] * self._px_size, self._px_size,
             self._px_size))
        pygame.display.flip()
