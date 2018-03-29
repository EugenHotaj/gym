"""TODO(ehotaj): DO NOT SUBMIT without one-line documentation for snake_env.

TODO(ehotaj): DO NOT SUBMIT without a detailed description of snake_env.
"""

import sys

import gym
from gym import error
from gym import spaces
from gym import utils
import numpy as np
import pygame


class Game(object):
  def __init__(self, width=20, height=20, snake_size=5):
    self._board = np.array((width, height))
    self._snake = self._generate_initial_snake(size=snake_size)
    while True:
      x, y = (np.random.rand_int(0, width), np.random.randint(height))
      self._apple = np.array([x, y])
      if x not in self._snake[:, 0] or y not in self._snake[:, 1]:
        break

  def _generate_initial_snake(self, size=5):
    snake = np.array(size, 2)
    # Subtract size from x-bounds because we always extends snake to the right.
    snake[0, 0] = np.random.randint(0, self._board.shape[0] - size)
    snake[0, 1] = np.random.randint(0, self._board.shape[1])

    # TODO(EugenHotaj): snake should generate in random configurations.
    for i in range(1, size):
      snake[i, 0] = snake[i-1, 0] + 1
      snake[i, 1] = snake[i-1]

    return snake


class SnakeEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self._is_pygame_init = False
    self._screen = None

  def step(self, action):
    pass

  def reset(self):
    pass

  def render(self, mode='human', close=False):
    """Only human mode is currently supported."""
    if close:
      return
    if not self._is_pygame_init:
      pygame.init()
      self._screen = pygame.display.set_mode((800, 600))
      self._screen.fill((0, 0, 0))

    pygame.display.flip()
