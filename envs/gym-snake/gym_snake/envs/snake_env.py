"""TODO(ehotaj): DO NOT SUBMIT without one-line documentation for snake_env.

TODO(ehotaj): DO NOT SUBMIT without a detailed description of snake_env.
"""

import sys

import gym
from gym import error
from gym import spaces
from gym import utils
import pygame


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
