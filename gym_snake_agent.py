"""Agent to solve (my own implementation) of Snake."""

import gym
import gym_snake
import numpy as np


class QFunctionAgent(object):
    def __init__(self, actions, dims):
        self._actions = actions

        self._gamma = 0.9
        self._alpha = 0.9
        self._eps = 0.9

        # width + height + action + bias
        self._weights = np.zeros(dims[0] + dims[1] + 1)
        self._prev_state_action = None

    def _compress_state_action(self, state, action):
        state_ = np.copy(state)
        predicate = np.where(state_ == 2)
        state_[predicate] = 100
        row_sum = np.sum(state_, axis=0)
        col_sum = np.sum(state_, axis=1)

        bias = [1]

        return np.concatenate((row_sum, col_sum, action, bias))

    def _find_max_action(self, state):
        max_q_val = 0
        max_action = None
        for action in self._actions:
            state = self._compress_state_action(state, action)
            q_val = np.dot(self._weights, state)
            if q_val > max_q_val:
                max_action = action
        return max_action

    def _update_q(self, reward, next_state):
        pass

    def act(self, state, reward):
        pass


env = gym.make('Snake-v0')
env.reset()
agent = QFunctionAgent(env.action_space, (10, 10))
action = env.action_space.sample()
while 1:
    state, reward, done, _ = env.step(action)
    action = agent.act(state, reward)
    if done:
        env.reset()
    env.render()
