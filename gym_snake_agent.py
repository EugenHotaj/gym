"""Agent to solve (my own implementation) of Snake."""

import gym
import gym_snake
import numpy as np


class QFunctionApproxAgent(object):
    def __init__(self, actions, dims):
        self._actions = actions

        self._gamma = 0.9
        self._alpha = 0.9
        self._eps = 0.9

        # width + height + action + bias
        self._weights = np.zeros(dims[0] + dims[1] + 1)
        self._prev_state_action = None

    def _phi(self, state, action):
        """Creates features from (state, action)."""
        state_ = np.copy(state)
        predicate = np.where(state_ == 2)
        state_[predicate] = 100
        row_sum = np.sum(state_, axis=0)
        col_sum = np.sum(state_, axis=1)
        bias = [1]
        return np.concatenate((bias, row_sum, col_sum, action))

    def _find_max_action(self, state):
        """Finds the action with the maximum expected reward in the `state`."""
        max_q_val = 0
        max_action = None
        for action in self._actions:
            state = self._phi(state, action)
            q_val = np.dot(self._weights, state)
            if q_val > max_q_val:
                max_action = action
        return max_action

    def _update_q(self, next_state, reward):
        """Performs "gradient descent" on the Q-Function weights.

        Note: no automatic differentiation is necessary since the partial
        derivatives were done by hand.
        """
        phi = self._phi(*self._prev_state_action)
        phi_prime = self._phi(next_state, self._find_max_action(next_state))
        q_prime = reward + self._gamma * phi_prime
        self._weights -= self._alpha * (q_prime - np.dot(phi, self._weights)) * phi

    def act(self, state, reward=None, done=False):
        if not self._prev_state_action:
            assert not reward
            assert not done
            action = self._find_max_action(state)
            self._prev_state_action = (state, action)
            return action

        _update_q(state_reward)

        if done:
          self._prev_state_action = None
          return None

        if np.random.uniform() < self._eps:
          action = self._action.sample()
          self._prev_state_action = (state, action)
          return action
        action = _find_max_action(self._prev_state_action[0])
        self._prev_state_action = (state, action)
        return action

env = gym.make('Snake-v0')
env.reset()
agent = QFunctionApproxAgent(env.action_space, (10, 10))
action = env.action_space.sample()
while 1:
    state, reward, done, _ = env.step(action)
    action = agent.act(state, reward)
    if done:
        env.reset()
    env.render()
