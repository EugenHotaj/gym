"""Approximate QLearning Agent to solve (my own implementation) of Snake.

For the environment implementation, refer to:
    https://www.github.com/EugenHotaj/gym-snake
"""

from absl import app
from absl import flags
import gym
import numpy as np
import gym_snake

FLAGS = flags.FLAGS

flags.DEFINE_string('render_mode', 'human',
                    'The rendering mode for the environment')


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class QFunctionApproxAgent(object):
    def __init__(self, actions, dims):
        self._actions = actions

        self._gamma = 0.99
        self._alpha = .01
        self._eps = 1

        # [actions]  x [bias + state]
        self._weights = np.random.uniform(size=(3, 7))
        self._prev_state = None

        self._step = 0
        self._episode = 0
        self._episode_reward = 0

    def _phi(self, state):
        """Creates features from (state, action)."""
        head = np.where(state == 2)
        right = state[(head[0], (head[1] + 1) % 10)]
        up = state[((head[0] - 1) % 10, head[1])]
        left = state[(head[0], (head[1] - 1) % 10)]
        down = state[((head[0] + 1) % 10, head[1])]

        # Has self close by
        snake_right = 1 if right == 1 else 0
        snake_up = 1 if up == 1 else 0
        snake_left = 1 if left == 1 else 0
        snake_down = 1 if down == 1 else 0
        snake_one_hot = [snake_right, snake_up, snake_left, snake_down]

        # Distance to apple
        head = np.array([head[1][0], head[0][0]])
        apple = np.where(state == 3)
        apple = np.array([apple[1][0], apple[0][0]])
        xdist = (apple[0] - head[0])
        ydist = (apple[1] - head[1])
        apple_dist = [xdist, ydist]

        # [1] is the bias term
        return np.concatenate(([1], snake_one_hot, apple_dist))

    def _update_q(self, next_state, reward, done):
        """Performs "gradient descent" on the Q-Function weights.

        Note: no automatic differentiation is necessary since the partial
        derivatives are computed manually.
        """
        phi = self._phi(self._prev_state)
        q = np.dot(self._weights, phi)
        if done:
            correction = reward * np.ones(shape=(3, 1))
        else:
            phi_prime = self._phi(next_state)
            q_prime = max(np.dot(self._weights, phi_prime))
            correction = (reward + self._gamma * q_prime) - q
        self._weights += (self._alpha * correction.reshape((3, 1))
                          * phi.reshape(1, 7))

    def act(self, state, reward, done):
        """Updates the internal Q Function weights and choses an action.

        Args:
            state: the current state of the environment.
            reward: the reward in the current state, `None` if this is the
                first state.
            done: whether the episode has ended.
        Returns:
            The next action to take.
        """
        self._step += 1
        # Anneal hyperparameters
        if self._step <= 100000 and self._step % 100 == 0:
            self._eps -= .00095

        if self._prev_state is None:
            self._episode += 1
        else:
            self._update_q(state, reward, done)
            self._episode_reward += reward

        if done:
            self._prev_state = None

            if self._episode % 100 == 0:
                print(('Step: %d; Alpha: %.3f, Eps: %.2f, Avg Ep Reward: %.2f;'
                       % (self._step, self._alpha, self._eps,
                           self._episode_reward / 100)))
                print(self._weights)
                self._episode_reward = 0
            return None

        if np.random.uniform() < self._eps:
            action = self._actions.sample()
        else:
            phi = self._phi(state)
            actions = np.dot(self._weights, phi)
            actions_dist = _softmax(actions)
            action = np.argmax(actions_dist)

        self._prev_state = state
        return action


def main(argv):
    del argv  # Unused.

    env = gym.make('Snake-v0')
    agent = QFunctionApproxAgent(env.action_space, (10, 10))
    state, reward, done, _ = env.reset()
    steps = 0
    render_mode = 'train'
    while True:
        steps += 1
        if steps > 400000:
            render_mode = 'human'
        env.render(render_mode)
        action = agent.act(state, reward, done)
        state, reward, done, _ = env.step(action) if not done else env.reset()


if __name__ == '__main__':
    app.run(main)
