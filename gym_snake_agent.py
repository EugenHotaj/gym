"""Agent to solve (my own implementation) of Snake."""
from absl import app
from absl import flags
import gym
import numpy as np
import gym_snake

FLAGS = flags.FLAGS

flags.DEFINE_string('render_mode', 'human',
                    'The rendering mode for the environment')


class QFunctionApproxAgent(object):
    def __init__(self, actions, dims):
        self._actions = actions

        self._gamma = 0.99
        self._alpha = .001
        self._eps = 1

        # bias + state + action
        self._weights = np.random.uniform(0, 1, 19)
        self._prev_state_action = None

        self._step = 0
        self._episode = 0
        self._episode_reward = 0

    def _phi(self, state, action):
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

        s = np.concatenate((snake_one_hot, apple_dist))
        phi = self._dim_scaling(s, action)
        return phi

    def _dim_scaling(self, phi_state, action):
        phi = np.zeros(len(phi_state) * self._actions.n)
        start_index = len(phi_state) * action
        phi[start_index:start_index+len(phi_state)] = phi_state
        return np.concatenate(([1], phi))

    def _find_max_action(self, state):
        """Finds the action with the maximum expected reward in the `state`."""
        max_action = 0
        max_q_val = np.dot(self._weights, self._phi(state, max_action))
        for action in range(1, self._actions.n):
            q_val = np.dot(self._weights, self._phi(state, action))
            if q_val > max_q_val:
                max_q_val = q_val
                max_action = action

        return max_action

    def _update_q(self, next_state, reward, done):
        """Performs "gradient descent" on the Q-Function weights.

        Note: no automatic differentiation is necessary since the partial
        derivatives were done by hand.
        """
        phi = self._phi(*self._prev_state_action)
        q = np.dot(self._weights, phi)
        if done:
            correction = reward
        else:
            phi_prime = self._phi(next_state,
                                  self._find_max_action(next_state))
            q_prime = np.dot(self._weights, phi_prime)
            correction = reward + (self._gamma * q_prime) - q
        self._weights += self._alpha * correction * phi

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
            self._eps -= .0007

        if not self._prev_state_action:
            self._episode += 1
        else:
            self._update_q(state, reward, done)
            self._episode_reward += reward

        if done:
            self._prev_state_action = None

            if self._episode % 100 == 0:
                print(('Step: %d; Alpha: %.2f, Eps: %.2f, Avg Ep Reward: %.2f;'
                       % (self._step, self._alpha, self._eps,
                           self._episode_reward / 100)))
                self._episode_reward = 0
            return None

        if not self._prev_state_action or np.random.uniform() < self._eps:
            action = self._actions.sample()
        else:
            action = self._find_max_action(self._prev_state_action[0])

        self._prev_state_action = (state, action)
        return action


def main(argv):
    del argv  # Unused.

    env = gym.make('Snake-v0')
    agent = QFunctionApproxAgent(env.action_space, (10, 10))
    state, reward, done, _ = env.reset()
    while True:
        env.render(FLAGS.render_mode)
        action = agent.act(state, reward, done)
        state, reward, done, _ = env.step(action) if not done else env.reset()


if __name__ == '__main__':
    app.run(main)
