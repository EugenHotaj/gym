"""OpenAI gym's cartpole world solved using DQN.

Taken from https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_cartpole.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from absl import app
from keras import models
from keras import layers
from keras import optimizers
from rl import memory
from rl import policy
from rl.agents import dqn

_ENV_NAME = 'CartPole-v0'


def _build_model(observation_space_shape, nb_actions):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(1,) + observation_space_shape))
    model.add(layers.Dense(16))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(16))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(16))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(nb_actions))
    model.add(layers.Activation('linear'))
    return model


def _build_agent(observation_space, action_space):
    observation_space_shape = observation_space.shape
    nb_actions = action_space.n
    model = _build_model(observation_space_shape, nb_actions)
    mem = memory.SequentialMemory(limit=50000, window_length=1)
    pol = policy.BoltzmannQPolicy()
    agent = dqn.DQNAgent(
        model=model, nb_actions=nb_actions, memory=mem, nb_steps_warmup=10,
        target_model_update=1e-2, policy=pol)
    agent.compile(optimizers.Adam(lr=1e-3), metrics=['mae'])
    return agent


def main(argv):
    del argv  # Unused.

    env = gym.make(_ENV_NAME)
    observation_space = env.observation_space
    action_space = env.action_space
    agent = _build_agent(observation_space, action_space)

    agent.fit(env, nb_steps=50000, visualize=True, verbose=2)
    agent.save_weights('dqn_{}_weights.h5f'.format(_ENV_NAME), overwrite=True)
    agent.test(env, nb_episodes=5, visualize=True)

if __name__ == '__main__':
    app.run(main)
