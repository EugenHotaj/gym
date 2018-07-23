"""OpenAI gym's mountain car world solved using DQN."""

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

_ENV_NAME = 'MountainCar-v0'

_WINDOW_LENGTH = 200


def _build_agent(input_shape, nb_actions):
    model = _build_model(input_shape, nb_actions)
    mem = memory.SequentialMemory(limit=1000000, window_length=_WINDOW_LENGTH)
    pol = policy.LinearAnnealedPolicy(
        policy.EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
        value_test=.05, nb_steps=1000000)
    agent = dqn.DQNAgent(
        model=model, nb_actions=nb_actions, policy=pol, memory=mem,
        nb_steps_warmup=200, target_model_update=1000, train_interval=4,
        delta_clip=1.)
    agent.compile(optimizers.Adam(lr=.00025), metrics=['mae'])
    return agent


def _build_model(input_shape, nb_actions):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(_WINDOW_LENGTH, ) + input_shape))
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(nb_actions))
    model.add(layers.Activation('linear'))
    return model


def main(argv):
    del argv  # Unused.

    env = gym.make(_ENV_NAME)
    observation_shape = env.observation_space.shape
    nb_actions = env.action_space.n
    agent = _build_agent(observation_shape, nb_actions)

    agent.fit(env, nb_steps=500000, visualize=False, verbose=2)
    agent.save_weights('dqn_{}_weights.h5f'.format(_ENV_NAME), overwrite=True)
    agent.test(env, nb_episodes=5, visualize=True)

if __name__ == '__main__':
    app.run(main)
