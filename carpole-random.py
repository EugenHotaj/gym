"""Cartpole solved using only random search.

Basically works 100% of the time.
"""

import multiprocessing

import gym
import numpy as np


def simulate_env(model, num_times=3, render=False):
    env = gym.make('CartPole-v0')
    total_reward = 0
    for _ in range(num_times):
        done = False
        state = env.reset()
        while not done:
            action = np.dot(model, state)
            if (action < 0):
                action = 0
            else:
                action = 1
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            total_reward += reward
        if render:
            print(total_reward)
    return total_reward / num_times


def main():
    pool = multiprocessing.Pool(processes=8)
    models = [np.random.uniform(size=(4,)) for i in range(10)]
    results = pool.map(simulate_env, models)
    idxs = np.argsort(results)
    best = models[idxs[-1]]
    simulate_env(best, num_times=1, render=True)


if __name__ == '__main__':
    main()
