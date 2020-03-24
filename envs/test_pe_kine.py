#!/usr/bin/python3
from pe_kine_env import PEKineEnv
import numpy as np
from numpy import random
from numpy import pi


if __name__ == '__main__':
    env=PEKineEnv()
    for _ in range(10):
        # specify evader's spawining position
        theta_e = random.uniform(-pi,pi)
        env.evaders_spawning_pool = np.array([[3*np.cos(theta_e),3*np.sin(theta_e)]])
        obs, _ = env.reset()
        print("\n---\n{}".format(obs))
        env.render(pause=1)
        # for _ in range(10):
        #     action_pursuers = np.zeros(2)
        #     action_evaders = np.zeros(2)
        #     obs, rew, done, _ = env.step(action_evaders,action_pursuers)
        #     # print("evader: {}, pursuers: {}, reward: {}".format(obs['evaders']['position'],obs['pursuers']['position'],rew[0]))
        #     env.render(pause=1)
