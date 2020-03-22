#!/usr/bin/python3
from pe_kine_env import PEKineEnv
import numpy as np
from numpy import random


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=1)
    for _ in range(10):
        obs, _ = env.reset()
        for _ in range(15):
            action_evader = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            action_pursuers = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=(env.num_pursuers,2))
            obs, rew, done, _ = env.step(action_evader,action_pursuers)
            # print("evader: {}, pursuers: {}, reward: {}".format(obs['evaders']['position'],obs['pursuers']['position'],rew[0]))
            env.render(pause=1./env.rate)
