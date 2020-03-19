#!/usr/bin/python3
from pe_kine_env import PEKineEnv
from numpy import random


if __name__ == '__main__':
    env=PEKineEnv()
    for _ in range(16):
        obs, _ = env.reset()
        print("evader_traj: {}, pursuers_traj: {}".format(obs['evader']['trajectory'],obs['pursuers']['trajectory']))
        env.render(pause=0.32)
    # obs, info = env.reset()
    # for st in range(30):
    #     env.step(random.randn(2))
    #     print("obs: {} \ninfo: {}".format(obs, info))
    #     env.render()
