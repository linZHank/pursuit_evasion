#!/usr/bin/python3
from pe_kine_env import PEKineEnv
from numpy import random


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=2)
    env.render(pause=10)
    # obs, info = env.reset()
    # for st in range(30):
    #     env.step(random.randn(2))
    #     print("obs: {} \ninfo: {}".format(obs, info))
    #     env.render()
