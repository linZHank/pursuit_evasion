#!/usr/bin/python3
from pe_dyna import PEDyna
import numpy as np
from numpy import random
from numpy import pi
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env=PEDyna(num_evaders=4, num_pursuers=4)
    # env.render()
    for ep in range(3):
        obs = env.reset()
        print("obs: {}".format(obs))
        # env.render(pause=1)
        # print(bool(sum(sum(env.distance_matrix>env.interfere_radius))))
                # print("\npursuers status: {} \nevaders status: {}".format(env.pursuers['status'],env.evaders['status']))
        # env.render(pause=1./env.rate)
        for st in range(10):
            env.render(pause=1./env.rate)
            actions = np.random.uniform(-4,4,size=(env.num_evaders+env.num_pursuers,2))
            env.step(actions)
        #     action_pursuers = np.random.uniform(-4,4,size=(env.num_pursuers,2))
        #     obs, rew, done, info = env.step(action_evaders,action_pursuers)
        #     print("\n-\nepisode: {}, step: {} \nstate: {} \naction_evaders: {} \naction_pursuers: {} \nstatus_evaders: {}, status_pursuers: {} \nreward: {}".format(ep+1, st+1, obs, action_evaders, action_pursuers, env.evaders['status'], env.pursuers['status'], rew))
        #     if info:
        #         print(info)
        #         break
