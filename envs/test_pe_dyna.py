#!/usr/bin/python3
from pe_dyna_env import PEDynaEnv
import numpy as np
from numpy import random
from numpy import pi
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env=PEDynaEnv(num_evaders=2,num_pursuers=10)
    for ep in range(4):
        env.reset()
        # print(bool(sum(sum(env.distance_matrix>env.interfere_radius))))
                # print("\npursuers status: {} \nevaders status: {}".format(env.pursuers['status'],env.evaders['status']))
        # env.render(pause=1./env.rate)
        for st in range(env.max_steps):
            env.render(pause=1./env.rate)
            action_evaders = np.random.uniform(-4,4,size=(env.num_evaders,2))
            action_pursuers = np.random.uniform(-4,4,size=(env.num_pursuers,2))
            obs, rew, done, info = env.step(action_evaders,action_pursuers)
            print("\n-\nepisode: {}, step: {} \nstate: {} \naction_evaders: {} \naction_pursuers: {} \nstatus_evaders: {}, status_pursuers: {} \nreward: {}".format(ep+1, st+1, obs, action_evaders, action_pursuers, env.evaders['status'], env.pursuers['status'], rew))
            if info:
                print(info)
                break
