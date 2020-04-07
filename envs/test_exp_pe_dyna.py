#!/usr/bin/python3
from exp_pe_dyna_env import PEDynaEnv
import numpy as np
from numpy import random
from numpy import pi
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env=PEDynaEnv(num_evaders=3,num_pursuers=4)
    for ep in range(10):
        env.reset()
        # print(bool(sum(sum(env.distance_matrix>env.interfere_radius))))
                # print("\npursuers status: {} \nevaders status: {}".format(env.pursuers['status'],env.evaders['status']))
        # env.render(pause=1./env.rate)
        for st in range(60):
            env.render(pause=1./env.rate)
            actions = np.random.uniform(-5,5,size=(env.num_evaders+env.num_pursuers,2))
            obs, rew, done, info = env.step(actions)
            print("\n-\nepisode: {}, step: {} \nstate: {} \nactions \nstatus_evaders: {}, status_pursuers: {} \nreward: {}".format(ep+1, st+1, obs, actions, env.evaders['status'], env.pursuers['status'], rew))
            if info:
                print(info)
                break
