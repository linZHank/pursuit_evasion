#!/usr/bin/python3
from pe_dyna_env import PEDynaEnv
import numpy as np
from numpy import random
from numpy import pi


if __name__ == '__main__':
    env=PEDynaEnv(num_evaders=2,num_pursuers=3)
    for _ in range(16):
        env.reset()
        print(bool(sum(sum(env.distance_matrix>env.interfere_radius))))
                # print("\npursuers status: {} \nevaders status: {}".format(env.pursuers['status'],env.evaders['status']))
        env.render(pause=1./env.rate)
        # for st in range(5):
        #     # action_pursuers = np.zeros(2)
        #     # action_evaders = np.zeros(2)
        #     action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=(env.num_evaders,2))
        #     action_pursuers = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=(env.num_pursuers,2))
        #     obs, rew, done, _ = env.step(action_evaders,action_pursuers)
        #     print("\n-\nepisode: {}, step: {} \nstate: {} \naction_evaders: {}, action_pursuers: {} \nstatus_evaders: {}, status_pursuers: {} \nreward: {}".format(ep+1, st+1, obs, action_evaders, action_pursuers, env.evaders['status'], env.pursuers['status'], rew))
        #     env.render(pause=0.5)
