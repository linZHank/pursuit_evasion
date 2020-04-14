#!/usr/bin/python3
from pe_kine_env import PEKineEnv
import numpy as np
from numpy import random
from numpy import pi


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=3, num_evaders=2)
    for ep in range(5):
        # specify evader's spawining position
        obs = env.reset()
        for i in range(100):
            print("\n---\n{}".format(obs))
            env.render(pause=1/env.rate)
            actions = np.random.randn(env.num_pursuers+env.num_evaders,2)
            obs, reward, done, _ = env.step(actions)
            if all(done):
                break
        # for st in range(5):
        #     # action_pursuers = np.zeros(2)
        #     # action_evaders = np.zeros(2)
        #     action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=(env.num_evaders,2))
        #     action_pursuers = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=(env.num_pursuers,2))
        #     obs, rew, done, _ = env.step(action_evaders,action_pursuers)
        #     print("\n-\nepisode: {}, step: {} \nstate: {} \naction_evaders: {}, action_pursuers: {} \nstatus_evaders: {}, status_pursuers: {} \nreward: {}".format(ep+1, st+1, obs, action_evaders, action_pursuers, env.evaders['status'], env.pursuers['status'], rew))
        #     env.render(pause=0.5)
