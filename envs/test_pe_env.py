#!/usr/bin/python3
from pe_env import PEEnv
import numpy as np
from numpy import random
from numpy import pi
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    env=PEEnv()
    # env.render()
    for ep in range(10):
        obs = env.reset()
        for st in range(env.max_episode_steps):
            env.render(pause=1./env.rate)
            actions = np.random.uniform(-4,4,size=(env.num_evaders+env.num_pursuers,2))
            obs, rew, done, info = env.step(actions)
            # img = obs[:,:,[2,1,0]]
            # cv2.imshow('map', cv2.resize(img, (360, 360)))
            # if cv2.waitKey(25) & 0xFF == ord('q'):
              # break
            # print("\nreward: {} \ndone: {}".format(rew, done))

            if info:
                print(info)
                break
    cv2.destroyAllWindows()
