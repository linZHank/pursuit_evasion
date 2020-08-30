import sys
import os
import numpy as np
import tensorflow as tf
import scipy.signal
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from envs.purnavs0 import PursuerNavigationScene0
from agents.dqn import ReplayBuffer, DeepQNet

if __name__=='__main__':
    env = PursuerNavigationScene0() # default resolution:(80,80)
    agent = DeepQNet(
        dim_obs=[8], 
        num_act=env.action_reservoir.shape[0], 
    ) 
    model_path = './saved_models/pursuer_navigation_scene0_discrete/dqn/9094'
    agent.q.q_net = tf.keras.models.load_model(model_path)
    agent.epsilon = 0.

    for ep in range(10):
        o, d, ep_ret = env.reset(), False, 0
        for st in range(env.max_episode_steps):
            cv2.imshow('map', cv2.resize(o[:,:,[2,1,0]], (640, 640)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            env.render(pause=1./env.rate)
            s = np.array([
                env.evaders['position'][0] - env.pursuers['position'][0],
                env.pursuers['position'][0],
                env.pursuers['velocity'][0],
            ]).reshape(-1)
            a = np.squeeze(agent.act(np.expand_dims(s, axis=0)))
            o,r,d,_ = env.step(a)
            s2 = np.array([
                env.evaders['position'][0] - env.pursuers['position'][0],
                env.pursuers['position'][0],
                env.pursuers['velocity'][0],
            ]).reshape(-1)
            ep_ret += r
            s = s2.copy()
            if d:
                print("EpReturn: {}".format(ep_ret))
                break 
