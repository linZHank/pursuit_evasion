import os
import numpy as np
import cv2
import scipy.signal
import random
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging

import tensorflow as tf
print(tf.__version__)
import tensorflow_probability as tfp
tfd = tfp.distributions

from envs.pe import PursuitEvasion
from agents.ppo import PPOAgent


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(self, size, gamma=.99, lam=.95, batch_size=64):
        self.img_buf = []
        self.odom_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.adv_buf = []
        self.val_buf = []
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.episode_start_idx, self.max_size = 0, 0, size
        self.batch_size = batch_size

    def store(self, img, odom, act, logp, rew, val):
        assert self.ptr <= self.max_size
        self.img_buf.append(img)
        self.odom_buf.append(odom)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.ptr += 1

    def finish_episode(self, last_val=0):
        ep_slice = slice(self.episode_start_idx, self.ptr)
        rews = np.array(self.rew_buf[ep_slice])
        vals = np.append(np.array(self.val_buf[ep_slice]), last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf += list(discount_cumsum(deltas, self.gamma*self.lam))
        # next line implement reward-to-go
        self.ret_buf += list(discount_cumsum(np.append(rews, last_val), self.gamma)[:-1])
        self.episode_start_idx = self.ptr

    def get(self):
        """
        Get a data dicts from replay buffer
        """
        # convert list to array
        img_buf = np.array(self.img_buf)
        odom_buf = np.array(self.odom_buf)
        act_buf = np.array(self.act_buf) 
        logp_buf = np.array(self.logp_buf)
        rew_buf = np.array(self.rew_buf) 
        ret_buf = np.array(self.ret_buf) 
        adv_buf = np.array(self.adv_buf) 
        # next three lines implement advantage normalization
        adv_mean = np.mean(adv_buf)
        adv_std = np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        # create data dict for training actor
        actor_data = dict(
            img = tf.convert_to_tensor(img, dtype=tf.float32),
            odom = tf.convert_to_tensor(odom, dtype=tf.float32),
            act = tf.convert_to_tensor(act, dtype=tf.float32),
            logp = tf.convert_to_tensor(logp, dtype=tf.float32),
            adv = tf.convert_to_tensor(adv, dtype=tf.float32)
        )
        # create data dict for training critic
        critic_data = dict(
            img = tf.convert_to_tensor(img, dtype=tf.float32),
            odom = tf.convert_to_tensor(odom, dtype=tf.float32),
            ret = tf.convert_to_tensor(ret, dtype=tf.float32)
        )

        return actor_data, critic_data


if __name__=='__main__':
    env = PursuitEvasion()
    agent_e = PPOAgent(name='ppo_evader')
    agent_p = PPOAgent(name='ppo_pursuer')
    # parameters
    num_episodes = 1
    num_steps = env.max_episode_steps
    buffer_size = int(1e3)
    on_policy_ep = 0
    for ep in range(num_episodes):
        obs = env.reset()
        num_e = env.num_evaders
        num_p = env.num_pursuers
        img = obs.copy()
        odoms_e = np.concatenate((env.evaders['position'],env.evaders['velocity']), axis=-1)
        odoms_p = np.concatenate((env.pursuers['position'],env.pursuers['velocity']), axis=-1)
        acts_e = np.zeros((num_e, 2))
        acts_p = np.zeros((num_p, 2))
        vals_e = np.zeros(num_e)
        logps_e = np.zeros(num_e)
        done = np.array([False]*(num_e+num_p))
        evader_buf_collection = [PPOBuffer(size=buffer_size)]*num_e
        pursuer_buf_collection = [PPOBuffer(size=buffer_size)]*num_p
        for st in range(num_steps):
            acts_e = np.zeros((num_e, 2))
            acts_p = np.zeros((num_p, 2))
            for ie in range(num_e):
                if not done[ie]:
                    acts_e[ie], vals_e[ie], logps_e[ie] = agent_e.pi_given_state(
                        np.expand_dims(img, axis=0),
                        np.expand_dims(odoms_e[ie], axis=0)
                    )
            obs, rew, done, info = env.step(np.concatenate((acts_e,acts_p), axis=0))
            img = obs.copy()
            odoms_e = np.concatenate((env.evaders['position'],env.evaders['velocity']), axis=-1)
            odoms_p = np.concatenate((env.pursuers['position'],env.pursuers['velocity']), axis=-1)
            if info:
                break




