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

    def create_datasets(self):
        # convert list to array
        img_buf = np.array(self.img_buf)
        odom_buf = np.array(self.odom_buf)
        act_buf = np.array(self.act_buf) 
        logp_buf = np.array(self.logp_buf)
        rew_buf = np.array(self.rew_buf) 
        ret_buf = np.array(self.ret_buf) 
        adv_buf = np.array(self.adv_buf) 
        val_buf = np.array(self.val_buf) 
        # next three lines implement advantage normalization
        adv_mean = np.mean(adv_buf)
        adv_std = np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        # create dataset for training actor
        actor_dataset = tf.data.Dataset.from_tensor_slices((img_buf, odom_buf, act_buf, logp_buf, adv_buf))
        actor_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        # create dataset for training critic
        critic_dataset = tf.data.Dataset.from_tensor_slices((img_buf, odom_buf, ret_buf))
        critic_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        
        return actor_dataset, critic_dataset

if __name__=='__main__':
    env = PursuitEvasion()
    agent = PPOAgent(name='test_ppo_agent')
    obs = env.reset()
    num_e = env.num_evaders
    num_p = env.num_pursuers
    img = obs.copy()
    odom = np.concatenate((env.evaders['position'][0],env.evaders['velocity'][0]), axis=-1)
    buf = PPOBuffer(size=10)
    for _ in range(10):
        acts = np.random.randn(num_e+num_p, 2)
        a, v, logp = agent.pi_given_state(np.expand_dims(img, axis=0), np.expand_dims(odom, axis=0))
        acts[0] = a
        obs, rew, done, info = env.step(acts)
        n_img = obs.copy()
        n_odom = np.concatenate((env.evaders['position'][0],env.evaders['velocity'][0]), axis=-1) 
        buf.store(img, odom, a, logp, rew[0], v[0])
        img = n_img.copy()
        odom = n_odom.copy()
        if info:
            break
    buf.finish_episode()
    ad, cd = buf.create_datasets()



# class PPOBuffer:
#     """
#     A buffer for storing trajectories experienced by a PPO agent interacting
#     with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
#     for calculating the advantages of state-action pairs.
#     """
#     def __init__(self, dim_img, dim_odom, dim_act, size, gamma=0.99, lam=0.95):
#         self.obs_buf = np.zeros((size, dim_img[0], dim_img[1], dim_img[2]), dtype=np.float32)
#         self.odom_buf = np.zeros(shape=(size, dim_odom), dtype=np.float32)
#         self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
#         self.adv_buf = np.zeros(size, dtype=np.float32)
#         self.rew_buf = np.zeros(size, dtype=np.float32)
#         self.ret_buf = np.zeros(size, dtype=np.float32)
#         self.val_buf = np.zeros(size, dtype=np.float32)
#         self.logp_buf = np.zeros(size, dtype=np.float32)
#         self.gamma, self.lam = gamma, lam
#         self.ptr, self.path_start_idx, self.max_size = 0, 0, size
# 
#     def store(self, img, odom, act, rew, val, logp):
#         assert self.ptr <= self.max_size     # buffer has to have room so you can store
#         self.img_buf[self.ptr] = img
#         self.odom_buf[self.ptr] = odom
#         self.act_buf[self.ptr] = act
#         self.rew_buf[self.ptr] = rew
#         self.val_buf[self.ptr] = val
#         self.logp_buf[self.ptr] = logp
#         self.ptr += 1
# 
#     def finish_path(self, last_val=0):
#         path_slice = slice(self.path_start_idx, self.ptr)
#         rews = np.append(self.rew_buf[path_slice], last_val)
#         vals = np.append(self.val_buf[path_slice], last_val)
#         # the next two lines implement GAE-Lambda advantage calculation
#         deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
#         self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
#         # the next line computes rewards-to-go, to be targets for the value function
#         self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
#         self.path_start_idx = self.ptr
#         # self.ptr, self.path_start_idx = 0, 0
# 
#     def get(self):
#         assert self.ptr == self.max_size    # buffer has to be full before you can get
#         self.ptr, self.path_start_idx = 0, 0
#         # the next two lines implement the advantage normalization trick
#         adv_mean = np.mean(self.adv_buf)
#         adv_std = np.std(self.adv_buf)
#         self.adv_buf = (self.adv_buf - adv_mean) / adv_std
#         data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
#                     adv=self.adv_buf, logp=self.logp_buf)
#         return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
    

# # Main
# # parameters
# epochs = 4
# steps_per_epoch = 100 # make sure this can cover several episodes
# # prepare
# env=PursuitEvasion()
# obs = env.reset() # obs is map
# num_evaders = env.num_evaders
# num_pursuers = env.num_pursuers
# total_steps = 0
# ep_ret = np.zeros(num_evaders+num_pursuers)
# ep_len = np.zeros(num_evaders+num_pursuers)
# for ep in range(epochs):
#     episode = 0
#     while episode < episodes_per_epoch:
#         odoms = compute_odometry(env)
#         imgs = np.stack([np.zeros_like(obs) for _ in range(num_evaders+num_pursuers)], axis=0) # prepare storing images
#         for ie in range(num_evaders):
#            imgs[ie] = obs.copy()+np.random.normal(loc=0, scale=0.01, size=obs.shape) # add noise N(0,0.01)
#         for ip in range(num_pursuers):
#            imgs[-num_pursuers+ip] = obs.copy()+np.random.normal(loc=0, scale=0.01, size=obs.shape) 
#         acts = np.zeros((num_evaders+num_pursuers,2))
#         vals = np.zeros(num_evaders+num_pursuers)
#         logps = np.zeros(num_evaders+num_pursuers)
#         # compute actions
#         acts = np.zeros((num_evaders+num_pursuers,2))
#         for ie in range(num_evaders):
#             if not done[ie]:
#                 acts[ie], vals[ie], logps[ie] = agent_eva.step(imgs[ie], odoms[ie])
#         for ip in range(num_pursuers):
#             if not done[-num_pursuers+ip]:
#                 acts[-num_pursuers+ip], vals[-num_pursuers+ip], logps[-num_pursuers+ip] =
#                 agent_pur.step(imgs[-num_pursuers+ip], odoms[-num_pursuers+ip])
#         # step env and obtain new obs
#         next_obs, rew, done, info = env.step(acts)
#         total_steps += 1
#         # store experiences
#         for ie in range(num_evaders):
#             if not done[ie] or rew[ie]:
#                 agent_eva.buffer.store(imgs[ie], odoms[ie], rews[ie], vals[ie], logps[ie]) 
#         for ip in range(num_pursuers):
#             if not done[-num_pursuers+ip] or rew[-num_pursuers+ip]:
#                 agent_pur.buffer.store(imgs[-num_pursuers+ip], odoms[-num_pursuers+ip], rews[-num_pursuers+ip], vals[-num_pursuers+ip], logps[-num_pursuers+ip]) 
#         obs = next_obs # THIS IS CRITICAL!!!
#         timeout = (env.step_counter==env.max_episode_steps)
# 
#         if all(done[:num_evaders]) or all(done[-num_pursuers:]):
#             episode += 1
#             if all(done): # timeout
#                 last_vals = np.zeros(num_evaders)
#                 odoms = compute_odometry(env)
#                 imgs = np.stack([np.zeros_like(obs) for _ in range(num_evaders+num_pursuers)], axis=0) # prepare storing images
#                 for ie in range(num_evaders):
#                     imgs[ie] = obs.copy()+np.random.normal(loc=0, scale=0.01, size=obs.shape) # add noise N(0,0.01)
#                     _, last_val[ie], _ = agent_eva.step(imgs[ie], odoms[ie])
#                 for ip in range(num_pursuers):
#                     imgs[-num_pursuers+ip] = obs.copy()+np.random.normal(loc=0, scale=0.01, size=obs.shape) 
                
        
                



