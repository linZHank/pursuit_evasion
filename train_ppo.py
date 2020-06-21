import sys
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

from envs.pe_env import PursuitEvasion

################################################################
"""
Build PPO flavored actor-critic
"""
class Actor(tf.Module):
    def _distribution(self, obs, state):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def __call__(self, obs, state, act=None):
        pi = self._distribution(obs, state)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class ActorCritic(tf.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64), activation='tanh'):
        super().__init__()
        self.actor = MLPGaussianActor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation)
        self.critic = MLPCritic(obs_dim=obs_dim, hidden_sizes=hidden_sizes, activation=activation)

    # @tf.function
    def step(self, obs):
        with tf.GradientTape() as t:
            with t.stop_recording():
                pi_dist = self.actor._distribution(obs)
                a = pi_dist.sample()
                logp_a = self.actor._log_prob_from_distribution(pi_dist, a)
                v = self.critic(obs)

        # return a.numpy(), v.numpy(), logp_a.numpy()
        return a, v, logp_a

    # @tf.function
    def act(self, obs):
        return self.step(obs)[0] 

def compute_odometry(env):
    """
    compute odometries of all players
    Args:
        env: PE environment
    Returns:
        odoms: (x, y, vx, vy)*num_players
    """
    num_e = env.num_evaders
    num_p = env.num_pursuers
    odoms = np.zeros((num_e+num_p,4)) # x,y,vx,vy
    for ie in range(num_e):
        odoms[ie] = np.concatenate((env.evaders['position'][ie], env.evaders['velocity'][ie]), axis=0)
    for ip in range(num_p):
        odoms[-num_p+ip] = np.concatenate((env.pursuers['position'][ip], env.pursuers['velocity'][ip]), axis=0)

    return odoms


# Main
# parameters
epochs = 4
steps_per_epoch = 100 # make sure this can cover several episodes
# prepare
env=PursuitEvasion()
obs = env.reset() # obs is map
num_evaders = env.num_evaders
num_pursuers = env.num_pursuers
total_steps = 0
ep_ret = np.zeros(num_evaders+num_pursuers)
ep_len = np.zeros(num_evaders+num_pursuers)
for ep in range(epochs):
    episode = 0
    while episode < episodes_per_epoch:
        imgs = np.stack([np.zeros_like(obs) for _ in range(num_evaders+num_pursuers)], axis=0) # prepare storing images
        for ie in range(num_evaders):
           imgs[ie] = obs.copy()+np.random.normal(loc=0, scale=0.01, size=obs.shape) # add noise N(0,0.01)
        for ip in range(num_pursuers):
           imgs[-num_pursuers+ip] = obs.copy()+np.random.normal(loc=0, scale=0.01, size=obs.shape) 
        odoms = compute_odometry(env)
        acts = np.zeros((num_evaders+num_pursuers,2))
        vals = np.zeros(num_evaders+num_pursuers)
        logps = np.zeros(num_evaders+num_pursuers)
        # compute actions
        acts = np.zeros((num_evaders+num_pursuers,2))
        for ie in range(num_evaders):
            if not done[ie]:
                acts[ie], vals[ie], logps[ie] = agent_eva.step(imgs[ie], odoms[ie])
        for ip in range(num_pursuers):
            if not done[-num_pursuers+ip]:
                acts[-num_pursuers+ip], vals[-num_pursuers+ip], logps[-num_pursuers+ip] =
                agent_pur.step(imgs[-num_pursuers+ip], odoms[-num_pursuers+ip])
        # step env and obtain new obs
        next_obs, rew, done, info = env.step(acts)
        # next_imgs = np.stack([np.zeros_like(next_obs) for _ in range(num_evaders+num_pursuers)], axis=0) 
        # for ie in range(num_evaders):
        #     next_imgs[ie] = next_obs.copy()+np.random.normal(loc=0, scale=0.01, size=obs.shape) # add noise N(0,0.01)
        # for ip in range(num_pursuers):
        #     next_imgs[-num_pursuers+ip] = next_obs.copy()+np.random.normal(loc=0, scale=0.01, size=obs.shape) 
        # next_odoms = compute_odometry(env)
        total_steps += 1
        # store experiences
        for ie in range(num_evaders):
            if not done[ie] or rew[ie]:
                agent_eva.buffer.store(imgs[ie], odoms[ie], rews[ie], vals[ie], logps[ie]) 
        for ip in range(num_pursuers):
            if not done[-num_pursuers+ip] or rew[-num_pursuers+ip]:
                agent_pur.buffer.store(imgs[-num_pursuers+ip], odoms[-num_pursuers+ip], rews[-num_pursuers+ip], vals[-num_pursuers+ip], logps[-num_pursuers+ip]) 
        obs = next_obs # THIS IS CRITICAL!!!

        
                



