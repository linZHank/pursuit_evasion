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
obs, ep_ret, ep_len = env.reset(), 0, 0 # obs is map
num_evaders = env.num_evaders
num_pursuers = env.num_pursuers
acts = np.zeros((num_evaders+num_pursuers,2))
vals = np.zeros(num_evaders+num_pursuers)
logps = np.zeros(num_evaders+num_pursuers)
odoms = compute_odometry(env)
#     for ep in range(epochs):
#         for st in range(steps_per_epoch):
#             # compute evaders action
#             for ie in range(num_evaders):
#                 if not done[ie]:
#                     acts[ie], vals[ie], logps[ie] = agent_eva.step(obs, odoms[ie])

