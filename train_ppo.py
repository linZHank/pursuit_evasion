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

from pe_env import PursuitEvasion

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

class GaussianActor(Actor):
    def __init__(self):
        pass


epochs = 4
steps_per_epoch = 100
if __name__='__main__':
    env=PursuitEvasion()
    obs, ep_ret, ep_len = env.reset(), 0, 0
    for ep in range(epoch):
        for st in range(steps_per_epoch):
            pass
