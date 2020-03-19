#!/usr/bin/python3
from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import random


class QTableAgent(object):
    def __init__(self, name, env):
        # fixed
        self.name = name
        # hyper-parameters
        self.warmup_steps = 1
        self.init_eps = 1.
        self.final_eps = 0.1


    def epsilon_greedy(self, state_index):
        """
        Take action based on epsilon_greedy
        Args:
            state_index: [i_dx, i_dy]
        Returns:
            action_index:
        """
        if random.uniform() > self.epsilon:
            action_index = np.argmax(self.q_table[state_index[0],state_index[1], state_index[2]])
        else:
            action_index = random.randint(self.action_space[0])
            print("!{} Take a random action: {}:{}".format(self.name, action_index, self.actions[action_index]))

        return action_index

    def linear_epsilon_decay(self, episode, decay_period):
        """
        Returns the current epsilon for the agent's epsilon-greedy policy. This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et al., 2015). The schedule is as follows:
            Begin at 1. until warmup_steps steps have been taken; then Linearly decay epsilon from 1. to final_eps in decay_period steps; and then Use epsilon from there on.
        Args:
            decay_period: int
            episode: int
        Returns:
        """
        episodes_left = decay_period + self.warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., self.init_eps-self.final_eps)
        self.epsilon = self.final_eps + bonus

    def train(self, state_index, action_index, next_state_index, reward):
        """
        Update Q-table
        """
        # Q(s,a)=Q(s,a)+alpha*(r+gamma*Q(s',a')-Q(s,a))
        pass


    def obs_to_state(self, obs):
        """
        Convert observation into indices in Q-table
        Args:
            obs: {target,catcher}
        Returns:
            state: array([dx, dy, cable_length])
            state_index: [dim_0, dim_1, ...], index of state in Q-table
        """
        pass

    def save_table(self, save_dir):
        save_dir = os.path.join(save_dir, self.name)
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        np.save(os.path.join(save_dir, 'q_table.npy'), self.q_table)
        print("\nQ-table saved!\n")
