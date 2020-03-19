#!/usr/bin/python3
"""
Task environment for pursuit-evasion, kinematics case.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi
from numpy import random
import time
import matplotlib
import matplotlib.pyplot as plt

import pdb


class PEKineEnv(object):
    """
    Pursuit-evasion kinematics env class
    """
    def __init__(self, num_pursuers=2):
        # fixed
        self.num_pursuers = num_pursuers
        self.world_length = 10.
        self.max_steps = 1000
        self.rate = 100 # Hz
        # variable, we use polar coord to represent objects location
        self.obstacle_circles = dict(
            position = np.zeros((1,2)),
            radius = np.array([self.world_length/8])
        )
        self.obstacle_rectangles = dict(
            position = np.array([[-self.world_length/8,-self.world_length/2],[-self.world_length/8,self.world_length/2-self.world_length/10]]),
            dimension = np.array([[self.world_length/4,self.world_length/10],[self.world_length/4,self.world_length/10]])
        )
        self.evader = dict(position=np.array([[self.world_length/7,0]]), velocity=np.zeros((1,2)))
        self.pursuers = dict(
            id = ['pursuer_'+str(i) for i in range(num_pursuers)],
            position = np.array([[-4*self.world_length/9, -4*self.world_length/9], [-4*self.world_length/9, 4*self.world_length/9]]),
            velocity = np.zeros([num_pursuers,2])
        )

        self.step_count = 0
        #
        fig, ax = plt.subplots(figsize=(16, 16))


    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: {target_loc: array([x,y]), catcher_loc: array([x,y], cabler_loc: array([x,y]...)
            info: 'coordinate type'
        """
        self.step_count = 0
        theta_e = random.uniform(-pi,pi)
        self.evader['position'] = np.array([[self.world_length/7*np.cos(theta_e),self.world_length/7*np.sin(theta_e)]])
        self.evader['velocity'] = np.zeros((1,2))
        pos_choice = np.array([-4*self.world_length/9, 4*self.world_length/9])
        self.pursuers['position'][0] = random.choice(pos_choice,2)
        self.pursuers['position'][1] = random.choice(pos_choice,2)
        while np.array_equal(self.pursuers['position'][0], self.pursuers['position'][1]):
            self.pursuers['position'][1] = random.choice(pos_choice,2)
        self.pursuers['velocity'] = np.zeros((self.num_pursuers,2))
        obs = dict(pursuers=self.pursuers, evader=self.evader)
        info = ''

        return obs, info

    def step(self, action):
        """
        Take a resolved velocity command
        Args:
            action: array([v_x,v_y])
        Returns:
            obs: {target_loc: array([x,y]), catcher_loc: array([x,y])
            reward:
            done: bool
            info: 'coordinate type'
        """
        pass

    def render(self,pause=1):
        fig, ax = plt.gcf(), plt.gca()
        # plot world boundary
        bound = plt.Rectangle((-self.world_length/2,-self.world_length/2), self.world_length, self.world_length, linewidth=2, color='k', fill=False)
        ax.add_patch(bound)
        # draw obstacles
        for ci in range(len(self.obstacle_circles['radius'])):
            obs_circle = plt.Circle((self.obstacle_circles['position'][ci]), self.obstacle_circles['radius'][ci], color='grey')
            ax.add_patch(obs_circle)
        for ri in range(self.obstacle_rectangles['position'].shape[0]):
            obs_rect = plt.Rectangle((self.obstacle_rectangles['position'][ri]),self.obstacle_rectangles['dimension'][ri,0], self.obstacle_rectangles['dimension'][ri,1], color='grey')
            ax.add_patch(obs_rect)

        # draw pursuers and evader
        plt.scatter(self.evader['position'][0,0], self.evader['position'][0,1], s=200, marker='*', color='crimson')
        plt.scatter(self.pursuers['position'][:,0], self.pursuers['position'][:,1], s=200, marker='o', color='deepskyblue')
        # set axis
        plt.axis(1.1/2*np.array([-self.world_length,self.world_length,-self.world_length,self.world_length]))

        plt.show(block=False)
        plt.pause(pause)
        plt.clf()
