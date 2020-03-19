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
        self.rate = 30 # Hz
        self.num_pursuers = num_pursuers
        self.world_length = 10.
        self.max_steps = self.rate*20 # 20s
        # variable, we use polar coord to represent objects location
        self.obstacle_circles = dict(
            names = ['circle_'+str(i) for i in range(1)],
            position = np.zeros((1,2)),
            radius = np.array([self.world_length/8])
        )
        self.obstacle_rectangles = dict(
            names = ['rectangle_'+str(i) for i in range(2)],
            position = np.array([[-self.world_length/8,-self.world_length/2],[-self.world_length/8,self.world_length/2-self.world_length/10]]),
            dimension = np.array([[self.world_length/4,self.world_length/10],[self.world_length/4,self.world_length/10]])
        )
        self.evader = dict(position=np.array([[self.world_length/7,0]]), velocity=np.zeros((1,2)), trajectory=[])
        self.pursuers = dict(
            names = ['pursuer_'+str(i) for i in range(num_pursuers)],
            position = np.array([[-4*self.world_length/9, -4*self.world_length/9], [-4*self.world_length/9, 4*self.world_length/9]]),
            velocity = np.zeros([num_pursuers,2]),
            trajectory = []
        )

        self.step_count = 0
        #
        fig, ax = plt.subplots(figsize=(16, 16))


    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: {evader, pursuers)
            info: ''
        """
        self.step_count = 0
        # reset evader
        theta_e = random.uniform(-pi,pi)
        self.evader['position'] = np.array([[self.world_length/7*np.cos(theta_e),self.world_length/7*np.sin(theta_e)]])
        self.evader['velocity'] = np.zeros((1,2))
        self.evader['trajectory'] = []
        self.evader['trajectory'].append(self.evader['position'])
        # reset pursuers
        pos_choice = np.array([-4*self.world_length/9, 4*self.world_length/9])
        self.pursuers['position'][0] = random.choice(pos_choice,2)
        self.pursuers['position'][1] = random.choice(pos_choice,2)
        while np.array_equal(self.pursuers['position'][0], self.pursuers['position'][1]):
            self.pursuers['position'][1] = random.choice(pos_choice,2)
        self.pursuers['velocity'] = np.zeros((self.num_pursuers,2))
        self.pursuers['trajectory'] = []
        self.pursuers['trajectory'].append(self.pursuers['position'])
        obs = dict(evader=self.evader, pursuers=self.pursuers)
        info = ''

        return obs, info

    def step(self, action_evader, action_pursuers):
        """
        Agents take velocity command
        Args:
            action_evader: array([[v_x,v_y]])
            action_pursuers: array([[v_x0,v_y0],[v_x1,v_y1],...])
        Returns:
            obs: {evader, pursuers)
            reward: +1 if pursuers succeeded
            done: bool
            info: 'whatever'
        """
        # set limitation for velocity commands
        action_evader = np.clip(action_evader, -self.world_length/4, self.world_length/4)
        action_pursuers = np.clip(action_pursuers, -self.world_length/4, self.world_length/4)
        # step evader
        if not self._collide_circles(self.evader['position'][0]+action_evader/self.rate) and not self._collide_rectangles(self.evader['position'][0]+action_evader/self.rate):
            if not self._out_of_bound(self.evader['position'][0]+action_evader/self.rate): # detect obstacles collision
                self.evader['position'][0] += action_evader/self.rate
        self.evader['trajectory'].append(self.evader['position'])
        # step pursuers
        for i in range(len(self.pursuers['names'])):
            if not self._collide_circles(self.pursuers['position'][i]+action_pursuers[i]/self.rate) and not self._collide_rectangles(self.pursuers['position'][i]+action_pursuers[i]/self.rate):
                if not self._out_of_bound(self.pursuers['position'][i]+action_pursuers[i]/self.rate): # detect obstacles collision
                    self.pursuers['position'][i] += action_pursuers[i]/self.rate
        self.pursuers['trajectory'].append(self.pursuers['position'])
        # collect results
        self.step_count += 1
        obs = dict(evader=self.evader, pursuers=self.pursuers)
        reward = 0
        done = False
        if self.step_count >= self.max_steps:
            done = True
        info = ''

        return obs, reward, done, info

    def render(self,pause=2):
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

    # circle collision detection
    def _collide_circles(self, pos):
        flag = False
        for i in range(len(self.obstacle_circles['names'])):
            if (pos[0]-self.obstacle_circles['position'][i,0])**2+(pos[1]-self.obstacle_circles['position'][i,1])**2 <= self.obstacle_circles['radius'][i]**2:
                flag = True
        return flag

    # rectangle collision detection
    def _collide_rectangles(self, pos):
        flag = False
        for i in range(len(self.obstacle_rectangles['names'])):
            if 0<=(pos[0]-self.obstacle_rectangles['position'][i,0])<=self.obstacle_rectangles['dimension'][i,0] and 0<=(pos[1]-self.obstacle_rectangles['position'][i,1])<=self.obstacle_rectangles['dimension'][i,1]:
                flag = True
        return flag

    # out of bound detection
    def _out_of_bound(self, pos):
        flag = False
        if np.absolute(pos[0])>=self.world_length/2 or np.absolute(pos[1])>=self.world_length/2:
            flag = True

        return flag
