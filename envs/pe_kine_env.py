#!/usr/bin/python3

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
    Pursuit-evasion kinematics env: N pursuers, 1 evader
    """
    def __init__(self, num_pursuers=2):
        # fixed
        self.rate = 20 # Hz
        self.num_pursuers = num_pursuers
        self.world_length = 10.
        self.interfere_radius = 0.1 # effective catch within this range
        self.max_steps = self.rate*30 # 30s
        self.obstacle_circles = dict(
            names = ['circle_'+str(i) for i in range(1)],
            position = np.zeros((1,2)),
            radius = np.array([2])
        )
        self.obstacle_rectangles = dict(
            names = ['rectangle_'+str(i) for i in range(2)],
            position = np.array([[-2,-self.world_length/2],[-2,self.world_length/2-1.5]]),
            dimension = np.array([[4,1.5],[4,1.5]])
        )
        self.pursuer_spawning_pool = np.array([-4,4])
        # variables
        self.step_counter = 0
        self.evaders = dict(names=['evaders_0'], position=np.zeros((1,2)), velocity=np.zeros((1,2)), trajectory=[])
        self.pursuers = dict(
            names = ['pursuer_'+str(i) for i in range(num_pursuers)],
            position = np.zeros((num_pursuers,2)),
            velocity = np.zeros((num_pursuers,2)),
            trajectory = []
        )
        self.values = np.zeros(num_pursuers) # distance from each pursuer to the evader
        # create figure
        fig, ax = plt.subplots(figsize=(16, 16))

    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: {evader, pursuers)
            info: ''
        """
        self.step_counter = 0
        # reset evader
        theta_e = random.uniform(-pi,pi)
        self.evaders['position'] = np.array([[3*np.cos(theta_e),3*np.sin(theta_e)]])
        self.evaders['velocity'] = np.zeros((1,2))
        self.evaders['trajectory'] = []
        self.evaders['trajectory'].append(self.evaders['position'][0])
        # reset pursuers
        for i in range(len(self.pursuers['names'])):
            self.pursuers['position'][i] = random.choice(self.pursuer_spawning_pool,2)+random.normal(0,0.1,2)
            while self.out_of_bound(self.pursuers['position'][i]):
                self.pursuers['position'][i] = random.choice(self.pursuer_spawning_pool,2)+random.normal(0,0.1,2)
        self.pursuers['velocity'] = np.zeros((self.num_pursuers,2))
        self.pursuers['trajectory'] = []
        self.pursuers['trajectory'].append(self.pursuers['position'])
        obs = np.concatenate((self.pursuers['position'].reshape(-1), self.evaders['position'][0]), axis=0)
        info = ''
        self.values = -np.linalg.norm(self.evaders['position'][0]-self.pursuers['position'], axis=1) # distance from each pursuer to the evader

        return obs, info

    def step(self, action_evaders, action_pursuers):
        """
        Agents take velocity command
        Args:
            action_evaders: array([[v_x,v_y]])
            action_pursuers: array([[v_x0,v_y0],[v_x1,v_y1],...])
        Returns:
            obs: {evaders, pursuers)
            reward: -|evaders-pursuers|
            done: bool
            info: ''
        """
        reward, done, info = np.zeros(self.num_pursuers), False, ''
        dist_prev = -self.values
        # set limitation for velocity commands
        action_evaders = np.clip(action_evaders, -self.world_length/4, self.world_length/4)
        action_pursuers = np.clip(action_pursuers, -self.world_length/4, self.world_length/4)
        # step evaders
        if not self.obstacles_collision(self.evaders['position'][0]+action_evaders/self.rate):
            if not self.out_of_bound(self.evaders['position'][0]+action_evaders/self.rate): # detect obstacles collision
                self.evaders['position'][0] += action_evaders/self.rate
            else:
                info = "{} out of bound".format(self.evaders['names'][0])
                done = True
        else:
            info = "{} collide obstacle".format(self.evaders['names'][0])
            done = True
        self.evaders['trajectory'].append(self.evaders['position'][0])
        # step pursuers
        for i in range(len(self.pursuers['names'])):
            if not self.obstacles_collision(self.pursuers['position'][i]+action_pursuers[i]/self.rate):
                if not self.out_of_bound(self.pursuers['position'][i]+action_pursuers[i]/self.rate): # detect obstacles collision
                    self.pursuers['position'][i] += action_pursuers[i]/self.rate
                else:
                    info = "{} out of bound".format(self.pursuers['names'][i])
                    done = True
            else:
                info = "{} collide obstacle".format(self.pursuers['names'][i])
                done = True
        self.pursuers['trajectory'].append(self.pursuers['position'])
        # update reward, done, info
        obs = np.concatenate((self.pursuers['position'].reshape(-1), self.evaders['position'][0]), axis=0)
        self.values = -np.linalg.norm(self.evaders['position'][0]-self.pursuers['position'], axis=1)
        # dist_curr = -self.values
        # reward = dist_prev - dist_curr
        reward = self.values
        if self.step_counter >= self.max_steps:
            info = "maximum step: {} reached".format(self.max_steps)
            done = True
        self.step_counter += 1

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
        # draw pursuers and evaders
        plt.scatter(self.evaders['position'][:,0], self.evaders['position'][:,1], s=400, marker='*', color='crimson', linewidth=2)
        plt.scatter(self.pursuers['position'][:,0], self.pursuers['position'][:,1], s=400, marker='o', color='deepskyblue', linewidth=2)
        # set axis
        plt.axis(1.1/2*np.array([-self.world_length,self.world_length,-self.world_length,self.world_length]))

        plt.show(block=False)
        plt.pause(pause)
        plt.clf()

    # Helper Functions
    # obstacles collision detection
    def obstacles_collision(self, pos):
        collision_flag = False
        if self._collide_circles(pos) or self._collide_rectangles(pos):
            collision_flag = True

        return collision_flag

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
    def out_of_bound(self, pos):
        flag = False
        if np.absolute(pos[0])>=self.world_length/2 or np.absolute(pos[1])>=self.world_length/2:
            flag = True

        return flag
