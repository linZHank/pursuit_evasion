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
    def __init__(self, num_evaders=1, num_pursuers=1):
        # fixed
        self.rate = 20 # Hz
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.world_length = 10.
        self.interfere_radius = 0.2 # effective catch within this range
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
        # variables
        self.evaders_spawning_pool = np.zeros([num_evaders, 2])
        self.pursuers_spawning_pool = np.zeros([num_pursuers, 2])
        self.step_counter = 0
        self.evaders = dict(
            names = ['evaders'+str(i) for i in range(num_evaders)],
            position = np.zeros((num_evaders,2)),
            velocity = np.zeros((num_evaders,2)),
            trajectory = [],
            status = ['']*num_evaders
        )
        self.pursuers = dict(
            names = ['pursuer_'+str(i) for i in range(num_pursuers)],
            position = np.zeros((num_pursuers,2)),
            velocity = np.zeros((num_pursuers,2)),
            trajectory = [],
            status = ['']*num_pursuers
        )
        # create figure
        fig, ax = plt.subplots(figsize=(16, 16))

    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: array([x_p0,y_p0,...,v_x_p0,v_y_p0,...,x_e0,y_e0,...,v_x_e0,v_y_e0])
            info: ''
        """
        self.step_counter = 0
        # reset evader
        for i in range(self.num_evaders):
            self.evaders['position'][i] = self.evaders_spawning_pool[i]
            while self.is_outbound(self.evaders['position'][i]) or self.is_occluded(self.evaders['position'][i]):
                self.evaders['position'][i] = random.uniform(-self.world_length/2, self.world_length/2, 2)
        self.evaders['velocity'] = np.zeros((self.num_evaders,2))
        self.evaders['trajectory'] = []
        self.evaders['trajectory'].append(self.evaders['position'])
        self.evaders['status'] = ['evading']*self.num_evaders
        # reset pursuers
        for i in range(self.num_pursuers):
            self.pursuers['position'][i] = self.pursuers_spawning_pool[i]
            while self.is_outbound(self.pursuers['position'][i]) or self.is_occluded(self.pursuers['position'][i]) or sum(self.interfere_evaders(self.pursuers['position'][i])):
                self.pursuers['position'][i] = random.uniform(-self.world_length/2, self.world_length/2, 2)
        self.pursuers['velocity'] = np.zeros((self.num_pursuers,2))
        self.pursuers['trajectory'] = []
        self.pursuers['trajectory'].append(self.pursuers['position'])
        self.pursuers['status'] = ['pursuing']*self.num_pursuers
        # create obs and info

        # obs = np.concatenate(
        #     (
        #         self.pursuers['position'].reshape(-1),
        #         self.pursuers['velocity'].reshape(-1),
        #         self.evaders['position'].reshape(-1),
        #         self.evaders['velocity'].reshape(-1)
        #     ), axis=0
        # )

        info = ''
        obs = np.concatenate((self.pursuers['position'].reshape(-1),self.evaders['position'].reshape(-1)), axis=0)

        return obs, info

    def step(self, action_evaders, action_pursuers):
        """
        Agents take velocity command
        Args:
            action_evaders: array([[v_x,v_y]])
            action_pursuers: array([[v_x0,v_y0],[v_x1,v_y1],...])
        Returns:
            obs: array([x_p0,y_p0,...,v_x_p0,v_y_p0,...,x_e0,y_e0,...,v_x_e0,v_y_e0])
            reward: -|evaders-pursuers|
            done: bool
            info: ''
        """
        # make sure actions are in right shapes
        assert action_evaders.shape == self.evaders['position'].shape
        assert action_pursuers.shape == self.pursuers['position'].shape
        # default reward, done, info
        reward, done, info = 0, False, ''
        # set limitation for velocity commands
        action_evaders = np.clip(action_evaders, -self.world_length/4, self.world_length/4)
        action_pursuers = np.clip(action_pursuers, -self.world_length/4, self.world_length/4)
        # step evaders
        temp_epos = self.evaders['position']+action_evaders/self.rate # possible next pos
        for i in range(self.num_evaders):
            if not self.is_outbound(temp_epos[i]):
                if not self.is_occluded(temp_epos[i]):
                    self.evaders['position'][i] = temp_epos[i]
                    # self.evaders['velocity'][i] = action_evaders[i]
                else:
                    # self.evaders['velocity'][i] = np.zeros(2)
                    self.evaders['status'][i] = 'occluded'
            else:
                # self.evaders['velocity'][i] = np.zeros(2)
                self.evaders['status'][i] = 'out'
        self.evaders['trajectory'].append(self.evaders['position'])
        # step pursuers
        temp_ppos = self.pursuers['position']+action_pursuers/self.rate
        for i in range(self.num_pursuers):
            if not self.is_outbound(temp_ppos[i]):
                if not self.is_occluded(temp_ppos[i]):
                    self.pursuers['position'][i] = temp_ppos[i]
                    self.pursuers['velocity'][i] = action_pursuers[i]
                    if sum(self.interfere_evaders(temp_ppos[i])):
                        self.pursuers['status'][i] = 'catching'
                else:
                    self.pursuers['velocity'][i] = np.zeros(2)
                    self.pursuers['status'][i] = 'occluded'
            else:
                self.pursuers['velocity'][i] = np.zeros(2)
                self.pursuers['status'][i] = 'out'
        self.pursuers['trajectory'].append(self.pursuers['position'])
        # update reward, done, info

        # obs = np.concatenate(
        #     (
        #         self.pursuers['position'].reshape(-1),
        #         self.pursuers['velocity'].reshape(-1),
        #         self.evaders['position'].reshape(-1),
        #         self.evaders['velocity'].reshape(-1)
        #     ), axis=0
        # )

        obs = np.concatenate((self.pursuers['position'].reshape(-1),self.evaders['position'].reshape(-1)), axis=0)
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
    def is_occluded(self, pos):
        collision_flag = False
        # detect collision with circles
        circle_flag = False
        for i in range(len(self.obstacle_circles['names'])):
            if (pos[0]-self.obstacle_circles['position'][i,0])**2+(pos[1]-self.obstacle_circles['position'][i,1])**2 <= self.obstacle_circles['radius'][i]**2:
                circle_flag = True
                break
        # detect collision with rectangles
        rect_flag = False
        for i in range(len(self.obstacle_rectangles['names'])):
            if 0<=(pos[0]-self.obstacle_rectangles['position'][i,0])<=self.obstacle_rectangles['dimension'][i,0] and 0<=(pos[1]-self.obstacle_rectangles['position'][i,1])<=self.obstacle_rectangles['dimension'][i,1]:
                rect_flag = True
                break
        # compute collision flag
        collision_flag = circle_flag or rect_flag

        return collision_flag

    # out of bound detection
    def is_outbound(self, pos):
        flag = False
        if np.absolute(pos[0])>=self.world_length/2 or np.absolute(pos[1])>=self.world_length/2:
            flag = True

        return flag

    def interfere_evaders(self, pos):
        flag = np.linalg.norm(pos-self.evaders['position'],axis=1) <= self.interfere_radius

        return flag
