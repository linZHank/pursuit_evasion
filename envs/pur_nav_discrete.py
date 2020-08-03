#!/usr/bin/python3
"""
Pursuer Navigation Environment: a toy environment for testing RL algorithms with continuous action space. 
    - Random placed obstacles (number<=14)
    - 1 Pursuer (controllable)
    - 1 Evader (still)
"""
import numpy as np
import cv2
from numpy import pi
from numpy import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, RegularPolygon, Circle
from matplotlib.collections import PatchCollection
from pur_nav import PursuerNavigation


class PursuerNavigationDiscrete(PursuerNavigation):

    def __init__(self, resolution=(80, 80)):
        super(PursuerNavigationDiscrete, self).__init__(resolution)
        self.name = 'pur_nav_discrete'
        self.action_options = np.array([
            [0.,0.],
            [-1.,0.],
            [1.,0.],
            [0.,-1.],
            [0.,1.]
        ])


    def step(self, action_index):
        """
        Agents take velocity command
        Args:
            action: index 0: [0,0], 1: [-1,0], 2: [1,0], 3: [0,-1], 4: [0,1]
        Returns:
            obs: map image
            reward: r_p0
            done
            info: episode result or ''
        """
        # Check input
        assert isinstance(action_index, int)
        assert 0<=action_index<=self.action_options.shape[0]
        action = self.action_options[action_index]
        action = np.clip(action, self.action_space_low, self.action_space_high)
        # Prepare, following should be identical to pur_nav
        bonus = 0 # add bonus when key event detected
        reward = 0
        done = False
        info = ''
        obs = np.zeros((self.resolution[0], self.resolution[1], 3))
        obs[:,:,1] = self.image[:,:,1]
        # Step evaders
        obs[:,:,0] = self.image[:,:,0]

        # Step pursuers
        if self.pursuers['status'][0] == 'active':
            d_vel = (action/self.pursuer_mass  - self.damping*self.pursuers['velocity'][0])/self.rate
            self.pursuers['velocity'][0] += d_vel
            self.pursuers['velocity'][0] = np.clip(self.pursuers['velocity'][0], -self.pursuer_max_speed, self.pursuer_max_speed)
            d_pos = self.pursuers['velocity'][0]/self.rate
            self.pursuers['position'][0] += d_pos # possible next pos
            if any(
                [
                    self._is_outbound(self.pursuers['position'][0], radius=self.pursuer_radius),
                    self._is_occluded(self.pursuers['position'][0], radius=self.pursuer_radius),
                ]
            ):
                self.pursuers['status'][0] = 'deactivated'
                bonus = -np.sqrt(2*self.action_space_high**2)*self.max_episode_steps/10.
            else:
                bonus = -np.linalg.norm(action)/10.
        else:
            action = np.zeros(2)
            self.pursuers['velocity'][0] = np.zeros(2)
        ## detect captures
        if self.pursuers['status'][0] == 'active': # status updated, check status again
            if self.evaders['status'][0] =='active':
                if np.linalg.norm(self.pursuers['position'][0] - self.evaders['position'][0]) <= self.interfere_radius:
                    self.evaders['status'][0] = 'deactivated'
                    bonus = np.sqrt(2*self.action_space_high**2)*self.max_episode_steps/10.
        ## record pursuers trajectory
        self.pursuers['trajectory'].append(self.pursuers['position'].copy())
        ## create pursuer patches, 圆滑世故
        self.pursuer_patches = []
        if self.pursuers['status'][0] == 'active':
            circle = Circle(
                xy=self.pursuers['position'][0], 
                radius=self.pursuer_radius, 
                fc='deepskyblue'
            )
            self.pursuer_patches.append(circle)
            obs[:,:,-1] = 0.9*np.transpose(
                self._get_image(
                    patch_list=[circle], 
                    radius=self.pursuer_radius
                )
            )
        # Create map image, obstacle channel no need to change 
        self.image[:,:,2] = obs[:,:,2] # B: pursuer channel
        # Finish step
        self.step_counter += 1
        ## reward
        reward += bonus
        ## done if deactivated
        done = (self.evaders['status'][0]=='deactivated' or self.pursuers['status'][0]=='deactivated')
        if self.step_counter == self.max_episode_steps:
            info = "Timeup"
        ## info
        if self.evaders['status'][0]=='deactivated':
            info = "Goal reached!"
        if self.pursuers['status'][0]=='deactivated':
            info = "Pursuer crashed"

        return obs, reward, done, info


# usage example refer to test script
