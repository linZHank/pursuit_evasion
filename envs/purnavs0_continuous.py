#!/usr/bin/python3
"""
Pursuer Navigation Environment (continuous action space): a toy environment for testing algorithms. 
    - Fixed obstacles (1 circle, 2 rectangles)
    - 1 Pursuer (controllable)
    - 1 Evader (still)
    - observation space: 3-channel image
    - action space: int scalar, discrete
"""
import numpy as np
import cv2
from numpy import pi
from numpy import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, RegularPolygon
from matplotlib.collections import PatchCollection
from .purnavs0 import PursuerNavigationScene0


class PursuerNavigationScene0Continuous(PursuerNavigationScene0):

    def __init__(self, resolution=(80, 80)):
        super(PursuerNavigationScene0Continuous, self).__init__(resolution)
        self.name = 'pursuer_navigation_scene0_continuous'
        self.action_space_shape = (2,)
        self.action_space_high = 2.*np.ones(self.action_space_shape)
        self.action_space_low = -self.action_space_high

    def step(self, action):
        """
        Agents take velocity command
        Args:
            action: array([fx_p0,fy_p0])
        Returns:
            obs: map image
            reward
            done
            info: episode result or ''
        """
        # Check input
        assert action.shape==(2,)
        action = np.clip(action, self.action_space_low, self.action_space_high)
        # Prepare
        prev_distance = np.linalg.norm(self.pursuers['position'] - self.evaders['position']) 
        bonus = 0 # add bonus when key event detected
        reward = 0
        done = False
        info = ''
        obs = np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint8)
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
            ## detect collision
            if any(
                [
                    self._is_outbound(self.pursuers['position'][0], radius=self.pursuer_radius),
                    self._is_occluded(self.pursuers['position'][0], radius=self.pursuer_radius),
                ]
            ):
                self.pursuers['status'][0] = 'deactivated'
            distance = np.linalg.norm(self.pursuers['position'] - self.evaders['position'])
            bonus = 10*(prev_distance - distance) - .1
        else:
            action = np.zeros(2)
            self.pursuers['velocity'][0] = np.zeros(2)
        ## detect captures
        if self.pursuers['status'][0] == 'active': # status updated, check status again
            if self.evaders['status'][0] =='active':
                if np.linalg.norm(self.pursuers['position'][0] - self.evaders['position'][0]) <= self.interfere_radius:
                    self.evaders['status'][0] = 'deactivated'
                    bonus = 100.
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
            obs[:,:,2] = 255*self._get_image(
                patch_list=[circle], 
                radius=self.pursuer_radius
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

