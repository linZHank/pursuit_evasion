#!/usr/bin/python3
"""
Pursuit-evasion environment:
    - Randomly placed obstacles
    - Multiple Obstacles (<=14)
    - 1 Pursuer
    - 1 Evader 
    - Homogeneous agents
Note:
    - Discrete action space
"""
import numpy as np
import cv2
from numpy import pi
from numpy import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, RegularPolygon, Circle
from matplotlib.collections import PatchCollection
from .pe1v1 import PursuitEvasionOneVsOne


class PursuitEvasionOneVsOneContinuous(PursuitEvasionOneVsOne):

    def __init__(self, resolution=(80, 80)):
        super(PursuitEvasionOneVsOneContinuous, self).__init__(resolution)
        self.name = 'pursuit_evasion_1v1_continuous'
        self.action_space_shape = (2,2)
        self.action_space_high = 2.*np.ones(self.action_space_shape)
        self.action_space_low = -self.action_space_high

    def step(self, action):
        """
        Agents take velocity command
        Args:
            action_indices: array([a_e, a_p])
        Returns:
            obs: map image
            reward: array([r_e, r_p])
            done: bool array([d_e, d_p])
            info: episode result or ''
        """
        # Check input 
        assert action.shape==self.action_space_shape
        action = np.clip(action, self.action_space_low, self.action_space_high)
        # Prepare
        prev_distance = np.linalg.norm(self.pursuers['position'] - self.evaders['position']) 
        bonus = np.zeros(self.num_evaders+self.num_pursuers) # add bonus when key event detected
        reward = np.zeros(self.num_evaders + self.num_pursuers)
        done = np.array([False]*(self.num_evaders + self.num_pursuers))
        info = ''
        obs = np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint8)
        obs[:,:,1] = self.image[:,:,1] # obstacle image
        # Step evaders
        obs[:,:,0] = self.image[:,:,0]
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie] == 'active':
                d_vel = (action[ie]/self.evader_mass - self.damping*self.evaders['velocity'][ie])/self.rate
                self.evaders['velocity'][ie] += d_vel
                self.evaders['velocity'][ie] = np.clip(self.evaders['velocity'][ie], -self.evader_max_speed, self.evader_max_speed)
                d_pos = self.evaders['velocity'][ie]/self.rate
                self.evaders['position'][ie] += d_pos # possible next pos
                if any(
                    [
                        self._is_outbound(self.evaders['position'][ie], radius=self.evader_radius),
                        self._is_occluded(self.evaders['position'][ie], radius=self.evader_radius),
                    ]
                ):
                    self.evaders['status'][ie] = 'deactivated'
                    bonus[ie] = -100. # -np.sqrt(2*self.action_space_high**2)*self.max_episode_steps/10.
                else:
                    bonus[ie] = .1 # -np.linalg.norm(action[ie])/10.
            else:
                action[ie] = np.zeros(2)
                self.evaders['velocity'][ie] = np.zeros(2)
        ## record evaders trajectory
        self.evaders['trajectory'].append(self.evaders['position'].copy())
        ## create evader patches, 八面玲珑
        self.evader_patches = []
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie] == 'active':
                octagon = RegularPolygon(
                    xy=self.evaders['position'][ie], 
                    numVertices=8, 
                    radius=self.evader_radius, 
                    fc='orangered'
                )
                self.evader_patches.append(octagon)
        obs[:,:,0] = 255*self._get_image(patch_list=self.evader_patches, radius=self.evader_radius) 
        ## Create map image, obstacle channel no need to change 
        self.image[:,:,0] = obs[:,:,0] # B: pursuer channel

        # Step pursuer
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip] == 'active':
                d_vel = (action[-self.num_pursuers+ip]/self.pursuer_mass  - self.damping*self.pursuers['velocity'][ip])/self.rate
                self.pursuers['velocity'][ip] += d_vel
                self.pursuers['velocity'][ip] = np.clip(self.pursuers['velocity'][ip], -self.pursuer_max_speed, self.pursuer_max_speed)
                d_pos = self.pursuers['velocity'][ip]/self.rate
                self.pursuers['position'][ip] += d_pos # possible next pos
                if any(
                    [
                        self._is_outbound(self.pursuers['position'][ip], radius=self.pursuer_radius),
                        self._is_occluded(self.pursuers['position'][ip], radius=self.pursuer_radius),
                    ]
                ):
                    self.pursuers['status'][ip] = 'deactivated'
                    bonus[-self.num_pursuers+ip] = -100. # -np.sqrt(2*self.action_space_high**2)*self.max_episode_steps/10.
                else:
                    bonus[-self.num_pursuers+ip] = -.1 # -np.linalg.norm(action[-self.num_pursuers+ip])/10.
            else: # make sure deactivated pursuers not moving
                action[-self.num_pursuers+ip] = np.zeros(2)
                self.pursuers['velocity'][ip] = np.zeros(2)
            ## detect captures
            if self.pursuers['status'][ip] == 'active': # status updated, check status again
                for ie in range(self.num_evaders):
                    if self.evaders['status'][ie] =='active':
                        if np.linalg.norm(self.pursuers['position'][ip] - self.evaders['position'][ie]) <= self.interfere_radius:
                            self.evaders['status'][ie] = 'deactivated'
                            bonus[ie] = -100. 
                            bonus[-self.num_pursuers+ip] = 100. 
        ## record pursuers trajectory
        self.pursuers['trajectory'].append(self.pursuers['position'].copy())
        ## create pursuer patches, 圆滑世故
        self.pursuer_patches = []
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip] == 'active':
                circle = Circle(
                    xy=self.pursuers['position'][ip], 
                    radius=self.pursuer_radius, 
                    fc='deepskyblue'
                )
                self.pursuer_patches.append(circle)
        obs[:,:,2] = 255*self._get_image(patch_list=self.pursuer_patches, radius=self.pursuer_radius) 
        ## Create map image, obstacle channel no need to change 
        self.image[:,:,2] = obs[:,:,2] # B: pursuer channel

        # Finish step
        self.step_counter += 1
        ## reward
        reward += bonus.copy()
        ## done if deactivated
        done = np.array([s=='deactivated' for s in self.evaders['status']] + [s=='deactivated' for s in self.pursuers['status']])
        if all(done[:self.num_evaders]):
            done[:] = True
        ## info
        if self.step_counter == self.max_episode_steps:
            info = "Timeup"
        if all(done[:self.num_evaders]): # pursuers win
            info = "All evaders deceased"
        if all(done[-self.num_pursuers:]): # evaders win
            info = "All pursuers deceased"

        return obs, reward, done, info

