#!/usr/bin/python3
"""
Pursuit-evasion environment:
    - Randomly placed obstacles
    - Multiple Obstacles (>=0)
    - Multiple Pursuers (>=1)
    - Multiple Evaders (>=1)
    - Homogeneous agents
Note:
    - Discrete action space
"""
import numpy as np
from numpy import pi
from numpy import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, RegularPolygon, Circle
from matplotlib.collections import PatchCollection
from pe_env import PursuitEvasion

class PursuitEvasionDiscrete(PursuitEvasion):
    """
    Dynamics Pursuit-evasion env: N pursuers, N evader (1<=N<=4)
    """
    def __init__(self, resolution=(100, 100)):
        super().__init__(resolution)
        self.action_reservoir = np.array([[0,0], [0,1], [0,-1], [-1,0], [1,0]])  # 0: None, 1: Up, 2: Down, 3: Left, 4: Right

    def step(self, action_indices):
        """
        Agents take velocity command
        Args:
            action_indices: array([a_e0, a_e1,...,a_pN])
        Returns:
            obs: map image
            reward:
            done: bool
            info: ''
        """
        # Check input and convert index into actual force
        assert action_indices.shape == (self.num_evaders+self.num_pursuers,)
        assert all(action_indices>=0) and all(action_indices<self.action_reservoir.shape[0])
        actions = np.zeros([self.num_evaders+self.num_pursuers, 2])
        for i in range(actions.shape[0]):
            actions[i] = self.action_reservoir[action_indices[i]]
        # Default reward, done, info
        bonus = np.zeros(self.num_evaders+self.num_pursuers) # add bonus when key event detected
        reward = np.zeros(self.num_evaders + self.num_pursuers)
        done = np.array([False]*(self.num_evaders + self.num_pursuers))
        info = ''
        # Limit input
        actions = np.clip(actions, self.action_space_low, self.action_space_high)
        # Step evaders
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie] == 'active':
                d_vel = (actions[ie]/self.evader_mass - self.damping*self.evaders['velocity'][ie])/self.rate
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
                    self._disable_evader(id=ie)
                    bonus[ie] = -self.max_episode_steps/10.
                else:
                    bonus[ie] = -np.linalg.norm(actions[ie])/10.
            else:
                actions[ie] = np.zeros(2)
                self.evaders['velocity'][ie] = np.zeros(2)
        ## record evaders trajectory
        self.evaders['trajectory'].append(self.evaders['position'].copy())
        ## create evader patches, 八面玲珑
        self.evader_patches = []
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie] == 'active':
                octagon = RegularPolygon(xy=self.evaders['position'][ie], numVertices=8, radius=self.evader_radius, fc='orangered')
                self.evader_patches.append(octagon)
        ## create evader map
        self.evader_map = self._get_map(patch_list=self.evader_patches, radius=self.evader_radius)

        # Step pursuers
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip] == 'active':
                d_vel = (actions[-self.num_pursuers+ip]/self.pursuer_mass  - self.damping*self.pursuers['velocity'][ip])/self.rate
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
                    self._disable_pursuer(id=ip)
                    bonus[-self.num_pursuers+ip] = -self.max_episode_steps/10.
                else:
                    bonus[-self.num_pursuers+ip] = -np.linalg.norm(actions[-self.num_pursuers+ip])/10.
            else:
                actions[-self.num_pursuers+ip] = np.zeros(2)
                self.pursuers['velocity'][ip] = np.zeros(2)
            ## detect captures
            if self.pursuers['status'][ip] == 'active': # status updated, check status again
                for ie in range(self.num_evaders):
                    if self.evaders['status'][ie] =='active':
                        if np.linalg.norm(self.pursuers['position'][ip] - self.evaders['position'][ie]) <= self.interfere_radius:
                            self._disable_evader(id=ie)
                            bonus[ie] = -self.max_episode_steps/10.
                            bonus[-self.num_pursuers+ip] = self.max_episode_steps/10.
        ## record pursuers trajectory
        self.pursuers['trajectory'].append(self.pursuers['position'].copy())
        ## create pursuer patches, 圆滑世故
        self.pursuer_patches = []
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip] == 'active':
                circle = Circle(xy=self.pursuers['position'][ip], radius=self.pursuer_radius, fc='deepskyblue')
                self.pursuer_patches.append(circle)
        ## create pursuer map
        self.pursuer_map = self._get_map(patch_list=self.pursuer_patches, radius=self.pursuer_radius)
        # Combine maps in the order of RGB 
        self.map[:,:,0] = 0.5*np.transpose(self.evader_map) # R
        self.map[:,:,1] = 0.5*np.transpose(self.obstacle_map) # G
        self.map[:,:,2] = 0.5*np.transpose(self.pursuer_map) # B
        # Output
        self.step_counter += 1
        ## obs
        obs = self.map.copy()
        ## reward
        reward += bonus.copy()
        ## done if deactivated
        done = np.array([s=='deactivated' for s in self.evaders['status']] + [s=='deactivated' for s in self.pursuers['status']])
        if self.step_counter == self.max_episode_steps:
            done = np.array([True]*(self.num_evaders + self.num_pursuers))
        ## info
        if all(done[:self.num_evaders]): # pursuers win
            info = "All evaders deceased"
        if all(done[-self.num_pursuers:]): # evaders win
            info = "All pursuers deceased"

        return obs, reward, done, info


