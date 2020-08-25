#!/usr/bin/python3
"""
Pursuer Navigation Environment: a toy environment for testing algorithms. 
    - Fixed obstacles (1 circle, 2 rectangles)
    - 1 Pursuer (controllable)
    - 1 Evader (still)
    - observation space: 3-channel image
    - action space: 2-dim array, continuous
"""
import numpy as np
import cv2
from numpy import pi
from numpy import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, RegularPolygon
from matplotlib.collections import PatchCollection
from .pe import PursuitEvasion


class PursuerNavigationScene0(PursuitEvasion):

    def __init__(self, resolution=(80, 80)):
        super(PursuerNavigationScene0, self).__init__(resolution)
        self.name = 'pursuer_navigation_scene0'
        self.num_evaders = 1
        self.num_pursuers = 1
        self.num_circles = 1
        self.num_rectangles = 2
        self.observation_space_shape = list(resolution) + [3]
        self.action_space_shape = (2,)
        self.action_space_high = 2.*np.ones(self.action_space_shape)
        self.action_space_low = -self.action_space_high

    def reset(self):
        """
        Reset obstacles and agents location
        Args:
        Return:
            obs: map image
        """

        # Prepare 
        self.step_counter = 0
        self.evaders = dict(
            name = ['e-0'],
            position = np.inf*np.ones((1, 2)),
            velocity = np.zeros((1, 2)),
            trajectory = [],
            status = ['deactivated'],
        )
        self.pursuers = dict(
            name = ['p-0'],
            position = np.inf*np.ones((1, 2)),
            velocity = np.zeros((1, 2)),
            trajectory = [],
            status = ['deactivated'],
        )
        self.spawning_pool = random.uniform(
            -self.world_length/2+.5, self.world_length/2-.5,
            size=(2,2)
        ) # .5 threshold to avoid spawning too close to the walls
        obs = np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint8)

        # Reset obstacles: you can add more shapes in the section below
        self.obstacle_patches = []
        circle = Circle(
            xy=[0,0],
            radius=1,
            fc='grey'
        )
        self.obstacle_patches.append(circle)
        north_box = Rectangle(
            xy=[-2,3],
            width=4,
            height=2,
            fc='grey'
        )
        self.obstacle_patches.append(north_box)
        south_box = Rectangle(
            xy=[-2,-5],
            width=4,
            height=2,
            fc='grey'
        )
        self.obstacle_patches.append(south_box)
        obs[:,:,1] = 255*self._get_image(
            patch_list=self.obstacle_patches, 
            radius=self.world_length/np.min(self.resolution)/2
        )

        # Reset Evaders 
        self.evaders['position'][0] = self.spawning_pool[0]
        while any(
            [
                self._is_occluded(self.evaders['position'][0], radius=self.evader_radius),
                self._is_interfered(self.evaders['position'][0], radius=2*self.evader_radius)
            ]
        ): # evaders are sneaky so that they can stay closer to each other
            self.evaders['position'][0] = random.uniform(-self.world_length/2+.3, self.world_length/2-.3, 2)
        self.evaders['velocity'] = np.zeros((self.num_evaders,2))
        self.evaders['trajectory'].append(self.evaders['position'].copy())
        self.evaders['status'] = ['active']*self.num_evaders
        self.spawning_pool[0] = self.evaders['position'][0].copy()
        ## create evader patches, 八面玲珑
        self.evader_patches = []
        octagon = RegularPolygon(
            xy=self.evaders['position'][0], 
            numVertices=8, 
            radius=self.evader_radius, 
            fc='orangered'
        )
        self.evader_patches.append(octagon)
        obs[:,:,0] = 255*self._get_image(
            patch_list=[octagon], 
            radius=self.evader_radius
        )

        # Reset Pursuers
        self.pursuers['position'][0] = self.spawning_pool[-1]
        while any(
            [
                self._is_occluded(self.pursuers['position'][0], radius=self.pursuer_radius),
                self._is_interfered(self.pursuers['position'][0], radius=2*self.interfere_radius)
            ]
        ): # pursuer has to work safely so that they don't want to start too close to others
            self.pursuers['position'][0] = random.uniform(-self.world_length/2+.3, self.world_length/2-.3, 2)
        self.pursuers['velocity'] = np.zeros((self.num_pursuers,2))
        self.pursuers['trajectory'].append(self.pursuers['position'].copy())
        self.pursuers['status'] = ['active']*self.num_pursuers
        self.spawning_pool[-1] = self.pursuers['position'][0].copy()
        ## create pursuer patches, 圆滑世故
        self.pursuer_patches = []
        circle = Circle(
            xy=self.pursuers['position'][0], 
            radius=self.pursuer_radius, 
            fc='deepskyblue'
        )
        self.pursuer_patches.append(circle)
        obs[:,:,-1] = 255*self._get_image(
            patch_list=[circle], 
            radius=self.pursuer_radius
        )
        # Create map image 
        self.image[:,:,0] = obs[:,:,0] # .9*np.transpose(obs[:,:,0]) # R: evader channel
        self.image[:,:,1] = obs[:,:,1] # .9*np.transpose(obs[:,:,1]) # G: obstacle channel
        self.image[:,:,2] = obs[:,:,2] # .9*np.transpose(obs[:,:,2]) # B: pursuer channel

        return obs

    def step(self, action):
        """
        Agents take velocity command
        Args:
            action: array([fx_p0,fy_p0])
        Returns:
            obs: map image
            reward: r_p0
            done
            info: episode result or ''
        """
        # Check input
        assert action.shape == (2,)
        action = np.clip(action, self.action_space_low, self.action_space_high)
        # Prepare
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
            if any(
                [
                    self._is_outbound(self.pursuers['position'][0], radius=self.pursuer_radius),
                    self._is_occluded(self.pursuers['position'][0], radius=self.pursuer_radius),
                ]
            ):
                self.pursuers['status'][0] = 'deactivated'
                bonus = -.1
            else:
                bonus = -.1
        else:
            action = np.zeros(2)
            self.pursuers['velocity'][0] = np.zeros(2)
        ## detect captures
        if self.pursuers['status'][0] == 'active': # status updated, check status again
            if self.evaders['status'][0] =='active':
                if np.linalg.norm(self.pursuers['position'][0] - self.evaders['position'][0]) <= self.interfere_radius:
                    self.evaders['status'][0] = 'deactivated'
                    bonus = 100
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
            obs[:,:,-1] = 255*self._get_image(
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

