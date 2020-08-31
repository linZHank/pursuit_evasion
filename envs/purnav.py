#!/usr/bin/python3
"""
Pursuer Navigation Environment (discrete action space): a toy environment for testing algorithms. 
    - Random obstacles 
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
from matplotlib.patches import Ellipse, Circle, RegularPolygon
from matplotlib.collections import PatchCollection
from .purnavs0 import PursuerNavigationScene0


class PursuerNavigation(PursuerNavigationScene0):

    def __init__(self, resolution=(80, 80)):
        super(PursuerNavigation, self).__init__(resolution)
        self.name = 'pursuer_navigation_discrete'
        self.max_num_ellipses = 7
        self.max_num_polygons = 7
        
    def reset(self):

        """
        Reset obstacles and agents location
        Args:
        Return:
            obs: map image
        """
        # Prepare 
        self.num_ellipses = random.randint(1,self.max_num_ellipses+1)
        self.num_polygons = random.randint(1,self.max_num_polygons+1)
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
        for _ in range(self.num_ellipses):
            ellipse = Ellipse(
                xy=random.uniform(-self.world_length/2, self.world_length/2, size=2), 
                width=random.uniform(self.world_length/10, self.world_length/7), 
                height=random.uniform(self.world_length/10, self.world_length/7), 
                angle=random.uniform(0,360), 
                fc='grey'
            )
            self.obstacle_patches.append(ellipse)
        for _ in range(self.num_polygons):
            reg_polygon = RegularPolygon(
                xy=random.uniform(-self.world_length/2, self.world_length/2, size=2), 
                numVertices=random.randint(4,7), 
                radius=random.uniform(self.world_length/10, self.world_length/7), 
                orientation=random.uniform(-pi,pi), 
                fc='grey'
            )
            self.obstacle_patches.append(reg_polygon)
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
        self.image[:,:,0] = obs[:,:,0] 
        self.image[:,:,1] = obs[:,:,1] 
        self.image[:,:,2] = obs[:,:,2] 

        return obs

