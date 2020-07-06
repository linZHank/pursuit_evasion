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
from .pe import PursuitEvasion


class PursuitEvasionOneVsOneContinuous(PursuitEvasion):
    """
    Dynamics Pursuit-evasion env: 1 pursuer, 1 evader, discrete action space 
    """
    def __init__(self, resolution=(80, 80)):
        super().__init__(resolution)
        self.name = '1p1e_continuous'
        self.num_evaders = 1
        self.num_pursuers = 1

    def reset(self):
        """
        Reset obstacles and agents location
        Args:
        Return:
            obs: map image
        """
        self.num_ellipses = random.randint(1,self.max_num_ellipses+1)
        self.num_polygons = random.randint(1,self.max_num_polygons+1)
        # Init evader dict
        self.evaders = dict(
            names = ['e-'+str(i) for i in range(self.num_evaders)],
            position = np.inf*np.ones((self.num_evaders, 2)),
            velocity = np.zeros((self.num_evaders, 2)),
            trajectory = [],
            status = ['deactivated']*self.num_evaders
        )
        # Init pursuer dict
        self.pursuers = dict(
            names = ['p-'+str(i) for i in range(self.num_pursuers)],
            position = np.inf*np.ones((self.num_pursuers, 2)),
            velocity = np.zeros((self.num_pursuers, 2)),
            trajectory = [],
            status = ['deactivated']*self.num_pursuers
        )
        # Generate spawning positions
        self.spawning_pool = random.uniform(-self.world_length/2+.3, self.world_length/2-.3, size=(self.num_evaders+self.num_pursuers,2))
        self.step_counter = 0
        # Reset obstacles: you can add more shapes in the section below
        self.obstacle_patches = []
        for _ in range(self.num_ellipses):
            ellipse = Ellipse(xy=random.uniform(-self.world_length/2, self.world_length/2, size=2), width=random.uniform(self.world_length/10, self.world_length/7), height=random.uniform(self.world_length/10, self.world_length/7), angle=random.uniform(0,360), fc='grey')
            self.obstacle_patches.append(ellipse)
        for _ in range(self.num_polygons):
            reg_polygon = RegularPolygon(xy=random.uniform(-self.world_length/2, self.world_length/2, size=2), numVertices=random.randint(4,7), radius=random.uniform(self.world_length/10, self.world_length/7), orientation=random.uniform(-pi,pi), fc='grey')
            self.obstacle_patches.append(reg_polygon)
        self.obstacle_map = self._get_map(patch_list=self.obstacle_patches, radius=self.world_length/np.min(self.resolution)/2)
        # Reset Evaders #
        for ie in range(self.num_evaders):
            self.evaders['position'][ie] = self.spawning_pool[ie]
            while any(
                [
                    self._is_occluded(self.evaders['position'][ie], radius=self.evader_radius),
                    self._is_interfered(self.evaders['position'][ie], radius=2*self.evader_radius)
                ]
            ): # evaders are sneaky so that they can stay closer to each other
                self.evaders['position'][ie] = random.uniform(-self.world_length/2+.3, self.world_length/2-.3, 2)
        self.evaders['velocity'] = np.zeros((self.num_evaders,2))
        self.evaders['trajectory'].append(self.evaders['position'].copy())
        self.evaders['status'] = ['active']*self.num_evaders
        self.spawning_pool[:self.num_evaders] = self.evaders['position'].copy()
        ## create evader patches, 八面玲珑
        self.evader_patches = []
        for ie in range(self.num_evaders):
            octagon = RegularPolygon(xy=self.evaders['position'][ie], numVertices=8, radius=self.evader_radius, fc='orangered')
            self.evader_patches.append(octagon)
        ## create evaders map
        self.evader_map = self._get_map(patch_list=self.evader_patches, radius=self.evader_radius)
        # Reset Pursuers #
        for ip in range(self.num_pursuers):
            self.pursuers['position'][ip] = self.spawning_pool[self.num_evaders+ip]
            while any(
                [
                    self._is_occluded(self.pursuers['position'][ip], radius=self.pursuer_radius),
                    self._is_interfered(self.pursuers['position'][ip], radius=2*self.interfere_radius)
                ]
            ): # pursuer has to work safely so that they don't want to start too close to others
                self.pursuers['position'][ip] = random.uniform(-self.world_length/2+.3, self.world_length/2-.3, 2)
        self.pursuers['velocity'] = np.zeros((self.num_pursuers,2))
        self.pursuers['trajectory'].append(self.pursuers['position'].copy())
        self.pursuers['status'] = ['active']*self.num_pursuers
        self.spawning_pool[-self.num_pursuers:] = self.pursuers['position'].copy()
        ## create pursuer patches, 圆滑世故
        self.pursuer_patches = []
        for ip in range(self.num_pursuers):
            circle = Circle(xy=self.pursuers['position'][ip], radius=self.pursuer_radius, fc='deepskyblue')
            self.pursuer_patches.append(circle)
        ## create pursuers map
        self.pursuer_map = self._get_map(patch_list=self.pursuer_patches, radius=self.pursuer_radius)
        # Create map in the order of RGB 
        self.map[:,:,0] = 0.5*np.transpose(self.evader_map)
        self.map[:,:,1] = 0.5*np.transpose(self.obstacle_map)
        self.map[:,:,2] = 0.5*np.transpose(self.pursuer_map)
        # Get obs
        obs = self.map.copy()

        return obs


# usage example
if __name__=='__main__':
    env = PursuitEvasionOneVsOneContinuous()
    for ep in range(4):
        obs = env.reset()
        for st in range(env.max_episode_steps):
            env.render(pause=1./env.rate)
            actions = np.random.uniform(-4,4,size=(env.num_evaders+env.num_pursuers,2))
            obs, rew, done, info = env.step(actions)
            img = obs[:,:,[2,1,0]]
            cv2.imshow('map', cv2.resize(img, (640, 640)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            print("\nevaders_pos: {} \npursuers_pos: {} \nreward: {} \ndone: {}".format(env.evaders['position'], env.pursuers['position'], rew, done))
            if info:
                print(info)
                time.sleep(2)
                break
    cv2.destroyAllWindows()
            

