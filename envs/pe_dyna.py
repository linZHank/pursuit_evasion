#!/usr/bin/python3
"""
Pursuit-evasion environment:
    - Randomly placed obstacles
    - Multiple Obstacles (>=0)
    - Multiple Pursuers (>=1)
    - Multiple Evaders (>=1)
    - Homogeneous agents
"""
import numpy as np
from numpy import pi
from numpy import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, RegularPolygon, Polygon
from matplotlib.collections import PatchCollection
# import matplotlib.image as mpimg
import cv2

class PEDyna(object):
    """
    Dynamics Pursuit-evasion env: N pursuers, N evader (1<=N<=8)
    """
    def __init__(self, train=True, num_evaders=1, num_pursuers=1, num_ellipses=1, num_polygons=1, resolution=(100, 100)):
        assert isinstance(num_evaders, int)
        assert isinstance(num_pursuers, int)
        assert num_evaders >= 1
        assert num_pursuers >= 1
        # Env specs
        self.name='dyna_mpme' # dynamic multi-pursuer multi-evader
        self.world_length = 10
        self.resolution = resolution
        self.num_evaders = num_evaders
        self.num_pursuers = num_pursuers
        self.num_ellipses = num_ellipses
        self.num_polygons = num_polygons
        if train:
            self.mode = 'train'
            self.num_evaders = random.randint(1,9)
            self.num_pursuers = random.randint(1,9)
            self.num_ellipses = random.randint(1,7)
            self.num_polygons = random.randint(1,7)
        else:
            self.mode = 'eval'
        # next 7 lines compute grid coordinates
        step_x, step_y = self.world_length/resolution[0], self.world_length/resolution[1]
        x_coords = np.linspace((-self.world_length+step_x)/2, (self.world_length-step_x)/2, resolution[0])
        y_coords = -np.linspace((-self.world_length+step_y)/2, (self.world_length-step_y)/2, resolution[1]) # don't forget the negative sign, so that y can go through from top to bottom
        self.pix_coords = np.zeros((resolution[0]*resolution[1], 2))
        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                self.pix_coords[i*len(x_coords)+j] = np.array([x_coords[i], y_coords[j]])
        # Place obstacles: you can add more shapes in the section below
        self.obstacles = []
        for i in range(self.num_ellipses):
            ellipse = Ellipse(xy=random.uniform(-self.world_length/2, self.world_length/2, size=2), width=random.uniform(self.world_length/10, self.world_length/7), height=random.uniform(self.world_length/10, self.world_length/7), angle=random.uniform(0,360), ec='k', fc='grey')
            self.obstacles.append(ellipse)
        for i in range(self.num_polygons):
            reg_polygon = RegularPolygon(xy=random.uniform(-self.world_length/2, self.world_length/2, size=2), numVertices=random.randint(4,7), radius=random.uniform(self.world_length/10, self.world_length/7), orientation=random.uniform(-pi,pi), ec='k', fc='grey')
            self.obstacles.append(reg_polygon)
        # generate obstacle map
        obst_pix = np.array([False]*self.pix_coords.shape[0])
        for op in self.obstacles:
            obst_pix = np.logical_or(obst_pix, op.contains_points(self.pix_coords, radius=self.world_length/np.min(resolution)/2))
        self.obst_map = obst_pix.reshape(resolution)

        # Prepare renderer
        self.fig = plt.figure(figsize=(20, 10))
        self.ax_env = self.fig.add_subplot(121)
        self.ax_img = self.fig.add_subplot(122)

    def reset(self):
        """
        Reset obstacles and agents location
        Args:
        Return:
            obs: array([x_p0,y_p0,...,vx_p0,vy_p0,...,x_e0,y_e0,...,vx_e0,vy_e0])
        """
        self.step_counter = 0

        # create obs
        obs = self._get_observation()

        return obs

    def _get_observation(self):
        pass

    def render(self, pause=2):
        self.ax_env = self.fig.get_axes()[0]
        self.ax_img = self.fig.get_axes()[1]
        self.ax_env.cla()
        # plot world boundary
        bound = plt.Rectangle((-self.world_length/2,-self.world_length/2), self.world_length, self.world_length, linewidth=2, color='k', fill=False)
        self.ax_env.add_patch(bound)
        # draw obstacles
        obst_collection = PatchCollection(self.obstacles, match_original=True) # match_origin prevent PatchCollection mess up original color
        self.ax_env.add_collection(obst_collection)
        # set axis
        self.ax_env.axis(1.1/2*np.array([-self.world_length,self.world_length,-self.world_length,self.world_length]))
        self.ax_env.set_xlabel('X', fontsize=20)
        self.ax_env.set_ylabel('Y', fontsize=20)
        self.ax_env.set_xticks(np.arange(-5, 6))
        self.ax_env.set_yticks(np.arange(-5, 6))
        # plt.grid(color='grey', linestyle=':', linewidth=0.5)
        self.ax_env.grid(color='grey', linestyle=':', linewidth=0.5)
        # Display env image
        self.ax_img.imshow(np.transpose(self.obst_map))
        # show
        plt.pause(pause) # 1/16x to 16x
        self.fig.show()
        # plt.show(block=False)
        # plt.pause(pause)
