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


class PEDyna(object):
    """
    Dynamics Pursuit-evasion env: N pursuers, N evader (1<=N<=8)
    """
    def __init__(self, train=True, num_evaders=1, num_pursuers=1, num_ellipses=1, num_polygons=1):
        assert isinstance(num_evaders, int)
        assert isinstance(num_pursuers, int)
        assert num_evaders >= 1
        assert num_pursuers >= 1
        # specs
        self.name='dyna_mpme' # dynamic multi-pursuer multi-evader
        self.world_length = 10
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
        # obstacles
        self.obstacles = []
        for i in range(self.num_ellipses):
            ellipse = Ellipse(xy=random.uniform(-self.world_length/2, self.world_length/2, size=2), width=random.uniform(self.world_length/10, self.world_length/7), height=random.uniform(self.world_length/10, self.world_length/7), angle=random.uniform(0,360))
            self.obstacles.append(ellipse)
        for i in range(self.num_polygons):
            reg_polygon = RegularPolygon(xy=random.uniform(-self.world_length/2, self.world_length/2, size=2), numVertices=random.randint(4,7), radius=random.uniform(self.world_length/10, self.world_length/7), orientation=random.uniform(-pi,pi))
            self.obstacles.append(reg_polygon)

        # render
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)

    def render(self, pause=2):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        # plot world boundary
        bound = plt.Rectangle((-self.world_length/2,-self.world_length/2), self.world_length, self.world_length, linewidth=2, color='k', fill=False)
        self.ax.add_patch(bound)
        # draw obstacles
        obst_patches = PatchCollection(self.obstacles)
        self.ax.add_collection(obst_patches)
        # set axis
        self.ax.axis(1.1/2*np.array([-self.world_length,self.world_length,-self.world_length,self.world_length]))
        self.ax.set_xlabel('X', fontsize=20)
        self.ax.set_ylabel('Y', fontsize=20)
        self.ax.set_xticks(np.arange(-5, 6))
        self.ax.set_yticks(np.arange(-5, 6))
        # plt.grid(color='grey', linestyle=':', linewidth=0.5)
        self.ax.grid(color='grey', linestyle=':', linewidth=0.5)
        # show
        plt.pause(pause) # 1/16x to 16x
        self.fig.show()
        # plt.show(block=False)
        # plt.pause(pause)
