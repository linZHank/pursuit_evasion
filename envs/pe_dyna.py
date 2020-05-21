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
from matplotlib.patches import Ellipse, RegularPolygon, Circle
from matplotlib.collections import PatchCollection
# import matplotlib.image as mpimg
# import cv2

class PEDyna(object):
    """
    Dynamics Pursuit-evasion env: N pursuers, N evader (1<=N<=4)
    """
    def __init__(self, num_evaders, num_pursuers, resolution=(100, 100)):
        # Env specs #
        self.name='dyna_mpme' # dynamic multi-pursuer multi-evader
        self.rate = 20 # Hz
        self.world_length = 10
        self.resolution = resolution
        self.num_evaders = num_evaders # random.randint(1,5)
        self.num_pursuers = num_pursuers # random.randint(1,5)
        self.num_ellipses = random.randint(1,7)
        self.num_polygons = random.randint(1,7)
        self.obstacle_patches = []
        self.evader_patches = []
        self.pursuer_patches = []
        self.interfere_radius = 0.4
        self.action_space_low = -2.
        self.action_space_high = 2.
        self.max_speed = 2. # max speed on one dirction, real max should be sqrt(2)*max_speed**2
        # next 7 lines compute grid coordinates
        step_x, step_y = self.world_length/resolution[0], self.world_length/resolution[1]
        x_coords = np.linspace((-self.world_length+step_x)/2, (self.world_length-step_x)/2, resolution[0])
        y_coords = -np.linspace((-self.world_length+step_y)/2, (self.world_length-step_y)/2, resolution[1]) # don't forget the negative sign, so that y goes from top to bottom
        self.pix_coords = np.zeros((resolution[0]*resolution[1], 2))
        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                self.pix_coords[i*len(x_coords)+j] = np.array([x_coords[i], y_coords[j]])
        # Init evaders and pursuers
        self.evader_radius = 0.1
        self.evader_mass = 0.4
        self.evaders = dict(
            names = ['e-'+str(i) for i in range(num_evaders)],
            position = np.inf*np.ones((num_evaders, 2)),
            velocity = np.zeros((num_evaders, 2)),
            trajectory = [],
            status = ['deactivated']*num_evaders
        )
        self.pursuer_radius = 0.1
        self.pursuer_mass = 0.4
        self.pursuers = dict(
            names = ['p-'+str(i) for i in range(num_pursuers)],
            position = np.inf*np.ones((num_pursuers, 2)),
            velocity = np.zeros((num_pursuers, 2)),
            trajectory = [],
            status = ['deactivated']*num_pursuers
        )
        # Prepare renderer #
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
        # Reset counter
        # self.num_evaders = random.randint(1,5)
        # self.num_pursuers = random.randint(1,5)
        self.num_ellipses = random.randint(1,7)
        self.num_polygons = random.randint(1,7)
        self.spawning_pool = random.uniform(-self.world_length/2, self.world_length/2, size=(self.num_evaders+self.num_pursuers,2))
        self.step_counter = 0
        # Reset obstacles: you can add more shapes in the section below #
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
                self.evaders['position'][ie] = random.uniform(-self.world_length/2, self.world_length/2, 2)
        self.evaders['velocity'] = np.zeros((self.num_evaders,2))
        self.evaders['trajectory'].append(self.evaders['position'].copy())
        self.evaders['status'] = ['active']*self.num_evaders
        self.spawning_pool[:self.num_evaders] = self.evaders['position'].copy()
        # create evader patches, 八面玲珑
        self.evader_patches = []
        for ie in range(self.num_evaders):
            octagon = RegularPolygon(xy=self.evaders['position'][ie], numVertices=8, radius=self.evader_radius, fc='orangered')
            self.evader_patches.append(octagon)
        # generate evaders map
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
                self.pursuers['position'][ip] = random.uniform(-self.world_length/2, self.world_length/2, 2)
        self.pursuers['velocity'] = np.zeros((self.num_pursuers,2))
        self.pursuers['trajectory'].append(self.pursuers['position'].copy())
        self.pursuers['status'] = ['active']*self.num_pursuers
        self.spawning_pool[self.num_evaders:] = self.pursuers['position'].copy()
        # create evader patches, 圆滑世故
        self.pursuer_patches = []
        for ip in range(self.num_pursuers):
            circle = Circle(xy=self.pursuers['position'][ip], radius=self.pursuer_radius, fc='deepskyblue')
            self.pursuer_patches.append(circle)
        # generate pursuers map
        self.pursuer_map = self._get_map(patch_list=self.pursuer_patches, radius=self.pursuer_radius)
        # Get obs
        obs = self._get_observation()

        return obs

    def step(self, actions):
        """
        Agents take velocity command
        Args:
            actions: array([[fx_e0,fy_e0],[fx_e1,fy_e1],...,[fx_pN,fy_pN]])
        Returns:
            obs: array([x_e0,y_e0,...,vx_e0,vy_e0,...,vx_pN,vy_pN])
            reward:
            done: bool
            info: ''
        """
        # Limit input
        assert actions.shape == (self.num_evaders+self.num_pursuers, 2)
        actions = np.clip(actions, self.action_space_low, self.action_space_high)
        # print("clipped actions: {}".format(actions))
        # Step evaders
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie] == 'active':
                d_vel = actions[ie]/self.evader_mass/self.rate # don't go too fast
                self.evaders['velocity'][ie] += d_vel
                self.evaders['velocity'][ie] = np.clip(self.evaders['velocity'][ie], -self.max_speed, self.max_speed)
                d_pos = self.evaders['velocity'][ie]/self.rate
                self.evaders['position'][ie] += d_pos # possible next pos
                if any(
                    [
                        self._is_outbound(self.evaders['position'][ie], radius=self.evader_radius),
                        self._is_occluded(self.evaders['position'][ie], radius=self.evader_radius),
                        self._is_interfered(self.evaders['position'][ie], radius=2*self.evader_radius)
                    ]
                ):
                    self._disable_evader(id=ie)
            else:
                actions[ie] = np.zeros(2)
                self.evaders['velocity'][ie] = np.zeros(2)
        print("\nevader status: {} \nevaders position: {}".format(self.evaders['status'], self.evaders['position'])) #debug
        ## record evaders trajectory
        self.evaders['trajectory'].append(self.evaders['position'].copy())
        ## create evader patches, 八面玲珑
        self.evader_patches = []
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie] == 'active':
                octagon = RegularPolygon(xy=self.evaders['position'][ie], numVertices=8, radius=self.evader_radius, fc='orangered')
                self.evader_patches.append(octagon)
        ## generate evaders map
        self.evader_map = self._get_map(patch_list=self.evader_patches, radius=self.evader_radius)
        # # step pursuers
        # for i in range(self.num_pursuers):
        #     if self.pursuers['status'][i] == 'active':
        #         self.pursuers['velocity'][i] += actions[i]/self.mass_pursuer/self.rate
        #         self.pursuers['velocity'][i] = np.clip(self.pursuers['velocity'][i], -2, 2)
        #         self.pursuers['position'][i] += self.pursuers['velocity'][i]/self.rate # possible next pos
        #         if self._is_outbound(self.pursuers['position'][i]) or self._is_occluded(self.pursuers['position'][i]):
        #             self._disable_pursuer(id=i)
        #     else:
        #         actions[i] = np.zeros(2)
        #         self.pursuers['velocity'][i] = np.zeros(2)
        # self.compute_distances()
        # # pursuers trajectory
        # coords = [] # [x0,y0,x1,y1,...]
        # for p in self.pursuers['position']:
        #     for c in p:
        #         coords.append(c)
        # self.pursuers['trajectory'].append(coords)
        # # default reward, done, info
        # reward, done, info = np.zeros(self.num_evaders+self.num_pursuers), np.array([False]*(self.num_evaders+self.num_pursuers)), ''
        # # update obs
        # obs = np.concatenate(
        #     (
        #         self.pursuers['position'].reshape(-1),
        #         self.pursuers['velocity'].reshape(-1),
        #         self.evaders['position'].reshape(-1),
        #         self.evaders['velocity'].reshape(-1)
        #     ), axis=0
        # )
        # # detect captures
        # bonus = np.zeros(self.num_pursuers+self.num_evaders)
        # for i in range(self.num_pursuers):
        #     if self.pursuers['status'][i] == 'active':
        #         for j in range(self.num_evaders):
        #             if self.distance_matrix[i,-self.num_evaders+j] <= self.interfere_radius:
        #                 self._disable_evader(id=j)
        #                 bonus[i] = 10.
        # # episode end reached
        # if self.step_counter+1 >= self.max_steps:
        #     for i in range(self.num_pursuers):
        #         self._disable_pursuer(id=i)
        #     bonus[-self.num_evaders:] = 10.*np.logical_not(np.array(done[-self.num_evaders:]))
        # # update reward, done, info
        # done[:self.num_pursuers] = [s=='deactivated' for s in self.pursuers['status']]
        # done[-self.num_evaders:] = [s=='deactivated' for s in self.evaders['status']]
        # # reward = [-1.*d for d in done]
        # reward = -1.*np.array(done) + bonus
        # if all(done[:self.num_pursuers]): # evaders win
        #     # reward[:self.num_pursuers] = -1. # [-1.]*self.num_pursuers
        #     reward[-self.num_evaders:] = [not(d)*1. for d in done[-self.num_evaders:]] # [1.]*self.num_evaders
        #     info = "All pursuers deceased"
        # if all(done[-self.num_evaders:]): # pursuers win
        #     # reward[:self.num_pursuers] = 1. # [1.]*self.num_pursuers
        #     # reward[-self.num_evaders:] = -1. # [-1.]*self.num_evaders
        #     info = "All evaders deceased"
        # self.step_counter += 1
        #
        # return obs, reward, done, info

    def render(self, pause=2):
        self.ax_env = self.fig.get_axes()[0]
        self.ax_img = self.fig.get_axes()[1]
        self.ax_env.cla()
        # Plot world boundary #
        bound = plt.Rectangle((-self.world_length/2,-self.world_length/2), self.world_length, self.world_length, linewidth=3, color='k', fill=False)
        self.ax_env.add_patch(bound)
        # Draw objects: obstacles, evaders, pursuers #
        patches_collection = PatchCollection(self.obstacle_patches+self.evader_patches+self.pursuer_patches, match_original=True) # match_origin prevent PatchCollection mess up original color
        self.ax_env.add_collection(patches_collection)
        # annotate evaders
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie]=='active':
                # plt.annotate(
                self.ax_env.annotate(
                    self.evaders['names'][ie], # pursuer name
                    (self.evaders['position'][ie,0], self.evaders['position'][ie,1]), # name label location
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
        # annotate pursuers
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip]=='active':
                # plt.annotate(
                self.ax_env.annotate(
                    self.pursuers['names'][ip], # pursuer name
                    (self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), # name label location
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
                # draw interfere circle
                interfere_circle = plt.Circle((self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), self.interfere_radius, color='deepskyblue', linestyle='dashed', fill=False)
                self.ax_env.add_patch(interfere_circle)
        # set axis
        self.ax_env.axis(1.1/2*np.array([-self.world_length,self.world_length,-self.world_length,self.world_length]))
        self.ax_env.set_xlabel('X', fontsize=20)
        self.ax_env.set_ylabel('Y', fontsize=20)
        self.ax_env.set_xticks(np.arange(-5, 6))
        self.ax_env.set_yticks(np.arange(-5, 6))
        # plt.grid(color='grey', linestyle=':', linewidth=0.5)
        self.ax_env.grid(color='grey', linestyle=':', linewidth=0.5)
        # Display env image
        map = np.zeros((self.resolution[0],self.resolution[1],3))
        map[:,:,0] = 0.5*np.transpose(self.evader_map)
        map[:,:,1] = 0.5*np.transpose(self.obstacle_map)
        map[:,:,2] = 0.5*np.transpose(self.pursuer_map)
        # self.ax_img.imshow(np.transpose(self.obstacle_map+self.evader_map+self.pursuer_map))
        self.ax_img.imshow(map)
        # show
        plt.pause(pause) # 1/16x to 16x
        self.fig.show()
        # plt.show(block=False)
        # plt.pause(pause)


    def _is_outbound(self, pos, radius):
        """
        Detect a given position is out of boundary or not
        """
        out_flag = False
        if np.absolute(pos[0])>=self.world_length/2-radius or np.absolute(pos[1])>=self.world_length/2+radius:
            out_flag = True
            # print("\nOUT!\n") #debug

        return out_flag

    def _is_occluded(self, pos, radius):
        """
        Detect a given position is occluded by
        """
        occ_flag = False
        for p in self.obstacle_patches:
            occ_flag = p.contains_point(pos, radius=radius)
            if occ_flag:
                break

        return occ_flag

    def _is_interfered(self, pos, radius):
        """
        Detect a given agent is interfered by other agents
        """
        int_flag = False
        sum_agent_pos = np.concatenate((self.evaders['position'], self.pursuers['position']))
        for ap in sum_agent_pos:
            int_flag = (0<np.linalg.norm(pos-ap)<=radius) # don't forget self
            if int_flag:
                break

        return int_flag

    def _get_map(self, patch_list, radius):
        patch_pix = np.array([False]*self.pix_coords.shape[0])
        for p in patch_list:
            patch_pix = np.logical_or(patch_pix, p.contains_points(self.pix_coords, radius=radius))
        map = patch_pix.reshape(self.resolution)

        return map

    def _get_observation(self):
        observation = np.concatenate(
            (
                self.evaders['position'].reshape(-1),
                self.evaders['velocity'].reshape(-1),
                self.pursuers['position'].reshape(-1),
                self.pursuers['velocity'].reshape(-1)
            ), axis=0
        )

        return observation
        # raise NotImplementedError

    def _disable_pursuer(self, id):
        self.pursuers['position'][id] = np.inf*np.ones(2)
        self.pursuers['velocity'][id] = np.zeros(2)
        self.pursuers['status'][id] = 'deactivated'

    def _disable_evader(self, id):
        self.evaders['position'][id] = np.inf*np.ones(2)
        self.evaders['velocity'][id] = np.zeros(2)
        self.evaders['status'][id] = 'deactivated'
