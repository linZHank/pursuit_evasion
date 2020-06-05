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

class PEEnv:
    """
    Dynamics Pursuit-evasion env: N pursuers, N evader (1<=N<=4)
    """
    def __init__(self, resolution=(100, 100)):
        # Env specs #
        self.name='mpme' # dynamic multi-pursuer multi-evader
        self.rate = 20 # Hz
        self.max_episode_steps = 1000
        self.resolution = resolution
        self.world_length = 10
        self.damping = 0.1
        self.max_num_evaders = 4
        self.max_num_pursuers = 4
        self.num_evaders = random.randint(1,self.max_num_evaders+1)
        self.num_pursuers = random.randint(1,self.max_num_pursuers+1)
        self.obstacle_patches = []
        self.evader_patches = []
        self.pursuer_patches = []
        self.interfere_radius = 0.4
        self.action_space_low = -2.
        self.action_space_high = 2.
        self.evader_max_speed = 2. # max speed on x or y
        self.pursuer_max_speed = 2. 
        self.evader_radius = 0.1
        self.evader_mass = 0.4
        self.pursuer_radius = 0.1
        self.pursuer_mass = 0.4
        self.max_num_ellipses = 7
        self.max_num_polygons = 7
        self.num_ellipses = random.randint(1,self.max_num_ellipses+1)
        self.num_polygons = random.randint(1,self.max_num_polygons+1)
        ## next 7 lines compute grid coordinates
        step_x, step_y = self.world_length/resolution[0], self.world_length/resolution[1]
        x_coords = np.linspace((-self.world_length+step_x)/2, (self.world_length-step_x)/2, resolution[0])
        y_coords = -np.linspace((-self.world_length+step_y)/2, (self.world_length-step_y)/2, resolution[1]) # don't forget the negative sign, so that y goes from top to bottom
        self.pix_coords = np.zeros((resolution[0]*resolution[1], 2))
        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                self.pix_coords[i*len(x_coords)+j] = np.array([x_coords[i], y_coords[j]])
        # Prepare renderer #
        self.map = np.zeros((self.resolution[0],self.resolution[1],3))
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)

    def reset(self):
        """
        Reset obstacles and agents location
        Args:
        Return:
            obs: array([x_p0,y_p0,...,vx_p0,vy_p0,...,x_e0,y_e0,...,vx_e0,vy_e0])
        """
        self.num_evaders = random.randint(1,self.max_num_evaders+1)
        self.num_pursuers = random.randint(1,self.max_num_pursuers+1)
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
        self.spawning_pool = random.uniform(-self.world_length/2, self.world_length/2, size=(self.num_evaders+self.num_pursuers,2))
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
                self.evaders['position'][ie] = random.uniform(-self.world_length/2, self.world_length/2, 2)
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
                self.pursuers['position'][ip] = random.uniform(-self.world_length/2, self.world_length/2, 2)
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
        # Check input
        assert actions.shape == (self.num_evaders+self.num_pursuers, 2)
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
                            bonus[ie] = -10.
                            bonus[-self.num_pursuers+ip] = 100.
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

    def render(self, pause=2):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        # Plot world boundary #
        bound = plt.Rectangle((-self.world_length/2,-self.world_length/2), self.world_length, self.world_length, linewidth=3, color='k', fill=False)
        self.ax.add_patch(bound)
        # Draw objects: obstacles, evaders, pursuers #
        patches_collection = PatchCollection(self.obstacle_patches+self.evader_patches+self.pursuer_patches, match_original=True) # match_origin prevent PatchCollection mess up original color
        self.ax.add_collection(patches_collection)
        ## depict evaders
        evader_trajectories = np.array(self.evaders['trajectory'])
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie]=='active':
                ### text annotation
                self.ax.annotate(
                    self.evaders['names'][ie], # pursuer name
                    (self.evaders['position'][ie,0], self.evaders['position'][ie,1]), # name label location
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
                ### draw evader trajectories
                self.ax.plot(evader_trajectories[:,ie,0], evader_trajectories[:,ie,1], linestyle='--', linewidth=0.5, color='orangered')
        ## depict pursuers
        pursuer_trajectories = np.array(self.pursuers['trajectory'])
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip]=='active':
                # text annotation
                self.ax.annotate(
                    self.pursuers['names'][ip], # pursuer name
                    (self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), # name label location
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
                ### draw pursuer trajectories
                self.ax.plot(pursuer_trajectories[:,ip,0], pursuer_trajectories[:,ip,1], linestyle='--', linewidth=0.5, color='deepskyblue')
                ### draw interfere circle
                interfere_circle = plt.Circle((self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), self.interfere_radius, color='deepskyblue', linestyle='dashed', fill=False)
                self.ax.add_patch(interfere_circle)
        # Set axis
        self.ax.axis(1.1/2*np.array([-self.world_length,self.world_length,-self.world_length,self.world_length]))
        self.ax.set_xlabel('X', fontsize=20)
        self.ax.set_ylabel('Y', fontsize=20)
        self.ax.set_xticks(np.arange(-5, 6))
        self.ax.set_yticks(np.arange(-5, 6))
        self.ax.grid(color='grey', linestyle=':', linewidth=0.5)
        ## pause
        plt.pause(pause) # 1/16x to 16x
        self.fig.show()
        # plt.show(block=False)
        # plt.pause(pause)


    def _is_outbound(self, pos, radius):
        """
        Detect a given position is out of boundary or not
        """
        out_flag = False
        if np.absolute(pos[0])>=self.world_length/2-radius or np.absolute(pos[1])>=self.world_length/2-radius:
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

    def _disable_pursuer(self, id):
        self.pursuers['position'][id] = np.inf*np.ones(2)
        self.pursuers['velocity'][id] = np.zeros(2)
        self.pursuers['status'][id] = 'deactivated'

    def _disable_evader(self, id):
        self.evaders['position'][id] = np.inf*np.ones(2)
        self.evaders['velocity'][id] = np.zeros(2)
        self.evaders['status'][id] = 'deactivated'
