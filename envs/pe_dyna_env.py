#!/usr/bin/python3

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi
from numpy import random
import time
import matplotlib.pyplot as plt


class PEDynaEnv(object):
    """
    Pursuit-evasion dynamics env: N pursuers, N evader, homogeneous
    """
    def __init__(self, num_evaders=1, num_pursuers=1):
        assert num_evaders >= 1
        assert num_pursuers >= 1
        assert isinstance(num_evaders, int)
        assert isinstance(num_pursuers, int)
        # world properties
        self.rate = 30 # Hz
        self.num_pursuers = num_pursuers # not recommend for large numbers
        self.num_evaders = num_evaders
        self.world_length = 10.
        self.interfere_radius = 0.4 # effective catch within this range
        self.max_steps = self.rate*60 # 1 min
        self.obstacle_circles = dict(
            names = ['circle_'+str(i) for i in range(1)],
            position = np.zeros((1,2)),
            radius = np.array([2])
        )
        self.obstacle_rectangles = dict(
            names = ['rectangle_'+str(i) for i in range(2)],
            position = np.array([[-2,-self.world_length/2],[-2,self.world_length/2-1.5]]),
            dimension = np.array([[4,1.5],[4,1.5]])
        )
        # pursuers and evaders properties
        self.radius_evader = 0.1
        self.radius_pursuer = 0.1
        self.mass_pursuer = 0.5
        self.mass_evader = 0.5
        self.observation_space = (num_evaders*4+num_pursuers*4,) # x,y,vx,vy
        self.action_space = (2,) # fx,fy
        # variables
        self.evaders_spawning_pool = np.zeros([num_evaders, 2]) # x,y,theta
        self.pursuers_spawning_pool = np.zeros([num_pursuers, 2])
        self.step_counter = 0
        self.evaders = dict(
            names = ['e_'+str(i) for i in range(num_evaders)],
            position = np.empty((num_evaders, 2)),
            velocity = np.empty((num_evaders, 2)),
            trajectory = [],
            status = ['deactivated']*num_evaders
        )
        self.pursuers = dict(
            names = ['p_'+str(i) for i in range(num_pursuers)],
            position = np.zeros((num_pursuers, 2)),
            velocity = np.zeros((num_pursuers, 2)),
            trajectory = [],
            status = ['deactivated']*num_pursuers
        )
        self.distance_matrix = np.zeros((num_evaders+num_pursuers,num_evaders+num_pursuers))
        self.distance_matrix[:] = np.nan
        # create figure
        fig, ax = plt.subplots(figsize=(16, 16))

    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: array([x_p0,y_p0,...,vx_p0,vy_p0,...,x_e0,y_e0,...,vx_e0,vy_e0])
        """
        self.step_counter = 0
        # reset evader
        for i in range(self.num_evaders):
            self.evaders['position'][i] = self.evaders_spawning_pool[i]
            while self._is_outbound(self.evaders['position'][i]) or self._is_occluded(self.evaders['position'][i]):
                self.evaders['position'][i] = random.uniform(-self.world_length/2, self.world_length/2, 2)
        self.evaders['velocity'] = np.zeros((self.num_evaders,2))
        self.evaders['trajectory'] = []
        # retrieve individual coordinates to prevent obj sync
        coords = [] # [x0,y0,x1,y1,...]
        for p in self.evaders['position']:
            for c in p:
                coords.append(c)
        self.evaders['trajectory'].append(coords)
        self.evaders['status'] = ['active']*self.num_evaders
        self.compute_distances()
        # reset pursuers
        for i in range(self.num_pursuers):
            self.pursuers['position'][i] = self.pursuers_spawning_pool[i]
            self.compute_distances()
            while any(
                [
                    self._is_outbound(self.pursuers['position'][i]),
                    self._is_occluded(self.pursuers['position'][i]),
                    sum(self.distance_matrix[i]<=2*self.interfere_radius)
                ]
            ):
                self.pursuers['position'][i] = random.uniform(-self.world_length/2, self.world_length/2, 2)
                self.compute_distances()
        self.pursuers['velocity'] = np.zeros((self.num_pursuers,2))
        self.pursuers['trajectory'] = []
        # retrieve individual coordinates to prevent obj sync
        coords = [] # [x0,y0,x1,y1,...]
        for p in self.pursuers['position']:
            for c in p:
                coords.append(c)
        self.pursuers['trajectory'].append(coords)
        self.pursuers['status'] = ['active']*self.num_pursuers
        # self.traj_pursuers.append([i for i in self.pursuers['position'][i]])
        # update distance matrix
        self.compute_distances()
        # create obs
        obs = np.concatenate(
            (
                self.pursuers['position'].reshape(-1),
                self.pursuers['velocity'].reshape(-1),
                self.evaders['position'].reshape(-1),
                self.evaders['velocity'].reshape(-1)
            ), axis=0
        )

        return obs

    def step(self, action_evaders, action_pursuers):
        """
        Agents take velocity command
        Args:
            action_evaders: array([[fx_e0,fy_e0],...])
            action_pursuers: array([[fx_p0,fy_p0],...])
        Returns:
            obs: array([x_p0,y_p0,...,v_x_p0,v_y_p0,...,x_e0,y_e0,...,v_x_e0,v_y_e0])
            reward:
            done: bool
            info: ''
        """
        # make sure actions are in right shapes
        assert action_evaders.shape == self.evaders['velocity'].shape
        assert action_pursuers.shape == self.pursuers['velocity'].shape
        # default reward, done, info
        reward, done, info = [0.]*(self.num_pursuers+self.num_evaders), [False]*(self.num_pursuers+self.num_evaders), ''
        # set limitation for velocity commands
        action_evaders = np.clip(action_evaders, -2, 2) # N
        action_pursuers = np.clip(action_pursuers, -2, 2)
        # step evaders
        for i in range(self.num_evaders):
            if self.evaders['status'][i] == 'active':
                d_vel = action_evaders[i]/self.mass_evader/self.rate
                self.evaders['velocity'][i] += d_vel
                d_pos = self.evaders['velocity'][i]/self.rate
                self.evaders['position'][i] += d_pos # possible next pos
                if self._is_outbound(self.evaders['position'][i]) or self._is_occluded(self.evaders['position'][i]):
                    self._disable_evader(id=i)
        self.compute_distances()
        # retrieve individual coordinates to prevent obj sync
        coords = [] # [x0,y0,x1,y1,...]
        for p in self.evaders['position']:
            for c in p:
                coords.append(c)
        self.evaders['trajectory'].append(coords)
        # step pursuers
        for i in range(self.num_pursuers):
            if self.pursuers['status'][i] == 'active':
                self.pursuers['velocity'][i] += action_pursuers[i]/self.mass_evader/self.rate
                self.pursuers['position'][i] += self.pursuers['velocity'][i]/self.rate # possible next pos
                if self._is_outbound(self.pursuers['position'][i]) or self._is_occluded(self.pursuers['position'][i]):
                    self._disable_pursuer(id=i)
        self.compute_distances()
        # retrieve individual coordinates to prevent obj sync
        coords = [] # [x0,y0,x1,y1,...]
        for p in self.pursuers['position']:
            for c in p:
                coords.append(c)
        self.pursuers['trajectory'].append(coords)
        # update obs
        obs = np.concatenate(
            (
                self.pursuers['position'].reshape(-1),
                self.pursuers['velocity'].reshape(-1),
                self.evaders['position'].reshape(-1),
                self.evaders['velocity'].reshape(-1)
            ), axis=0
        )
        # detect pe interfere
        for i in range(self.num_pursuers):
            for j in range(self.num_evaders):
                if self.distance_matrix[i,self.num_pursuers+j] <= self.interfere_radius:
                    self._disable_evader(id=j)
        # update reward, done, info
        done[-self.num_evaders:] = [s=='deactivated' for s in self.evaders['status']]
        done[:self.num_pursuers] = [s=='deactivated' for s in self.pursuers['status']]
        reward = [-1*d for d in done]
        if all(done[-self.num_evaders:]): # pursuers win
            reward[:self.num_pursuers] = [1.]*self.num_pursuers
            reward[-self.num_evaders:] = [-1.]*self.num_evaders
            info = "All evaders deceased"
        if all(done[:self.num_pursuers]): # evaders win
            reward[:self.num_pursuers] = [-1.]*self.num_pursuers
            reward[-self.num_evaders:] = [1.]*self.num_evaders
            info = "All pursuers deceased"
        if self.step_counter >= self.max_steps:
            info = "maximum step: {} reached".format(self.max_steps)
            done = [True]*(self.num_pursuers+self.num_evaders)
        self.step_counter += 1

        return obs, reward, done, info

    def render(self,pause=2):
        fig, ax = plt.gcf(), plt.gca()
        ax.cla()
        # plot world boundary
        bound = plt.Rectangle((-self.world_length/2,-self.world_length/2), self.world_length, self.world_length, linewidth=2, color='k', fill=False)
        ax.add_patch(bound)
        # draw obstacles
        for ci in range(len(self.obstacle_circles['radius'])):
            obs_circle = plt.Circle((self.obstacle_circles['position'][ci]), self.obstacle_circles['radius'][ci], color='grey')
            ax.add_patch(obs_circle)
        for ri in range(self.obstacle_rectangles['position'].shape[0]):
            obs_rect = plt.Rectangle((self.obstacle_rectangles['position'][ri]),self.obstacle_rectangles['dimension'][ri,0], self.obstacle_rectangles['dimension'][ri,1], color='grey')
            ax.add_patch(obs_rect)
        # draw evaders and annotate
        traj_es = np.array(self.evaders['trajectory'])
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie]=='active':
                plt.scatter(self.evaders['position'][ie,0], self.evaders['position'][ie,1], s=200, marker='*', color='orangered', linewidth=2)
                evader_circle = plt.Circle((self.evaders['position'][ie,0], self.evaders['position'][ie,1]), self.radius_evader, color='darkorange', fill=False)
                ax.add_patch(evader_circle)
                plt.annotate(
                    self.evaders['names'][ie], # this is the text
                    (self.evaders['position'][ie,0], self.evaders['position'][ie,1]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
                # draw trajectory
                plt.plot(traj_es[:,2*ie], traj_es[:,2*ie+1],  linestyle='--', linewidth=0.5, color='orangered')
        # draw pursuers
        traj_ps = np.array(self.pursuers['trajectory'])
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip]=='active':
                pursuer_circle = plt.Circle((self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), self.radius_evader, color='deepskyblue')
                ax.add_patch(pursuer_circle)
                plt.annotate(
                    self.pursuers['names'][ip], # this is the text
                    (self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
                # draw interfere circle
                interfere_circle = plt.Circle((self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), self.interfere_radius, color='deepskyblue', linestyle='-.', fill=False)
                ax.add_patch(interfere_circle)
                # draw trajectory
                plt.plot(traj_ps[:,2*ip], traj_ps[:,2*ip+1],  linestyle='--', linewidth=0.5, color='deepskyblue')
        # set axis
        plt.axis(1.1/2*np.array([-self.world_length,self.world_length,-self.world_length,self.world_length]))
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_xticks(np.arange(-5, 6))
        ax.set_yticks(np.arange(-5, 6))
        plt.grid(color='grey', linestyle=':', linewidth=0.5)
        # show
        plt.show(block=False)
        plt.pause(pause)

    # Helper Functions
    def compute_distances(self):
        """
        Compute distances between each others
        Args:
        Returns:
            self.distance_matrix
        """
        # obtain list of poses
        pos_all = np.concatenate((self.pursuers['position'],self.evaders['position']), axis=0)
        for i in reversed(range(1,pos_all.shape[0])):
            for j in range(i):
                self.distance_matrix[i,j] = np.linalg.norm(pos_all[i]-pos_all[j])
                self.distance_matrix[j,i] = np.linalg.norm(pos_all[i]-pos_all[j])

    # obstacles collision detection
    def _is_occluded(self, pos):
        collision_flag = False
        # detect collision with circles
        circle_flag = False
        for i in range(len(self.obstacle_circles['names'])):
            if (pos[0]-self.obstacle_circles['position'][i,0])**2+(pos[1]-self.obstacle_circles['position'][i,1])**2 <= self.obstacle_circles['radius'][i]**2:
                circle_flag = True
                break
        # detect collision with rectangles
        rect_flag = False
        for i in range(len(self.obstacle_rectangles['names'])):
            if 0<=(pos[0]-self.obstacle_rectangles['position'][i,0])<=self.obstacle_rectangles['dimension'][i,0] and 0<=(pos[1]-self.obstacle_rectangles['position'][i,1])<=self.obstacle_rectangles['dimension'][i,1]:
                rect_flag = True
                break
        # compute collision flag
        collision_flag = circle_flag or rect_flag

        return collision_flag

    # out of bound detection
    def _is_outbound(self, pos):
        flag = False
        if np.absolute(pos[0])>=self.world_length/2 or np.absolute(pos[1])>=self.world_length/2:
            flag = True

        return flag

    def _disable_pursuer(self, id):
        self.pursuers['position'][id] = np.zeros(2)
        self.pursuers['velocity'][id] = np.zeros(2)
        self.pursuers['status'][id] = 'deactivated'

    def _disable_evader(self, id):
        self.evaders['position'][id] = np.zeros(2)
        self.evaders['velocity'][id] = np.zeros(2)
        self.evaders['status'][id] = 'deactivated'
