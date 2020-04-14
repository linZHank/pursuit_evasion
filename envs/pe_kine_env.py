#!/usr/bin/python3

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi
from numpy import random
import time
import matplotlib
import matplotlib.pyplot as plt

import pdb


class PEKineEnv(object):
    """
    Pursuit-evasion kinematics env: N pursuers, 1 evader
    """
    def __init__(self, num_pursuers=1, num_evaders=1):
        assert isinstance(num_evaders, int)
        assert isinstance(num_pursuers, int)
        # world properties
        self.name = 'dyna_p'+str(num_pursuers)+'e'+str(num_evaders)
        self.rate = 20 # Hz
        self.num_pursuers = num_pursuers
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
        self.observation_space = (num_evaders*2+num_pursuers*2, ) # x,y,vx,vy
        self.action_space = (2,) # fx,fy
        self.step_counter = 0
        # pursuers and evaders properties
        self.radius_pursuer = 0.1
        self.radius_evader = 0.1
        self.evaders_spawning_pool = np.zeros([num_evaders, 2])
        self.pursuers_spawning_pool = np.zeros([num_pursuers, 2])
        self.evaders = dict(
            names = ['e_'+str(i) for i in range(num_evaders)],
            position = np.empty((num_evaders, 2)),
            trajectory = [],
            status = ['deactivated']*num_evaders
        )
        self.pursuers = dict(
            names = ['p_'+str(i) for i in range(num_pursuers)],
            position = np.zeros((num_pursuers, 2)),
            trajectory = [],
            status = ['deactivated']*num_pursuers
        )
        self.distance_matrix = np.zeros((num_evaders+num_pursuers,num_evaders+num_pursuers))
        self.distance_matrix[:] = np.nan
        # create figure
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = ax1 = self.fig.add_subplot(111)

    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: array([x_p0,y_p0,...,x_eN,y_eN])
            info: ''
        """
        self.step_counter = 0
        # reset evader
        for ie in range(self.num_evaders):
            self.evaders['position'][ie] = self.evaders_spawning_pool[ie]
            while self._is_outbound(self.evaders['position'][ie]) or self._is_occluded(self.evaders['position'][ie]):
                self.evaders['position'][ie] = random.uniform(-self.world_length/2, self.world_length/2, 2)
        self.evaders['trajectory'] = []
        self.evaders['trajectory'].append(self.evaders['position'].copy().reshape(-1))
        self.evaders['status'] = ['active']*self.num_evaders
        self.compute_distances()
        # reset pursuers
        for ip in range(self.num_pursuers):
            self.pursuers['position'][ip] = self.pursuers_spawning_pool[ip]
            self.compute_distances()
            while any(
                [
                    self._is_outbound(self.pursuers['position'][ip]),
                    self._is_occluded(self.pursuers['position'][ip]),
                    sum(self.distance_matrix[ip]<=2*self.interfere_radius)
                ]
            ):
                self.pursuers['position'][ip] = random.uniform(-self.world_length/2, self.world_length/2, 2)
                self.compute_distances()
        self.pursuers['trajectory'] = []
        self.pursuers['trajectory'].append(self.pursuers['position'].copy().reshape(-1))
        self.pursuers['status'] = ['active']*self.num_pursuers
        self.compute_distances()
        # create obs
        obs = np.concatenate((self.pursuers['position'].reshape(-1), self.evaders['position'].reshape(-1)), axis=0)

        return obs

    def step(self, actions):
        """
        Agents take velocity command
        Args:
            actions: array([[vx_p_0,vy_p_0],...,[vx_e_N,vy_e_N]])
        Returns:
            obs: array([x_p0,y_p0,...,x_eN,y_eN])
            reward: array([r_p0,...,r_eN])
            done: array([done_p0,...,done_eN])
            info: ''
        """
        # make sure actions are in right shapes
        assert actions.shape == (self.num_pursuers+self.num_evaders, 2)
        # default reward, done, info
        reward, done, info = np.zeros(self.num_pursuers+self.num_evaders), [False]*(self.num_pursuers+self.num_evaders), ''
        # set limitation for velocity commands
        actions = np.clip(actions, -2, 2) # N
        # step evaders
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie] == 'active':
                d_pos = actions[-self.num_evaders+ie]/self.rate
                self.evaders['position'][ie] += d_pos # possible next pos
                if self._is_outbound(self.evaders['position'][ie]) or self._is_occluded(self.evaders['position'][ie]):
                    self._disable_evader(id=ie)
            else:
                actions[-self.num_evaders+ie] = np.zeros(2)
        self.compute_distances()
        # evaders trajectory
        self.evaders['trajectory'].append(self.evaders['position'].copy().reshape(-1))
        # step pursuers
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip] == 'active':
                self.pursuers['position'][ip] += actions[ip]/self.rate # possible next pos
                if self._is_outbound(self.pursuers['position'][ip]) or self._is_occluded(self.pursuers['position'][ip]):
                    self._disable_pursuer(id=ip)
            else:
                actions[ip] = np.zeros(2)
        self.compute_distances()
        # pursuers trajectory
        self.pursuers['trajectory'].append(self.pursuers['position'].copy().reshape(-1))
        # update obs
        obs = np.concatenate(
            (
                self.pursuers['position'].reshape(-1),
                self.evaders['position'].reshape(-1)
            ), axis=0
        )
        # detect captures
        bonus = np.zeros(self.num_pursuers+self.num_evaders)
        for i in range(self.num_pursuers):
            if self.pursuers['status'][i] == 'active':
                for j in range(self.num_evaders):
                    if self.distance_matrix[i,-self.num_evaders+j] <= self.interfere_radius:
                        self._disable_evader(id=j)
                        bonus[i] = 10.
        # episode end reached
        if self.step_counter+1 >= self.max_steps:
            for i in range(self.num_pursuers):
                self._disable_pursuer(id=i)
            bonus[-self.num_evaders:] = 10.*np.logical_not(np.array(done[-self.num_evaders:]))
        # update reward, done, info
        done[:self.num_pursuers] = [s=='deactivated' for s in self.pursuers['status']]
        done[-self.num_evaders:] = [s=='deactivated' for s in self.evaders['status']]
        # reward = [-1.*d for d in done]
        reward = -1.*np.array(done) + bonus
        if all(done[:self.num_pursuers]): # evaders win
            # reward[:self.num_pursuers] = -1. # [-1.]*self.num_pursuers
            reward[-self.num_evaders:] = [not(d)*1. for d in done[-self.num_evaders:]] # [1.]*self.num_evaders
            info = "All pursuers deceased"
        if all(done[-self.num_evaders:]): # pursuers win
            # reward[:self.num_pursuers] = 1. # [1.]*self.num_pursuers
            # reward[-self.num_evaders:] = -1. # [-1.]*self.num_evaders
            info = "All evaders deceased"
        self.step_counter += 1

        return obs, reward, done, info

    def render(self, pause=2):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        # plot world boundary
        bound = plt.Rectangle((-self.world_length/2,-self.world_length/2), self.world_length, self.world_length, linewidth=2, color='k', fill=False)
        self.ax.add_patch(bound)
        # draw obstacles
        for ci in range(len(self.obstacle_circles['radius'])):
            obs_circle = plt.Circle((self.obstacle_circles['position'][ci]), self.obstacle_circles['radius'][ci], color='grey')
            self.ax.add_patch(obs_circle)
        for ri in range(self.obstacle_rectangles['position'].shape[0]):
            obs_rect = plt.Rectangle((self.obstacle_rectangles['position'][ri]),self.obstacle_rectangles['dimension'][ri,0], self.obstacle_rectangles['dimension'][ri,1], color='grey')
            self.ax.add_patch(obs_rect)
        # draw evaders and annotate
        traj_es = np.array(self.evaders['trajectory'])
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie]=='active':
                self.ax.scatter(self.evaders['position'][ie,0], self.evaders['position'][ie,1], s=100, marker='*', color='orangered', linewidth=2)
                evader_circle = plt.Circle((self.evaders['position'][ie,0], self.evaders['position'][ie,1]), self.radius_evader, color='darkorange', fill=False)
                self.ax.add_patch(evader_circle)
                # plt.annotate(
                self.ax.annotate(
                    self.evaders['names'][ie], # this is the text
                    (self.evaders['position'][ie,0], self.evaders['position'][ie,1]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
                # draw trajectory
                self.ax.plot(traj_es[:,2*ie], traj_es[:,2*ie+1],  linestyle='--', linewidth=0.5, color='orangered')
        # draw pursuers
        traj_ps = np.array(self.pursuers['trajectory'])
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip]=='active':
                pursuer_circle = plt.Circle((self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), self.radius_evader, color='deepskyblue')
                self.ax.add_patch(pursuer_circle)
                # plt.annotate(
                self.ax.annotate(
                    self.pursuers['names'][ip], # this is the text
                    (self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
                # draw interfere circle
                interfere_circle = plt.Circle((self.pursuers['position'][ip,0], self.pursuers['position'][ip,1]), self.interfere_radius, color='deepskyblue', linestyle='-.', fill=False)
                self.ax.add_patch(interfere_circle)
                # draw trajectory
                self.ax.plot(traj_ps[:,2*ip], traj_ps[:,2*ip+1],  linestyle='--', linewidth=0.5, color='deepskyblue')
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
        self.pursuers['status'][id] = 'deactivated'

    def _disable_evader(self, id):
        self.evaders['position'][id] = np.zeros(2)
        self.evaders['status'][id] = 'deactivated'
