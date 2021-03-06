#!/usr/bin/env python3
"""
Pursuit-evasion environment:
    - Randomly placed obstacles
    - Multiple Obstacles (<=14)
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
import cv2


class PursuitEvasion:
    """
    Dynamics Pursuit-evasion env: N pursuers, N evader (1<=N<=4)
    """
    def __init__(self, resolution=(80, 80)):
        # Env specs #
        self.name='pursuit-evasion' # dynamic multi-pursuer multi-evader
        self.rate = 15 # Hz
        self.max_episode_steps = 1000
        self.resolution = resolution
        self.world_length = 10
        self.damping = 0.2
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
        self.image = np.zeros((self.resolution[0],self.resolution[1],3))
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)

    def reset(self):
        """
        Reset obstacles and agents location
        Args:
        Return:
            obs: map image
        """
        # Prepare 
        self.num_evaders = random.randint(1,self.max_num_evaders+1)
        self.num_pursuers = random.randint(1,self.max_num_pursuers+1)
        self.num_ellipses = random.randint(1,self.max_num_ellipses+1)
        self.num_polygons = random.randint(1,self.max_num_polygons+1)
        self.step_counter = 0
        self.evaders = dict(
            name = ['e-'+str(i) for i in range(self.num_evaders)],
            position = np.inf*np.ones((self.num_evaders, 2)),
            velocity = np.zeros((self.num_evaders, 2)),
            trajectory = [],
            status = ['deactivated']*self.num_evaders,
        )
        self.pursuers = dict(
            name = ['p-'+str(i) for i in range(self.num_pursuers)],
            position = np.inf*np.ones((self.num_pursuers, 2)),
            velocity = np.zeros((self.num_pursuers, 2)),
            trajectory = [],
            status = ['deactivated']*self.num_pursuers,
        )
        self.spawning_pool = random.uniform(
            -self.world_length/2+.2, self.world_length/2-.2,
            size=(self.num_evaders+self.num_pursuers,2)
        ) # .2 threshold to avoid spawning too close to the walls
        img_e = np.zeros((self.resolution[0], self.resolution[1]), dtype=np.uint8)
        img_o = np.zeros((self.resolution[0], self.resolution[1]), dtype=np.uint8)
        img_p = np.zeros((self.resolution[0], self.resolution[1]), dtype=np.uint8)
        obs = np.zeros((self.num_evaders+self.num_pursuers+1, self.resolution[0], self.resolution[1]), dtype=np.uint8)
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
        obs[self.num_evaders] = 255*self._get_image(
            patch_list=self.obstacle_patches, 
            radius=self.world_length/np.min(self.resolution)/2
        )
        img_o = obs[self.num_evaders]
        # Reset Evaders 
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
            octagon = RegularPolygon(
                xy=self.evaders['position'][ie], 
                numVertices=8, 
                radius=self.evader_radius, 
                fc='orangered'
            )
            self.evader_patches.append(octagon)
            obs[ie] = 255*self._get_image(patch_list=[octagon], radius=self.evader_radius) 
            img_e += obs[ie]
        ## create evaders map
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
            circle = Circle(
                xy=self.pursuers['position'][ip], 
                radius=self.pursuer_radius, 
                fc='deepskyblue'
            )
            self.pursuer_patches.append(circle)
            obs[-self.num_pursuers+ip] = 255*self._get_image(patch_list=[circle], radius=self.pursuer_radius) 
            img_p += obs[-self.num_pursuers+ip]
        # Create map image 
        self.image[:,:,0] = img_e # R: evader channel
        self.image[:,:,1] = img_o # G: obstacle channel
        self.image[:,:,2] = img_p # B: pursuer channel

        return obs

    def step(self, actions):
        """
        Agents take velocity command
        Args:
            actions: array([[fx_e0,fy_e0],[fx_e1,fy_e1],...,[fx_pN,fy_pN]])
        Returns:
            obs: map image
            reward: array([r_e0,...,r_pN])
            done: array([d_e0,...,d_pN]), bool
            info: episode result or ''
        """
        # Check input
        assert actions.shape == (self.num_evaders+self.num_pursuers, 2)
        actions = np.clip(actions, self.action_space_low, self.action_space_high)
        # Prepare
        bonus = np.zeros(self.num_evaders+self.num_pursuers) # add bonus when key event detected
        reward = np.zeros(self.num_evaders + self.num_pursuers)
        done = np.array([False]*(self.num_evaders + self.num_pursuers))
        info = ''
        img_e = np.zeros((self.resolution[0], self.resolution[1]), dtype=np.uint8)
        img_p = np.zeros((self.resolution[0], self.resolution[1]), dtype=np.uint8)
        obs = np.zeros((self.num_evaders+self.num_pursuers+1, self.resolution[0], self.resolution[1]), dtype=np.uint8)
        obs[self.num_evaders] = self.image[:,:,1]
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
                    self.evaders['status'][ie] = 'deactivated'
                    bonus[ie] = -100 # -np.sqrt(2*self.action_space_high**2)*self.max_episode_steps/10.
                else:
                    bonus[ie] = 0 # -np.linalg.norm(actions[ie])/10.
            else:
                actions[ie] = np.zeros(2)
                self.evaders['velocity'][ie] = np.zeros(2)
        ## record evaders trajectory
        self.evaders['trajectory'].append(self.evaders['position'].copy())
        ## create evader patches, 八面玲珑
        self.evader_patches = []
        for ie in range(self.num_evaders):
            if self.evaders['status'][ie] == 'active':
                octagon = RegularPolygon(
                    xy=self.evaders['position'][ie], 
                    numVertices=8, 
                    radius=self.evader_radius, 
                    fc='orangered'
                )
                self.evader_patches.append(octagon)
                obs[ie] = 255*self._get_image(patch_list=[octagon], radius=self.evader_radius) 
                img_e += obs[ie]

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
                    self.pursuers['status'][ip] = 'deactivated'
                    bonus[-self.num_pursuers+ip] = -100 # -np.sqrt(2*self.action_space_high**2)*self.max_episode_steps/10.
                else:
                    bonus[-self.num_pursuers+ip] = 0 # -np.linalg.norm(actions[-self.num_pursuers+ip])/10.
            else:
                actions[-self.num_pursuers+ip] = np.zeros(2)
                self.pursuers['velocity'][ip] = np.zeros(2)
            ## detect captures
            if self.pursuers['status'][ip] == 'active': # status updated, check status again
                for ie in range(self.num_evaders):
                    if self.evaders['status'][ie] =='active':
                        if np.linalg.norm(self.pursuers['position'][ip] - self.evaders['position'][ie]) <= self.interfere_radius:
                            self.evaders['status'][ie] = 'deactivated'
                            bonus[ie] = -100 # -np.sqrt(2*self.action_space_high**2)*self.max_episode_steps/10.
                            bonus[-self.num_pursuers+ip] = 100 # np.sqrt(2*self.action_space_high**2)*self.max_episode_steps/10.
        ## record pursuers trajectory
        self.pursuers['trajectory'].append(self.pursuers['position'].copy())
        ## create pursuer patches, 圆滑世故
        self.pursuer_patches = []
        for ip in range(self.num_pursuers):
            if self.pursuers['status'][ip] == 'active':
                circle = Circle(
                    xy=self.pursuers['position'][ip], 
                    radius=self.pursuer_radius, 
                    fc='deepskyblue'
                )
                self.pursuer_patches.append(circle)
                obs[-self.num_pursuers+ip] = 255*self._get_image(patch_list=[circle], radius=self.pursuer_radius) 
                img_p += obs[-self.num_pursuers+ip]
        # Create map image, obstacle channel no need to change 
        self.image[:,:,0] = img_e # R: evader channel
        self.image[:,:,2] = img_p # B: pursuer channel
        # Finish step
        self.step_counter += 1
        ## reward
        reward += bonus.copy()
        ## done if deactivated
        done = np.array([s=='deactivated' for s in self.evaders['status']] + [s=='deactivated' for s in self.pursuers['status']])
        if self.step_counter == self.max_episode_steps:
            info = "Timeup"
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
                    self.evaders['name'][ie], # pursuer name
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
                    self.pursuers['name'][ip], # pursuer name
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

    def _is_outbound(self, pos, radius):
        """
        Detect a given position is out of boundary or not
        """
        out_flag = False
        if np.absolute(pos[0])>=self.world_length/2-radius or np.absolute(pos[1])>=self.world_length/2-radius:
            out_flag = True

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

    def _get_image(self, patch_list, radius):
        """
        Create image on map according to patches in list
        """
        patch_pix = np.array([False]*self.pix_coords.shape[0])
        for p in patch_list:
            patch_pix = np.logical_or(patch_pix, p.contains_points(self.pix_coords, radius=radius))
        image = patch_pix.reshape(self.resolution)

        return np.transpose(image)


if __name__ == '__main__':
    env=PursuitEvasion()
    for ep in range(10):
        obs = env.reset()
        for st in range(env.max_episode_steps):
            cv2.imshow('map', cv2.resize(env.image[:,:,[2,1,0]], (720, 720)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            env.render(pause=1./env.rate)
            actions = np.random.uniform(-4,4,size=(env.num_evaders+env.num_pursuers,2))
            obs, rew, done, info = env.step(actions)
            print("\nevaders_pos: {} \npursuers_pos: {} \nreward: {} \ndone: {}".format(env.evaders['position'], env.pursuers['position'], rew, done))
            if info:
                print(info)
                break
    cv2.destroyAllWindows()
