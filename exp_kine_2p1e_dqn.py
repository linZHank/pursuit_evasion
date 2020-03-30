#!/usr/bin/python3
"""
Evaluate trained DQN agent
"""
import sys
import os
import numpy as np
from numpy import random
from numpy import pi

from envs.pe_kine_env import PEKineEnv
from agents.dqn import DQNAgent
from agents.agent_utils import dqn_utils


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=2)
    agent_p0 = DQNAgent()
    agent_p1 = DQNAgent()
    model_path = sys.path[0]+'/saved_models/p1e1_kine/dqn/2020-03-26-00-39/agent_p/active_model-6479248.h5'
    agent_p0.load_model(model_path)
    agent_p0.epsilon = 0.01
    agent_p1.load_model(model_path)
    agent_p1.epsilon = 0.01

    num_episodes = 20
    num_steps = 200 # env.max_steps
    success_counter = 0

    for ep in range(num_episodes):
        # specify evader's spawining position
        theta_e = random.uniform(-pi,pi)
        env.evaders_spawning_pool[0] = np.array([3*np.cos(theta_e),3*np.sin(theta_e)])
        evader_speed = random.uniform(-pi/2,pi/2)
        env.pursuers_spawning_pool = np.array([np.random.choice([-4,4],2)+np.random.normal(0,0.1,2), np.random.choice([-4,4],2)+np.random.normal(0,0.1,2)])
        done, total_reward = False, []
        state, _ = env.reset()
        success_p0, success_p1 = False, False
        for st in range(num_steps):
            state_p0 = np.concatenate((state[0:2],state[-2:]))
            state_p1 = state[-4:]
            # action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            action_evaders = dqn_utils.circular_action(state[-2:],speed=evader_speed)
            # action_evaders = np.zeros((1,2))
            ia_p0, action_p0 = agent_p0.epsilon_greedy(state_p0)
            ia_p1, action_p1 = agent_p1.epsilon_greedy(state_p1)
            action_pursuers = np.concatenate((action_p0,action_p1),axis=0)
            next_state, reward, done, info = env.step(action_evaders, action_pursuers)
            next_state_p0 = np.concatenate((next_state[0:2],next_state[-2:]))
            next_state_p1 = next_state[-4:]
            # rew, done, success = dqn_utils.adjust_reward(env, num_steps, state, reward, done, next_state)
            env.render(pause=1./env.rate)
            # total_reward.append(rew)
            state = next_state
            if env.pursuers['status'][0] == 'catching':
                success_p0 = True
            if env.pursuers['status'][1] == 'catching':
                success_p1 = True
            # log step
            print("\n-\nepisode: {}, step: {} \nstate: {} \naction_p0: {}->{} \naction_p1: {}->{} \nnext_state: {}\n-\n".format(ep+1, st+1, state, ia_p0, action_p0, ia_p1, action_p1, next_state))
            if success_p0 or success_p1:
                # if success:
                success_counter += 1
                break
        # log episode
        print(" \n---\nepisode: {} success: {}".format(ep+1, success_counter))
