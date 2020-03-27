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
    env=PEKineEnv()
    agent_p = DQNAgent()
    model_path = sys.path[0]+'/saved_models/p1e1_kine/dqn/2020-03-26-00-39/agent_p/active_model-6479248.h5'
    agent_p.load_model(model_path)
    agent_p.epsilon = 0.01

    num_episodes = 20
    num_steps = 200 # env.max_steps
    success_counter = 0

    for ep in range(num_episodes):
        # specify evader's spawining position
        theta_e = random.uniform(-pi,pi)
        env.evaders_spawning_pool[0] = np.array([3*np.cos(theta_e),3*np.sin(theta_e)])
        evader_speed = random.uniform(-pi/2,pi/2)
        done, total_reward = False, []
        state, _ = env.reset()
        for st in range(num_steps):
            # action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            action_evaders = dqn_utils.circular_action(state[-2:],speed=evader_speed)
            # action_evaders = np.zeros((1,2))
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            next_state, reward, done, info = env.step(action_evaders, action_pursuers)
            rew, done, success = dqn_utils.adjust_reward(env, num_steps, state, reward, done, next_state)
            env.render(pause=1./env.rate)
            total_reward.append(rew)
            state = next_state
            # log step
            print("\n-\nepisode: {}, step: {} \nstate: {} \naction: {}->{} \nnext_state: {} \nreward: {}, succeeded: {}\n-\n".format(ep+1, st+1, state, ia, action_pursuers, next_state, rew, success_counter))
            if done:
                if success:
                    success_counter += 1
                break
        # log episode
        print(" \n---\nepisode: {}. episodic_return: {}".format(ep+1, sum(total_reward)))
