#!/usr/bin/python3
"""
Evaluate trained DQN agent
"""
import sys
import os
import numpy as np
from numpy import random

import utils
from envs.pe_kine_env import PEKineEnv
from agents.dqn import DQNAgent
from agents.agent_utils import dqn_utils


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=1)
    agent_p = DQNAgent()
    model_path = sys.path[0]+'/saved_models/p1e1_kine/dqn/2020-03-22-13-23/agent_p/active_model-100000.h5'
    agent_p.load_model(model_path)
    agent_p.epsilon = 0.01

    num_episodes = 3
    num_steps = 50 # env.max_steps
    for ep in range(num_episodes):
        done, total_reward = False, []
        state, _ = env.reset()
        for st in range(num_steps):
            # action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            action_evaders = utils.cirluar_action(state[-2:],speed=1)
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            next_state, rew, done, _ = env.step(action_evaders, action_pursuers)
            env.render(pause=1./env.rate)
            total_reward.append(rew[0])
            state = next_state
            # log step
            print("episode: {}, step: {} \nstate: {} \naction: {}->{} \nnext_state: {} \nreward: {}".format(ep+1, st+1, state, ia, action_pursuers, next_state, rew[0]))
            if done:
                break
        # log episode
        print(" \n---\nepisode: {}. episodic_return: {}".format(ep+1, sum(total_reward)))
