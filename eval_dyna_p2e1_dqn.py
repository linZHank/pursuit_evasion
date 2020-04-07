#!/usr/bin/python3
"""
Evaluate trained DQN agent
"""
import sys
import os
import numpy as np
from numpy import random
from numpy import pi

from envs.pe_dyna_env import PEDynaEnv
from agents.dqn import DQNAgent
from agents.agent_utils import dqn_utils


if __name__ == '__main__':
    env=PEDynaEnv(num_evaders=1, num_pursuers=2)
    agent_p0 = DQNAgent(env=env, name='p_0')
    agent_p1 = DQNAgent(env=env, name='p_1')
    agent_e0 = DQNAgent(env=env, name='e_0')
    model_dir = sys.path[0]+'/saved_models/p2e1_dyna/dqn/2020-04-02-00-01/'
    model_path_p0 = os.path.join(model_dir,'p_0','models','430834.h5')
    model_path_p1 = os.path.join(model_dir,'p_1','models','429981.h5')
    model_path_e0 = os.path.join(model_dir,'e_0','models','518810.h5')
    agent_p0.load_model(model_path_p0)
    agent_p1.load_model(model_path_p1)
    agent_e0.load_model(model_path_e0)
    agent_p0.epsilon = 0.01
    agent_p1.epsilon = 0.01
    agent_e0.epsilon = 0.01

    num_episodes = 20
    num_steps = env.max_steps
    pwin_counter, ewin_counter = 0, 0

    for ep in range(num_episodes):
        done = [False]*3
        total_reward_p0, total_reward_p1, total_reward_e0 = [], [], []
        state = env.reset()
        print("\n---\n{}".format(state))
        for st in range(num_steps):
            env.render(pause=1./env.rate)
            # take actions, no action will take if deactivated
            action_evaders = np.zeros((1,2))
            action_pursuers = np.zeros((2,2))
            if env.pursuers['status'][0] == 'active':
                ia_p0, action_pursuers[0] = agent_p0.epsilon_greedy(state)
            if env.pursuers['status'][1] == 'active':
                ia_p1, action_pursuers[1] = agent_p1.epsilon_greedy(state)
            if env.evaders['status'][0] == 'active':
                ia_e0, action_evaders[0] = agent_e0.epsilon_greedy(state)
            # step env
            next_state, reward, done, info = env.step(action_evaders, action_pursuers)
            # adjust reward according to distances
            for i in range(env.num_pursuers):
                reward[i] += -sum(env.distance_matrix[i]<=env.interfere_radius)/num_steps
            # append reward
            if not done[0]:
                total_reward_p0.append(reward[0])
            if not done[1]:
                total_reward_p1.append(reward[1])
            if not done[2]:
                total_reward_e0.append(reward[2])
            if info == "All evaders deceased":
                pwin_counter += 1
            if info == "All pursuers deceased":
                ewin_counter += 1
            state = next_state
            # log step
            print("\n-\nepisode: {}, step: {} \nstate: {} \naction_pursuers: {} \naction_evaders: {} \nnext_state: {} \nreward: {}, \npursuers won: {}, evaders won: {}\n-\n".format(ep+1, st+1, state, action_pursuers, action_evaders, next_state, reward, pwin_counter, ewin_counter))
            state = next_state
            if info:
                break
        # log episode
        print(" \n---\nepisode: {}. episodic_return: {} \n---\n".format(ep+1, (sum(total_reward_p0),sum(total_reward_p1),sum(total_reward_e0))))
