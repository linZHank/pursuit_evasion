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

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


if __name__ == '__main__':
    num_pursuers, num_evaders = 4, 4
    env=PEDynaEnv(num_evaders=num_evaders, num_pursuers=num_pursuers)
    agent_p = DQNAgent(env=env, name='p_eval')
    # agent_e = DQNAgent(env=env, name='e_0')
    model_dir = sys.path[0]+'/saved_models/dqn/dyna_p4e4/2020-04-05-12-15/pursuer/models'
    model_path = os.path.join(model_dir,'1052374.h5')
    # model_path_e0 = os.path.join(model_dir,'e_0','models','518810.h5')
    agent_p.load_model(model_path)
    # agent_e0.load_model(model_path_e0)
    agent_p.epsilon = 0.01
    # agent_e0.epsilon = 0.01

    num_episodes = 20
    num_steps = env.max_steps
    pwin_counter, ewin_counter = 0, 0

    for ep in range(num_episodes):
        total_reward_p = []
        # reset env and get state from it
        agent_done = [False]*(num_evaders+num_pursuers) # agent done is one step after env done, so that -1 reward can be recorded
        obs = env.reset()
        for st in range(num_steps):
            env.render(pause=1./env.rate)
            # convert obs to states
            states = dqn_utils.obs_to_states(obs, num_pursuers, num_evaders)
            # take actions, no action will take if deactivated
            ia_p = np.zeros(num_pursuers, dtype=int)
            action_evaders = np.zeros((num_evaders,2))
            action_pursuers = np.zeros((num_pursuers,2))
            for i in range(num_evaders):
                if not agent_done[i]:
                    ia_p[i], action_pursuers[i] = agent_p.epsilon_greedy(states[i])
            # step env
            next_obs, reward, done, info = env.step(action_evaders, action_pursuers)
            next_states = dqn_utils.obs_to_states(next_obs, num_pursuers, num_evaders)
            # adjust reward then store transitions
            for i in range(env.num_pursuers):
                if not agent_done[i]: # env.pursuers['status'][i] == 'active:
                    reward[i] += -sum(env.distance_matrix[i,:num_pursuers]<=env.interfere_radius)/num_steps
                else:
                    reward[i] = 0.
            total_reward_p.append(reward[:num_pursuers])
            # count team wins
            if info == "All evaders deceased":
                pwin_counter += 1
            if info == "All pursuers deceased":
                ewin_counter += 1
            # log step
            logging.info("\n-\nepisode: {}, step: {} \naction_pursuers: {} \nreward: {} \ninfo: {} \npursuers won: {}, evaders won: {}\n-\n".format(ep+1, st+1, action_pursuers, reward, info, pwin_counter, ewin_counter))
            obs = next_obs
            states = next_states
            if info:
                break
        # log episode
        logging.info("\n===\nepisode: {} \nround lasts: {} steps \nepisodic_returns: {} \n===\n".format(ep+1, st+1, np.sum(np.array(total_reward_p),axis=0)))
