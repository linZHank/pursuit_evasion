#!/usr/bin/python3
"""
Evaluate trained DQN agent with 2 pursuers pursuing 1 evader with circular motion
"""
import sys
import os
import numpy as np
from numpy import random
from numpy import pi

from envs.exp_pe_dyna_env import PEDynaEnv
from agents.exp_dqn import DQNAgent
from agents.agent_utils import dqn_utils

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


if __name__ == '__main__':
    num_pursuers, num_evaders = 2, 1
    env=PEDynaEnv(num_evaders=num_evaders, num_pursuers=num_pursuers)
    agent_p = DQNAgent(env=env, name='p_eval')
    # agent_e = DQNAgent(env=env, name='e_0')
    model_dir = sys.path[0]+'/saved_models/dqn/dyna_p2e1/2020-04-09-20-59/pursuer/models'
    model_path = os.path.join(model_dir,'340000.h5')
    # model_path_e0 = os.path.join(model_dir,'e_0','models','518810.h5')
    agent_p.load_model(model_path)
    # agent_e0.load_model(model_path_e0)
    agent_p.epsilon = 0.01
    # agent_e0.epsilon = 0.01

    num_episodes = 20
    num_steps = 200
    pwin_counter, ewin_counter = 0, 0

    for ep in range(num_episodes):
        theta_e = random.uniform(-pi,pi)
        env.evaders_spawning_pool[0] = np.array([3*np.cos(theta_e),3*np.sin(theta_e)])
        speed_evader = 0.4
        total_reward_p = []
        # reset env and get state from it
        agent_done = [False]*(num_evaders+num_pursuers) # agent done is one step after env done, so that -1 reward can be recorded
        obs = env.reset()
        for st in range(num_steps):
            env.render(pause=1./env.rate)
            # convert obs to states
            states = dqn_utils.obs_to_states_solo(obs, num_pursuers)
            # take actions, no action will take if deactivated
            i_a = np.zeros(num_pursuers+num_evaders, dtype=int)
            actions = np.zeros((num_pursuers+num_evaders,2))
            env.evaders['velocity'][0] = dqn_utils.circular_action(env.evaders['position'][0], speed_evader)
            for i in range(num_pursuers):
                if not agent_done[i]:
                    i_a[i], actions[i] = agent_p.epsilon_greedy(states[i])
            # step env
            next_obs, reward, done, info = env.step(actions)
            next_states = dqn_utils.obs_to_states_solo(next_obs, num_pursuers)
            # adjust reward then store transitions
            for i in range(env.num_pursuers):
                if agent_done[i]: # env.pursuers['status'][i] == 'active:
                    reward[i] = 0.
            total_reward_p.append(reward[:num_pursuers])
            # count team wins
            if info == "All evaders deceased":
                pwin_counter += 1
            if info == "All pursuers deceased":
                ewin_counter += 1
            # log step
            logging.info("\n-\nepisode: {}, step: {} \nactions: {} \nreward: {} \ninfo: {} \npursuers won: {}, evaders won: {}\n-\n".format(ep+1, st+1, actions, reward, info, pwin_counter, ewin_counter))
            obs = next_obs
            states = next_states
            if info:
                break
        # log episode
        logging.info("\n===\nepisode: {} \nround lasts: {} steps \nepisodic_returns: {} \n===\n".format(ep+1, st+1, np.sum(np.array(total_reward_p),axis=0)))
