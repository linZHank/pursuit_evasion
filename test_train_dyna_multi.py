#!/usr/bin/python3
"""
Just a test script before full work process
"""
import sys
import os
import numpy as np
from numpy import random
from numpy import pi
from datetime import datetime
import matplotlib.pyplot as plt

from envs.pe_dyna_env import PEDynaEnv
from agents.dqn import DQNAgent
from agents.agent_utils import dqn_utils

import tensorflow as tf
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


if __name__ == '__main__':
    num_evaders, num_pursuers = 2, 2
    env=PEDynaEnv(num_evaders=num_evaders, num_pursuers=num_pursuers)
    agent_p = DQNAgent(env=env, name='pursuer')
    agent_e = DQNAgent(env=env, name='evader')
    num_episodes = 20
    num_steps = 100
    num_samples = 1 # sample k times to train q-net
    episodic_returns_p = np.zeros((num_episodes, num_pursuers))
    episodic_returns_e = np.zeros((num_episodes, num_evaders))
    sedimentary_returns_p = np.zeros((num_episodes, num_pursuers))
    sedimentary_returns_e = np.zeros((num_episodes, num_evaders))
    # step_counter = [1, 1, 1]
    pwin_counter, ewin_counter = 0, 0
    fig_r = plt.figure(figsize=(12,8))
    axs = []
    for i in range(num_pursuers+num_evaders):
        axs.append(fig_r.add_subplot(num_pursuers+num_evaders,1,i+1))
    fig_r.suptitle('Averaged Returns')

    for ep in range(num_episodes):
        done = [False]*(num_evaders+num_pursuers)
        # done_p0, done_p1, done_e0 = False, False, False
        total_reward_p = [[0.]]*num_pursuers # np.zeros(num_pursuers)
        total_reward_e = [[0.]]*num_evaders # np.zeros(num_evaders)
        # reset env and get state from it
        obs = env.reset()
        print("\n---\n{}".format(obs))
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(num_episodes/4))
        agent_e.linear_epsilon_decay(episode=ep, decay_period=int(num_episodes/4))
        # env.render(pause=0.5)
        for st in range(num_steps):
            # env.render(pause=1./env.rate)
            # convert obs to states
            states = dqn_utils.obs_to_states(obs, num_pursuers, num_evaders)
            # take actions, no action will take if deactivated
            ia_e = np.zeros(num_evaders, dtype=int)
            ia_p = np.zeros(num_pursuers, dtype=int)
            action_evaders = np.zeros((num_evaders,2))
            action_pursuers = np.zeros((num_pursuers,2))
            for i in range(num_evaders):
                if env.evaders['status'][i] == 'active':
                    ia_e[i], action_evaders[i] = agent_e.epsilon_greedy(states[num_pursuers+i])
            for i in range(num_evaders):
                if env.pursuers['status'][i] == 'active':
                    ia_p[i], action_pursuers[i] = agent_p.epsilon_greedy(states[i])
            # step env
            next_obs, reward, done, info = env.step(action_evaders, action_pursuers)
            next_states = dqn_utils.obs_to_states(next_obs, num_pursuers, num_evaders)
            # adjust reward according to distances
            for i in range(env.num_pursuers):
                if env.pursuers['status'] == 'active':
                    reward[i] += -sum(env.distance_matrix[i]<=env.interfere_radius)/num_steps
            # pursuers agent store transitions and train
            for i in range(num_pursuers):
                if not done[i]:
                    agent_p.store([states[i], ia_p[i], reward[i], done[i], next_states[i]])
                    # total_reward_p[i] += reward[i]
                    total_reward_p[i].append(reward[i])
            if ep >= agent_p.warmup_episodes:
                agent_p.train()
            # evaders agent store transitions and train
            for i in range(num_evaders):
                if not done[num_pursuers+i]:
                    agent_e.store([states[num_pursuers+i], ia_e[i], reward[num_pursuers+i], done[num_pursuers+i], next_states[num_pursuers+i]])
                    # total_reward_e[i] += reward[num_pursuers+i]
                    total_reward_e[i].append(reward[num_pursuers+i])
            if ep >= agent_e.warmup_episodes:
                agent_e.train()
            # count team wins
            if info == "All evaders deceased":
                pwin_counter += 1
            if info == "All pursuers deceased":
                ewin_counter += 1
            # log step
            print("\n-\nepisode: {}, step: {}, epsilon: {} \nstate: {} \naction_pursuers: {} \naction_evaders: {} \nnext_state: {} \nreward: {} \ninfo: {} \npursuers won: {}, evaders won: {}\n-\n".format(ep+1, st+1, agent_p.epsilon, obs, action_pursuers, action_evaders, next_obs, reward, info, pwin_counter, ewin_counter))
            obs = next_obs
            states = next_states
            if info:
                break
        # summarize episode
        episodic_returns_p[ep] = np.array([sum(total_reward_p[i]) for i in range(num_pursuers)])
        episodic_returns_e[ep] = np.array([sum(total_reward_e[i]) for i in range(num_evaders)])
        # sedimentary returns
        sedimentary_returns_p[ep] = np.sum(episodic_returns_p[:ep+1],axis=0)/(ep+1)
        sedimentary_returns_e[ep] = np.sum(episodic_returns_e[:ep+1],axis=0)/(ep+1)
        # plot ave_returns
        axs = fig_r.get_axes()
        for i in range(num_pursuers):
            axs[i].cla()
            axs[i].plot(np.arange(ep)+1, sedimentary_returns_p[:ep,i], color='deepskyblue', label='pursuer '+str(i))
            axs[i].legend()
            axs[i].grid(color='grey', linewidth=0.25)
        for i in range(num_evaders):
            axs[num_pursuers+i].cla()
            axs[num_pursuers+i].plot(np.arange(ep)+1, sedimentary_returns_e[:ep,i], color='orangered', label='evader '+str(i))
            axs[num_pursuers+i].legend()
            axs[num_pursuers+i].grid(color='grey', linewidth=0.25)
        plt.pause(1./100)
        fig_r.show()
        # plt.show(block=False)

    fig_path = os.path.join(os.path.dirname(agent_p.model_dir), 'ave_returns.png')
    fig_r.savefig(fig_path)

    # save model
    agent_p.save_model()
    agent_e.save_model()
    # save replay buffer
    agent_p.save_memory()
    agent_e.save_memory()
    # save returns
    np.save(os.path.join(agent_p.model_dir, 'ep_returns.npy'), episodic_returns_p)
    np.save(os.path.join(agent_e.model_dir, 'ep_returns.npy'), episodic_returns_e)
