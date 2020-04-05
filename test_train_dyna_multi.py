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

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.WARNING)


if __name__ == '__main__':
    num_evaders, num_pursuers = 2, 2
    env=PEDynaEnv(num_evaders=num_evaders, num_pursuers=num_pursuers)
    agent_p = DQNAgent(env=env, name='pursuer')
    agent_p.warmup_episodes=10
    agent_p.decay_period=20
    agent_p.update_epoch=25
    agent_e = DQNAgent(env=env, name='evader')
    num_episodes = 50
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
        agent_done = [False]*(num_evaders+num_pursuers)
        # done_p0, done_p1, done_e0 = False, False, False
        total_reward_p = []
        total_reward_e = []
        # total_reward_p = np.zeros(num_pursuers)
        # total_reward_e = np.zeros(num_evaders)
        # reset env and get state from it
        obs = env.reset()
        # print("\n---\n{}".format(obs))
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(num_episodes/4))
        agent_e.linear_epsilon_decay(episode=ep, decay_period=int(num_episodes/4))
        # env.render(pause=0.5)
        for st in range(num_steps):
            # env.render(pause=1./env.rate)
            # env.render(pause=4) # debug
            # convert obs to states
            states = dqn_utils.obs_to_states(obs, num_pursuers, num_evaders)
            # take actions, no action will take if deactivated
            ia_e = np.zeros(num_evaders, dtype=int)
            ia_p = np.zeros(num_pursuers, dtype=int)
            action_evaders = np.zeros((num_evaders,2))
            action_pursuers = np.zeros((num_pursuers,2))
            for i in range(num_evaders):
                if not agent_done[i]: # env.evaders['status'][i] == 'active':
                    ia_p[i], action_pursuers[i] = agent_p.epsilon_greedy(states[i])
            for j in range(num_evaders):
                if not agent_done[-num_evaders+j]: # if env.pursuers['status'][i] == 'active':
                    ia_e[j], action_evaders[j] = agent_e.epsilon_greedy(states[-num_evaders+j])
            # step env
            next_obs, reward, done, info = env.step(action_evaders, action_pursuers)
            next_states = dqn_utils.obs_to_states(next_obs, num_pursuers, num_evaders)
            # adjust reward then store transitions
            for i in range(env.num_pursuers):
                if not agent_done[i]: # env.pursuers['status'][i] == 'active:
                    reward[i] += -sum(env.distance_matrix[i]<=env.interfere_radius)/num_steps
                    agent_p.replay_memory.store([states[i], ia_p[i], reward[i], done[i], next_states[i]])
                else:
                    reward[i] = 0.
                # total_reward_p[i] += reward[i]
            total_reward_p.append(reward[:num_pursuers])
            for j in range(env.num_evaders):
                if not agent_done[-num_evaders+j]: # env.evaders['status'][i] == 'active':
                    agent_e.replay_memory.store([states[-num_evaders+j], ia_e[j], reward[-num_evaders+j], done[-num_evaders+j], next_states[-num_evaders+j]])
                else:
                    reward[-num_evaders+j] = 0.
                # total_reward_e[i] += reward[-num_evaders+i]
            total_reward_e.append(reward[-num_evaders:])
            # print("\n-\nepisode: {}, step: {} \nagent done: {}, done: {} \nreward: {} \n".format(ep+1, st+1, agent_done, done, reward)) # debug
            # train
            if ep >= agent_p.warmup_episodes:
                for _ in range(num_samples):
                    agent_p.train()
            if ep >= agent_e.warmup_episodes:
                for _ in range(num_samples):
                    agent_e.train()
            # count team wins
            if info == "All evaders deceased":
                pwin_counter += 1
            if info == "All pursuers deceased":
                ewin_counter += 1
            # log step
            logging.info("\n-\nepisode: {}, step: {}, epsilon: {} \nstate: {} \naction_pursuers: {} \naction_evaders: {} \nnext_state: {} \nreward: {} \ninfo: {} \npursuers won: {}, evaders won: {}\n-\n".format(ep+1, st+1, agent_p.epsilon, obs, action_pursuers, action_evaders, next_obs, reward, info, pwin_counter, ewin_counter))
            obs = next_obs
            states = next_states
            agent_done = [done[i] for i in range(len(done))]
            if info:
                break
        # summarize episode
        # episodic_returns_p[ep] = total_reward_p
        # episodic_returns_p[ep] = total_reward_e
        episodic_returns_p[ep] = np.sum(np.array(total_reward_p),axis=0)
        episodic_returns_e[ep] = np.sum(np.array(total_reward_e),axis=0)
        # episodic_returns_p[ep] = np.array([sum(total_reward_p[i]) for i in range(num_pursuers)])
        # episodic_returns_e[ep] = np.array([sum(total_reward_e[i]) for i in range(num_evaders)])
        # print("\n---\n ep_ret_p: {} \n ep_ret_e: {} \n---\n".format(episodic_returns_p[ep],episodic_returns_e[ep])) # debug
        # sedimentary returns
        sedimentary_returns_p[ep] = np.sum(episodic_returns_p[:ep+1],axis=0)/(ep+1)
        sedimentary_returns_e[ep] = np.sum(episodic_returns_e[:ep+1],axis=0)/(ep+1)
        # plot ave_returns
        # axs = fig_r.get_axes()
        # for i in range(num_pursuers):
        #     axs[i].cla()
        #     axs[i].plot(np.arange(ep)+1, sedimentary_returns_p[:ep,i], color='deepskyblue', label='pursuer '+str(i))
        #     axs[i].legend()
        #     axs[i].grid(color='grey', linewidth=0.25)
        # for i in range(num_evaders):
        #     axs[num_pursuers+i].cla()
        #     axs[num_pursuers+i].plot(np.arange(ep)+1, sedimentary_returns_e[:ep,i], color='orangered', label='evader '+str(i))
        #     axs[num_pursuers+i].legend()
        #     axs[num_pursuers+i].grid(color='grey', linewidth=0.25)
        # plt.pause(1./100)
        # fig_r.show()
        # plt.show(block=False)
        fig_r.clf()
        axs = []
        for i in range(num_pursuers+num_evaders):
            axs.append(fig_r.add_subplot(num_pursuers+num_evaders,1,i+1))
            if i < num_pursuers:
                axs[i].plot(np.arange(ep+1)+1, sedimentary_returns_p[:ep+1,i], color='deepskyblue', label='pursuer '+str(i))
            else:
                axs[i].plot(np.arange(ep+1)+1, sedimentary_returns_e[:ep+1,i-num_pursuers], color='orangered', label='evader '+str(i))
            y_ticks = np.arange(-1.,1.4,0.4)
            if not i==num_pursuers+num_evaders-1:
                axs[i].set_xticklabels([])
            axs[i].set_yticks(y_ticks)
            axs[i].set_xlim(0, ep+1)
            axs[i].set_ylim(y_ticks[0]-0.1, y_ticks[-1]+0.1)
            axs[i].legend(loc='lower right')
            axs[i].grid(color='grey', linewidth=0.2)
        fig_r.suptitle('Averaged Returns')
        plt.pause(1./1000)
        fig_r.show()

    # print("\n====\nep return: {} \nsed return: {}".format((episodic_returns_p,episodic_returns_e), (sedimentary_returns_p, sedimentary_returns_e))) # debug
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