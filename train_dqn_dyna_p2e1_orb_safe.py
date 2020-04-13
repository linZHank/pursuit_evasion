#!/usr/bin/python3
"""
Experimental training with 2 DQN pursuers vs 1 circular motion evader experience collection
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
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


if __name__ == '__main__':
    num_pursuers, num_evaders = 2, 1
    env=PEDynaEnv(num_evaders=num_evaders, num_pursuers=num_pursuers)
    agent_p = DQNAgent(
        env=env,
        name='pursuer',
        dim_state=12,
        layer_sizes=[128,128],
        learning_rate=0.0001,
        # update_epoch=1000,
        warmup_episodes=100
    )
    #
    num_episodes = 10000
    num_steps = 200
    num_samples = 1 # sample k times to train q-net
    episodic_returns_p = np.zeros((num_episodes, num_pursuers))
    sedimentary_returns_p = np.zeros((num_episodes, num_pursuers))
    # step_counter = [1, 1, 1]
    pwin_counter, ewin_counter = 0, 0
    fig_r = plt.figure(figsize=(12,8))

    for ep in range(num_episodes):
        theta_e = random.uniform(-pi,pi)
        env.evaders_spawning_pool[0] = np.array([3*np.cos(theta_e),3*np.sin(theta_e)])
        speed_evader = 0.4
        # step-wise reward storage
        total_reward_p = []
        # reset env and get state from it
        agent_done = [False]*(num_evaders+num_pursuers) # agent done is one step after env done, so that -1 reward can be recorded
        obs = env.reset()
        agent_p.linear_epsilon_decay(episode=ep, decay_period=2000) #int(num_episodes/4))
        for st in range(num_steps):
            # env.render(pause=1./env.rate) # render env will slow down training
            # convert obs to states
            states = dqn_utils.obs_to_states(obs, num_pursuers, num_evaders)
            # take actions, no action will take if deactivated
            i_a = np.zeros(num_pursuers+num_evaders, dtype=int)
            actions = np.zeros((num_pursuers+num_evaders,2))
            env.evaders['velocity'][0] = dqn_utils.circular_action(env.evaders['position'][0], speed_evader)
            for i in range(num_pursuers):
                if not agent_done[i]:
                    i_a[i], actions[i] = agent_p.epsilon_greedy(states[i])
            # step env
            next_obs, reward, done, info = env.step(actions)
            next_states = dqn_utils.obs_to_states(next_obs, num_pursuers, num_evaders)
            # adjust reward then store transitions
            for i in range(env.num_pursuers):
                if not agent_done[i]: # env.pursuers['status'][i] == 'active:
                    if reward[i] > 1 or reward[i] <= -1.:
                        agent_p.replay_memory.store([states[i], i_a[i], reward[i], True, next_states[i]])
                    else:
                        eff_d = (env.distance_matrix[i,:num_pursuers]<env.interfere_radius)*env.distance_matrix[i,:num_pursuers]
                        if not np.nansum(eff_d)==0:
                            reward[i] += np.clip(-.1/np.nansum(eff_d), -(num_pursuers-1), 0)
                        agent_p.replay_memory.store([states[i], i_a[i], reward[i], False, next_states[i]])
                else: # agent deactivated
                    reward[i] = 0.
            total_reward_p.append(reward[:num_pursuers])
            # train
            if ep >= agent_p.warmup_episodes:
                for _ in range(num_samples):
                    agent_p.train()
            # count team wins
            if info == "All evaders deceased":
                pwin_counter += 1
            if info == "All pursuers deceased":
                ewin_counter += 1
            # log step
            logging.info("\n-\nepisode: {}, step: {}, epsilon: {} \nstate: {} \nactions: {} \nnext_state: {} \nreward: {} \ninfo: {} \npursuers won: {}, evaders won: {}\n-\n".format(ep+1, st+1, agent_p.epsilon, states, actions[:num_pursuers], next_states, reward, info, pwin_counter, ewin_counter))
            obs = next_obs
            states = next_states
            agent_done = [done[i] for i in range(len(done))]
            if info:
                break
        # summarize episode
        episodic_returns_p[ep] = np.sum(np.array(total_reward_p),axis=0)
        # sedimentary returns
        sedimentary_returns_p[ep] = np.sum(episodic_returns_p[:ep+1],axis=0)/(ep+1)
        logging.info("\n===\nepisode: {} \ngame lasts: {} steps \nepisodic_returns: {} \nsedimentray_returns: {} \n===\n".format(ep+1, st+1, episodic_returns_p[ep], sedimentary_returns_p[ep]))
        # uncomment following to plot episodic average returns, but will slow down training
        ###################################################################################
        # fig_r.clf()
        # axs = []
        # for i in range(num_pursuers):
        #     axs.append(fig_r.add_subplot(num_pursuers,1,i+1))
        #     axs[i].plot(np.arange(ep+1)+1, sedimentary_returns_p[:ep+1,i], color='deepskyblue', label='pursuer '+str(i))
        #     # axs[i].plot(np.arange(ep+1)+1, episodic_returns_p[:ep+1,i], color='deepskyblue', label='pursuer '+str(i))
        #     if not i==num_pursuers-1:
        #         axs[i].set_xticklabels([])
        #     axs[i].set_xlim(0, ep+1)
        #     axs[i].legend(loc='upper right')
        #     axs[i].grid(color='grey', linewidth=0.2)
        # plt.tight_layout()
        # plt.pause(1./1000)
        # fig_r.show()
        ###################################################################################

    # save averaged returns figure
    fig_r.clf()
    axs = []
    for i in range(num_pursuers):
        axs.append(fig_r.add_subplot(num_pursuers,1,i+1))
        axs[i].plot(np.arange(ep+1)+1, sedimentary_returns_p[:ep+1,i], color='deepskyblue', label='pursuer '+str(i))
        if not i==num_pursuers-1:
            axs[i].set_xticklabels([])
        axs[i].set_xlim(0, ep+1)
        axs[i].legend(loc='upper right')
        axs[i].grid(color='grey', linewidth=0.2)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(agent_p.model_dir), 'ave_returns.png')
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    fig_r.savefig(fig_path)

    # save model
    agent_p.save_model()
    # save replay buffer
    # agent_p.save_memory()
    # save hyper-parameters
    agent_p.save_params()
    # save returns
    np.save(os.path.join(agent_p.model_dir, 'ep_returns.npy'), episodic_returns_p)
