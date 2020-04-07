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
    env=PEDynaEnv(num_evaders=1, num_pursuers=2)
    agent_p0 = DQNAgent(env=env, name='p_0')
    agent_p1 = DQNAgent(env=env, name='p_1')
    agent_e0 = DQNAgent(env=env, name='e_0')
    # date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # model_dir = sys.path[0]+"/saved_models/p2e1_dyna/test_dqn/"+date_time
    num_episodes = 200
    num_steps = 100
    num_samples = 1 # sample k times to train q-net
    episodic_returns_p0, episodic_returns_p1, episodic_returns_e0 = [], [], []
    sedimentary_returns_p0, sedimentary_returns_p1, sedimentary_returns_e0 = [], [], []
    # step_counter = [1, 1, 1]
    pwin_counter, ewin_counter = 0, 0
    fig_r = plt.figure(figsize=(12,8))
    ax_p0 = fig_r.add_subplot(3,1,1)
    ax_p1 = fig_r.add_subplot(3,1,2, sharex = ax_p0)
    ax_e0 = fig_r.add_subplot(3,1,3, sharex = ax_p0)
    fig_r.suptitle('Averaged Returns')

    for ep in range(num_episodes):
        done = [False] * 3
        # done_p0, done_p1, done_e0 = False, False, False
        total_reward_p0 = []
        total_reward_p1 = []
        total_reward_e0 = []
        # reset env and get state from it
        state = env.reset()
        print("\n---\n{}".format(state))
        agent_p0.linear_epsilon_decay(episode=ep, decay_period=int(num_episodes/5))
        agent_p1.linear_epsilon_decay(episode=ep, decay_period=int(num_episodes/5))
        agent_e0.linear_epsilon_decay(episode=ep, decay_period=int(num_episodes/5))
        # env.render(pause=0.5)
        for st in range(num_steps):
            # env.render(pause=1./env.rate)
            # env.render(pause=0.5)
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
            # store transitions and train
            if not done[0]:
                agent_p0.store([state, ia_p0, reward[0], done[0], next_state])
                if ep >= agent_p0.warmup_episodes:
                    agent_p0.train()
                # if not step_counter[0] % agent_p0.update_step:
                #     agent_p0.save_model(model_dir)
                #     agent_p0.qnet_stable.set_weights(agent_p0.qnet_active.get_weights())
                # step_counter[0] += 1
                total_reward_p0.append(reward[0])
            if not done[1]:
                agent_p1.store([state, ia_p1, reward[1], done[1], next_state])
                if ep >= agent_p1.warmup_episodes:
                    agent_p1.train()
                # if not step_counter[1] % agent_p1.update_step:
                #     agent_p1.save_model(model_dir)
                #     agent_p1.qnet_stable.set_weights(agent_p1.qnet_active.get_weights())
                # step_counter[1] += 1
                total_reward_p1.append(reward[1])
            if not done[2]:
                agent_e0.store([state, ia_e0, reward[2], done[2], next_state])
                if ep >= agent_e0.warmup_episodes:
                    agent_e0.train()
                # if not step_counter[2] % agent_e0.update_step:
                #     agent_e0.save_model(model_dir)
                #     agent_e0.qnet_stable.set_weights(agent_e0.qnet_active.get_weights())
                # step_counter[2] += 1
                total_reward_e0.append(reward[2])
            if info == "All evaders deceased":
                pwin_counter += 1
            if info == "All pursuers deceased":
                ewin_counter += 1
            # log step
            print("\n-\nepisode: {}, step: {}, epsilon: {} \nstate: {} \naction_pursuers: {} \naction_evaders: {} \nnext_state: {} \nreward: {} \ninfo: {} \npursuers won: {}, evaders won: {}\n-\n".format(ep+1, st+1, agent_p0.epsilon, state, action_pursuers, action_evaders, next_state, reward, info, pwin_counter, ewin_counter))
            state = next_state
            if info:
                break
        # summarize episode
        episodic_returns_p0.append(sum(total_reward_p0))
        episodic_returns_p1.append(sum(total_reward_p1))
        episodic_returns_e0.append(sum(total_reward_e0))
        # sedimentary returns
        sed_return_p0 = sum(episodic_returns_p0)/(ep+1)
        sedimentary_returns_p0.append(sed_return_p0)
        sedimentary_returns_p1.append(sum(episodic_returns_p1)/(ep+1))
        sedimentary_returns_e0.append(sum(episodic_returns_e0)/(ep+1))
        # plot ave_returns
        (ax_p0, ax_p1, ax_e0) = fig_r.get_axes()
        ax_p0.cla()
        ax_p0.plot(np.arange(ep+1)+1, sedimentary_returns_p0, color='deepskyblue', label='pursuer 0')
        ax_p0.legend()
        ax_p0.grid(color='grey', linewidth=0.5)
        ax_p1.cla()
        ax_p1.plot(np.arange(ep+1)+1, sedimentary_returns_p1, color='deepskyblue', label='pursuer 1')
        ax_p1.legend()
        ax_p1.grid(color='grey', linewidth=0.5)
        ax_e0.cla()
        ax_e0.set_xlabel("Episode", fontsize=14)
        ax_e0.plot(np.arange(ep+1)+1, sedimentary_returns_e0, color='orangered', label='evader 0')
        ax_e0.legend()
        ax_e0.grid(color='grey', linewidth=0.5)
        plt.pause(1./100)
        fig_r.show()
        # plt.show(block=False)

    fig_path = os.path.join(os.path.dirname(agent_p0.model_dir), 'ave_returns.png')
    fig_r.savefig(fig_path)

    # save model
    agent_p0.save_model()
    agent_p1.save_model()
    agent_e0.save_model()
    # save replay buffer
    agent_p0.save_memory()
    agent_p1.save_memory()
    agent_e0.save_memory()
    # save returns
    np.save(os.path.join(agent_p0.model_dir, 'ep_returns.npy'), episodic_returns_p0)
    np.save(os.path.join(agent_p1.model_dir, 'ep_returns.npy'), episodic_returns_p1)
    np.save(os.path.join(agent_e0.model_dir, 'ep_returns.npy'), episodic_returns_e0)
