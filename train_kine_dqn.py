#!/usr/bin/python3
"""
Training a single pursuers behavior using multiple instances
"""
import sys
import os
import time
from datetime import datetime
import numpy as np
from numpy import random
import matplotlib
import matplotlib.pyplot as plt

import utils
from envs.pe_kine_env import PEKineEnv
from agents.dqn import DQNAgent


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=1)
    agent_p = DQNAgent()
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_dir = sys.path[0]+"/saved_models/p1e1_kine/dqn/"+date_time+"/agent_p"
    # train parameters
    num_episodes = 2000
    num_steps = env.max_steps-1 # ensure done only when collision occured
    num_epochs = 1
    episodic_returns = []
    sedimentary_returns = []
    ep = 0
    step_counter = 1
    t_start = time.time()
    # train
    while ep < num_episodes:
        done, total_reward = False, []
        state, _ = env.reset()
        # evader_speed = random.choice([-1,1])
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(3*num_episodes/5))
        for st in range(num_steps):
            # action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            # action_evaders = utils.cirluar_action(state[-2:],speed=evader_speed)
            action_evaders = np.zeros(2)
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            next_state, rewards, done, info = env.step(action_evaders, action_pursuers)
            if not info:
                rew = rewards[0]
            else:
                rew = -10
            # store transitions
            agent_p.replay_memory.store([state, ia, rew, done, next_state])
            # train K epochs
            for i in range(num_epochs):
                agent_p.train()
            if not step_counter % agent_p.update_step:
                agent_p.qnet_stable.set_weights(agent_p.qnet_active.get_weights())
            # step summary
            print("\n-\nepisode: {}, step: {}, epsilon: {} \nstate: {} \naction: {}->{} \nnext_state: {} \nreward: {} \ninfo: {} \n-\n".format(ep+1, st+1, agent_p.epsilon, state, ia, action_pursuers, next_state, rew, info))
            # render, comment out following line to maximize training speed
            # env.render(pause=1./env.rate)
            total_reward.append(rew)
            step_counter += 1
            if done:
                break
        # save model
        if not (ep+1) % 1000:
            agent_p.save_model(model_dir)
        # summarize episode
        episodic_returns.append(sum(total_reward))
        sed_return = sum(episodic_returns)/(ep+1)
        sedimentary_returns.append(sed_return)
        ep += 1
        print("\n---\nepisode: {}, episodic_return: {} \n---\n".format(ep+1, sum(total_reward)))
    agent_p.save_model(model_dir)
    t_end = time.time()
    print("Training duration: {}".format(time.strftime("%H:%M:%S", time.gmtime(t_end-t_start))))

    # save rewards
    np.save(os.path.join(os.path.dirname(model_dir), 'ep_returns.npy'), episodic_returns)
    # plot ave_returns
    fig, ax = plt.subplots()
    plt.plot(np.arange(len(sedimentary_returns))+1, sedimentary_returns)
    ax.grid()
    ax.set(xlabel='Episode', ylabel='Accumulated returns')
    plt.savefig(os.path.join(os.path.dirname(model_dir), 'ave_returns.png'))
