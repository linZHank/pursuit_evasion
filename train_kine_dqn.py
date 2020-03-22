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
from agents.agent_utils import dqn_utils
# from numpy import random


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=1)
    agent_p = DQNAgent()
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_dir = sys.path[0]+"/saved_models/p1e1_kine/dqn/"+date_time+"/agent_p"
    # train parameters
    num_episodes = 1000
    num_steps = 100 # env.max_steps
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
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(3*num_episodes/5))
        for st in range(num_steps):
            # action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            action_evaders = np.zeros(2)
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            next_state, rew, done, _ = env.step(action_evaders, action_pursuers)
            # store transitions
            agent_p.replay_memory.store([state, ia, rew[0], done, next_state])
            # train K epochs
            for i in range(num_epochs):
                agent_p.train()
            if not step_counter % agent_p.update_step:
                agent_p.qnet_stable.set_weights(agent_p.qnet_active.get_weights())
            # step summary
            print("episode: {}, step: {}, epsilon: {} \nstate: {} \naction: {}->{} \nnext_state: {} \nreward: {}".format(ep+1, st+1, agent_p.epsilon, state, ia, action_pursuers, next_state, rew[0]))
            # render, comment out following line to maximize training speed
            # env.render(pause=1./env.rate)
            total_reward.append(rew[0])
            step_counter += 1
            if done:
                break
        # save model
        if not (ep+1) % 100:
            agent_p.save_model(model_dir)
        # summarize episode
        episodic_returns.append(sum(total_reward))
        sed_return = (sum(total_reward)+sum(episodic_returns))/(ep+1)
        sedimentary_returns.append(sed_return)
        ep += 1
        print("\n---\nepisode: {}, episodic_return: {}\n---\n".format(ep+1, total_reward))
    agent_p.save_model(model_dir)
    t_end = time.time()
    print("Training duration: {}".format(time.strftime("%H:%M:%S", time.gmtime(t_end-t_start))))

    # save rewards
    # plot ave_returns
    fig, ax = plt.subplots()
    plt.plot(np.arange(len(sedimentary_returns))+1, sedimentary_returns)
    plt.show()
