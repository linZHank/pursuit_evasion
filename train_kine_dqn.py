#!/usr/bin/python3
"""
Training a single pursuers behavior using multiple instances
"""
import sys
import os
import time
from datetime import datetime
import numpy as np
from numpy import pi
from numpy import random
import matplotlib
import matplotlib.pyplot as plt

from envs.pe_kine_env import PEKineEnv
from agents.dqn import DQNAgent
from agents.agent_utils import dqn_utils


if __name__ == '__main__':
    env=PEKineEnv()
    agent_p = DQNAgent()
    agent_p.warmup_episodes = 0
    agent_p.update_step = 2000
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_dir = sys.path[0]+"/saved_models/p1e1_kine/dqn/"+date_time+"/agent_p"
    # train parameters
    num_episodes = 12000
    num_steps = 200
    num_epochs = 1
    episodic_returns = []
    sedimentary_returns = []
    ep = 0
    step_counter = 1
    success_counter = 0
    t_start = time.time()
    # train
    while ep < num_episodes:
        # specify evader's spawining position
        theta_e = random.uniform(-pi,pi)
        env.evaders_spawning_pool[0] = np.array([3*np.cos(theta_e),3*np.sin(theta_e)])
        # evader_speed = random.uniform(-pi/2,pi/2)
        # reset env
        done, total_reward = False, []
        state, _ = env.reset()
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(2*num_episodes/5))
        for st in range(num_steps):
            # action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            # action_evaders = dqn_utils.cirluar_action(state[-4:-2],speed=evader_speed)
            action_evaders = np.zeros((1,2))
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            next_state, reward, done, info = env.step(action_evaders, action_pursuers)
            rew, done, success = dqn_utils.adjust_reward(env, num_steps, next_state, reward, done, info)
            # store transitions
            agent_p.replay_memory.store([state, ia, rew, done, next_state])
            # train K epochs
            if ep >= agent_p.warmup_episodes:
                for i in range(num_epochs):
                    agent_p.train()
            if not step_counter % agent_p.update_step:
                agent_p.qnet_stable.set_weights(agent_p.qnet_active.get_weights())
            # step summary
            print("\n-\nepisode: {}, step: {}, epsilon: {} \nstate: {} \naction: {}->{} \nnext_state: {} \nreward: {} \ninfo: {}, succeeded: {}\n-\n".format(ep+1, st+1, agent_p.epsilon, state, ia, action_pursuers, next_state, rew, info, success_counter))
            # render, comment out following line to maximize training speed
            # env.render(pause=1./env.rate)
            total_reward.append(rew)
            step_counter += 1
            state = next_state
            if done:
                if success:
                    success_counter += 1
                break
        # save model
        if not (ep+1) % 1000:
            agent_p.save_model(model_dir)
        # summarize episode
        episodic_returns.append(sum(total_reward))
        sed_return = sum(episodic_returns)/(ep+1)
        sedimentary_returns.append(sed_return)
        ep += 1
        print("\n---\nepisode: {}, episodic_return: {}\n---\n".format(ep+1, sum(total_reward)))
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
