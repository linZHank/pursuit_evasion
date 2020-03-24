#!/usr/bin/python3
"""
Just a test script before full work process
"""
import sys
import os
import numpy as np
from numpy import random

from envs.pe_kine_env import PEKineEnv
from agents.dqn import DQNAgent
from agents.agent_utils import dqn_utils


if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=1)
    agent_p = DQNAgent()
    num_episodes = 10
    num_steps = 400
    num_epochs = 1
    step_counter = 1
    for ep in range(num_episodes):
        # specify evader's spawining position
        theta_e = random.uniform(-pi,pi)
        env.evaders_spawning_pool = np.array([[3*np.cos(theta_e),3*np.sin(theta_e)]])
        # reset env and get state from it
        state, _ = env.reset()
        evader_speed = random.choice([-1,1])
        # agent_p.linear_epsilon_decay(episode=ep, decay_period=int(3*num_episodes/5))
        for st in range(num_steps):
            action_evaders = dqn_utils.circular_action(state[-2:],speed=evader_speed)
            # action_evaders = np.zeros(2)
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            next_state, rewards, done, info = env.step(action_evaders, action_pursuers)
            rew, done, success = dqn_utils.adjust_reward(env, num_steps, next_state, rewards[0], done, info)
            env.render(pause=1./env.rate)
            state = next_state
            # store transitions
            agent_p.replay_memory.store([state, ia, rew, done, next_state])
            # train K epochs
            for _ in range(num_epochs):
                agent_p.train()
            if not step_counter % agent_p.update_step:
                agent_p.qnet_stable.set_weights(agent_p.qnet_active.get_weights())
            # log step
            print("\n-\nepisode: {}, step: {}, epsilon: {} \nstate: {} \naction: {}->{} \nnext_state: {} \nreward: {}".format(ep+1, st+1, agent_p.epsilon, state, ia, action_pursuers, next_state, rew))
            print(info)
            step_counter += 1
            if done:
                if success:
                    success_counter += 1
                break
