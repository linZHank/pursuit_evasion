#!/usr/bin/python3
"""
Just a test script before full work process
"""
import sys
import os
import numpy as np
from numpy import random

import utils
from envs.pe_kine_env import PEKineEnv
from agents.dqn import DQNAgent

if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=1)
    agent_p = DQNAgent()
    num_episodes = 10
    num_steps = 100
    num_epochs = 1
    step_counter = 1
    for ep in range(num_episodes):
        state, _ = env.reset()
        # evader_speed = random.choice([-1,1])
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(3*num_episodes/5))
        for st in range(num_steps):
            # action_evaders = utils.cirluar_action(state[-2:],speed=evader_speed)
            action_evaders = np.zeros(2)
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            next_state, rewards, done, info = env.step(action_evaders, action_pursuers)
            if not info:
                rew = rewards[0]
            else:
                rew = -10./num_steps
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
            print("episode: {}, step: {}, epsilon: {} \nstate: {} \naction: {}->{} \nnext_state: {} \nreward: {}".format(ep+1, st+1, agent_p.epsilon, state, ia, action_pursuers, next_state, rew))
            print(info)
            step_counter += 1
            if done:
                break
