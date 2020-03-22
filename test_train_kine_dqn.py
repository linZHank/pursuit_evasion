#!/usr/bin/python3
"""
Just a test script before full work process
"""
import numpy as np
from numpy import random

from envs.pe_kine_env import PEKineEnv
from agents.dqn import DQNAgent
from agents.agent_utils import dqn_utils

if __name__ == '__main__':
    env=PEKineEnv(num_pursuers=1)
    agent_p = DQNAgent()
    num_episodes = 20
    num_steps = 20
    num_epochs = 1
    step_counter = 1
    for ep in range(num_episodes):
        state, _ = env.reset()
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(3*num_episodes/5))
        for st in range(num_steps):
            action_evaders = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            ia, action_pursuers = agent_p.epsilon_greedy(state)
            next_state, rew, done, _ = env.step(action_evaders, action_pursuers)
            env.render(pause=1./env.rate)
            state = next_state
            # store transitions
            agent_p.replay_memory.store([state, ia, rew[0], done, next_state])
            # train K epochs
            for _ in range(num_epochs):
                agent_p.train()
            if not step_counter % agent_p.update_step:
                agent_p.qnet_stable.set_weights(agent_p.qnet_active.get_weights())
            # log step
            print("episode: {}, step: {}, epsilon: {} \nstate: {} \naction: {}->{} \nnext_state: {} \nreward: {}".format(ep+1, st+1, agent_p.epsilon, state, ia, action_pursuers, next_state, rew))
            env.render()
            step_counter += 1
            if done:
                break
