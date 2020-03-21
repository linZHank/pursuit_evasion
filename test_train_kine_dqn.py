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
    agent_p = DQNAgent(env=env)
    num_episodes = 20
    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = dqn_utils.obs_to_state(obs)
        agent_p.linear_epsilon_decay(episode=ep, decay_period=int(3*num_episodes/5))
        print("epsilon: {}".format(agent_p.epsilon))
        for _ in range(15):
            action_evader = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=2)
            action_pursuers = random.uniform(low=-env.world_length/4,high=env.world_length/4,size=(env.num_pursuers,2))
            obs, val, done, _ = env.step(action_evader,action_pursuers)
            next_state = dqn_utils.obs_to_state(obs)
            print("next_state: {}, value: {}".format(next_state,val))
            env.render(pause=1./env.rate)
            state = next_state
