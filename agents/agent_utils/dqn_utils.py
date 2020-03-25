import numpy as np
from numpy import random


def obs_to_state(obs):
    """
    Convert env's raw observation into agent's state input
    Args:
        obs: dict(evaders, pursuers)
    Returns:
        state: array([x_p, y_p, x_e, y_e])
    """
    state = np.concatenate((obs['pursuers']['position'][0],obs['evaders']['position'][0]), axis=0)

    return state

def circular_action(pos, speed):
    """
    Generate action to enable agent moving in circle regarding to (0,0)
    Args:
        pos: array([x,y])
        speed: tangential velocity, +: counter-clockwise
    Returns:
        action: array([v_x,v_y])
    """
    rot = np.array([[0,-1], [1,0]]) # rotation matrix
    vec_vel = np.dot(rot, pos)
    norm_vec_vel = vec_vel/np.linalg.norm(vec_vel)
    action = speed*norm_vec_vel.reshape(1,-1)

    return action

def adjust_reward(env, num_steps, state, reward, done):
    """
    Args:
        env: env object
        state: array([x_p,y_p,x_e,y_e])
    Returns:
        reward: scalar
        done: boolean
    """
    success = False
    if env.pursuers['status'][0] == 'occluded' or env.pursuers['status'][0] == 'out':
        reward = -1
        done = True
    elif env.pursuers['status'][0] == 'catching':
        reward = env.world_length
        done = True
        success = True
    else:
        reward = -1./num_steps

    return reward, done, success
