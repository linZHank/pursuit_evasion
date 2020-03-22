import numpy as np

def cirluar_action(pos, speed):
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
    action = speed * norm_vec_vel

    return action
