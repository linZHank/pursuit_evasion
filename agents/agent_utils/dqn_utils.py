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
