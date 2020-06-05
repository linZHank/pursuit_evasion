#!/usr/bin/python3
"""
Pursuit-evasion environment:
    - Randomly placed obstacles
    - Multiple Obstacles (>=0)
    - Multiple Pursuers (>=1)
    - Multiple Evaders (>=1)
    - Homogeneous agents
Note:
    - Discrete action space
"""
#import numpy as np
#from numpy import pi
#from numpy import random
#import time
#import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse, RegularPolygon, Circle
#from matplotlib.collections import PatchCollection
from pe_env import PursuitEvasion

class PursuitEvasionDiscrete(PursuitEvasion):
    """
    Dynamics Pursuit-evasion env: N pursuers, N evader (1<=N<=4)
    """
    def __init__(self, resolution=(100, 100)):
        super().__init__(resolution)

