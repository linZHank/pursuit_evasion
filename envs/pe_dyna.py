#!/usr/bin/python3
"""
Pursuit-evasion environment:
    - Randomly placed obstacles
    - Multiple Obstacles (>=0)
    - Multiple Pursuers (>=1)
    - Multiple Evaders (>=1)
"""
import numpy as np
from numpy import pi
from numpy import random
import time
import matplotlib.pyplot as plt


class PEDyna(object):
    """
    Pursuit-evasion dynamics env: N pursuers, N evader, homogeneous
    """
    def __init__(self, num_evaders=1, num_pursuers=1):
        assert isinstance(num_evaders, int)
        assert isinstance(num_pursuers, int)
        assert num_evaders >= 1
        assert num_pursuers >= 1
        # specs
