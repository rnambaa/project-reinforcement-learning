import numpy as np


def buy5(hour: int):
    if hour == 5:
        return 1
    else:
        return 0


def buy5sell11(hour: int):
    if hour == 5:
        return 1
    elif hour == 11:
        return -1
    else:
        return 0


def random_uniform():
    """selects continuous random action between -1 and 1"""
    return np.random.uniform(-1, 1)


def random_normal():
    """selects continuous random action between -1 and 1 from normal distribution"""
    return np.random.normal(0, 1)
