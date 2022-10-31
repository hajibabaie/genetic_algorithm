import numpy as np


def sphere(x):

    return np.sum(np.square(x), axis=1)[0]
