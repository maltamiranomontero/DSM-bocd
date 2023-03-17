import numpy as np


def generate_data(varx, mean0, var0, T, num_cp):
    """Generate partitioned data of T observations according to number
    of changepoints `num_cp` with hyperpriors `mean0` and `var0`.
    """
    data = []
    cps = []
    meanx = mean0
    cps = np.random.randint(T, size=num_cp)
    for t in range(0, T):
        if t in cps:
            meanx = np.random.normal(mean0, var0)
        data.append(np.random.normal(meanx, varx))
    return data, cps
