import numpy as np


def find_cp(R, lag=20):
    n = len(R)
    CPs = [0]
    last_CP = 0
    for i in range(n):
        candidate = i-np.argmax(R[i, :])
        if candidate > last_CP+lag:
            if (candidate not in CPs):
                CPs.append(candidate)
                last_CP = np.max(CPs)
        if candidate < last_CP:
            try:
                CPs.remove(last_CP)
                last_CP = np.max(CPs)
            except Exception:
                pass
    return CPs
