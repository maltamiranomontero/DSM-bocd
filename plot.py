import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm

import numpy as np

def plot_posterior(T, data, R, cps=None):
    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1, ax2 = axes

    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    
    ax2.set_xlim([0, T])
    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1), extent=[0,T,0,T])
    ax2.plot(np.argmax(R,axis=1), c='r')

    if cps:
        for cp in cps:
            ax1.axvline(cp, c='red', ls='dotted')
            ax2.axvline(cp, c='red', ls='dotted')

    plt.tight_layout()
    plt.show()