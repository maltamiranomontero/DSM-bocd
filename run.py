import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm

import numpy as np

from bocpd import bocpd
from hazard import ConstantHazard
from distribution import GaussianUnknownMean
from generate_data import generate_data

def plot_posterior(T, data, cps, R):
    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)

    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted')
        #ax2.axvline(cp, c='red', ls='dotted')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    T      = 1000   # Number of observations.
    hazard = ConstantHazard(100)
    mean0  = 0      # The prior mean on the mean parameter.
    var0   = 2      # The prior variance for mean parameter.
    varx   = 1      # The known variance of the data.

    data, cps      = generate_data(varx, mean0, var0, T, hazard)
    model          = GaussianUnknownMean(mean0, var0, varx)
    R  = bocpd(data, hazard, model)

    plot_posterior(T, data, cps, R)
