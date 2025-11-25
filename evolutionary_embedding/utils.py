import os
import numpy as np
from matplotlib import pyplot as plt

def cart2pol(Y:np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to polar coordinates.
    """
    x           = Y[..., 0]
    y           = Y[..., 1]
    rho         = np.sqrt(x ** 2 + y ** 2)
    phi         = np.arctan2(y, x) % (2 * np.pi)
    Y[..., 0]   = rho
    Y[..., 1]   = phi
    return Y


def pol2cart(Y:np.ndarray) ->  np.ndarray:
    """
    Convert polar coordinates to Cartesian coordinates.
    """
    rho         = Y[..., 0]
    phi         = Y[..., 1]
    x           = rho * np.cos(phi)
    y           = rho * np.sin(phi)
    Y[..., 0]   = x
    Y[..., 1]   = y
    return Y

def log_interim(Y, labels=None, polar=True, stretch_offset=35, out_dir=None, iter=iter):
    """
    Log evolutionary embedding plots
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig         = plt.figure(figsize=(15, 15))
    ax          = fig.add_subplot(projection='polar')
    num_tsteps  = Y.shape[0]

    for t in range(num_tsteps):
        ax.scatter(Y[t, :, 1], Y[t, :, 0] + stretch_offset * t, c=labels[t], s=10, alpha=0.5)  # stretch out the donuts for plotting

    plt.savefig(os.path.join(out_dir, str(iter) + '.png'))
    plt.close()

def gen_synthetic_data(means, stds, size=100):
    """
    Generate synthetic data. Means (t steps, n gaussians)
    """

    X                               = None
    means                           = np.array(means)
    stds                            = np.array(stds)

    num_tsteps                      = len(means[:, 0])
    num_clusters                    = len(means[0, :])
    for i in range((num_tsteps)):
        dist                        = None
        for j in range((num_clusters)):
            if dist is None:
                dist                = np.random.normal(loc=[means[i,j],means[i,j],means[i,j]], scale=[stds[i,j],stds[i,j],stds[i,j]], size=(size,3))
            else:
                dist                = np.concatenate((dist, np.random.normal(loc=[means[i,j],means[i,j],means[i,j]], scale=[stds[i,j],stds[i,j],stds[i,j]], size=(size,3))), axis=0)

        dist                        = dist[np.newaxis,...]
        if X is None:
            X                       = dist
        else:
            X                       = np.concatenate((X,dist), axis=0)

    labels                          = None

    for i in range(num_clusters):
        if labels is None:
            labels                  = np.full((size), i)
        else:
            labels                  = np.concatenate((labels, np.full((size), i)), axis=0)

    labels                          = labels[np.newaxis, ...]
    labels                          = labels.repeat(num_tsteps, axis=0)
    return X, labels
