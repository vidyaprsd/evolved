"""
Evolutionary Embeddings
"""
import random
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from utils import cart2pol, pol2cart, gen_synthetic_data, log_interim
from tsne_funcs import x2p, pca

def alignment_dist_grad(phi1: np.ndarray, phi2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute angular similarity and gradient between two angle arrays.
    """
    val     = phi1 - phi2
    sim     = np.abs(np.cos(val))
    grad    = np.where(sim > 0, -np.sin(val), np.sin(val))
    return -sim, grad


def displacement_dist_grad(rho: np.ndarray, radius: np.ndarray, sigma: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gaussian distance to a target radius and its gradient.
    """
    exp     = np.exp(-(rho - radius) ** 2 / (2 * sigma ** 2))
    value   = exp / (sigma * np.sqrt(2 * np.pi))
    grad    = -((rho - radius) / sigma ** 2) * value
    return -value, grad

def semantic_dist_grad(Y, P_values):
    """
    Compute t-SNE loss and gradient. Source: L. van der Maaten (2008)
    """
    n                       = Y.shape[0]
    no_dims                 = Y.shape[-1]
    sum_Y                   = np.sum(np.square(Y), 1)
    num                     = -2. * np.dot(Y, Y.T)
    num                     = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))

    num[range(n), range(n)] = 0.
    Q                       = num / np.sum(num)
    Q                       = np.maximum(Q, 1e-12)

    # Compute gradient
    PQ                      = P_values - Q
    sem_grad                = np.zeros((n, no_dims))

    for i in range(n):
        # '''
        # tsne loss (KL Divergence) gradient
        sem_grad[i, :]      = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
    sem_dist                = np.sum(P_values * np.log(P_values / Q))

    return sem_dist, sem_grad

def tsne(X=np.array([]),
         no_dims=2,
         initial_dims=50,
         perplexity=30.0,
         disp_wt = 1,
         align_wt = 0.05,
         exp_name='test',
         labels = None,
         log_iter = 100,):


    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    initial_momentum                = 0.5
    final_momentum                  = 0.8
    min_gain                        = 0.01
    eta                             = 500
    max_iter                        = 2000

    init_disp_sigma                 = 20 #initial Gaussian sigma for point movements around ring during opt
    final_disp_sigma                = 8 #final Gaussian sigma

    radius                          = 20 #use same dummy radius offset for optimization; stretched out later for vis
    radii_buffer                    = 20 #point initialization band around desired radius

    out_dir                         = f"{exp_name}_{int(time.time())}"

    num_tsteps, n, _                = X.shape
    radii                           = [radius] * num_tsteps
    radii[0]                        = 0

    # Initialization
    X_red                           = np.zeros((num_tsteps, n, initial_dims))
    Y                               = np.zeros((num_tsteps, n, no_dims))
    P                               = np.zeros((num_tsteps, n, n))

    sem_updt                        = np.zeros((num_tsteps, n, no_dims))
    sem_grad                        = np.zeros((num_tsteps, n, no_dims))
    disp_updt                       = np.zeros((num_tsteps, n))
    align_updt                      = np.zeros((num_tsteps, n))
    gains                           = np.ones((n, no_dims))

    for t in range(num_tsteps):
        # Initialize low-dim points around rings
        radius_t                    = radii[t]
        rho_points_i                = np.random.uniform(max(0, radius_t - radii_buffer), radius_t + radii_buffer, n)[:, None]
        phi_points_i                = np.random.uniform(0, 2 * np.pi, n)[:, None]
        Y[t]                        = np.concatenate((rho_points_i, phi_points_i), axis=1)

        # PCA + compute P-values
        X_red[t]                    = pca(X[t], initial_dims).real if initial_dims < X.shape[-1] else X[t]
        P[t]                        = x2p(X_red[t], tol=1e-5, perplexity=perplexity)
        P[t]                        = (P[t] + P[t].T)
        P[t]                       /= np.sum(P[t])
        P[t]                       *= 4  # early exaggeration
        P[t]                        = np.maximum(P[t], 1e-12)


    # Main optimization loop
    for iter in range(max_iter):
        semantic_loss               = 0
        displacement_loss           = 0
        alignment_loss              = 0

        for t in range(num_tsteps):
            # Convert o Cartesian for t-SNE Euclidean distances
            Y                       = pol2cart(Y)

            # -----------------------
            # Semantic (t-SNE) Loss
            # -----------------------

            # Compute KL-divergence and gradient for current timestep
            sem_dist, sem_grad      = semantic_dist_grad(Y[t], P[t])

            momentum                = initial_momentum if iter < 20 else final_momentum
            gains                   = (gains + 0.2) * ((sem_grad > 0) != (sem_updt[t] > 0)) + (gains * 0.8) * ((sem_grad > 0) == (sem_updt[t] > 0))
            gains[gains < min_gain] = min_gain

            # Compute semantic update
            sem_updt[t]             = momentum * sem_updt[t] - eta * (gains * sem_grad)

            # Accumulate semantic loss
            semantic_loss          += sem_dist

            # -----------------------
            # Displacement Loss
            # -----------------------
            Y = cart2pol(Y)
            # Adaptive sigma schedule for controlling spread
            disp_sigma = (
                init_disp_sigma if iter < 0.5 * max_iter else
                0.75 * init_disp_sigma if iter < 0.625 * max_iter else
                0.5 * init_disp_sigma if iter < 0.75 * max_iter else
                final_disp_sigma
            )

            # Compute Gaussian distance and gradient relative to target radius
            disp_dist, disp_grad    = displacement_dist_grad(Y[t, :, 0], np.full(Y[t, :, 0].shape, radii[t]), sigma=disp_sigma)
            # Update displacement (rho) gradient
            disp_updt[t]            = disp_wt * eta * disp_grad
            # Accumulate displacement loss
            displacement_loss      += np.sum(disp_dist)

            # -----------------------
            # Alignment Loss
            # -----------------------

            # Skip last timestep for alignment (needs t+1)
            if t < (num_tsteps - 1):
                # Compute Cosine distance and gradient between pairs of corresp instances
                align_dist, align_grad  = alignment_dist_grad(Y[t, :, 1], Y[t + 1, :, 1])
                # Update alignment (phi) gradient
                align_updt[t]           = align_wt * align_grad
                # Accumulate alignment loss
                alignment_loss         += np.sum(align_dist)

        # -----------------------
        # Normalize losses for logging
        # -----------------------
        semantic_loss               = semantic_loss / (num_tsteps )
        displacement_loss           = displacement_loss / (num_tsteps * n)
        alignment_loss              = alignment_loss / (num_tsteps * n)

        # -----------------------
        # Apply semantic (t-SNE) update (Cartesian)
        # -----------------------
        Y                           = pol2cart(Y)
        Y                           = Y + sem_updt  # update based on tsne grad

        # -----------------------
        # Displacement & alignment updates (Polar)
        # -----------------------
        Y                           = cart2pol(Y)
        Y[..., 0]                  += disp_updt
        Y[..., 1]                  += align_updt
        Y[..., 1]                   = Y[..., 1] % (2 * np.pi)

        if log_iter > 0 and (iter + 1) % log_iter == 0:
            log_interim(Y, labels=labels, polar=True, out_dir=out_dir, iter=iter)
            print("Iteration %d: C_s %f C_d %f C_a %f" % (iter + 1, semantic_loss, displacement_loss, alignment_loss))

        # Stop lying about P-values
        if iter == 100:
            P                       = P / 4.

    return Y

if __name__ == "__main__":

    name                            = 'test'
    random.seed(10)
    np.random.seed(10)

    X, labels                       = gen_synthetic_data([[1,1,1],[1,3,3.5],[1,2.5,4]], [[0.5,0.5,0.5],[0.2,0.4,0.35],[0.1,0.1,0.1]])
    Y                               = tsne(X, no_dims = 2, initial_dims=3, perplexity = 20.0, exp_name = name, labels=labels)