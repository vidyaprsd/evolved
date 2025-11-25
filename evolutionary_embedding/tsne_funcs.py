import numpy as np

def Hbeta(D=np.array([]), beta=1.0):
    """
        Source: L. van der Maaten (2008)
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P       = np.exp(-D.copy() * beta)
    sumP    = sum(P)
    H       = np.log(sumP) + beta * np.sum(D * P) / sumP
    P       = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Source: L. van der Maaten (2008)
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d)  = X.shape
    sum_X   = np.sum(np.square(X), 1)
    D       = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P       = np.zeros((n, n))
    beta    = np.ones((n, 1))
    logU    = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin             = -np.inf
        betamax             = np.inf
        Di                  = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff               = H - logU
        tries               = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin     = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax     = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP)      = Hbeta(Di, beta[i])
            Hdiff           = H - logU
            tries           += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def pca(X=np.array([]), no_dims=50):
    """
        Source: L. van der Maaten (2008)
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d)                  = X.shape
    X                       = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M)                  = np.linalg.eig(np.dot(X.T, X))
    Y                       = np.dot(X, M[:, 0:no_dims])
    return Y