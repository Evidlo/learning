#!/usr/bin/env python
## Evan Widloski - 2020-02-08
## Simple Principle Component Analysis Exercise

import numpy as np
import matplotlib.pyplot as plt

mean = (0, 0)
covariance = ((3**2, 0), (0, 1**2))
N = 1000
x = np.random.multivariate_normal(mean, covariance, N)

# plt.scatter(x[:, 0], x[:, 1])
# plt.axis('equal')
# plt.show()

sample_mean = x.mean(axis=0)
residual = x - sample_mean
sample_cov = residual.T @ residual / (N - 1)

def dumb_pca(sample_cov):
    # (x_1n - x_bar)
    vectors = sample_cov / sample_cov.diagonal()
    return vectors[np.flip(sample_cov.diagonal().argsort())]
