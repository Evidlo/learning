#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# ----- generate -----
# %% generate

# generate coefficients for random cubic
# c_3, c_2, c_1, c_0 = np.random.randint(-5, 5, size=4)
c_3, c_2, c_1, c_0 = 0, 2, -1, -7
f_true = lambda x: c_3 * x**3 + c_2 * x**2 + c_1 * x + c_0

# generate random data points
x = np.random.random(10)
y = f_true(x) + np.random.normal(scale=0.1, size=len(x))

# ----- solve -----
# %% solve
#
#   Y              A           C
#
# | y₁ |    | x₁³ x₁² x₁ 1 | | c₃ |
# | .. | =  |      ..      | | c₂ |
# | yₙ |    | xₙ³ xₙ² xₙ 1 | | c₁ |
#                            | c₀ |
#
# C = (A'A)⁻¹A'Y

A = x[:, None] ** np.repeat([[3, 2, 1, 0]], len(x), axis=0)
C = np.linalg.inv(A.T @ A) @ A.T @ y
f = lambda v: C[3] * v**3 + C[2] * v**2 + C[1] ** v + C[0]

# ----- plot -----
# %% plot

plt.plot(x, y, 'ro')
# plot curve
t = np.linspace(0, 1, 100)
plt.plot(t, f_true(t), 'b')
plt.plot(t, f(t), 'g')
plt.legend(['points', 'true', 'found'])
plt.show()
