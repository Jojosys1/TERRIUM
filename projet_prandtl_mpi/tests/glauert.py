#!/usr/bin/env python3
"""Solution de reference de Glauert (series de Fourier) pour la ligne portante, aile trapezoidale.
Usage : glauert.py envergure croot ctip incidence_deg vrillage_deg alpha_l0_deg
Affiche : CL CDi lambda      (a0 = 2 pi)"""
import sys
import numpy as np

b, cr, ct, alpha, twist, al0 = map(float, sys.argv[1:7])
a0, K = 2*np.pi, 80
theta = np.arange(1, K+1)*np.pi/(2*K)            # points de collocation
n = np.arange(1, 2*K, 2)                         # harmoniques impaires (charge symetrique)
eta = np.abs(np.cos(theta))                      # |y| / (b/2), avec y = -(b/2) cos(theta)
c = cr + (ct-cr)*eta
rhs = np.deg2rad(alpha + twist*eta - al0)
M = np.sin(np.outer(theta, n))*(4*b/(a0*c)[:, None] + n[None, :]/np.sin(theta)[:, None])
A = np.linalg.solve(M, rhs)
S = 0.5*b*(cr+ct)
AR = b*b/S
print(f"{np.pi*AR*A[0]:.8f} {np.pi*AR*np.sum(n*A**2):.8f} {AR:.6f}")
