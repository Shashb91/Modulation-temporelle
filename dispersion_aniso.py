import numpy as np

rho = (1000, 1000)
c = (500, 1000)
E = (rho[0]*c[0]**2, rho[1]*c[1]**2)
eps = 0.3
alpha = 0.25
f_mt = 30
T = 1/f_mt

rho = (alpha*rho[0] + (1-alpha)*rho[1], rho[0]*rho[1]/(alpha*rho[1]+(1-alpha)*rho[0]))
E = alpha*E[0] + (1-alpha)*E[1]

rho = np.array([[rho[0]*(1-eps), rho[0]*(1+eps)], [rho[1]*(1-eps), rho[1]*(1+eps)]])
E = np.array([E*(1-eps), E*(1+eps)])

omega = np.linspace(-np.pi*f_mt, np.pi*f_mt, 100)