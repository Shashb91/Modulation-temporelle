from dataclasses import dataclass

c = 2800
rho = 1200
fmax = 200

xc = (x0, xf) = (0, 400)
tc = (t0, tf) = (0, 0.125)

M = 400
dx = xf / M
dt = 0.95 * dx / c
N = int(tf/dt)

