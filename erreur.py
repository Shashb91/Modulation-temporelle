from donnee import*
from LaxWendroff import*
import numpy as np

def erreur1D(t, dxc = (1e0,1e-7), Ndx = 10):
    dx_li = np.linspace(dxc[0], dxc[1], Ndx)
    eps = np.zeros_like((Ndx,2))
    for i in range(Ndx):
        data1, data2 = Donnee1D(), Donnee1D()
        data1.dx, data1.M = dx,data1.xc[1]/dx[i]
        data2.dx, data2.M = dx,data2.xc[1]/data2.dx
        U = LaxWendroff1D(data1)
        u = analytique1D(data2)
        eps[] = sum([U[t,i,:] - u[t,i,:] for i in range(data1.M)])/data1.M
