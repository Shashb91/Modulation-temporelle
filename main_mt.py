"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff et d'ADER4 1D/2D en mileu
modulé en temps
============================================================
"""

from schema_mt import *
from tracer import *

data = Donnee1D(M = 400, label = "ADER4_mt", e = 9.4e9, eps = 0, omega = 10000*2*np.pi, tc = (0, 0.1), xc = (0,600), CFL = 0.85, f = 20)
U = ADER41D_mt(data)

anim1D(data)