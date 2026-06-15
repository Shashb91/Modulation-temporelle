"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff et d'ADER4 1D/2D en mileu
modulé en temps

Les modules concernant la modulation temporelle sont indicés
par _mt

Parametres :
- CFL = 0.6
- e = 9.4e9
============================================================
"""

from schema_mt import *
from tracer import *
from sauvegarde import *


f_mt = 100
data1 = Donnee1D(M = 400, label = "ADER4_mt f=" + str(f_mt) + " Hz", e = 9.4e9, eps = 0.5, omega = f_mt*2*np.pi,
                 tc = (0, 0.155), xc = (0,600), CFL = 0.25, f = 20)
U = ADER41D_mt(data1)

anim1D(data1)
sauvegarder(data1)