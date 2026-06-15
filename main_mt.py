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


f_mt = 5
data1 = Donnee1D(M = 1000, label = "ADER4_mt f=" + str(f_mt) + " Hz", e = 9.4e9, eps = 0, omega = f_mt*2*np.pi,
                 tc = (0, 0.175), xc = (0,550), CFL = 0.5, f = 10, rho_mt=rho_echelon, E_mt=E_echelon)
U = ADER41D_mt(data1, alpha = 0.2)
#data1 = charger('.save/ADER4_mt f=20 Hz_06-15_16-27-44.pkl')
anim1D(data1, interval = 1)
sauvegarder(data1)