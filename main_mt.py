"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff et d'ADER4 1D/2D en mileu
modulé en temps

Les modules concernant la modulation temporelle sont indicés
par _mt
============================================================
"""

from schema_mt import *
from tracer import *
from sauvegarde import *

f_mt = 100
data = Donnee1D(M = 450, label = "ADER4_mt f=" + str(f_mt) + " Hz", e = 9.4e9, eps = 0.1, omega = f_mt*2*np.pi, tc = (0, 0.2), xc = (0,600), CFL = 0.6, f = 20)
U = ADER41D_mt(data)

nom_fichier = data.label + "_w-" + str(data.omega)
anim1D(data)
sauvegarder(data)