"""
============================================================
Auteur : Shashankan BALASSOUPRAMANIANE
Date : 01/06/2026
Implementation de Lax-Wendroff et d'ADER4 1D/2D en mileu
anisotrope

Les modules concernant la modulation anisotrope sont indicés
par _aniso

Parametres :
- CFL = 0.6
- e = 9.4e9
============================================================
"""

data = Donnee2D(label = "LW2D_aniso", )