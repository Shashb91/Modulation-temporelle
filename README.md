# Modélisation numérique de la propagation d'ondes en milieux 2D modulés en temps

**Auteur** : Shashankan Balassoupramaniane (Centrale Méditerranée)
**Encadrements** : Bruno Lombard (LMA), Michaël Darche (LMA), Marie Touboul (ENSTA)

Ce dépôt rassemble les outils Python développés lors d'un stage de recherche au **Laboratoire de Mécanique et d'Acoustique (LMA)** (Centrale Méditerranée). Il est dédié à la simulation de la propagation d'ondes acoustiques 1D et 2D à l'aide de schémas aux différences finies d'ordre élevé.

---

## Contexte scientifique

La propagation d'onde dans un milieu non dissipatif est décrite par le système hyperbolique du premier ordre :

$$\displaystyle\frac{\partial \mathbf u}{\partial t} + \mathbf A \frac{\partial \mathbf u}{\partial x} + \mathbf B \frac{\partial \mathbf u}{\partial y} = \mathbf F$$

Le projet explore quatre régimes physiques de complexité croissante :
1. **Homogène isotrope :** Paramètres mécaniques ($\rho$, $\kappa$) uniformes et constants.
2. **Modulé en temps :** Propriétés dépendant explicitement du temps ($\rho(t)$, $\kappa(t)$).
3. **Stratifié :** Milieu bicouche modélisé soit par **homogénéisation anisotrope**, soit par traitement direct des **interfaces** (micro-structuré).
4. **Stratifié & modulé :** Couplage inédit combinant variabilités spatiale et temporelle.

---

## Schémas numériques

La résolution repose sur deux familles de schémas explicites :
* **Lax–Wendroff** (ordre 2 en espace et en temps) ;
* **ADER4** (*Arbitrary DERivatives* ordre 4).

Deux configurations d'émission sont proposées : par **terme source ponctuel** ou par **condition initiale de Cauchy** (front d'onde formé). Les solutions sont validées par comparaison avec des solutions analytiques et par analyse d'erreur en norme $\mathbf L^1$.

---

## Architecture du projet

L'ensemble des paramètres, de la grille, de la solution $\mathbf u$ et du bilan d'énergie $E$ est encapsulé dans une classe centrale (`Donnee1D` / `Donnee2D`). Le code s'articule autour des modules suivants :

* **Schémas de calcul :** `schema.py` (homogène), `schema_mt.py` (modulations temporelles via splitting de Strang), `schema_aniso.py` (milieux stratifiés).
* **Physique & Source :** `modulation.py` (profils temporels et leurs dérivées), `source.py` (ondelettes).
* **Analyse & Rendu :** `analytique.py`, `erreur.py` (convergence $L^1$), `tracer.py` (animations et profils), `sauvegarde.py` (sérialisation `pickle`).
* **Scripts de pilotage :** `main.py`, `main_mt.py`, `main_aniso.py`.

---

## Utilisation rapide

```python
from donnee import Donnee1D
from schema import ADER41D
from tracer import anim1D
from sauvegarde import sauvegarder

# Initialisation et résolution
data = Donnee1D(c = 1500, kappa = 2.25e9, rho = 1000, M=150, CFL=0.6, xc=(0, 300), tc=(0, 0.2), f=20)
ADER41D(data)

# Visualisation de l'onde
anim1D(data, interval=30)
sauvegarder(data)
