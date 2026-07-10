import pickle
import os
import numpy as np
from datetime import datetime
from donnee import Donnee2D, Donnee1D
chemin = ".save/"

def sauvegarder(instance, nom_fichier = "", opt = False):
    """
    Sauvegarde une instance uniquement de Donnee1D ou Donnee2D dans un fichier binaire (.pkl).
    :param instance: Donnee1D ou Donnee2D à sauvegarder.
    :param nom_fichier: Le nom ou le chemin du fichier de destination :
            - le fichier sera placé automatiquement dans un dossier .save/, ou un autre dossier, nommé comme '.nom_du_dossier/'
            - le fichier sera horodaté automatiquement si nom_fichier finit par un '_', sous le format MOIS-JOUR_HEURE-MINUTES-SECONDES
            - le fichier sera automatiquement placé sous l'extension '.pkl'
    :param opt: Bool, True si la sauvegarde s'effectue automatiquement, False sinon.
    """
    if nom_fichier == "" or nom_fichier.endswith('/'): nom_fichier += instance.label + "_"
    if not nom_fichier.startswith('.'): nom_fichier = chemin + nom_fichier
    if nom_fichier.endswith('_'): nom_fichier += datetime.now().strftime('%m-%d_%H-%M-%S')
    if not nom_fichier.endswith('.pkl'): nom_fichier += '.pkl'
    if not opt: opt = ((input("Voulez vous sauvegarder ? O/N : ")) == "O")
    if opt:
        try:
            with open(nom_fichier, 'wb') as fichier:
                pickle.dump(instance, fichier)
            print(f"Instance sauvegardée avec succès sous : '{nom_fichier}'")
        except Exception as e:
            print(f"Impossible de sauvegarder l'objet : {e}")
    else:
        print("Sauvegarde annulée")


def charger(nom_fichier: str) -> Donnee2D/Donnee1D :
    """
    Charge une instance de Donnee1D ou Donnee2D depuis un fichier binaire (.pkl).
    :param nom_fichier: Le nom ou le chemin du fichier à charger.
    :return: L'instance de la classe restaurée, ou None en cas d'erreur.
    """
    if not nom_fichier.startswith('.'): nom_fichier = chemin + nom_fichier
    if not nom_fichier.endswith('.pkl'): nom_fichier += '.pkl'

    if not os.path.exists(nom_fichier):
        print(f"Le fichier '{nom_fichier}' n'existe pas.")
        return None

    try:
        with open(nom_fichier, 'rb') as fichier:
            instance = pickle.load(fichier)
        print(f"Instance chargée avec succès depuis : '{nom_fichier}'\n")
        print(instance)
        print("\n")
        return instance
    except Exception as e:
        print(f"Impossible de charger le fichier : {e}")
        return None

def compresser(instance):
    """
    Compresse une instance de Donnee1D ou Donnee2D, afin de ne pouvoir uniquement tracer les courbes en animations (conservant les données sur tous les 30 pts temporels)
    :param instance: Donnee1D ou Donnee2D, à compresser
    :return: instance_lite: Donnee1D ou Donnee2D, compressée
    """
    instance_lite = None
    if type(instance) == Donnee2D:
        instance_lite = Donnee2D()
    elif type(instance) == Donnee1D:
        instance_lite = Donnee1D()

    instance_lite.copy(instance)
    instance_lite.N = instance_lite.N//30 + 1
    instance_lite.U = np.concatenate((instance_lite.U[::30,...], instance_lite.U[-1,...]), axis = 0)
    instance_lite.t = np.linspace(instance_lite.t[0], instance_lite.t[-1], instance_lite.N)
    instance_lite.dt *= 30
    instance_lite.E = np.concatenate((instance_lite.E[::30,...],instance_lite.E[-1,...]), axis = 0)
    instance_lite.label += " compressé"
    return instance_lite

def sauvegarder_compresser(instance, nom_fichier = "", opt = False):
    """
    Sauvegarde une instance uniquement de Donnee1D ou Donnee2D dans un fichier binaire (.pkl) en version lite
    :param instance: Donnee1D ou Donnee2D à sauvegarder.
    :param nom_fichier: Le nom ou le chemin du fichier de destination :
            - le fichier sera placé automatiquement dans un dossier .save/, ou un autre dossier, nommé comme '.nom_du_dossier/'
            - le fichier sera horodaté automatiquement si nom_fichier finit par un '_', sous le format MOIS-JOUR_HEURE-MINUTES-SECONDES
            - le fichier sera automatiquement placé sous l'extension '.pkl'
    :param opt: Bool, True si la sauvegarde s'effectue automatiquement, False sinon.
    """
    sauvegarder(compresser(instance), nom_fichier, opt)