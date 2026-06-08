import pickle
import os
from datetime import datetime
chemin = ".save/"

def sauvegarder(instance, nom_fichier = "") -> None:
    """
    Sauvegarde d'une instance uniquement de Donnee1D ou Donnee2D dans un fichier binaire (.pkl).
    :param instance: Donnee1D ou Donnee2D à sauvegarder.
    :param nom_fichier: Le nom ou le chemin du fichier de destination.
    """
    if not nom_fichier.startswith('.'): nom_fichier = chemin + nom_fichier
    if nom_fichier == "" or nom_fichier.endswith('_'): nom_fichier += datetime.now().strftime('%m-%d_%H-%M-%S')
    if not nom_fichier.endswith('.pkl'): nom_fichier += '.pkl'

    try:
        with open(nom_fichier, 'wb') as fichier:
            pickle.dump(instance, fichier)
        print(f"Instance sauvegardée avec succès sous : '{nom_fichier}'")
    except Exception as e:
        print(f"Impossible de sauvegarder l'objet : {e}")


def charger(nom_fichier: str):
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
        print(f"Instance chargée avec succès depuis : '{nom_fichier}'")
        return instance
    except Exception as e:
        print(f"Impossible de charger le fichier : {e}")
        return None