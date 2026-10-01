"""Exceptions du paquet.

Une hiérarchie courte, pour que chaque couche sache quoi attraper :

- :class:`FinlamError` : base ; la CLI l'attrape pour afficher un message
  propre et sortir avec un code non nul, le service pour renvoyer un code
  HTTP adapté ;
- :class:`ContractError` : contrat JSON invalide (clé manquante, type
  inattendu, classe inconnue) — faute de l'appelant, 422 côté service ;
- :class:`DatasetError` : jeu de données absent ou inexploitable ;
- :class:`GallicaError` : accès à l'API Gallica (ARK invalide, pas d'ALTO,
  trop d'échecs réseau) ;
- :class:`ConversionError` : écriture ALTO/METS impossible ;
- :class:`PreprocessError` : prétraitement d'image impossible (OpenCV absent,
  image illisible).

Les erreurs de programmation (bugs) ne sont pas converties : elles doivent
remonter telles quelles, avec leur trace.
"""

from __future__ import annotations


class FinlamError(Exception):
    """Erreur attendue du paquet, porteuse d'un message destiné à l'utilisateur."""


class ContractError(FinlamError):
    """Le contrat JSON ne respecte pas le format attendu."""

    def __init__(self, message: str, page_id: str | None = None, field: str | None = None):
        where = " ".join(p for p in (f"page {page_id}" if page_id else "", field or "") if p)
        super().__init__(f"{where} : {message}" if where else message)
        self.page_id = page_id
        self.field = field


class DatasetError(FinlamError):
    """Jeu de données absent, incomplet ou incohérent."""


class GallicaError(FinlamError):
    """Échec d'accès à un document Gallica."""


class ConversionError(FinlamError):
    """Échec de production d'un fichier ALTO ou METS."""


class PreprocessError(FinlamError):
    """Prétraitement d'image impossible (dépendance absente, image illisible)."""
