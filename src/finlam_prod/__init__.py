"""finlam-prod : industrialisation du pipeline FINLAM de structuration de la presse.

Le consortium FINLAM (LITIS, TEKLIA, BnF) a produit un prototype qui
segmente une page de journal en blocs, les ordonne et les regroupe en
articles (Mocaër et al., 2026, arXiv 2607.15082). Ce paquet couvre ce
qui vient après le prototype : le contrat d'échange JSON, la conversion
vers les formats d'archivage de la BnF (ALTO, METS), le contrôle
qualité, l'évaluation avec les métriques du consortium, et l'exposition
en service. Une doublure légère (règles + gradient boosting) produit des
sorties au bon format tant que celles du prototype ne sont pas branchées.
"""

from finlam_prod.data.schema import Issue, Page, Zone, ZoneClass

__all__ = ["Issue", "Page", "Zone", "ZoneClass"]
__version__ = "0.3.0"
