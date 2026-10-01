"""Le pipeline et sa doublure.

``run`` enchaîne classification, ordre de lecture et regroupement en
articles et écrit le contrat JSON ; ``rules`` et ``ml`` sont les doublures
(règles, scikit-learn) utilisées en attendant les sorties du prototype
FINLAM (YOLO → séparateurs → LayoutReader → regroupeur).
"""

from finlam_prod.pipeline.base import ArticleSegmenter, ReadingOrderer, ZoneClassifier
from finlam_prod.pipeline.run import Pipeline, read_issue_json, write_issue_json

__all__ = [
    "ArticleSegmenter",
    "Pipeline",
    "ReadingOrderer",
    "ZoneClassifier",
    "read_issue_json",
    "write_issue_json",
]
