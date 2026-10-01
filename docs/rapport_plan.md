# Plan du rapport final (ANR et consortium) : utilisabilité du pipeline à l'échelle

1. **Contexte** : FINLAM, 8 M de documents de presse, dernière phase = industrialisation ; le pipeline
   modulaire retenu (Mocaër et al., 2026) et ce qui manquait entre le prototype et l'archivage.
2. **Architecture déployée** : contrat JSON, conversion ALTO/METS, QC, service, orchestration
   (`docs/architecture.md`) ; flux de fichiers entrants et sortants ; profils BnF.
3. **Données d'évaluation** : jeux La Liberté et multi-titres, constats sur les annotations
   (`docs/donnees_finlam.md`), jeu gold (`docs/protocole_gold.md`).
4. **Résultats** : métriques du consortium avec IC95 %, par strate (titre, période, densité) ;
   évaluation en chaîne ; comparaison avec l'OLR d'Europeana Newspapers sur fascicules communs.
5. **Contrôle qualité en production** : taux de pages bloquées, causes (tableaux, texte dégradé,
   XSD), coût de relecture.
6. **Coût et passage à l'échelle** : secondes par page (GPU pour le prototype, CPU pour l'archivage),
   projection sur la collection, dimensionnement.
7. **Limites et suite** : continuation inter-colonnes (texte), profils BnF, généralisation hors
   La Liberté, maintenance des modèles.
