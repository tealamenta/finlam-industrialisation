# État de l'art : extraction de la structure logique des journaux numérisés

Synthèse de la revue de cadrage (64 références vérifiées, grille d'extraction séparée). Question :
quelles méthodes, données, formats et protocoles existent pour la séparation d'articles, l'ordre de
lecture et la classification de zones, et lesquels sont transférables à une chaîne de production
ALTO/METS de la BnF ?

## 1. Cinq familles de méthodes

**Règles et géométrie.** Découpage XY, détection de séparateurs, colonnes. C'est ce que les
prestataires ont fait à l'échelle industrielle pour Europeana Newspapers (CCS docWORKS, 2 M de pages
en mode article ; Neudecker & Antonacopoulos, DAS 2016). Forces : coût nul, débit, explicabilité.
Faiblesses documentées : échec quasi systématique sur les grands tableaux, titres multi-colonnes,
continuations ; aucune métrique d'articles publique. Reste la référence « plancher » et le repli sans
modèle. Variante moderne : XY-Cut++ (2025), hiérarchie de masques.

**Modèles graphiques.** CRF au pixel puis règles de regroupement (LITIS, Journal de Rouen, 2012 :
85,8 % d'articles corrects), modèle de Markov 2D encodant l'ordre de lecture (ICDAR 2019). Première
génération apprise, corpus petits ; filiation directe du consortium FINLAM (LITIS).

**Détection d'objets et segmentation sur image.** FCN/U-Net (Meier et al. 2017, Liebl & Burghardt
2020), Newspaper Navigator (Library of Congress, 2020, 16 M de pages), impresso (Barman et al. 2021,
visuel + texte), Chronicling Germany (2024), YOLO. Robustes, rapides sur GPU, entraînables sur
quelques milliers de pages. C'est la brique « blocs » du pipeline FINLAM (YOLO, mAP@50 72 %). Ne
donnent ni l'ordre ni les articles ; sensibles au domaine.

**Multimodal texte + image.** LayoutLMv3 (pré-entraînement unifié), LayoutReader (ordre de lecture,
2021 ; utilisé tel quel, fine-tuné, dans le pipeline FINLAM), STRAS (séparation d'articles par indices
textuels, 2025), Ar-Q-Former (transformer multimodal, 2025 : +19 à +22 points d'AR-F1 sur NewsEye NLF
et BnF, mais 10 points sous les règles ajustées au corpus), GNN de CITlab/NewsEye. C'est la famille qui
traite ce que la géométrie ne peut pas : la continuation d'un article dans la colonne suivante.

**Bout-en-bout et modèles de fondation.** DAN/DANIEL (LITIS, manuscrits), Tiramisu (FINLAM 2026 :
meilleur Jaccard article 89 %, mais mAP@50 39 % et 34 fois plus lent sur CPU que le pipeline), VLM
(NewsBench 2026 : un détecteur de mise en page vaut +0,22 pour Tesseract, plus que le choix du
reconnaisseur). Le consortium conclut lui-même au pipeline modulaire pour le déploiement.

## 2. Données et formats

Europeana Newspapers (METS/ALTO au niveau article, profil ENMAP, 12 M de pages recherchables ; le
jeu « mode article » de la BnF, 790 000 pages, ALTO 1.4, est **déjà dans Gallica**), NewsEye
(compétition ICPR 2020 sur la segmentation de blocs, séparation d'articles CITlab), impresso, Newspaper
Navigator, OCR-D, Bibliothèque nationale du Luxembourg, Living with Machines. Tous convergent vers
METS/ALTO + IIIF et des jeux ouverts. Jeux FINLAM : multi-titres (721 pages) et *La Liberté* (8 836
pages, résultats de l'article) ; vérité terrain dérivée des METS/ALTO existants, gold standard en cours.

## 3. Évaluation

Articles : F1 par couples, ARI, AS@0.5 (appariement hongrois), Jaccard (FINLAM). Ordre : tau de
Kendall, BLEU sur séquences de blocs (LayoutReader, FINLAM), signification de l'ordre de lecture
(Clausner et al. 2013). OCR : CER/WER, cMER avec IC95 % (HIPE-OCRepair 2026, où l'équipe BnF-Mistral
est première), impact aval sur la recherche et le TAL (van Strien et al. 2020 ; Hill & Hengchen 2019 ;
Chiron, Moreux et al. 2017). Segmentation évaluée par son effet sur le texte (Boillet et al., IJDAR
2022, TEKLIA). Qualité sans vérité terrain en production : alignement croisé de moteurs (Ben Salah,
Moreux et al. 2015).

## 4. Ce qu'on en retient pour l'industrialisation à la BnF

1. Le pipeline modulaire est le bon choix de déploiement ; la recherche continue sur le bout-en-bout.
2. La brique manquante côté modèles est la continuation inter-colonnes, qui demande le texte ;
   c'est la famille multimodale à suivre, pas une règle de plus.
3. Les formats sont acquis (METS/ALTO au niveau article) ; la compatibilité avec le « mode article »
   déjà en production est une exigence, pas une option.
4. L'évaluation doit reprendre les métriques du consortium, ajouter des intervalles de confiance et un
   jeu gold stratifié ; l'évaluation isolée avec l'ordre annoté est optimiste.
5. Le contrôle qualité en production peut s'appuyer sur des échecs documentés (tableaux) et sur
   l'estimation de qualité sans vérité terrain, tradition de la BnF depuis 2015.
