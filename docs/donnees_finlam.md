# Ce que les annotations FINLAM contiennent vraiment

Constats établis sur `Teklia/Newspapers-finlam` (split test, 48 pages, 11 065 zones) le 21/09/2026,
en cherchant pourquoi des méthodes géométriquement correctes obtenaient des scores très bas. Ils
conditionnent ce qu'on peut raisonnablement apprendre et évaluer sur ce jeu, et donc la manière
dont un rapport d'utilisabilité doit lire ses propres chiffres.

## 1. Les polygones se touchent bord à bord

Largeur médiane des zones : 12 % de la page (6 à 8 colonnes) ; hauteur médiane : 1,2 % (un
paragraphe, parfois une ligne). Les bords gauches `x0` se concentrent sur une valeur par colonne
(5, 19, 35, 48, 63, 78…) et les largeurs valent exactement l'entraxe des colonnes : **les polygones
d'annotation incluent la gouttière**. Toute détection de colonnes par « vide » horizontal renvoie une
seule colonne. La détection retenue regroupe les bords gauches (`features.columns_from_zones`).

## 2. `zone_orders` : la séquence des articles est un artefact d'export

Dans l'ordre annoté, les zones d'un même `article_id` sont contiguës (48/48 pages), et les articles
se suivent **dans l'ordre lexicographique de leurs identifiants** : `1, 10, 11, …, 19, 2, 20, 3…`
(14 pages sur 48 le montrent sans ambiguïté ; les autres ont moins de 10 articles, donc ordre
lexicographique = ordre numérique). Les identifiants suivent l'ordre de création par les
annotateurs. **L'ordre inter-articles n'est donc pas une cible d'apprentissage** ; la métrique
`article_seq_tau` plafonne pour toutes les méthodes (0,59–0,71).

## 3. L'ordre intra-article est lui aussi bruité

Articles contenus dans une seule colonne (338 sur 530) : l'ordre annoté n'est haut→bas que dans
**76 %** des cas (tau 0,87 contre le simple tri par `y0`). Articles multi-colonnes : tau 0,64 avec
une lecture colonne par colonne, 22 % parfaits. Une part de l'ordre annoté reflète l'ordre de saisie,
pas la lecture. Le plafond réaliste de `intra_article_tau` pour une méthode géométrique est donc
autour de 0,75, valeur atteinte (`columns_top_down` : 0,74).

## 4. `article_id` ne désigne pas « un titre = un article »

| | test | train |
|---|---|---|
| articles sans aucun ARTICLE-TITLE | 37 % | 40 % |
| exactement un titre | 33 % | 32 % |
| deux titres ou plus | 30 % | 27 % |

Un « article » annoté est souvent un bloc éditorial (une rubrique de brèves, une suite de
dépêches sous un même chapeau). La règle « un titre ouvre un article » est fausse par
construction dans deux cas sur trois ; c'est le plafond de `title_rule` (0,60) et de
`column_title` (0,41).

## 5. Ce qui se prédit et ce qui ne se prédit pas (classifieur de paires, classes vérité terrain)

| lien candidat | n | taux de base | précision | rappel |
|---|---|---|---|---|
| zone suivante dans la colonne | 21 257 | 0,84 | 0,90 | 0,96 |
| zone pleine largeur → colonnes couvertes | 3 516 | 0,68 | 0,83 | 0,93 |
| haut de la colonne suivante | 26 651 | 0,31 | 0,56 | 0,60 |
| colonne suivante, même hauteur | 13 985 | 0,43 | 0,50 | 0,66 |

La continuité *dans* une colonne se lit dans la géométrie. La continuation *vers la colonne
suivante* ne se lit pas : précision à peine supérieure au taux de base. Décider qu'un paragraphe en
haut de colonne 4 continue l'article de la colonne 3 demande de comprendre le texte (reprise d'un
sujet, phrase coupée). C'est le cas d'usage d'un modèle de langue, et le cœur du problème de
recherche FINLAM.

## 6. *La Liberté* : des lignes, pas des blocs, et des coordonnées en fraction

Le jeu des résultats de l'article n'a pas la même granularité que le multi-titres : 205 zones par
page en moyenne (88 951 zones pour 433 pages de test), 23 articles par page, hauteur médiane de zone
1,2 % de la page. Ce sont des **lignes de texte** regroupées en articles, pas des blocs. Les métriques
du consortium (AS@0.5, Jaccard, BLEU_block) sont donc calculées sur des lignes ; un F1 d'articles à
80 % s'y lit comme « 80 % des articles retrouvés à IoU ≥ 0,5 sur l'union de leurs lignes ». Question
à poser au consortium : le pipeline déployé travaille-t-il au bloc (YOLO) puis redistribue-t-il les
lignes, ou évalue-t-il directement au bloc ?

Les polygones y sont en **fraction de page** (0-1) alors que le multi-titres est en pourcentage
(0-100). Le chargeur détecte l'échelle (`polygon_scale`) ; sans cela toutes les règles géométriques
étaient fausses en silence (exactitude de classes 2 %, aucune erreur levée). Leçon pour la chaîne
de production : **valider les bornes de coordonnées à l'entrée** (le QC le fait : `geometry`).

Les 16 classes se ramènent aux 13 du paquet sans classe inconnue ; `TEXT` fait 84 % des zones,
`TITLE` 8 %, `INSIDEHEADING` et `SUBTITLE` 2 % chacun. Aucune zone `ASIDE`, `TABLECONTENT`,
`ADVERTISEMENT`, `ANNOUNCEMENT`, `CAPTION` ni `AUTHOR` dans le split test : les classes rares de la
taxonomie n'y sont pas mesurables.

## Conséquences pour l'évaluation et le rapport

- Rapporter `intra_article_tau` et `article_seq_tau` séparément, et ne pas interpréter un tau global
  bas comme un échec de la méthode.
- Pour la séparation d'articles, comparer les méthodes *en chaîne* (classes prédites, ordre prédit) :
  l'évaluation « isolée » avec l'ordre annoté est optimiste, car cet ordre encode déjà les frontières.
- Avant toute industrialisation, demander au consortium les conventions d'annotation (définition
  d'`article_id`, origine de `zone_orders`) et, si possible, le jeu *La Liberté* (8 836 pages) dont
  l'article Tiramisu décrit des annotations d'ordre de lecture et de sections plus contrôlées.
