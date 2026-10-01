# Protocole de constitution d'un jeu « gold » pour la presse

L'article du consortium (Mocaër et al., 2026) indique que la vérité terrain de *La Liberté* est
dérivée des METS/ALTO existants (OCR ancien, erreurs de segmentation et d'ordre) et qu'un gold
standard est « en cours ». Un rapport d'utilisabilité à l'échelle de la collection a besoin de ce
jeu ; voici comment le constituer pour qu'il réponde aux questions de production.

## 1. À quoi il sert

Trois questions, dans l'ordre : (a) quelle est la qualité réelle de la séparation d'articles sur
les mises en page de Gallica, pas seulement sur *La Liberté* 1925-1928 ; (b) où le pipeline échoue
(titre, période, densité) pour régler les seuils du contrôle qualité ; (c) suivre la non-régression
d'une version à l'autre.

## 2. Échantillonnage stratifié

Trois strates, tirées au hasard dans les 8 M de documents de presse de Gallica :

| strate | modalités | pourquoi |
|---|---|---|
| titre | ~20 titres, nationaux et régionaux, quotidiens et hebdomadaires | la mise en page est propre à un titre |
| période | 1830-1880, 1880-1914, 1914-1945, après 1945 | typographie, colonnage, publicité évoluent |
| densité | tercile de zones par page (mesuré sur l'OCR existant) | les pages denses concentrent les erreurs |

Une page par cellule au minimum ; viser **300 à 500 pages**, pages de une sur-représentées (×2), car
c'est là que titres pleine largeur et suites d'articles abondent.

## 3. Taille et incertitude

Sur 48 pages, l'IC95 % du F1 d'articles mesure ± 6 points (`results/`) ; sur 433 pages (test *La
Liberté*) il serait de l'ordre de ± 2 points. Pour distinguer deux versions à 2 points près, 400
pages sont nécessaires ; le bootstrap sur les pages (`finlam_prod.evaluation.finlam.bootstrap_ci`)
donne l'intervalle réel une fois le jeu constitué.

## 4. Consignes d'annotation (à faire valider par le consortium)

- **Article** : une unité éditoriale avec ou sans titre ; les brèves d'une rubrique sont des
  articles distincts si elles ont chacune un titre ou un chapeau, sinon la rubrique est un article.
  Cette règle doit être écrite, car 40 % des « articles » du jeu public n'ont pas de titre et 27 %
  en ont plusieurs.
- **Ordre de lecture** : intra-article seulement (l'ordre entre articles est une convention, pas une
  vérité) ; colonne par colonne, de haut en bas, une continuation renvoie à son article.
- **Classes** : la taxonomie de l'article (17 libellés), avec un guide d'exemples par classe.
- **Double annotation** sur 10 % des pages pour mesurer l'accord inter-annotateurs (kappa sur les
  classes, AS@0.5 entre annotateurs pour les articles) : c'est le plafond que le pipeline peut viser.

## 5. Outillage et coût

Callico (TEKLIA, ICDAR 2024) ou Arkindex, déjà utilisés par le consortium, exportent en METS/ALTO
que `data/mets_alto.py` relit. À 10-15 minutes par page dense, 400 pages représentent 70 à 100
heures d'annotation, plus la double annotation ; c'est le poste de coût à budgéter dans le rapport.

## 6. Ce qu'on en tire pour la production

Les strates deviennent les axes du tableau de bord qualité : un F1 par titre et par période, des
seuils de QC réglés strate par strate, et un échantillon de relecture humaine proportionné au risque.
