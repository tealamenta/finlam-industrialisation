"""Pages de presse synthétiques, annotées comme FINLAM.

Sert aux tests unitaires et aux démonstrations hors ligne : on génère
des pages à colonnes, avec en-tête, titres pleine largeur ou de
colonne, paragraphes, illustrations et légendes, et une vérité terrain
cohérente (classe, ordre de lecture, article). Les textes sont tirés
d'un petit lexique pour donner aux descripteurs textuels une chance de
fonctionner ; ils n'ont aucune prétention de réalisme.
"""

from __future__ import annotations

import random

from finlam_prod.data.schema import NO_ARTICLE, Page, Zone, ZoneClass

_WORDS = [
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "ministre",
    "chambre",
    "député",
    "conseil",
    "séance",
    "projet",
    "loi",
    "vote",
    "paris",
    "marseille",
    "lyon",
    "toulouse",
    "mardi",
    "jeudi",
    "hier",
    "demain",
    "matin",
    "soir",
    "gouvernement",
    "affaire",
    "procès",
    "tribunal",
    "cour",
    "témoin",
    "accusé",
    "avocat",
    "grève",
    "ouvriers",
    "usine",
    "patron",
    "match",
    "équipe",
    "course",
    "cheval",
    "prix",
    "départ",
    "arrivée",
    "victoire",
    "record",
    "théâtre",
    "concert",
    "pièce",
    "acteur",
    "salle",
    "public",
    "succès",
    "ville",
    "quartier",
    "rue",
    "place",
    "marché",
    "bourse",
    "cours",
    "rente",
    "actions",
    "hausse",
    "baisse",
    "marchandises",
    "blé",
    "vin",
    "sucre",
    "navire",
    "port",
    "arrivée",
]


def _sentence(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(_WORDS) for _ in range(n_words)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def _paragraph(rng: random.Random, n_lines: int) -> str:
    return "\n".join(_sentence(rng, rng.randint(6, 11)) for _ in range(n_lines))


def _title(rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS).upper() for _ in range(rng.randint(2, 5)))


def make_page(
    seed: int = 0,
    n_columns: int = 4,
    issue_id: str = "issue-synth",
    index: int = 1,
    article_offset: int = 0,
) -> Page:
    """Génère une page annotée : en-tête, titre de une, colonnes d'articles.

    Vérité terrain produite : classe de zone, ordre de lecture (en-tête,
    titre de une, puis colonnes de gauche à droite et de haut en bas) et
    identifiant d'article (nouvel article à chaque titre).
    """
    rng = random.Random(seed)
    zones: list[Zone] = []
    order = 0
    article = article_offset

    def add(x0, y0, x1, y1, text, klass, art):
        nonlocal order
        y1 = min(y1, 97.0)
        if y1 - y0 < 0.5:
            return
        zones.append(
            Zone(
                id=f"p{index}_z{len(zones)}",
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                text=text,
                zone_class=int(klass),
                order=order,
                article_id=art,
            )
        )
        order += 1

    # En-tête (hors article)
    if index == 1:
        add(20, 1, 80, 6, "LE JOURNAL SYNTHÉTIQUE", ZoneClass.HEADER_TITLE, NO_ARTICLE)
    add(
        5,
        6.5,
        40,
        8,
        f"Numéro {seed} — Page {index}",
        ZoneClass.HEADER_TEXT,
        NO_ARTICLE,
    )

    top = 9.0
    # Titre pleine largeur (une fois sur deux)
    if rng.random() < 0.5:
        article += 1
        add(5, top, 95, top + 3, _title(rng), ZoneClass.ARTICLE_TITLE, article)
        add(
            5,
            top + 3.2,
            95,
            top + 4.6,
            _sentence(rng, 8),
            ZoneClass.ARTICLE_SUBTITLE,
            article,
        )
        top += 5.5
        spanning_article = article
    else:
        spanning_article = None

    gutter = 1.5
    col_w = (90 - gutter * (n_columns - 1)) / n_columns
    for col in range(n_columns):
        x0 = 5 + col * (col_w + gutter)
        x1 = x0 + col_w
        y = top
        current = spanning_article if (col == 0 and spanning_article) else None
        while y < 92:
            kind = rng.random()
            if current is None or kind < 0.25:
                article += 1
                current = article
                h = rng.uniform(1.5, 3.0)
                add(x0, y, x1, y + h, _title(rng), ZoneClass.ARTICLE_TITLE, current)
                y += h + 0.4
                if rng.random() < 0.3:
                    add(
                        x0,
                        y,
                        x1,
                        y + 1.2,
                        _sentence(rng, 6),
                        ZoneClass.ARTICLE_SUBTITLE,
                        current,
                    )
                    y += 1.6
            elif kind < 0.35:
                h = rng.uniform(6, 12)
                add(x0, y, x1, y + h, "", ZoneClass.ARTICLE_ILLUSTRATION, current)
                y += h + 0.3
                add(x0, y, x1, y + 1.2, _sentence(rng, 5), ZoneClass.CAPTION, current)
                y += 1.6
            elif kind < 0.42:
                add(
                    x0,
                    y,
                    x1,
                    y + 1.1,
                    rng.choice(["Jean Dupont", "P. Martin", "— X."]),
                    ZoneClass.AUTHOR,
                    current,
                )
                y += 1.5
            else:
                n_lines = rng.randint(3, 14)
                h = n_lines * 0.55
                add(
                    x0,
                    y,
                    x1,
                    min(y + h, 92),
                    _paragraph(rng, n_lines),
                    ZoneClass.ARTICLE_TEXT,
                    current,
                )
                y += h + 0.4
    return Page(
        id=f"synth-{issue_id}-{index}",
        issue_id=issue_id,
        index=index,
        zones=zones,
        width_px=1600,
        height_px=2400,
        source="synthetic",
    )


def make_pages(n_pages: int, seed: int = 0, pages_per_issue: int = 4) -> list[Page]:
    """Génère ``n_pages`` pages réparties en fascicules de ``pages_per_issue`` pages."""
    pages = []
    for i in range(n_pages):
        issue_id = f"synth-issue-{seed}-{i // pages_per_issue}"
        index = i % pages_per_issue + 1
        offset = (
            0
            if index == 1
            else max(
                (z.article_id for p in pages if p.issue_id == issue_id for z in p.zones),
                default=0,
            )
        )
        pages.append(
            make_page(
                seed=seed * 10_000 + i,
                n_columns=random.Random(i).choice([3, 4, 5]),
                issue_id=issue_id,
                index=index,
                article_offset=offset,
            )
        )
    return pages
