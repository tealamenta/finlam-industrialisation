"""Ligne de commande ``finlam-prod``.

- ``download``  : jeux FINLAM (multi-titres) et La Liberté → cache JSON Lines ;
- ``train``     : entraîne la doublure (règles + boosting) ;
- ``run``       : applique un pipeline à un fascicule (cache ou ARK Gallica) et écrit
                  JSON, ALTO, METS, rapport QC ;
- ``convert``   : contrat JSON → ALTO + METS validés ;
- ``qc``        : contrôle qualité d'un JSON ou d'un METS/ALTO existant ;
- ``evaluate``  : métriques du consortium (AS@0.5, Jaccard, BLEU) avec IC95 %, sur la
                  doublure ou sur des sorties du prototype ;
- ``compare``   : deux structurations d'un même fascicule (ex. OLR Europeana vs pipeline) ;
- ``preprocess``: image d'une page pliée → pli détecté, éclairage normalisé, moitiés
                  redressées (extra ``image``) ;
- ``dewarp``    : photo d'une page pliée en 3D / gondolée → page à plat (UVDoc, extra
                  ``dewarp``), puis le prétraitement ci-dessus ;
- ``serve``     : service HTTP.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

import typer

from finlam_prod.config import get_settings
from finlam_prod.data.schema import Issue, group_by_issue
from finlam_prod.errors import DatasetError, FinlamError
from finlam_prod.logging_utils import configure_logging
from finlam_prod.pipeline.run import Pipeline

app = typer.Typer(
    help="Industrialisation du pipeline FINLAM : conversion, QC, évaluation, service."
)
SETTINGS = get_settings()
DEFAULT_CACHE = SETTINGS.cache_dir
DEFAULT_RESULTS = SETTINGS.results_dir
DATASETS = ("finlam", "la-liberte")
LOGGER = logging.getLogger(__name__)


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging(SETTINGS, level="DEBUG" if verbose else None)
    LOGGER.debug("configuration : %s", SETTINGS.describe())


def _module(dataset: str) -> ModuleType:
    if dataset == "la-liberte":
        from finlam_prod.data import la_liberte

        return la_liberte
    if dataset == "finlam":
        from finlam_prod.data import finlam

        return finlam
    raise typer.BadParameter("--dataset doit valoir finlam ou la-liberte")


@app.command()
def download(
    dataset: str = typer.Option("la-liberte", help="la-liberte (jeu de l'article) ou finlam"),
    split: str = typer.Option("test", help="train, validation/val, test ou all"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE),
    limit: int | None = typer.Option(None, help="Nombre de pages max (essai rapide)"),
    images: bool = typer.Option(False, help="Enregistrer aussi les images"),
) -> None:
    """Télécharge un jeu FINLAM depuis Hugging Face et le met en cache."""
    from finlam_prod.data.finlam import describe

    module = _module(dataset)
    for name in module.SPLITS if split == "all" else [split]:
        pages = module.download_split(name, cache_dir, limit=limit, with_images=images)
        typer.echo(f"{dataset} / {name}: {describe(pages)}")


def _pipeline(models_dir: Path | None) -> Pipeline:
    from finlam_prod.pipeline.run import Pipeline

    if models_dir is None:
        return Pipeline()
    from finlam_prod.pipeline.ml import BoundarySegmenter, GradientBoostingZoneClassifier
    from finlam_prod.pipeline.registry import check_compatibility, load_metadata

    meta = load_metadata(models_dir)  # vérifie les empreintes, avertit si non versionné
    if meta is not None:
        check_compatibility(meta)
        LOGGER.info(
            "doublure %s (%s/%s, %d pages)",
            meta.model_version,
            meta.dataset,
            meta.split,
            meta.n_pages,
        )
    return Pipeline(
        GradientBoostingZoneClassifier.load(models_dir / "zone_classifier.joblib"),
        segmenter=BoundarySegmenter.load(models_dir / "segmenter.joblib"),
    )


@app.command()
def train(
    dataset: str = typer.Option("la-liberte"),
    split: str = typer.Option("train", help="Split d'entraînement (validation : plus léger)"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE),
    models_dir: Path = typer.Option(Path("models")),
    limit: int | None = typer.Option(None, help="Limiter le nombre de pages d'entraînement"),
    synthetic: bool = typer.Option(False, help="Pages synthétiques (hors ligne)"),
) -> None:
    """Entraîne la doublure (classifieur de zones + frontières d'articles)."""
    from finlam_prod.data.finlam import load_split
    from finlam_prod.pipeline.ml import BoundarySegmenter, GradientBoostingZoneClassifier

    if synthetic:
        from finlam_prod.data.synthetic import make_pages

        pages = make_pages(limit or 80, seed=1)
    else:
        pages = load_split(split, cache_dir, dataset)
        if limit:
            pages = pages[:limit]
    from finlam_prod.evaluation.tracking import log_models
    from finlam_prod.pipeline.registry import write_metadata

    models_dir.mkdir(parents=True, exist_ok=True)
    GradientBoostingZoneClassifier().fit(pages).save(models_dir / "zone_classifier.joblib")
    BoundarySegmenter().fit(pages).save(models_dir / "segmenter.joblib")
    meta = write_metadata(
        models_dir,
        dataset="synthetic" if synthetic else dataset,
        split=split,
        n_pages=len(pages),
        notes="doublure : classifieur de lignes + frontières d'articles",
    )
    log_models(models_dir, meta, experiment=SETTINGS.mlflow_experiment)
    typer.echo(
        f"Doublure entraînée sur {len(pages)} pages ({dataset}/{split}) → {models_dir} "
        f"[version {meta.model_version}]"
    )


@app.command()
def evaluate(
    dataset: str = typer.Option("la-liberte"),
    split: str = typer.Option("test"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE),
    models_dir: Path | None = typer.Option(
        SETTINGS.models_dir, help="Doublure entraînée (sinon règles seules) ; FINLAM_MODELS_DIR"
    ),
    predictions: Path | None = typer.Option(
        None, help="JSON de sorties du prototype à évaluer à la place de la doublure"
    ),
    results_dir: Path = typer.Option(DEFAULT_RESULTS),
    n_boot: int = typer.Option(1000),
    synthetic: bool = typer.Option(False),
) -> None:
    """Évalue avec les métriques du consortium (AS@0.5, Jaccard, BLEU_block) et IC95 %."""
    from finlam_prod.evaluation.run import evaluate_pipeline, evaluate_predictions

    if synthetic:
        from finlam_prod.data.synthetic import make_pages

        gold = make_pages(20, seed=2)
        tag = "synthetic"
    else:
        from finlam_prod.data.finlam import load_split

        gold = load_split(split, cache_dir, dataset)
        tag = f"{dataset.replace('-', '_')}_{split}"
    if predictions is not None:
        from finlam_prod.pipeline.run import read_issue_json

        predicted = [
            p for path in sorted(predictions.glob("*.json")) for p in read_issue_json(path).pages
        ]
        report = evaluate_predictions(gold, predicted, n_boot=n_boot)
        tag += "_prototype"
    else:
        pipeline = _pipeline(models_dir)
        report = evaluate_pipeline(pipeline, gold, n_boot=n_boot)
        tag += "_" + ("doublure" if models_dir else "regles")
    results_dir.mkdir(parents=True, exist_ok=True)
    from finlam_prod.evaluation.tracking import log_report

    model_version = None
    if models_dir is not None:
        from finlam_prod.pipeline.registry import load_metadata

        meta = load_metadata(models_dir, verify=False)
        model_version = meta.model_version if meta else None
    log_report(
        report,
        {
            "dataset": dataset if not synthetic else "synthetic",
            "split": split,
            "pipeline": tag.rsplit("_", 1)[-1],
            "models_dir": models_dir,
            "model_version": model_version or "",
        },
        experiment=SETTINGS.mlflow_experiment,
    )
    report.per_page.to_csv(results_dir / f"eval_{tag}_pages.csv", index=False)
    report.table().to_csv(results_dir / f"eval_{tag}.csv", index=False)
    (results_dir / f"eval_{tag}.md").write_text(report.markdown(), encoding="utf-8")
    typer.echo(report.markdown())


def _load_issue(source: str, cache_dir: Path, work_dir: Path, pages: str | None) -> Issue:
    if source.startswith(("ark:", "bpt6k", "http")):
        from finlam_prod.data.gallica import GallicaClient, normalise_ark

        wanted = [int(p) for p in pages.split(",")] if pages else None
        fetched = GallicaClient().fetch_issue(source, work_dir / "gallica", pages=wanted)
        return Issue(
            id=normalise_ark(source), name=f"Gallica {normalise_ark(source)}", pages=fetched
        )
    from finlam_prod.data.finlam import load_split

    for dataset in DATASETS:
        for split in ("test", "validation", "val", "train"):
            try:
                loaded = load_split(split, cache_dir, dataset)
            except DatasetError:
                continue
            issues = {i.id: i for i in group_by_issue(loaded)}
            if source in issues:
                return issues[source]
    raise typer.BadParameter(
        f"Fascicule {source!r} introuvable (ni ARK Gallica, ni identifiant en cache)."
    )


def _write_outputs(issue: Issue, target: Path, pipeline_name: str) -> dict[str, object]:
    from finlam_prod.convert.alto import validate_alto, write_alto
    from finlam_prod.convert.mets import validate_mets, write_mets
    from finlam_prod.pipeline.run import write_issue_json
    from finlam_prod.qc.checks import check_issue, findings_table, summarise

    write_issue_json(issue, target / "pipeline.json", pipeline_name)
    alto_errors = 0
    for page in issue.pages:
        alto_errors += len(validate_alto(write_alto(page, target / f"f{page.index}.alto.xml")))
    mets_errors = validate_mets(write_mets(issue, target / "mets.xml"))
    findings = check_issue(issue)
    findings_table(findings).to_csv(target / "qc_findings.csv", index=False)
    summary: dict[str, object] = dict(summarise(findings, len(issue.pages)))
    summary.update(
        {
            "alto_xsd_errors": alto_errors,
            "mets_xsd_errors": len(mets_errors),
            "pipeline": pipeline_name,
        }
    )
    (target / "qc_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


@app.command()
def run(
    source: str = typer.Argument(..., help="ARK Gallica ou identifiant de fascicule en cache"),
    out_dir: Path = typer.Option(Path("results/run")),
    cache_dir: Path = typer.Option(DEFAULT_CACHE),
    models_dir: Path | None = typer.Option(SETTINGS.models_dir),
    pages: str | None = typer.Option(None, help="Vues à traiter, ex. 1,2,3 (Gallica)"),
) -> None:
    """Applique le pipeline à un fascicule et écrit JSON, ALTO, METS et rapport QC."""
    issue = _load_issue(source, cache_dir, out_dir, pages)
    preprocess_summary = None
    if SETTINGS.preprocess and any(p.image_path for p in issue.pages):
        from finlam_prod.preprocess.issue import REPORT_FILE, preprocess_issue

        model = None
        if SETTINGS.dewarp:
            from finlam_prod.preprocess.dewarp import fetch_weights, load_model

            model = load_model(fetch_weights(models_dir or Path("models")))
        target = out_dir / issue.id
        target.mkdir(parents=True, exist_ok=True)
        issue, report = preprocess_issue(issue, dewarp_model=model)
        report.write(target / REPORT_FILE)
        preprocess_summary = report.summary
    pipeline = _pipeline(models_dir)
    predicted = pipeline.run_issue(issue)
    summary = _write_outputs(predicted, out_dir / predicted.id, pipeline.name)
    if preprocess_summary is not None:
        summary["preprocess"] = preprocess_summary
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command()
def convert(
    json_path: Path = typer.Argument(..., help="Contrat JSON (pipeline.json)"),
    out_dir: Path | None = typer.Option(None, help="Dossier de sortie (défaut : celui du JSON)"),
) -> None:
    """Convertit un contrat JSON en ALTO (par page) et METS (fascicule), validés XSD, avec QC."""
    from finlam_prod.pipeline.run import read_issue_json

    issue = read_issue_json(json_path)
    summary = _write_outputs(issue, out_dir or json_path.parent, "convert")
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command()
def qc(path: Path = typer.Argument(..., help="pipeline.json ou mets.xml")) -> None:
    """Contrôle qualité d'un contrat JSON ou d'un METS/ALTO existant."""
    from finlam_prod.qc.checks import check_issue, findings_table, summarise

    issue = _read_any(path)
    findings = check_issue(issue)
    typer.echo(findings_table(findings).to_string(index=False) if findings else "Aucune anomalie.")
    typer.echo(json.dumps(summarise(findings, len(issue.pages)), indent=2, ensure_ascii=False))


def _read_any(path: Path) -> Issue:
    if path.suffix.lower() == ".json":
        from finlam_prod.pipeline.run import read_issue_json

        return read_issue_json(path)
    from finlam_prod.data.mets_alto import read_mets_alto

    result = read_mets_alto(path)
    for warning in result.warnings[:10]:
        logging.getLogger(__name__).warning(warning)
    return result.issue


@app.command()
def compare(
    reference: Path = typer.Argument(
        ..., help="Structuration de référence (mets.xml ou pipeline.json)"
    ),
    candidate: Path = typer.Argument(
        ..., help="Structuration à comparer (mets.xml ou pipeline.json)"
    ),
    n_boot: int = typer.Option(1000),
) -> None:
    """Compare deux structurations d'un même fascicule (ex. OLR Europeana Newspapers vs pipeline).

    Les zones doivent être les mêmes dans les deux (mêmes blocs ALTO) ; les
    articles de la référence servent de vérité, ceux du candidat sont évalués
    avec AS@0.5, Jaccard et BLEU_block.
    """
    from finlam_prod.evaluation.run import evaluate_predictions

    ref, cand = _read_any(reference), _read_any(candidate)
    for a, b in zip(ref.pages, cand.pages, strict=True):
        b.id = a.id  # alignement par position de page
    report = evaluate_predictions(ref.pages, cand.pages, n_boot=n_boot)
    typer.echo(report.markdown())


@app.command()
def preprocess(
    image: Path = typer.Argument(..., help="image de la page (JPEG, PNG, TIFF)"),
    out: Path | None = typer.Option(None, help="image corrigée (défaut : <nom>_pre.png)"),
    max_angle: float = typer.Option(3.0, help="inclinaison maximale corrigée, en degrés"),
) -> None:
    """Prétraitement d'une page scannée pliée : pli, éclairage, inclinaison par moitié."""
    from finlam_prod.errors import PreprocessError
    from finlam_prod.preprocess import preprocess_page

    try:
        import cv2
    except ImportError as exc:
        raise PreprocessError(
            "installer l'extra image : pip install 'finlam-prod[image]'"
        ) from exc
    data = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise PreprocessError(f"image illisible : {image}")
    result = preprocess_page(data, max_angle=max_angle)
    target = out or image.with_name(f"{image.stem}_pre.png")
    if not cv2.imwrite(str(target), result.image):
        raise PreprocessError(f"écriture impossible : {target}")
    typer.echo(json.dumps({"output": str(target), **result.to_dict()}, indent=2))


@app.command()
def dewarp(
    image: Path = typer.Argument(..., help="photo de la page (JPEG, PNG)"),
    out: Path | None = typer.Option(None, help="image corrigée (défaut : <nom>_flat.png)"),
    weights_dir: Path = typer.Option(
        Path("models"), help="dossier des poids UVDoc (téléchargés une fois, 32 Mo)"
    ),
    no_preprocess: bool = typer.Option(False, help="géométrie seule, sans éclairage ni pli"),
) -> None:
    """Dégauchissement appris (UVDoc) d'une page photographiée, puis prétraitement."""
    from finlam_prod.errors import PreprocessError
    from finlam_prod.preprocess import preprocess_page
    from finlam_prod.preprocess.dewarp import dewarp_page, fetch_weights, load_model

    try:
        import cv2
    except ImportError as exc:
        raise PreprocessError(
            "installer l'extra image : pip install 'finlam-prod[image]'"
        ) from exc
    data = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if data is None:
        raise PreprocessError(f"image illisible : {image}")
    model = load_model(fetch_weights(weights_dir))
    flat = dewarp_page(data, model)
    report: dict[str, object] = {"dewarp": "uvdoc"}
    output: Any = flat
    if not no_preprocess:
        result = preprocess_page(flat)
        output = result.image
        report.update(result.to_dict())
    target = out or image.with_name(f"{image.stem}_flat.png")
    if not cv2.imwrite(str(target), output):
        raise PreprocessError(f"écriture impossible : {target}")
    typer.echo(json.dumps({"output": str(target), **report}, indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    models_dir: Path | None = typer.Option(SETTINGS.models_dir),
) -> None:
    """Lance le service HTTP (FastAPI) : POST /issues → ALTO, METS, QC."""
    import uvicorn

    from finlam_prod.service.api import create_app

    uvicorn.run(create_app(models_dir), host=host, port=port)


def main() -> None:
    """Point d'entrée : erreur attendue → une ligne et code 1 ; bug → trace complète."""
    try:
        app()
    except FinlamError as exc:
        typer.secho(f"Erreur : {exc}", err=True, fg=typer.colors.RED)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
