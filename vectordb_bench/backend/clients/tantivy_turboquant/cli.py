from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class TantivyTurboquantTypedDict(CommonTypedDict):
    index_path: Annotated[
        str,
        click.option(
            "--index-path",
            type=click.Path(),
            default="/tmp/tantivy_turboquant_bench",
            show_default=True,
            help="Directory for the on-disk Tantivy index + raw_vectors.bin sampler store",
        ),
    ]
    metric: Annotated[
        str,
        click.option(
            "--metric",
            type=click.Choice(["cosine", "ip", "l2"], case_sensitive=False),
            default="cosine",
            show_default=True,
            help="Distance metric (cosine = normalize + inner product)",
        ),
    ]
    heap_mb: Annotated[
        int,
        click.option(
            "--heap-mb",
            type=int,
            default=200,
            show_default=True,
            help="Per-thread Tantivy IndexWriter heap budget (MB)",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TantivyTurboquantTypedDict)
def TantivyTurboquant(**parameters: Unpack[TantivyTurboquantTypedDict]):
    from ..api import MetricType
    from .config import TantivyTurboquantConfig, TantivyTurboquantIndexConfig

    metric_str = parameters["metric"].lower()
    metric = {
        "cosine": MetricType.COSINE,
        "ip": MetricType.IP,
        "l2": MetricType.L2,
    }[metric_str]

    run(
        db=DB.TantivyTurboquant,
        db_config=TantivyTurboquantConfig(
            db_label=parameters["db_label"],
            index_path=parameters["index_path"],
        ),
        db_case_config=TantivyTurboquantIndexConfig(
            metric_type=metric,
            heap_mb=parameters["heap_mb"],
        ),
        **parameters,
    )
