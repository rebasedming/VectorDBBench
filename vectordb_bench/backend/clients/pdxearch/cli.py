from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class PDXearchTypedDict(CommonTypedDict):
    extension_path: Annotated[
        str,
        click.option(
            "--extension-path",
            type=click.Path(exists=True, dir_okay=False),
            required=True,
            help="Absolute path to the locally built pdxearch.duckdb_extension binary",
        ),
    ]
    db_path: Annotated[
        str,
        click.option(
            "--db-path",
            type=click.Path(),
            default="/tmp/pdxearch_bench.duckdb",
            show_default=True,
            help="Path to the on-disk DuckDB file (table persists; index is rebuilt per subprocess)",
        ),
    ]
    table_name: Annotated[
        str,
        click.option(
            "--table-name",
            type=str,
            default="pdxearch_bench",
            show_default=True,
            help="Table name to load vectors into",
        ),
    ]
    metric: Annotated[
        str,
        click.option(
            "--metric",
            type=click.Choice(["cosine", "l2sq"], case_sensitive=False),
            default="cosine",
            show_default=True,
            help="Distance metric for the PDXEARCH index",
        ),
    ]
    quantization: Annotated[
        str,
        click.option(
            "--quantization",
            type=click.Choice(["u8", "f32"], case_sensitive=False),
            default="u8",
            show_default=True,
            help="Index quantization (u8 = 8-bit scalar quantization)",
        ),
    ]
    n_probe: Annotated[
        int,
        click.option(
            "--n-probe",
            type=int,
            default=24,
            show_default=True,
            help="Build-time n_probe (clusters probed per row group)",
        ),
    ]
    n_probe_search: Annotated[
        int | None,
        click.option(
            "--n-probe-search",
            type=int,
            default=None,
            help="Override n_probe at search time via SET pdxearch_n_probe",
        ),
    ]
    seed: Annotated[
        int | None,
        click.option(
            "--seed",
            type=int,
            default=None,
            help="Optional reproducibility seed for index build",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(PDXearchTypedDict)
def PDXearch(**parameters: Unpack[PDXearchTypedDict]):
    from ..api import MetricType
    from .config import PDXearchConfig, PDXearchIndexConfig

    metric = MetricType.COSINE if parameters["metric"].lower() == "cosine" else MetricType.L2

    run(
        db=DB.PDXearch,
        db_config=PDXearchConfig(
            db_label=parameters["db_label"],
            extension_path=parameters["extension_path"],
            db_path=parameters["db_path"],
            table_name=parameters["table_name"],
        ),
        db_case_config=PDXearchIndexConfig(
            metric_type=metric,
            quantization=parameters["quantization"].lower(),
            n_probe=parameters["n_probe"],
            n_probe_search=parameters["n_probe_search"],
            seed=parameters["seed"],
        ),
        **parameters,
    )
