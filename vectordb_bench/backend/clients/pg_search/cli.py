import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class PgSearchTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Postgres database password",
            default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
            show_default="$POSTGRES_PASSWORD",
        ),
    ]
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="Postgres database port",
            default=5432,
            show_default=True,
        ),
    ]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]
    vector_cluster_probes: Annotated[
        int,
        click.option(
            "--vector-cluster-probes",
            type=int,
            help="paradedb.vector_cluster_probes: per-segment cluster probe cap for "
            "vector ORDER BY queries. Higher = better recall, linearly more work.",
            default=50,
            show_default=True,
        ),
    ]
    vector_rerank_multiplier: Annotated[
        float,
        click.option(
            "--vector-rerank-multiplier",
            type=float,
            help="paradedb.vector_rerank_multiplier: over-fetch factor for "
            "exact-distance rerank. >1.0 engages a heap-side rerank pass that "
            "reloads full-precision vectors for ceil(k * multiplier) candidates "
            "and re-sorts by exact distance. 1.0 disables rerank.",
            default=1.0,
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(PgSearchTypedDict)
def PgSearch(**parameters: Unpack[PgSearchTypedDict]):
    from .config import PgSearchConfig, PgSearchIndexConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.PgSearch,
        db_config=PgSearchConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=PgSearchIndexConfig(
            vector_cluster_probes=parameters["vector_cluster_probes"],
            vector_rerank_multiplier=parameters["vector_rerank_multiplier"],
        ),
        **parameters,
    )
