from pathlib import Path
from pprint import pformat

import typer
from dotenv import load_dotenv
from loguru import logger

from juddges.data.database import MongoInterface
from juddges.schema import POLARS_SCHEMAS

BATCH_SIZE = 100
SHARD_SIZE = 50_000

load_dotenv()


def main(
    uri: str = typer.Option(..., envvar="MONGO_URI"),
    db: str = typer.Option(...),
    collection: str = typer.Option(...),
    file_name: Path = typer.Option(..., exists=False),
    filter_empty_content: bool = typer.Option(True),
    ignore_mongo_id: bool = typer.Option(True),
    batch_size: int = typer.Option(BATCH_SIZE),
    shard_size: int = typer.Option(SHARD_SIZE),
    enforce_schema: str | None = typer.Option(
        None,
        help="Name of the schema in juddges.schema.SCHEMAS",
    ),
) -> None:
    filter_query = {}
    if filter_empty_content:
        filter_query["full_text"] = {"$exists": True}

    fields_to_ignore = ["embedding"]
    if ignore_mongo_id:
        fields_to_ignore.append("_id")

    if enforce_schema is not None:
        schema = POLARS_SCHEMAS[enforce_schema]
        logger.info(f"Dumping collection with schema:\n{pformat(schema)}")
    else:
        schema = None

    with MongoInterface(
        uri=uri,
        db_name=db,
        collection_name=collection,
        batch_size=batch_size,
    ) as db:
        db.dump_collection(
            file_name=file_name,
            shard_size=shard_size,
            clean_legacy_shards=False,
            filter_query=filter_query,
            fields_to_ignore=fields_to_ignore,
            schema=schema,
        )


if __name__ == "__main__":
    typer.run(main)
