import os
import warnings
from pathlib import Path
from typing import Any, Callable, Generator, Iterator

import pandas as pd
from loguru import logger
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import BulkWriteError
from pymongo.results import BulkWriteResult
from tqdm import tqdm, trange


def get_mongo_collection(
    mongo_uri: str | None = None,
    mongo_db: str | None = None,
    collection_name: str = "pl-court",
) -> Collection:
    """Legacy"""
    warnings.warn(
        "get_mongo_collection is deprecated and will be removed in a future version. "
        "Use MongoInterface class instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    uri = mongo_uri or os.environ.get("MONGO_URI")
    assert uri, "Mongo URI is required"
    db_name = mongo_db or os.environ.get("MONGO_DB_NAME")
    assert db_name, "Mongo DB name is required"

    client: MongoClient = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]


class MongoInterface:
    def __init__(self, uri: str, db_name: str, collection_name: str, batch_size: int = 0) -> None:
        self.client: MongoClient | None = None
        self.collection: Collection | None = None
        self.batch_size = batch_size

        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name

    def __enter__(self) -> "MongoInterface":
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        if self.client:
            self.client.close()

    def update_documents(
        self,
        documents: list[dict[str, Any]],
        upsert: bool = True,
    ) -> BulkWriteResult:
        assert self.collection is not None, "Collection not initialized"

        update_batch = [
            UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=upsert) for doc in documents
        ]

        try:
            write_results = self.collection.bulk_write(update_batch, ordered=False)
        except BulkWriteError as err:
            logger.error(err)
            raise
        else:
            if write_results.matched_count != write_results.modified_count:
                logger.error(
                    f"Matched count {write_results.matched_count} != modified count {write_results.modified_count}"
                )
            return write_results

    def dump_collection(
        self,
        file_name: Path,
        shard_size: int,
        filter_query: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        assert self.collection is not None, "Collection not initialized"

        if filter_query is None:
            query = {}
        else:
            query = filter_query

        total_num_docs = self.collection.count_documents(query)
        for offset in trange(0, total_num_docs, shard_size, desc="Chunks", disable=not verbose):
            docs = list(
                tqdm(
                    self.collection.find(query, {"embedding": 0}, batch_size=self.batch_size)
                    .skip(offset)
                    .limit(shard_size),
                    total=shard_size,
                    leave=False,
                    desc="Documents in chunk",
                    disable=not verbose,
                )
            )
            shard_idx = offset // shard_size
            dumped_f_name = self._save_docs_shard(docs, file_name, shard_idx)
            logger.info(f"Dumped {shard_idx}-th batch of documents to {dumped_f_name}")

    @staticmethod
    def _save_docs_shard(
        docs: list[dict[str, Any]], file_name: Path, shard_idx: int | None
    ) -> Path:
        if shard_idx is not None:
            file_name = file_name.with_name(f"{file_name.stem}_{shard_idx:02d}{file_name.suffix}")
        pd.DataFrame(docs).to_parquet(file_name)
        return file_name


class BatchedDatabaseCursor:
    """MongoDB cursor wrapper that returns documents in batches.
    - Cursor is consumed in batches of specified size.
    - Prefetch option loads all documents into memory before iterating.
    """

    def __init__(self, cursor: Cursor, batch_size: int, prefetch: bool) -> None:
        self.cursor = cursor
        self.batch_size = batch_size
        self.prefetch = prefetch

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        if self.prefetch:
            logger.info("Prefetching document ids from database")
            iterable = [batch for batch in self.cursor]
        else:
            iterable = self.cursor

        def gen_batches() -> Generator[list[dict[str, Any]], None, None]:
            """Credit: https://stackoverflow.com/a/61809417"""
            chunk: list[dict[str, Any]] = []
            for i, row in enumerate(iterable):
                if i % self.batch_size == 0 and i > 0:
                    yield chunk
                    del chunk[:]
                chunk.append(row)
            yield chunk

        return gen_batches()


class BatchDatabaseUpdate:
    """Updates database in batches using provided update function.
    - Update function takes document id and returns dictionary with updated fields:
        def update_func (document: dict[str, Any]) -> dict[str, Any]:
    - Updated document may be constrained to only necessary fields (_id must be present).
    - Update fields may or may not be already present in the database.
    - Update is called specified documents.
    """

    def __init__(
        self,
        mongo_uri: str,
        mongo_db_name: str,
        mongo_collection_name: str,
        update_func: Callable[[dict[str, Any]], dict] | None = None,
    ) -> None:
        self.mongo_uri = mongo_uri
        self.mongo_db_name = mongo_db_name
        self.mongo_collection_name = mongo_collection_name
        self.update_func = update_func

    def __call__(self, documents: list[dict[str, Any]]) -> BulkWriteResult:
        update_batch: list[dict[str, Any]] = []

        for doc in documents:
            if self.update_func is not None:
                update_data = self.update_func(doc)
            else:
                update_data = doc

            update_batch.append(update_data)

        with MongoInterface(
            uri=self.mongo_uri,
            db_name=self.mongo_db_name,
            collection_name=self.mongo_collection_name,
        ) as db:
            return db.update_documents(update_batch)
