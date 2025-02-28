import os
from typing import Any, Callable, Generator, Iterator

from loguru import logger
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import BulkWriteError
from pymongo.results import BulkWriteResult


def get_mongo_collection(
    mongo_uri: str | None = None,
    mongo_db: str | None = None,
    collection_name: str = "pl-court",
) -> Collection:
    uri = mongo_uri or os.environ.get("MONGO_URI")
    assert uri, "Mongo URI is required"
    db_name = mongo_db or os.environ.get("MONGO_DB_NAME")
    assert db_name, "Mongo DB name is required"

    client: MongoClient = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]


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
        update_batch: list[UpdateOne] = []

        for doc in documents:
            if self.update_func is not None:
                update_data = self.update_func(doc)
            else:
                update_data = doc

            update_batch.append(UpdateOne({"_id": doc["_id"]}, {"$set": update_data}, upsert=True))

        collection = get_mongo_collection(
            mongo_uri=self.mongo_uri,
            mongo_db=self.mongo_db_name,
            collection_name=self.mongo_collection_name,
        )

        try:
            write_results = collection.bulk_write(update_batch, ordered=False)
        except BulkWriteError as err:
            logger.error(err)
            raise
        else:
            if write_results.matched_count != write_results.modified_count:
                logger.error(
                    f"Matched count {write_results.matched_count} != modified count {write_results.modified_count}"
                )
            return write_results
