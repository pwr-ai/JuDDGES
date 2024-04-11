import os

from pymongo import MongoClient
from pymongo.collection import Collection


def get_mongo_collection(
    mongo_uri: str | None = None,
    mongo_db: str | None = None,
    collection_name: str = "judgements",
) -> Collection:
    uri = mongo_uri or os.environ.get("MONGO_URI")
    assert uri, "Mongo URI is required"
    db = mongo_db or os.environ.get("MONGO_DB_NAME")
    assert db, "Mongo DB name is required"

    client = MongoClient(uri)
    db = client[db]
    return db[collection_name]
