import os

from pymongo import MongoClient

if os.environ.get("MONGO_URI", None) is None:
    raise Exception("Missing `MONGO_URI` environment variable.")


if os.environ.get("MONGO_DB_NAME", None) is None:
    raise Exception("Missing `MONGO_DB_NAME` environment variable.")


def get_mongo_collection(collection_name: str):
    client = MongoClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB_NAME"]]
    return db[collection_name]
