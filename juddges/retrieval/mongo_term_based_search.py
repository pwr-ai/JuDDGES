from juddges.data.database import get_mongo_collection
from juddges.data_models import Judgment


def search_judgements(query: str, max_docs: int = 100):
    collection = get_mongo_collection()
    return list(
        collection.aggregate(
            [
                {
                    "$search": {
                        "index": "text",
                        "text": {"query": query, "path": Judgment.TEXT.value},
                        "highlight": {"path": Judgment.TEXT.value},
                    }
                },
                {"$limit": max_docs},
                {
                    "$project": {
                        "_id": 0,
                        "signature": 1,
                        "text": 1,
                        "excerpt": 1,
                        "score": {"$meta": "searchScore"},
                        "highlights": {"$meta": "searchHighlights"},
                    }
                },
            ]
        )
    )


def search_judgements_by_signature(
    signature: str, max_docs: int = 100, max_distance: int = 2
):
    collection = get_mongo_collection()
    return list(
        collection.aggregate(
            [
                {
                    "$search": {
                        "index": "text",
                        "text": {
                            "query": signature,
                            "path": Judgment.SIGNATURE.value,
                            "fuzzy": {"maxEdits": 1},
                        },
                        "highlight": {"path": Judgment.SIGNATURE.value},
                    }
                },
                {"$limit": max_docs},
                {
                    "$project": {
                        "_id": 0,
                        "signature": 1,
                        "text": 1,
                        "excerpt": 1,
                        "score": {"$meta": "searchScore"},
                        "highlights": {"$meta": "searchHighlights"},
                    }
                },
            ]
        )
    )
