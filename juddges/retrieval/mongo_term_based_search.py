from juddges.data.database import get_mongo_collection
from juddges.data_models import Judgment

RETURNED_ATTRIBUTES = {
    Judgment.ID.value: 0,
    Judgment.SIGNATURE.value: 1,
    Judgment.TEXT.value: 1,
    Judgment.EXCERPT.value: 1,
    Judgment.COURT_NAME.value: 1,
    Judgment.DEPARTMENT_NAME.value: 1,
    Judgment.DATE.value: 1,
    "score": {"$meta": "searchScore"},
    "highlights": {"$meta": "searchHighlights"},
}


def search_judgments(query: str, max_docs: int = 100):
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
                {"$project": RETURNED_ATTRIBUTES},
            ]
        )
    )


def search_judgments_by_signature(signature: str, max_docs: int = 100):
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
                {"$project": RETURNED_ATTRIBUTES},
            ]
        )
    )
