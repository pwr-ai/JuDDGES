def run_hybrid_search(
    collection,
    collection_name: str,
    embedding,
    query: str,
    limit: int = 10,
    vector_priority: float = 1,
    text_priority: float = 1,
):
    num_candidates = limit * 10
    
    vector_search = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": num_candidates,
            "limit": limit,
        }
    }

    make_array = {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}}

    add_rank = {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}}

    def make_compute_score_doc(priority, score_field_name):
        return {
            "$addFields": {
                score_field_name: {"$divide": [1.0, {"$add": ["$rank", priority, 1]}]}
            }
        }

    def make_projection_doc(score_field_name):
        return {
            "$project": {
                score_field_name: 1,
                "_id": "$docs._id",
                "excerpt": "$docs.excerpt",
                "text": "$docs.text",
                "department_name": "$docs.department_name",
                "signature": "$docs.signature",
            }
        }

    text_search = {
        "$search": {
            "index": "text_index",
            "text": {"query": query, "path": "text"},
        }
    }

    limit_results = {"$limit": limit}

    combine_search_results = {
        "$group": {
            "_id": "$_id",
            "vs_score": {"$max": "$vs_score"},
            "ts_score": {"$max": "$ts_score"},
            "excerpt": {"$first": "$excerpt"},
            "text": {"$first": "$text"},
            "department_name": {"$first": "$department_name"},
            "signature": {"$first": "$signature"},
        }
    }

    project_combined_results = {
        "$project": {
            "_id": 1,
            "excerpt": 1,
            "text": 1,
            "department_name": 1,
            "signature": 1,
            "score": {
                "$let": {
                    "vars": {
                        "vs_score": {"$ifNull": ["$vs_score", 0]},
                        "ts_score": {"$ifNull": ["$ts_score", 0]},
                    },
                    "in": {"$add": ["$$vs_score", "$$ts_score"]},
                }
            },
        }
    }

    sort_results = {"$sort": {"score": -1}}

    pipeline = [
        vector_search,
        make_array,
        add_rank,
        make_compute_score_doc(vector_priority, "vs_score"),
        make_projection_doc("vs_score"),
        {
            "$unionWith": {
                "coll": collection_name,
                "pipeline": [
                    text_search,
                    limit_results,
                    make_array,
                    add_rank,
                    make_compute_score_doc(text_priority, "ts_score"),
                    make_projection_doc("ts_score"),
                ],
            }
        },
        combine_search_results,
        project_combined_results,
        sort_results,
        limit_results,
    ]

    return collection.aggregate(pipeline)
