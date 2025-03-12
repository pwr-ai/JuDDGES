import os
from pprint import pprint

from dotenv import load_dotenv
from weaviate.collections.classes.grpc import MetadataQuery

import weaviate

load_dotenv()
WV_HOST = os.getenv("WV_URL", "localhost")
WV_PORT = int(os.getenv("WV_PORT", 8080))
WV_GRPC_PORT = int(os.getenv("WV_GRPC_PORT", 50051))
WV_API_KEY = os.getenv("WV_API_KEY", None)

QUERY_PROMPT = "zapytanie: {query}"

# NOTE: This is standalone example, for convenience you can use judgments/data/weaviate_db.py
with weaviate.connect_to_local(
    host=WV_HOST,
    port=WV_PORT,
    grpc_port=WV_GRPC_PORT,
    auth_credentials=weaviate.auth.Auth.api_key(WV_API_KEY),
) as client:
    coll = client.collections.get("judgment_chunks")
    response = coll.query.hybrid(
        query=QUERY_PROMPT.format(query="oskarżony handlował narkotykami"),
        limit=2,
        return_metadata=MetadataQuery(distance=True),
    )

for o in response.objects:
    print(
        f"{o.properties['judgment_id']} - {o.properties['chunk_id']}".center(
            100,
            "=",
        )
    )
    pprint(o.properties["chunk_text"])
