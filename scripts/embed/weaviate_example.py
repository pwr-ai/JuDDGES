import os
from pprint import pprint

from dotenv import load_dotenv

import weaviate
from weaviate.collections.classes.grpc import MetadataQuery

load_dotenv()
WV_HOST = os.environ["WV_URL"]
WV_PORT = int(os.environ["WV_PORT"])
WV_GRPC_PORT = int(os.environ["WV_GRPC_PORT"])
WV_API_KEY = os.environ["WV_API_KEY"]

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
        query=QUERY_PROMPT.format(query="drug dealer"),
        target_vector="base",
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
