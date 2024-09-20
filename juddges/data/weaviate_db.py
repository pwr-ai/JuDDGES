import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import weaviate
import weaviate.classes.config as wvcc
from weaviate.auth import Auth, _APIKey


class WeaviateDatabase(ABC):
    def __init__(self, host: str, port: str, grpc_port: str, api_key: str | None):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.__api_key = api_key

        self.client: weaviate.WeaviateClient

    def __enter__(self) -> "WeaviateDatabase":
        self.client = weaviate.connect_to_local(
            host=self.host,
            port=self.port,
            grpc_port=self.grpc_port,
            auth_credentials=self.api_key,
        )
        self.create_collections()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if hasattr(self, "client"):
            self.client.close()

    def __del__(self) -> None:
        self.__exit__(None, None, None)

    @property
    def api_key(self) -> _APIKey | None:
        if self.__api_key is not None:
            return Auth.api_key(self.__api_key)
        return None

    @abstractmethod
    def create_collections(self) -> None:
        pass

    def insert_batch(
        self,
        collection: weaviate.collections.Collection,
        objects: list[dict[str, Any]],
    ) -> None:
        with collection.batch.dynamic() as wv_batch:
            for obj in objects:
                wv_batch.add_object(**obj)
                if wv_batch.number_errors > 0:
                    break
            if wv_batch.number_errors > 0:
                errors = [err.message for err in collection.batch.results.objs.errors.values()]
                raise ValueError(f"Error ingesting batch: {errors}")

    def get_uuids(self, collection: weaviate.collections.Collection) -> list[str]:
        return [str(obj.uuid) for obj in collection.iterator(return_properties=[])]

    def _safe_create_collection(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.client.collections.create(*args, **kwargs)
        except weaviate.exceptions.UnexpectedStatusCodeError as err:
            if (
                re.search(r"class name (\w+?) already exists", err.message)
                and err.status_code == 422
            ):
                pass
            else:
                raise


class WeaviateJudgementsDatabase(WeaviateDatabase):
    JUDGMENTS_COLLECTION: ClassVar[str] = "judgements"
    JUDGMENT_CHUNKS_COLLECTION: ClassVar[str] = "judgement_chunks"

    @property
    def judgements_collection(self) -> weaviate.collections.Collection:
        return self.client.collections.get(self.JUDGMENTS_COLLECTION)

    @property
    def judgement_chunks_collection(self) -> weaviate.collections.Collection:
        return self.client.collections.get(self.JUDGMENT_CHUNKS_COLLECTION)

    def create_collections(self) -> None:
        self._safe_create_collection(
            name=self.JUDGMENTS_COLLECTION,
            properties=[
                wvcc.Property(name="judgement_id", data_type=wvcc.DataType.TEXT),
            ],
        )
        self._safe_create_collection(
            name=self.JUDGMENT_CHUNKS_COLLECTION,
            properties=[
                wvcc.Property(name="chunk_id", data_type=wvcc.DataType.INT),
                wvcc.Property(name="chunk_text", data_type=wvcc.DataType.TEXT),
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
            references=[
                wvcc.ReferenceProperty(
                    name="judgementChunk",
                    target_collection=self.JUDGMENTS_COLLECTION,
                )
            ],
        )

    @staticmethod
    def uuid_from_judgement_chunk_id(judgement_id: str, chunk_id: int) -> str:
        return weaviate.util.generate_uuid5(f"{judgement_id}_chunk_{chunk_id}")
