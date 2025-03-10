import os
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import weaviate.classes.config as wvcc
from dotenv import load_dotenv
from loguru import logger
from weaviate.classes.init import Auth

import weaviate
from juddges.settings import ROOT_PATH

load_dotenv(ROOT_PATH / ".env", override=True)

logger.info(f"Environment variables loaded from {ROOT_PATH / '.env'} file")


class WeaviateDatabase(ABC):
    def __init__(
        self,
        host: str = os.environ["WV_URL"],
        port: str = os.environ["WV_PORT"],
        grpc_port: str = os.environ["WV_GRPC_PORT"],
        api_key: str = os.environ["WV_API_KEY"],
    ):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.__api_key = api_key

        self.client: weaviate.WeaviateClient

    def __enter__(self) -> "WeaviateDatabase":
        self.client = weaviate.connect_to_custom(
            http_host=self.host,
            http_port=self.port,
            http_secure=False,
            grpc_host=self.host,
            grpc_port=self.grpc_port,
            grpc_secure=False,
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
    def api_key(self):
        if self.__api_key is not None:
            logger.info(f"API key: {self.__api_key[:5]}...")
            return Auth.api_key(self.__api_key)
        logger.error("No API key provided")
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
                errors = [
                    err.message for err in collection.batch.results.objs.errors.values()
                ]
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
                wvcc.Property(name="docket_number", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="judgement_date", data_type=wvcc.DataType.DATE),
                wvcc.Property(name="publication_date", data_type=wvcc.DataType.DATE),
                wvcc.Property(name="last_update", data_type=wvcc.DataType.DATE),
                wvcc.Property(name="court_id", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="department_id", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="judgement_type", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="excerpt", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="content", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="presiding_judge", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="decision", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="judges", data_type=wvcc.DataType.TEXT_ARRAY),
                wvcc.Property(name="legal_bases", data_type=wvcc.DataType.TEXT_ARRAY),
                wvcc.Property(name="publisher", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="recorder", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="reviser", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="keywords", data_type=wvcc.DataType.TEXT_ARRAY),
                wvcc.Property(name="num_pages", data_type=wvcc.DataType.INT),
                wvcc.Property(name="full_text", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="volume_number", data_type=wvcc.DataType.INT),
                wvcc.Property(name="volume_type", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="court_name", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="department_name", data_type=wvcc.DataType.TEXT),
                wvcc.Property(
                    name="text_legal_bases",
                    data_type=wvcc.DataType.OBJECT_ARRAY,
                    nested_properties=[
                        wvcc.Property(name="text", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="art", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="isap_id", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="title", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="address", data_type=wvcc.DataType.TEXT),
                    ],
                ),
                wvcc.Property(name="thesis", data_type=wvcc.DataType.TEXT),
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
        )
        self._safe_create_collection(
            name=self.JUDGMENT_CHUNKS_COLLECTION,
            properties=[
                wvcc.Property(name="judgment_id", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="chunk_id", data_type=wvcc.DataType.INT),
                wvcc.Property(name="chunk_text", data_type=wvcc.DataType.TEXT),
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_transformers(),
            references=[
                wvcc.ReferenceProperty(
                    name="judgment_chunks",
                    target_collection=self.JUDGMENTS_COLLECTION,
                )
            ],
        )

    @staticmethod
    def uuid_from_judgement_chunk_id(judgement_id: str, chunk_id: int) -> str:
        return weaviate.util.generate_uuid5(f"{judgement_id}_chunk_{chunk_id}")
