import os
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional

import weaviate.classes.config as wvcc
from dotenv import load_dotenv
from loguru import logger
from weaviate.classes.init import Auth

import weaviate
from juddges.settings import ROOT_PATH

logger.info(f"Environment variables loaded from {ROOT_PATH / '.env'} file")
load_dotenv(ROOT_PATH / ".env", override=True)


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

        self.client: weaviate.WeaviateAsyncClient

    async def __aenter__(self) -> "WeaviateDatabase":
        self.client = weaviate.use_async_with_custom(
            http_host=self.host,
            http_port=self.port,
            http_secure=False,
            grpc_host=self.host,
            grpc_port=self.grpc_port,
            grpc_secure=False,
            auth_credentials=self.api_key,
        )
        await self.client.connect()
        await self.async_create_collections()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self.client and self.client.is_connected():
            await self.client.close()

    async def close(self) -> None:
        await self.__aexit__(None, None, None)

    @property
    def api_key(self):
        if self.__api_key is not None:
            return Auth.api_key(self.__api_key)
        logger.error("No API key provided")
        return None

    @abstractmethod
    async def async_create_collections(self) -> None:
        pass

    async def async_insert_batch(
        self,
        collection: weaviate.collections.Collection,
        objects: list[dict[str, Any]],
    ) -> None:
        try:
            response = await collection.data.insert_many(objects)
            if response.has_errors:
                # Log each error with detailed information
                for i, err in enumerate(response.errors):
                    obj_id = objects[i].get("id", "unknown")
                    logger.error(f"Error inserting object {obj_id}: {err.message}")
                # Raise a consolidated error
                errors = [f"Object {i}: {err.message}" for i, err in enumerate(response.errors)]
                raise ValueError(f"Error ingesting batch: {errors}")
        except Exception as e:
            logger.error(f"Exception during batch insertion: {str(e)}")
            raise

    async def async_get_uuids(self, collection: weaviate.collections.Collection) -> list[str]:
        result = []
        async for obj in collection.iterator(return_properties=[]):
            result.append(str(obj.uuid))
        return result

    async def async_safe_create_collection(self, *args: Any, **kwargs: Any) -> None:
        try:
            await self.client.collections.create(*args, **kwargs)
        except weaviate.exceptions.UnexpectedStatusCodeError as err:
            if (
                re.search(r"class name (\w+?) already exists", err.message)
                and err.status_code == 422
            ):
                pass
            else:
                raise


class WeaviateJudgmentsDatabase(WeaviateDatabase):
    JUDGMENTS_COLLECTION: ClassVar[str] = "judgments"
    JUDGMENT_CHUNKS_COLLECTION: ClassVar[str] = "judgment_chunks"

    @property
    def judgments_collection(self) -> weaviate.collections.Collection:
        return self.client.collections.get(self.JUDGMENTS_COLLECTION)

    @property
    def judgment_chunks_collection(self) -> weaviate.collections.Collection:
        return self.client.collections.get(self.JUDGMENT_CHUNKS_COLLECTION)

    @property
    async def judgments_properties(self) -> list[str]:
        """Get list of property names for the judgments collection.

        Returns:
            list[str]: List of property names in the judgments collection.
        """
        config = await self.judgments_collection.config.get()
        return [prop.name for prop in config.properties]

    @property
    async def judgment_chunks_properties(self) -> list[str]:
        """Get list of property names for the judgment chunks collection.

        Returns:
            list[str]: List of property names in the judgment chunks collection.
        """
        config = await self.judgment_chunks_collection.config.get()
        return [prop.name for prop in config.properties]

    def get_collection(self, collection_name: str) -> weaviate.collections.Collection:
        return self.client.collections.get(collection_name)

    async def async_create_collections(self) -> None:
        await self.async_safe_create_collection(
            name=self.JUDGMENTS_COLLECTION,
            properties=[
                wvcc.Property(
                    name="judgment_id",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="source",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="docket_number",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="judgment_date",
                    data_type=wvcc.DataType.DATE,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="publication_date",
                    data_type=wvcc.DataType.DATE,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="last_update",
                    data_type=wvcc.DataType.DATE,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="court_id",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="department_id",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="judgment_type",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="excerpt",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="xml_content",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="presiding_judge",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="decision",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="judges",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="legal_bases",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="publisher",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="recorder",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="reviser",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="keywords",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="num_pages",
                    data_type=wvcc.DataType.INT,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="full_text",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="volume_number",
                    data_type=wvcc.DataType.INT,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="volume_type",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="court_name",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="department_name",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="extracted_legal_bases",
                    data_type=wvcc.DataType.OBJECT_ARRAY,
                    index_filterable=True,
                    index_searchable=True,
                    nested_properties=[
                        wvcc.Property(
                            name="text",
                            data_type=wvcc.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                        wvcc.Property(
                            name="art",
                            data_type=wvcc.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                        wvcc.Property(
                            name="isap_id",
                            data_type=wvcc.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                        wvcc.Property(
                            name="title",
                            data_type=wvcc.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                        wvcc.Property(
                            name="address",
                            data_type=wvcc.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                    ],
                ),
                wvcc.Property(
                    name="thesis",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="references",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="country",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="court_type",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="submission_date",
                    data_type=wvcc.DataType.DATE,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="finality",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="related_docket_numbers",
                    data_type=wvcc.DataType.OBJECT_ARRAY,
                    index_filterable=True,
                    index_searchable=True,
                    nested_properties=[
                        wvcc.Property(
                            name="judgment_id",
                            data_type=wvcc.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                        wvcc.Property(
                            name="docket_number",
                            data_type=wvcc.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                        wvcc.Property(
                            name="judgment_date",
                            data_type=wvcc.DataType.DATE,
                            index_filterable=True,
                        ),
                        wvcc.Property(
                            name="judgment_type",
                            data_type=wvcc.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                    ],
                ),
                wvcc.Property(
                    name="challenged_authority",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="official_collection",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="glosa_information",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="reasons_for_judgment",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="dissenting_opinion",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="judge_rapporteur",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
        )
        await self.async_safe_create_collection(
            name=self.JUDGMENT_CHUNKS_COLLECTION,
            properties=[
                wvcc.Property(
                    name="judgment_id",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
                wvcc.Property(
                    name="chunk_id",
                    data_type=wvcc.DataType.INT,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="chunk_text",
                    data_type=wvcc.DataType.TEXT,
                    index_filterable=True,
                    index_searchable=True,
                ),
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_transformers(),
            references=[
                wvcc.ReferenceProperty(
                    name="judgment_chunks",
                    target_collection=self.JUDGMENTS_COLLECTION,
                )
            ],
        )

    # For backward compatibility
    async def create_collections(self) -> None:
        await self.async_create_collections()
        
    async def insert_batch(self, collection, objects):
        await self.async_insert_batch(collection, objects)
        
    async def get_uuids(self, collection):
        return await self.async_get_uuids(collection)
        
    async def _safe_create_collection(self, *args, **kwargs):
        await self.async_safe_create_collection(*args, **kwargs)

    @staticmethod
    def uuid_from_judgment_chunk_id(judgment_id: str, chunk_id: int) -> str:
        return weaviate.util.generate_uuid5(f"{judgment_id}_chunk_{chunk_id}")
