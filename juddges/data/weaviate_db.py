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
        await self.create_collections()
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
    async def create_collections(self) -> None:
        pass

    async def insert_batch(
        self,
        collection: weaviate.collections.Collection,
        objects: list[dict[str, Any]],
    ) -> None:
        response = await collection.data.insert_many(objects)
        if response.has_errors:
            errors = [err.message for err in response.errors]
            raise ValueError(f"Error ingesting batch: {errors}")

    async def get_uuids(self, collection: weaviate.collections.Collection) -> list[str]:
        result = []
        async for obj in collection.iterator(return_properties=[]):
            result.append(str(obj.uuid))
        return result

    async def _safe_create_collection(self, *args: Any, **kwargs: Any) -> None:
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

    async def create_collections(self) -> None:
        await self._safe_create_collection(
            name=self.JUDGMENTS_COLLECTION,
            properties=[
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
                    name="content",
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
                    name="text_legal_bases",
                    data_type=wvcc.DataType.OBJECT_ARRAY,
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
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
        )
        await self._safe_create_collection(
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

    async def update_judgments_vectorizer(
        self, vectorizer_type: str = "text2vec_transformers"
    ) -> None:
        """Update the vectorizer configuration for the judgments collection.

        Args:
            vectorizer_type (str, optional): Type of vectorizer to use.
                Defaults to "text2vec_transformers".

        Returns:
            None

        Raises:
            ValueError: If an unsupported vectorizer type is provided.
        """
        logger.info(
            f"Updating vectorizer for {self.JUDGMENTS_COLLECTION} to {vectorizer_type}"
        )

        if vectorizer_type == "text2vec_transformers":
            vectorizer_config = wvcc.Configure.Vectorizer.text2vec_transformers()
        elif vectorizer_type == "none":
            vectorizer_config = wvcc.Configure.Vectorizer.none()
        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

        try:
            await self.judgments_collection.config.update_vectorizer(
                vectorizer_config=vectorizer_config
            )
            logger.info(
                f"Successfully updated vectorizer for {self.JUDGMENTS_COLLECTION}"
            )
        except Exception as e:
            logger.error(f"Failed to update vectorizer: {str(e)}")
            raise

    @staticmethod
    def uuid_from_judgment_chunk_id(judgment_id: str, chunk_id: int) -> str:
        return weaviate.util.generate_uuid5(f"{judgment_id}_chunk_{chunk_id}")


async def main():
    """Run the update vectorizer functionality."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update Weaviate vectorizer configuration"
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="text2vec_transformers",
        choices=["text2vec_transformers", "none"],
        help="Vectorizer type to use (default: text2vec_transformers)",
    )
    args = parser.parse_args()

    try:
        async with WeaviateJudgmentsDatabase() as db:
            await db.update_judgments_vectorizer(vectorizer_type=args.vectorizer)
    except Exception as e:
        logger.error(f"Error updating vectorizer: {e}")
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
