import os
from abc import ABC, abstractmethod
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger

import weaviate
from juddges.settings import ROOT_PATH
from weaviate import WeaviateClient
from weaviate.collections import Collection

logger.info(f"Environment variables loaded from {ROOT_PATH / '.env'} file")
load_dotenv(ROOT_PATH / ".env", override=True)


class BaseWeaviateDB(ABC):
    def __init__(self, client: Optional[WeaviateClient] = None):
        """Initialize the Weaviate database connection.

        Args:
            client: Optional pre-configured Weaviate client. If not provided, one will be created.
        """
        self.client = client
        self._collection = None

    def __enter__(self):
        """Set up the database connection when entering context."""
        if self.client is None:
            self.client = weaviate.connect_to_custom(
                http_host=os.getenv("WEAVIATE_HOST", "localhost"),
                http_port=int(os.getenv("WEAVIATE_PORT", "8084")),
                http_secure=False,
                grpc_host=os.getenv("WEAVIATE_HOST", "localhost"),
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
                grpc_secure=False,
                auth_credentials=weaviate.auth.AuthApiKey(
                    api_key=os.getenv("WEAVIATE_API_KEY", "")
                ),
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        if self.client:
            self.client.close()
            self.client = None
            self._collection = None

    @property
    def collection(self) -> Collection:
        """Get the current collection."""
        if self._collection is None:
            raise ValueError("Collection not initialized. Call safe_create_collection first.")
        return self._collection

    def insert_batch(self, objects: List[dict], batch_size: int = 100) -> None:
        """Insert a batch of objects into the collection.

        Args:
            objects: List of objects to insert
            batch_size: Size of each batch
        """
        try:
            with self.collection.batch.dynamic() as batch:
                for obj in objects:
                    try:
                        batch.add_object(
                            properties=obj.get("properties", {}),
                            vector=obj.get("vector"),
                            uuid=obj.get("id"),
                        )
                    except Exception as e:
                        logger.error(f"Failed to insert object {obj.get('id')}: {str(e)}")

        except Exception as e:
            logger.error(f"Batch insertion failed: {str(e)}")
            raise

    @abstractmethod
    def insert(self, *args, **kwargs):
        """Abstract method for inserting a single object."""
        pass

    @abstractmethod
    def search(self, *args, **kwargs):
        """Abstract method for searching objects."""
        pass

    @abstractmethod
    def delete(self, *args, **kwargs):
        """Abstract method for deleting objects."""
        pass

    def get_uuids(self, collection: weaviate.collections.Collection) -> list[str]:
        result = []
        for obj in collection.iterator(return_properties=[]):
            result.append(str(obj.uuid))
        return result

    def close(self) -> None:
        self.__exit__(None, None, None)
