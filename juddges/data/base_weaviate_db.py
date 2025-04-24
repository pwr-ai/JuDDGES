import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, List

from dotenv import load_dotenv
from loguru import logger
import weaviate
from weaviate import WeaviateClient
from weaviate.collections import Collection
from weaviate.collections.classes.config import Configure, Property
from weaviate.connect import ConnectionParams

from juddges.settings import ROOT_PATH

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
            connection_params = ConnectionParams.from_params(
                host=os.getenv("WEAVIATE_HOST", "localhost"),
                port=int(os.getenv("WEAVIATE_PORT", "8080")),
                scheme=os.getenv("WEAVIATE_SCHEME", "http"),
                headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")},
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
                timeout_config=60
            )
            self.client = weaviate.WeaviateClient(connection_params=connection_params)
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

    def safe_create_collection(
        self,
        name: str,
        description: str = "",
        properties: Optional[List[Property]] = None,
        vectorizer_config: Optional[dict] = None
    ) -> Collection:
        """Safely create a collection if it doesn't exist.
        
        Args:
            name: Name of the collection
            description: Optional description
            properties: List of property configurations
            vectorizer_config: Optional vectorizer configuration
            
        Returns:
            The created or existing collection
        """
        try:
            # Check if collection exists
            if self.client.collections.exists(name):
                logger.info(f"Collection {name} already exists")
                self._collection = self.client.collections.get(name)
                return self._collection

            # Create new collection
            logger.info(f"Creating collection {name}")
            config = Configure.new(
                name=name,
                description=description,
                vectorizer_config=vectorizer_config or {"none": {"vectorizeClassName": False}},
                properties=properties or []
            )
            self._collection = self.client.collections.create(config)
            logger.info(f"Successfully created collection {name}")
            return self._collection

        except Exception as e:
            logger.error(f"Error creating collection {name}: {str(e)}")
            raise

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
                            uuid=obj.get("id")
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