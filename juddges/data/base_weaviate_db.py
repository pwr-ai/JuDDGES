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
            weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
            weaviate_scheme = os.getenv("WEAVIATE_SCHEME", "http")

            # Check if using public instance
            if weaviate_host not in ["localhost", "127.0.0.1", "weaviate"]:
                # Public instance - use REST API only (no GRPC)
                logger.info(f"Connecting to public Weaviate instance: {weaviate_host} (REST-only on port 8084)")

                # Use connect_to_custom with dummy GRPC values
                # The client will fall back to REST when GRPC is unavailable
                self.client = weaviate.connect_to_custom(
                    http_host=weaviate_host,
                    http_port=int(os.getenv("WEAVIATE_PORT", "8084")),
                    http_secure=False,
                    grpc_host=weaviate_host,  # Same host but GRPC won't be used
                    grpc_port=8085,  # Dummy port, won't be used
                    grpc_secure=False,
                    auth_credentials=weaviate.auth.AuthApiKey(
                        api_key=os.getenv("WEAVIATE_API_KEY", "")
                    ),
                    skip_init_checks=True,  # Skip init checks to avoid GRPC requirement
                )

                # Force disable GRPC by redirecting grpc_search to http_search
                try:
                    if hasattr(self.client, '_connection'):
                        # Store original http_search method
                        original_http_search = self.client._connection.http_search

                        # Replace grpc_search with http_search
                        self.client._connection.grpc_search = original_http_search

                        # Also set grpc stub to None
                        if hasattr(self.client._connection, '_grpc_stub'):
                            self.client._connection._grpc_stub = None

                        logger.info("Disabled GRPC connection - redirected all queries to REST API")
                except Exception as e:
                    logger.warning(f"Could not redirect GRPC to HTTP: {e}")

                logger.info("Connected using REST API (GRPC disabled)")
            else:
                # Local instance - use HTTP and GRPC
                logger.info(f"Connecting to local Weaviate instance: {weaviate_host}")
                self.client = weaviate.connect_to_custom(
                    http_host=weaviate_host,
                    http_port=int(os.getenv("WEAVIATE_PORT", "8084")),
                    http_secure=False,
                    grpc_host=weaviate_host,
                    grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "8085")),
                    grpc_secure=False,
                    auth_credentials=weaviate.auth.AuthApiKey(
                        api_key=os.getenv("WEAVIATE_API_KEY", "")
                    ),
                    skip_init_checks=True,
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
