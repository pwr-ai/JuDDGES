"""Weaviate REST API client for legal document operations.

This module provides a clean interface for interacting with Weaviate's REST API,
specifically tailored for legal document retrieval and updates.
"""

import os
from typing import Any, Dict, List, Optional

import requests
import weaviate.util
from loguru import logger


class WeaviateRestClient:
    """Client for Weaviate REST API operations on LegalDocuments collection."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8084,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        """Initialize Weaviate REST client.

        Args:
            host: Weaviate server host
            port: Weaviate server port
            api_key: Optional API key for authentication
            timeout: Default timeout for requests in seconds
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

        # Build headers
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def from_env(cls) -> "WeaviateRestClient":
        """Create client from environment variables.

        Expected env vars:
        - WEAVIATE_HOST (default: localhost)
        - WEAVIATE_PORT (default: 8084)
        - WEAVIATE_API_KEY (optional)

        Returns:
            Configured WeaviateRestClient instance
        """
        return cls(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_PORT", "8084")),
            api_key=os.getenv("WEAVIATE_API_KEY"),
        )

    def fetch_documents(
        self,
        max_documents: int = 1000,
        chunk_size: int = 1000,
        search_query: Optional[str] = None,
        document_type_filter: Optional[str] = None,
        use_cursor: bool = True,
        search_mode: str = "hybrid",
        force_cursor: bool = False,
        skip_documents: int = 0,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> List[Dict[str, Any]]:
        """Fetch documents using Weaviate REST API with cursor-based pagination.

        Args:
            max_documents: Maximum number of documents to fetch (returns all valid docs up to this limit)
            chunk_size: Number of documents to fetch per request (max 1000 per chunk)
            search_query: Optional search query for keyword/semantic/hybrid search (ignored if force_cursor=True)
            document_type_filter: Optional filter by document_type (e.g., "judgment", "tax_interpretation")
            use_cursor: Use cursor-based pagination (no 10K limit) instead of offset (default: True)
            search_mode: Search mode - "keyword" (BM25), "semantic" (vector), or "hybrid" (default: "hybrid")
            force_cursor: Skip search query and use cursor pagination to iterate through ALL documents
            skip_documents: Number of documents to skip before starting to collect results (default: 0)
            sort_by: Optional field to sort by (e.g., "_creationTimeUnix", "document_number"). Use None for no sorting.
            sort_order: Sort order - "asc" (ascending, default) or "desc" (descending)

        Returns:
            List of all valid document properties with full_text (no sampling)

        Note:
            Weaviate's cursor API doesn't support filters or search with 'after' parameter.
            If search_query or document_type_filter is provided, falls back to offset pagination.
            All valid documents up to max_documents are returned - no random sampling.
            - keyword: BM25 full-text search (best for exact term matching)
            - semantic: Vector similarity search (best for conceptual similarity)
            - hybrid: Combines BM25 + vector (alpha=0.5, balanced approach)

            RECOMMENDED: Use force_cursor=True WITHOUT search_query to iterate through all documents
            and bypass the 10K offset limit. The search_query will be ignored when force_cursor=True.

            The skip_documents parameter allows you to skip the first N documents in the result set,
            useful for resuming interrupted extraction jobs or distributing work across multiple runs.
        """
        # Weaviate limitations:
        # 1. Cursor API ('after') cannot be used with sort - they are mutually exclusive
        # 2. Hybrid/semantic search is not compatible with sort
        # 3. Sort only works with: offset pagination + (plain queries OR BM25 keyword search)
        has_filters = bool(search_query or document_type_filter)

        # If sort is requested, we MUST use offset pagination (not cursor)
        if sort_by:
            if use_cursor:
                logger.warning(
                    f"Sort requires offset pagination (cursor pagination doesn't support sorting). "
                    f"Switching to offset pagination (max 10K limit applies)."
                )
                use_cursor = False

            # Check if search mode is compatible with sort
            if search_query and search_mode in ["hybrid", "semantic"]:
                logger.warning(
                    f"Sort is not compatible with {search_mode} search in Weaviate. "
                    f"Disabling sort. To use sort, switch to search_mode='keyword' (BM25) or remove search query."
                )
                sort_by = None  # Disable sorting to avoid errors

        # If force_cursor is enabled, remove search query to ensure cursor pagination works
        if force_cursor and search_query:
            logger.info(
                f"force_cursor=True: Ignoring search query '{search_query}' to enable cursor pagination. "
                "Fetching ALL documents up to max_documents limit."
            )
            search_query = None
            has_filters = bool(document_type_filter)  # Recalculate without search_query

        if use_cursor and not has_filters:
            return self._fetch_documents_cursor(
                max_documents=max_documents,
                chunk_size=chunk_size,
                search_query=search_query,
                document_type_filter=document_type_filter,
                search_mode=search_mode,
                skip_documents=skip_documents,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        else:
            if has_filters and use_cursor:
                logger.warning(
                    "Cursor pagination not supported with filters/search. "
                    "Using offset pagination (max 10K limit). "
                    "Remove search query or use force_cursor=True to bypass this limit."
                )
            return self._fetch_documents_offset(
                max_documents=max_documents,
                chunk_size=chunk_size,
                search_query=search_query,
                document_type_filter=document_type_filter,
                search_mode=search_mode,
                skip_documents=skip_documents,
                sort_by=sort_by,
                sort_order=sort_order,
            )

    def _fetch_documents_cursor(
        self,
        max_documents: int,
        chunk_size: int,
        search_query: Optional[str],
        document_type_filter: Optional[str],
        search_mode: str = "hybrid",
        skip_documents: int = 0,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> List[Dict[str, Any]]:
        """Fetch documents using cursor-based pagination (no 10K limit).

        Args:
            max_documents: Maximum number of documents to fetch
            chunk_size: Number of documents to fetch per request
            search_query: Optional search query
            document_type_filter: Optional document type filter
            search_mode: Search mode - "keyword", "semantic", or "hybrid"
            skip_documents: Number of documents to skip before collecting results
            sort_by: Optional field to sort by (e.g., "_creationTimeUnix")
            sort_order: Sort order - "asc" or "desc"

        Returns:
            List of all valid documents up to max_documents (no sampling)
        """
        chunk_size = min(chunk_size, 1000)  # Weaviate max per request

        # Build filter info for logging
        filter_info = []
        if search_query:
            filter_info.append(f"search='{search_query}' ({search_mode})")
        if document_type_filter:
            filter_info.append(f"type={document_type_filter}")
        filter_str = f" with filters: {', '.join(filter_info)}" if filter_info else ""

        if skip_documents > 0:
            logger.info(
                f"Fetching up to {max_documents} documents from {self.base_url} in chunks of {chunk_size}{filter_str} (cursor-based), skipping first {skip_documents} documents..."
            )
        else:
            logger.info(
                f"Fetching up to {max_documents} documents from {self.base_url} in chunks of {chunk_size}{filter_str} (cursor-based)..."
            )

        all_documents = []
        skipped_count = 0
        cursor = None
        iteration = 0

        # Use larger chunks during skip phase for faster skipping
        skip_chunk_size = 1000  # Always use max size when skipping

        while len(all_documents) < max_documents or skipped_count < skip_documents:
            iteration += 1

            # If we're still skipping, use large chunks and minimal fields
            if skipped_count < skip_documents:
                # Calculate how many more to skip
                remaining_skip = skip_documents - skipped_count
                current_limit = min(skip_chunk_size, remaining_skip)
                # Use minimal query during skip phase (only IDs)
                query = self._build_graphql_query_cursor_minimal(
                    limit=current_limit,
                    cursor=cursor,
                    document_type_filter=document_type_filter,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )
            else:
                # Normal fetching after skip phase with full fields
                current_limit = min(chunk_size, max_documents - len(all_documents))
                query = self._build_graphql_query_cursor(
                    limit=current_limit,
                    cursor=cursor,
                    search_query=search_query,
                    document_type_filter=document_type_filter,
                    search_mode=search_mode,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )

            try:
                if cursor:
                    if skipped_count < skip_documents:
                        if iteration % 50 == 0:  # Log every 50th iteration to reduce noise
                            logger.info(f"Skipping chunk {iteration}: limit={current_limit}, cursor={cursor[:20]}... (skipped: {skipped_count}/{skip_documents})")
                    else:
                        logger.info(f"Fetching chunk {iteration}: limit={current_limit}, cursor={cursor[:20]}...")
                else:
                    logger.info(f"Fetching chunk {iteration}: limit={current_limit} (initial)")

                response = requests.post(
                    f"{self.base_url}/v1/graphql",
                    headers=self.headers,
                    json={"query": query},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                if "errors" in data:
                    logger.error(f"GraphQL errors: {data['errors']}")
                    raise Exception(f"GraphQL query failed: {data['errors']}")

                documents = data.get("data", {}).get("Get", {}).get("LegalDocuments", [])

                if not documents:
                    logger.info(f"No more documents available at iteration {iteration}")
                    break

                # If we're in skip phase, just count and don't add to results
                if skipped_count < skip_documents:
                    skipped_count += len(documents)
                    if iteration % 50 == 0 or skipped_count >= skip_documents:  # Log progress periodically
                        logger.info(f"Skipped {len(documents)} documents (total skipped: {skipped_count}/{skip_documents})")
                else:
                    all_documents.extend(documents)
                    logger.info(f"Fetched {len(documents)} documents (total: {len(all_documents)})")

                # Extract cursor from last document's _additional field
                if documents and documents[-1].get("_additional", {}).get("id"):
                    cursor = documents[-1]["_additional"]["id"]
                else:
                    logger.warning("No cursor found in response, stopping pagination")
                    break

                # Break if we got fewer documents than requested (end of data)
                if len(documents) < current_limit:
                    logger.info("Reached end of available documents")
                    break

            except Exception as e:
                logger.error(f"Failed to fetch chunk at iteration {iteration}: {e}")
                raise

        logger.info(f"Fetched {len(all_documents)} total documents from Weaviate using cursor pagination")

        # Filter documents with non-empty full_text
        valid_docs = [
            doc
            for doc in all_documents
            if doc.get("full_text") and len(doc.get("full_text", "").strip()) > 100
        ]

        logger.info(f"Found {len(valid_docs)} documents with valid full_text")

        if not valid_docs:
            logger.warning("No documents with valid full_text found!")
            return []

        logger.info(f"Returning all {len(valid_docs)} valid documents for extraction")
        return valid_docs

    def _fetch_documents_offset(
        self,
        max_documents: int,
        chunk_size: int,
        search_query: Optional[str],
        document_type_filter: Optional[str],
        search_mode: str = "hybrid",
        skip_documents: int = 0,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> List[Dict[str, Any]]:
        """Fetch documents using offset-based pagination (legacy, max 10K limit).

        Args:
            max_documents: Maximum number of documents to fetch (capped at 10K due to Weaviate limit)
            chunk_size: Number of documents to fetch per request
            search_query: Optional search query
            document_type_filter: Optional document type filter
            search_mode: Search mode - "keyword", "semantic", or "hybrid"
            skip_documents: Number of documents to skip before collecting results
            sort_by: Optional field to sort by (e.g., "_creationTimeUnix")
            sort_order: Sort order - "asc" or "desc"

        Returns:
            List of all valid documents up to max_documents (no sampling)
        """
        # Fetch documents in chunks with pagination
        # Note: Weaviate has a hard limit of offset < 10000
        max_offset = 10000  # Weaviate offset limit

        # Start offset at skip_documents position
        if skip_documents >= max_offset:
            logger.warning(
                f"skip_documents ({skip_documents}) exceeds Weaviate offset limit ({max_offset}). "
                "Use cursor-based pagination with force_cursor=True for skipping beyond 10K."
            )
            return []

        max_documents = min(max_documents, max_offset - skip_documents)  # Enforce Weaviate limit
        chunk_size = min(chunk_size, 1000)

        # Build filter info for logging
        filter_info = []
        if search_query:
            filter_info.append(f"search='{search_query}' ({search_mode})")
        if document_type_filter:
            filter_info.append(f"type={document_type_filter}")
        filter_str = f" with filters: {', '.join(filter_info)}" if filter_info else ""

        if skip_documents > 0:
            logger.info(
                f"Fetching up to {max_documents} documents from {self.base_url} in chunks of {chunk_size}{filter_str} (offset-based, max 10K), starting at offset {skip_documents}..."
            )
        else:
            logger.info(
                f"Fetching up to {max_documents} documents from {self.base_url} in chunks of {chunk_size}{filter_str} (offset-based, max 10K)..."
            )

        all_documents = []
        offset = skip_documents  # Start from skip position

        while len(all_documents) < max_documents and offset < max_offset:
            # Calculate how many more documents we need
            remaining = min(max_documents - len(all_documents), max_offset - offset)
            current_limit = min(chunk_size, remaining)

            # Build GraphQL query
            query = self._build_graphql_query(
                limit=current_limit,
                offset=offset,
                search_query=search_query,
                document_type_filter=document_type_filter,
                search_mode=search_mode,
                sort_by=sort_by,
                sort_order=sort_order,
            )

            try:
                logger.info(f"Fetching chunk: offset={offset}, limit={current_limit}")
                response = requests.post(
                    f"{self.base_url}/v1/graphql",
                    headers=self.headers,
                    json={"query": query},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                if "errors" in data:
                    error_msg = str(data["errors"])
                    if "query maximum results exceeded" in error_msg or offset >= max_offset:
                        logger.warning(
                            f"Reached Weaviate offset limit at {offset}. Using {len(all_documents)} documents."
                        )
                        break
                    logger.error(f"GraphQL errors: {data['errors']}")
                    raise Exception(f"GraphQL query failed: {data['errors']}")

                documents = data.get("data", {}).get("Get", {}).get("LegalDocuments", [])

                if not documents:
                    logger.info(f"No more documents available at offset {offset}")
                    break

                all_documents.extend(documents)
                logger.info(f"Fetched {len(documents)} documents (total: {len(all_documents)})")

                offset += len(documents)

                # Break if we got fewer documents than requested (end of data)
                if len(documents) < current_limit:
                    logger.info("Reached end of available documents")
                    break

            except Exception as e:
                if "query maximum results exceeded" in str(e) and len(all_documents) > 0:
                    logger.warning(
                        f"Hit Weaviate offset limit at {offset}. Continuing with {len(all_documents)} documents."
                    )
                    break
                logger.error(f"Failed to fetch chunk at offset {offset}: {e}")
                raise

        logger.info(f"Fetched {len(all_documents)} total documents from Weaviate")

        # Filter documents with non-empty full_text
        valid_docs = [
            doc
            for doc in all_documents
            if doc.get("full_text") and len(doc.get("full_text", "").strip()) > 100
        ]

        logger.info(f"Found {len(valid_docs)} documents with valid full_text")

        if not valid_docs:
            logger.warning("No documents with valid full_text found!")
            return []

        logger.info(f"Returning all {len(valid_docs)} valid documents for extraction")
        return valid_docs

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single document by its ID.

        Args:
            document_id: Document ID to fetch

        Returns:
            Document data or None if not found
        """
        weaviate_uuid = weaviate.util.generate_uuid5(document_id)
        url = f"{self.base_url}/v1/objects/LegalDocuments/{weaviate_uuid}"

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Document {document_id} not found in Weaviate")
                return None
            else:
                logger.error(f"Error fetching document {document_id}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Exception fetching document {document_id}: {e}")
            return None

    def update_document(
        self,
        document_id: str,
        properties: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> bool:
        """Update a single document with PATCH request.

        Args:
            document_id: Document ID to update
            properties: Dictionary of properties to update
            timeout: Optional timeout override

        Returns:
            True if successful, False otherwise
        """
        weaviate_uuid = weaviate.util.generate_uuid5(document_id)
        url = f"{self.base_url}/v1/objects/LegalDocuments/{weaviate_uuid}"

        try:
            response = requests.patch(
                url,
                headers=self.headers,
                json={"properties": properties},
                timeout=timeout or self.timeout,
            )
            response.raise_for_status()
            logger.debug(f"✓ Updated {document_id} with {len(properties)} properties")
            return True

        except requests.exceptions.HTTPError as e:
            # Enhanced logging for 422 validation errors
            if e.response and e.response.status_code == 422:
                logger.warning(
                    f"✗ Failed to update {document_id} with 422 validation error. "
                    f"Response: {e.response.text[:500]}"
                )
            else:
                logger.warning(f"✗ Failed to update {document_id}: {e}")
            return False

        except Exception as e:
            logger.warning(f"✗ Error updating {document_id}: {e}")
            return False

    def batch_update(
        self,
        batch_objects: List[Dict[str, Any]],
        action: str = "MERGE",
        timeout: Optional[int] = None,
    ) -> tuple[int, int, List[Dict[str, str]]]:
        """Update multiple documents using batch API.

        Args:
            batch_objects: List of objects with 'id', 'class', 'properties'
            action: Batch action - "MERGE" (update only specified fields) or "PUT" (replace all)
            timeout: Optional timeout override

        Returns:
            Tuple of (successful_count, failed_count, errors_list)
        """
        url = f"{self.base_url}/v1/batch/objects"

        successful = 0
        failed = 0
        errors = []

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json={"objects": batch_objects, "action": action},
                timeout=timeout or 60,
            )
            response.raise_for_status()
            result = response.json()

            # Process batch results
            if isinstance(result, list):
                for item_result in result:
                    if item_result.get("result", {}).get("status") == "SUCCESS":
                        successful += 1
                    else:
                        failed += 1
                        error_msg = item_result.get("result", {}).get("errors", {})
                        errors.append({
                            "document_id": item_result.get("id", "unknown"),
                            "error": str(error_msg),
                        })
            else:
                # Assume all successful if no detailed result
                successful = len(batch_objects)

        except requests.exceptions.HTTPError as e:
            # Batch failed - mark all as failed
            failed = len(batch_objects)
            for obj in batch_objects:
                errors.append({
                    "document_id": obj.get("id", "unknown"),
                    "error": f"Batch API error: {e}",
                    "status_code": e.response.status_code if e.response else None,
                })
            logger.error(f"Batch API request failed: {e}")

        except Exception as e:
            failed = len(batch_objects)
            for obj in batch_objects:
                errors.append({
                    "document_id": obj.get("id", "unknown"),
                    "error": f"Batch processing error: {e}",
                })
            logger.error(f"Batch processing failed: {e}")

        return successful, failed, errors

    def _build_graphql_query(
        self,
        limit: int,
        offset: int,
        search_query: Optional[str] = None,
        document_type_filter: Optional[str] = None,
        search_mode: str = "hybrid",
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> str:
        """Build GraphQL query for document fetching.

        Args:
            limit: Maximum number of documents to return
            offset: Offset for pagination
            search_query: Optional search query
            document_type_filter: Optional document type filter
            search_mode: Search mode - "keyword" (BM25), "semantic" (vector), or "hybrid"
            sort_by: Optional field to sort by (e.g., "_creationTimeUnix")
            sort_order: Sort order - "asc" or "desc"

        Returns:
            GraphQL query string
        """
        # Build where clause for document type filter
        where_clause = ""
        if document_type_filter:
            where_clause = f"""
                where: {{
                    path: ["document_type"],
                    operator: Equal,
                    valueText: "{document_type_filter}"
                }}
            """

        # Build sort clause
        sort_clause = ""
        if sort_by:
            # Weaviate uses lowercase for sort order
            order = sort_order.lower() if sort_order else "asc"
            sort_clause = f"""
                sort: [{{
                    path: ["{sort_by}"]
                    order: {order}
                }}]
            """

        # Build query method based on search mode
        if search_query:
            if search_mode == "keyword":
                # Pure BM25 keyword search
                query_method = f"""
                    bm25: {{
                        query: "{search_query}"
                    }}
                    {where_clause}
                    {sort_clause}
                    limit: {limit}
                    offset: {offset}
                """
            elif search_mode == "semantic":
                # Pure vector semantic search
                query_method = f"""
                    nearText: {{
                        concepts: ["{search_query}"]
                    }}
                    {where_clause}
                    {sort_clause}
                    limit: {limit}
                    offset: {offset}
                """
            else:  # hybrid (default)
                # Hybrid search (BM25 + vector)
                query_method = f"""
                    hybrid: {{
                        query: "{search_query}",
                        alpha: 0.5
                    }}
                    {where_clause}
                    {sort_clause}
                    limit: {limit}
                    offset: {offset}
                """
        else:
            # No search query - just filter
            query_method = f"""
                {where_clause}
                {sort_clause}
                limit: {limit}
                offset: {offset}
            """

        # Full GraphQL query
        return f"""
            {{
                Get {{
                    LegalDocuments({query_method}) {{
                        document_id
                        document_type
                        full_text
                        language
                        document_number
                    }}
                }}
            }}
        """

    def _build_graphql_query_cursor_minimal(
        self,
        limit: int,
        cursor: Optional[str],
        document_type_filter: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> str:
        """Build minimal GraphQL query for fast cursor-based skipping (only fetches IDs).

        Args:
            limit: Maximum number of documents to return
            cursor: Cursor (document ID) to start after (None for first page)
            document_type_filter: Optional document type filter
            sort_by: Optional field to sort by
            sort_order: Sort order - "asc" or "desc"

        Returns:
            GraphQL query string with minimal fields
        """
        # Build where clause for document type filter
        where_clause = ""
        if document_type_filter:
            where_clause = f"""
                where: {{
                    path: ["document_type"],
                    operator: Equal,
                    valueText: "{document_type_filter}"
                }}
            """

        # Build sort clause
        sort_clause = ""
        if sort_by:
            order = sort_order.lower() if sort_order else "asc"
            sort_clause = f"""
                sort: [{{
                    path: ["{sort_by}"]
                    order: {order}
                }}]
            """

        # Cursor pagination without search - minimal fields
        if cursor:
            query_method = f"""
                {where_clause}
                {sort_clause}
                limit: {limit}
                after: "{cursor}"
            """
        else:
            query_method = f"""
                {where_clause}
                {sort_clause}
                limit: {limit}
            """

        # Minimal query - only fetch _additional.id for cursor
        return f"""
            {{
                Get {{
                    LegalDocuments({query_method}) {{
                        _additional {{
                            id
                        }}
                    }}
                }}
            }}
        """

    def _build_graphql_query_cursor(
        self,
        limit: int,
        cursor: Optional[str],
        search_query: Optional[str] = None,
        document_type_filter: Optional[str] = None,
        search_mode: str = "hybrid",
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> str:
        """Build GraphQL query for cursor-based pagination.

        Args:
            limit: Maximum number of documents to return
            cursor: Cursor (document ID) to start after (None for first page)
            search_query: Optional search query
            document_type_filter: Optional document type filter
            search_mode: Search mode - "keyword" (BM25), "semantic" (vector), or "hybrid"
            sort_by: Optional field to sort by
            sort_order: Sort order - "asc" or "desc"

        Returns:
            GraphQL query string
        """
        # Build where clause for document type filter
        where_clause = ""
        if document_type_filter:
            where_clause = f"""
                where: {{
                    path: ["document_type"],
                    operator: Equal,
                    valueText: "{document_type_filter}"
                }}
            """

        # Build sort clause
        sort_clause = ""
        if sort_by:
            order = sort_order.lower() if sort_order else "asc"
            sort_clause = f"""
                sort: [{{
                    path: ["{sort_by}"]
                    order: {order}
                }}]
            """

        # Build query method based on search mode with cursor
        if search_query:
            # Build search clause based on mode
            if search_mode == "keyword":
                search_clause = f"""
                    bm25: {{
                        query: "{search_query}"
                    }}
                """
            elif search_mode == "semantic":
                search_clause = f"""
                    nearText: {{
                        concepts: ["{search_query}"]
                    }}
                """
            else:  # hybrid (default)
                search_clause = f"""
                    hybrid: {{
                        query: "{search_query}",
                        alpha: 0.5
                    }}
                """

            # Add cursor if provided
            if cursor:
                query_method = f"""
                    {search_clause}
                    {where_clause}
                    {sort_clause}
                    limit: {limit}
                    after: "{cursor}"
                """
            else:
                query_method = f"""
                    {search_clause}
                    {where_clause}
                    {sort_clause}
                    limit: {limit}
                """
        else:
            # Cursor pagination without search
            if cursor:
                query_method = f"""
                    {where_clause}
                    {sort_clause}
                    limit: {limit}
                    after: "{cursor}"
                """
            else:
                query_method = f"""
                    {where_clause}
                    {sort_clause}
                    limit: {limit}
                """

        # Full GraphQL query - MUST include _additional { id } for cursor
        return f"""
            {{
                Get {{
                    LegalDocuments({query_method}) {{
                        document_id
                        document_type
                        full_text
                        language
                        document_number
                        _additional {{
                            id
                        }}
                    }}
                }}
            }}
        """
