from typing import ClassVar

import weaviate
import weaviate.classes.config as wvcc
from juddges.data.base_weaviate_db import WeaviateDatabase


class WeaviateJudgmentsDatabase(WeaviateDatabase):
    """Database for court judgments."""

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
                # Source of the data, can be one of: [pl-court, nsa, tax-interpretation]
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
                # Plain-text references to legal acts
                wvcc.Property(
                    name="references",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    index_filterable=True,
                    index_searchable=True,
                ),
                # the country of origin of the judgment (one of [Poland, England])
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
                wvcc.Property(
                    name="x",
                    data_type=wvcc.DataType.NUMBER,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="y",
                    data_type=wvcc.DataType.NUMBER,
                    index_filterable=True,
                ),
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_transformers(),
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
                wvcc.Property(
                    name="x",
                    data_type=wvcc.DataType.NUMBER,
                    index_filterable=True,
                ),
                wvcc.Property(
                    name="y",
                    data_type=wvcc.DataType.NUMBER,
                    index_filterable=True,
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

    @staticmethod
    def uuid_from_judgment_chunk_id(judgment_id: str, chunk_id: int) -> str:
        return weaviate.util.generate_uuid5(f"{judgment_id}_chunk_{chunk_id}")
