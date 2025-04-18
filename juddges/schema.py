import polars as pl

JUDGMENT_SCHEMA = {
    "source": "Source of the data, can be one of: [pl-court, nsa]",
    "judgment_id": "unique identifier of the judgment",
    "docket_number": "signature of judgment (unique within court)",
    "judgment_date": "date of judgment",
    "publication_date": "date of judgment publication",
    "last_update": "date of last update of judgment",
    "court_id": "system unique identifier of the court",
    "department_id": "system unique identifier of the court's department",
    "judgment_type": "type of the judgment (one of)",
    "excerpt": "First 500 characters of the judgment",
    "xml_content": "Full content of judgment in XML format",
    "presiding_judge": "chairman judge name",
    "decision": "decision",
    "judges": "list of judge names participating in the judgment",
    "legal_bases": "legal acts which are bases for the judgment",
    "publisher": "name of the person publishing the judgment",
    "recorder": "name of the person recording the judgment",
    "reviser": "name of the person revising the judgment",
    "keywords": "list of phrases representing the themes/topics of the judgment",
    "num_pages": "number of pages in the judgment",
    "full_text": "full text of the judgment",
    "volume_number": "volume number",
    "volume_type": "type of volume",
    "court_name": "name of the court where the judgment was made",
    "department_name": "name of the department within the court where the judgment was made",
    "extracted_legal_bases": "textual representation of the legal bases for the judgment (with references to online repository)",
    "references": "Plain-text references to legal acts",
    "thesis": "thesis of the judgment",
    "country": "the country of origin of the judgment (one of [Poland, England])",
    "court_type": "type of the court (one of ['ordinary court', 'administrative court', 'crown court'])",
}

POLARS_JUDGMENT_SCHEMA = {
    "source": pl.String,
    "judgment_id": pl.String,
    "docket_number": pl.String,
    "judgment_date": pl.Datetime,
    "publication_date": pl.Datetime,
    "last_update": pl.Datetime,
    "court_id": pl.String,
    "department_id": pl.String,
    "judgment_type": pl.String,
    "excerpt": pl.String,
    "xml_content": pl.String,
    "presiding_judge": pl.String,
    "decision": pl.Null,
    "judges": pl.List(pl.String),
    "legal_bases": pl.List(pl.String),
    "publisher": pl.String,
    "recorder": pl.String,
    "reviser": pl.String,
    "keywords": pl.List(pl.String),
    "num_pages": pl.Int64,
    "full_text": pl.String,
    "volume_number": pl.Int64,
    "volume_type": pl.String,
    "court_name": pl.String,
    "department_name": pl.String,
    "extracted_legal_bases": pl.List(
        pl.Struct(
            [
                pl.Field("address", pl.String),
                pl.Field("art", pl.String),
                pl.Field("isap_id", pl.String),
                pl.Field("text", pl.String),
                pl.Field("title", pl.String),
            ]
        )
    ),
    "references": pl.List(pl.String),
    "thesis": pl.String,
    "country": pl.String,
    "court_type": pl.String,
}


SCHEMAS = {
    "pl-court": JUDGMENT_SCHEMA,
}

POLARS_SCHEMAS = {
    "pl-court": POLARS_JUDGMENT_SCHEMA,
}
