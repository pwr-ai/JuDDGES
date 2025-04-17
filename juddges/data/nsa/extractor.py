"""Data extraction module for NSA (National Administrative Court) judgments.

This module provides functionality to extract structured data from raw NSA judgment HTML pages.
The extracted data can be converted to pandas DataFrame or PyArrow Table formats for further processing.
"""

import re
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pytz
from bs4 import BeautifulSoup, Tag
from mpire import WorkerPool
from pandas import Timestamp

# Timezone for date handling
WARSAW_TZ = pytz.timezone("Europe/Warsaw")

# Mapping of field names to their descriptions
DESCRIPTION_MAP = {
    "kraj": "country where the court is located",
    "rodzaj sądu": "type of the court",
    "źródło": "source of the judgment data",
    "id": "unique identifier of the judgment",
    "Sygnatura": "signature of judgment (unique within court)",
    "Powołane przepisy": "textual representation of the legal bases for the judgment (with references to online repository)",
    "Sygn. powiązane": "related docket numbers",
    "Sędziowie": "list of judge names participating in the judgment",
    "Data orzeczenia": "date of judgment",
    "Rodzaj orzeczenia": "type of the judgment (one of)",
    "przewodniczący": "chairman judge name",
    "sprawozdawca": "judge rapporteur",
    "ustawa": "law",
    "dziennik_ustaw": "journal of laws",
    "art": "article",
    "link": "link to the law",
    "Data wpływu": "date of submission",
    "Sąd": "name of the court where the judgment was made",
    "Symbol z opisem": "type of case with the detailed description",
    "Hasła tematyczne": "list of phrases representing the themes/topics of the judgment",
    "Skarżony organ": "challenged authority",
    "Treść wyniku": "decision",
    "Publikacja w u.z.o.": "published in official collection of judgments jurisprudence of the voivodeship administrative courts and the supreme administrative court",
    "Info. o glosach": "information on glosa(s)",
    "Sentencja": "full text of the judgment",
    "Uzasadnienie": "reasons for judgment",
    "Tezy": "thesis of the judgment",
    "Zdanie odrębne": "dissenting opinion",
    "Prawomocność": "finality",
}

# Mapping of Polish field names to English field names
FIELD_MAP = {
    "kraj": "country",
    "rodzaj sądu": "court_type",
    "źródło": "source",
    "id": "judgment_id",
    "Sygnatura": "docket_number",
    "Powołane przepisy": "extracted_legal_bases",
    "Sygn. powiązane": "related_docket_numbers",
    "Sędziowie": "judges",
    "Data orzeczenia": "judgment_date",
    "Rodzaj orzeczenia": "judgment_type",
    "przewodniczący": "presiding_judge",
    "sprawozdawca": "judge_rapporteur",
    "ustawa": "law",
    "dziennik_ustaw": "journal",
    "art": "article",
    "link": "link",
    "Data wpływu": "submission_date",
    "Sąd": "court_name",
    "Symbol z opisem": "case_type_description",
    "Hasła tematyczne": "keywords",
    "Skarżony organ": "challenged_authority",
    "Treść wyniku": "decision",
    "Publikacja w u.z.o.": "official_collection",
    "Info. o glosach": "glosa_information",
    "Sentencja": "full_text",
    "Uzasadnienie": "reasons_for_judgment",
    "Tezy": "thesis",
    "Zdanie odrębne": "dissenting_opinion",
    "Prawomocność": "finality",
}

# Order of fields in the output data
ORDER = [
    "country",
    "court_type",
    "source",
    "judgment_id",
    "docket_number",
    "judgment_type",
    "finality",
    "judgment_date",
    "submission_date",
    "court_name",
    "judges",
    "presiding_judge",
    "judge_rapporteur",
    "case_type_description",
    "keywords",
    "related_docket_numbers",
    "challenged_authority",
    "decision",
    "extracted_legal_bases",
    "official_collection",
    "glosa_information",
    "thesis",
    "full_text",
    "reasons_for_judgment",
    "dissenting_opinion",
]

# Fields that should be treated as lists
LIST_TYPE_FIELDS = {
    "Hasła tematyczne",
    "Symbol z opisem",
    "Sędziowie",
    "Treść wyniku",
    "Info. o glosach",
    "Publikacja w u.z.o.",
}

# PyArrow schema for structured data storage
PYARROW_SCHEMA = pa.schema(
    [
        ("country", pa.string(), True),
        ("court_type", pa.string(), True),
        ("source", pa.string(), True),
        ("judgment_id", pa.string()),
        ("docket_number", pa.string(), True),
        ("judgment_type", pa.string(), True),
        ("finality", pa.string(), True),
        ("judgment_date", pa.timestamp("s", tz=WARSAW_TZ), True),
        ("submission_date", pa.timestamp("s", tz=WARSAW_TZ), True),
        ("court_name", pa.string(), True),
        ("judges", pa.list_(pa.string()), True),
        ("presiding_judge", pa.string(), True),
        ("judge_rapporteur", pa.string(), True),
        ("case_type_description", pa.list_(pa.string()), True),
        ("keywords", pa.list_(pa.string()), True),
        (
            "related_docket_numbers",
            pa.list_(
                pa.struct(
                    [
                        ("judgment_id", pa.string()),
                        ("docket_number", pa.string()),
                        ("judgment_date", pa.timestamp("s", tz=WARSAW_TZ)),
                        ("judgment_type", pa.string()),
                    ]
                )
            ),
            True,
        ),
        ("challenged_authority", pa.string(), True),
        ("decision", pa.list_(pa.string()), True),
        (
            "extracted_legal_bases",
            pa.list_(
                pa.struct(
                    [
                        ("link", pa.string()),
                        ("article", pa.string()),
                        ("journal", pa.string()),
                        ("law", pa.string()),
                    ]
                )
            ),
            True,
        ),
        ("official_collection", pa.list_(pa.string()), True),
        ("glosa_information", pa.list_(pa.string()), True),
        ("thesis", pa.large_string(), True),
        ("full_text", pa.large_string(), True),
        ("reasons_for_judgment", pa.large_string(), True),
        ("dissenting_opinion", pa.large_string(), True),
    ]
)

CONSTANT_FIELDS = {
    "country": "Poland",
    "court_type": "administrative court",
    "source": "nsa",
}


class NSADataExtractor:
    """Extractor for structured data from raw NSA judgment HTML pages.

    This class provides methods to extract and transform data from raw NSA judgment HTML pages
    into structured formats (dictionaries, pandas DataFrames, or PyArrow Tables).
    """

    def __init__(self) -> None:
        """Initialize the extractor and validate field mappings."""
        assert (set(FIELD_MAP.values()) - set(ORDER)) == {"journal", "law", "article", "link"}
        assert (set(ORDER) - set(FIELD_MAP.values())) == set()

    def extract_data_from_pages(
        self, pages: Sequence[str], doc_ids: Sequence[str], n_jobs: int | None = None
    ) -> list[dict[str, Any]]:
        """Extract data from multiple HTML pages in parallel.

        Args:
            pages: Sequence of HTML pages to extract data from
            doc_ids: Sequence of document IDs corresponding to the pages
            n_jobs: Number of parallel jobs to use (None for number of cores)

        Returns:
            List of dictionaries containing extracted data for each page
        """
        extracted_data = []
        with WorkerPool(n_jobs) as pool:
            args = (
                {"page": page, "doc_id": doc_id}
                for page, doc_id in zip(pages, doc_ids, strict=True)
            )
            for item in pool.map(
                self.extract_data,
                args,
                progress_bar=True,
                iterable_len=len(pages),
                progress_bar_options={"desc": "Extracting data from pages"},
            ):
                extracted_data.append(item)
        return extracted_data

    def extract_data_from_pages_to_df(
        self, pages: Iterable[str], doc_ids: Iterable[str], n_jobs: int | None = None
    ) -> pd.DataFrame:
        """Extract data from pages and convert to pandas DataFrame.

        Args:
            pages: Iterable of HTML pages to extract data from
            doc_ids: Iterable of document IDs corresponding to the pages
            n_jobs: Number of parallel jobs to use (None for automatic)

        Returns:
            DataFrame containing extracted data with columns in ORDER
        """
        extracted_data = self.extract_data_from_pages(pages, doc_ids, n_jobs)
        return pd.DataFrame(extracted_data, columns=ORDER)

    def extract_data_from_pages_to_pyarrow(
        self, pages: Iterable[str], doc_ids: Iterable[str], n_jobs: int | None = None
    ) -> pa.Table:
        """Extract data from pages and convert to PyArrow Table.

        Args:
            pages: Iterable of HTML pages to extract data from
            doc_ids: Iterable of document IDs corresponding to the pages
            n_jobs: Number of parallel jobs to use (None for automatic)

        Returns:
            PyArrow Table containing extracted data with schema PYARROW_SCHEMA
        """
        extracted_data = self.extract_data_from_pages(pages, doc_ids, n_jobs)
        table = pa.Table.from_pylist(extracted_data, schema=self.pyarrow_schema)
        return table.select(ORDER)

    @property
    def pyarrow_schema(self) -> pa.Schema:
        """Get the PyArrow schema for the extracted data."""
        return PYARROW_SCHEMA

    def extract_data(self, page: str, doc_id: str) -> dict[str, Any]:
        """Extract structured data from a single HTML page.

        Args:
            page: HTML content of the judgment page
            doc_id: Document ID of the judgment

        Returns:
            Dictionary containing extracted data fields
        """
        soup = BeautifulSoup(page, "html.parser")
        extracted_data = (
            CONSTANT_FIELDS
            | {FIELD_MAP["id"]: doc_id}
            | self._extract_number_and_type(soup)
            | self._extract_table(soup)
            | self._extract_text_sections(soup)
        )
        assert set(extracted_data.keys()) - set(ORDER) == set()
        extracted_data = {k: extracted_data[k] for k in ORDER if k in extracted_data}
        return extracted_data

    def _extract_number_and_type(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Extract docket number and judgment type from the header."""
        number_and_type = soup.find("span", class_="war_header").get_text(strip=True)
        if number_and_type.startswith("- ") and " - " not in number_and_type:
            number_and_type = " " + number_and_type
        if len(number_and_type.split(" - ")) != 2:
            print(number_and_type)
        number, judgment_type = number_and_type.split(" - ")
        number, judgment_type = number.strip(), judgment_type.strip()
        if not number:
            number = None
        if not judgment_type:
            judgment_type = None
        return {FIELD_MAP["Sygnatura"]: number, FIELD_MAP["Rodzaj orzeczenia"]: judgment_type}

    def _extract_text_sections(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Extract text sections from the page (e.g., full text, reasons, thesis)."""
        extracted_data = {}
        section_headers = soup.find_all("div", class_="lista-label")
        for header in section_headers:
            next_section = header.find_next("span", class_="info-list-value-uzasadnienie")
            if next_section:
                header_text = header.get_text(strip=True)
                extracted_data[FIELD_MAP[header_text]] = self._extract_text_preserve_paragraphs(
                    next_section
                )
        return extracted_data

    def _extract_table(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Extract data from the table of the judgment."""
        extracted_data = {}
        rows = soup.find_all("tr", class_="niezaznaczona")
        for row in rows:
            label = row.find("td", class_="lista-label")
            value = row.find("td", class_="info-list-value")

            if label and value:
                label_text = label.get_text(strip=True)
                value_text = value.decode_contents().strip()
                if "Powołane przepisy" in label_text:
                    extracted_data[FIELD_MAP["Powołane przepisy"]] = self._extract_regulations(
                        value
                    )
                elif "Sygn. powiązane" in label_text:
                    extracted_data[FIELD_MAP["Sygn. powiązane"]] = self._extract_related(value)
                elif "<br/>" in value_text or label_text in LIST_TYPE_FIELDS:
                    extracted_data |= self._extract_fields_with_br(label_text, value_text)
                elif "Data orzeczenia" in label_text:
                    extracted_data |= self._extract_date_finality(value)
                else:
                    extracted_data[FIELD_MAP[label_text]] = value_text
        for date_field in ["Data orzeczenia", "Data wpływu"]:
            if FIELD_MAP[date_field] in extracted_data:
                extracted_data[FIELD_MAP[date_field]] = self._to_datetime(
                    extracted_data[FIELD_MAP[date_field]]
                )
        return extracted_data

    def _extract_text_data_with_br_tags(self, value: str) -> list[str]:
        return [item.strip() for item in value.split("<br/>") if item.strip()]

    def _extract_text_preserve_paragraphs(self, html_element: Tag) -> str:
        """Extract text from HTML while preserving paragraph structure."""
        paragraphs = html_element.find_all(["p", "br"])
        text_parts = []
        for paragraph in paragraphs:
            text = paragraph.get_text(strip=True)
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)

    def _extract_regulations(self, value: Tag) -> list[dict[str, Any]]:
        """Extract legal regulations from the HTML table cell."""
        chunks = []
        x = 0
        chunks.append([])
        for i in value.contents:
            if "<br/>" not in str(i):
                if i.get_text(strip=True) != "":
                    chunks[x].append(i)
            else:
                x += 1
                chunks.append([])

        chunks = [chunk for chunk in chunks if len(chunk) > 0]
        regulations = []

        current = {}
        for chunk in chunks:
            if (
                isinstance(chunk[0], Tag)
                and "class" in chunk[0].attrs
                and chunk[0].attrs["class"][0] == "nakt"
            ):
                if FIELD_MAP["ustawa"] in current:
                    regulations.append(current)
                    current = {}
                current[FIELD_MAP["ustawa"]] = chunk[0].get_text(strip=True)
            else:
                if FIELD_MAP["dziennik_ustaw"] in current:
                    regulations.append(current)
                    current = {}
                current[FIELD_MAP["dziennik_ustaw"]] = chunk[0].get_text(strip=True)
                current[FIELD_MAP["art"]] = (
                    chunk[1].get_text(strip=True) if len(chunk) > 1 else None
                )
                current[FIELD_MAP["link"]] = chunk[0]["href"] if isinstance(chunk[0], Tag) else None
        if current:
            regulations.append(current)
        return regulations

    def _extract_related(self, value: Tag) -> list[dict[str, Any]]:
        """Extract related judgments from the table cell.
        Returns list of dictionaries with related judgment details (sygnatura, rodzaj orzeczenia, data orzeczenia, id)."""
        related = []
        for a in value.find_all("a"):
            number_type_date = a.get_text(strip=True)
            if number_type_date.startswith("- ") and " - " not in number_type_date:
                number_type_date = " " + number_type_date
            assert " z " in number_type_date
            number, judgment_type, date = re.match(r"(.*) - (.*) z (.*)", number_type_date).groups()
            link = a["href"]
            if not number:
                number = None
            if not judgment_type:
                judgment_type = None
            if not date:
                date = None
            else:
                date = self._to_datetime(date)
            related.append(
                {
                    FIELD_MAP["Sygnatura"]: number,
                    FIELD_MAP["Rodzaj orzeczenia"]: judgment_type,
                    FIELD_MAP["Data orzeczenia"]: date,
                    FIELD_MAP["id"]: link,
                }
            )
        return related

    def _extract_fields_with_br(self, label_text: str, value_text: str) -> dict[str, Any]:
        """Extract fields that contain <br/> tags or are list-type fields."""
        data = dict()
        data[FIELD_MAP[label_text]] = self._extract_text_data_with_br_tags(value_text)
        if label_text == "Sędziowie":
            function = [re.findall(r"/([^/]*)/", j) for j in data[FIELD_MAP[label_text]]]
            data[FIELD_MAP[label_text]] = [
                re.sub(r"/[^/]*/", "", s).strip() for s in data[FIELD_MAP[label_text]]
            ]
            function_map = {f[0]: j for f, j in zip(function, data[FIELD_MAP[label_text]]) if f}
            if "przewodniczący" in function_map and "sprawozdawca" in function_map:
                data[FIELD_MAP["przewodniczący"]] = function_map["przewodniczący"]
                data[FIELD_MAP["sprawozdawca"]] = function_map["sprawozdawca"]
            elif "przewodniczący sprawozdawca" in function_map:
                data[FIELD_MAP["przewodniczący"]] = function_map["przewodniczący sprawozdawca"]
                data[FIELD_MAP["sprawozdawca"]] = function_map["przewodniczący sprawozdawca"]
        return data

    def _extract_date_finality(self, value: Tag) -> dict[str, Any]:
        """Extract judgment date and finality status from the HTML table cell."""
        date = dict()
        date_value = value.find_all("td")[0].get_text(strip=True)
        judgment_type = value.find_all("td")[1].get_text(strip=True)
        if len(judgment_type) == 0:
            judgment_type = None
        date[FIELD_MAP["Data orzeczenia"]] = date_value
        date[FIELD_MAP["Prawomocność"]] = judgment_type
        return date

    def _to_datetime(self, date_str: str) -> Timestamp:
        """Convert date string to timestamp in Warsaw timezone."""
        date = pd.to_datetime(date_str)
        return WARSAW_TZ.localize(date)
