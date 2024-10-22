from typing import Any, Iterable

import pandas as pd
from mpire import WorkerPool
import re
from bs4 import BeautifulSoup, Tag

FIELD_MAP = {
    "id": "id",
    "Sygnatura": "Docket number",
    "Powołane przepisy": "The cited provisions",
    "Sygn. powiązane": "Related docket numbers",
    "Sędziowie": "Judges",
    "Data orzeczenia": "The day of the judgment",
    "Rodzaj orzeczenia": "Type of decision",
    "przewodniczący": "Presiding judge",
    "sprawozdawca": "Judge rapporteur",
    "ustawa": "Law",
    "dziennik_ustaw": "Journal of laws",
    "art": "Article",
    "Data wpływu": "Date of submission",
    "Sąd": "Court",
    "Symbol z opisem": "Type of case with the detailed description",
    "Hasła tematyczne": "Keywords",
    "Skarżony organ": "Challenged authority",
    "Treść wyniku": "Nature of the verdict",
    "Publikacja w u.z.o.": "Published in official collection of judgments Jurisprudence of the Voivodeship Administrative Courts and the Supreme Administrative Court",
    "Info. o glosach": "Information on glosa(s)",
    "Sentencja": "Sentence of the judgment",
    "Uzasadnienie": "Reasons for judgment",
    "Tezy": "Theses",
    "Zdanie odrębne": "Dissenting opinion",
    "Prawomocność": "Finality",
}

ORDER = [
    "id",
    "Docket number",
    "Type of decision",
    "Finality",
    "The day of the judgment",
    "Date of submission",
    "Court",
    "Judges",
    "Presiding judge",
    "Judge rapporteur",
    "Type of case with the detailed description",
    "Keywords",
    "Related docket numbers",
    "Challenged authority",
    "Nature of the verdict",
    "The cited provisions",
    "Published in official collection of judgments Jurisprudence of the Voivodeship Administrative Courts and the Supreme Administrative Court",
    "Information on glosa(s)",
    "Theses",
    "Sentence of the judgment",
    "Reasons for judgment",
    "Dissenting opinion",
]


class NSADataExtractor:
    def __init__(self) -> None:
        assert (set(FIELD_MAP.values()) - set(ORDER)) == {"Journal of laws", "Law", "Article"}
        assert (set(ORDER) - set(FIELD_MAP.values())) == set()

    def extract_data_from_pages(
        self, pages: Iterable[str], doc_ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        extracted_data = []
        with WorkerPool() as pool:
            args = (
                {"page": page, "doc_id": doc_id}
                for page, doc_id in zip(pages, doc_ids, strict=True)
            )
            for item in pool.map(self.extract_data, args, progress_bar=True):
                extracted_data.append(item)
        return extracted_data

    def extract_data_from_pages_to_df(
        self, pages: Iterable[str], doc_ids: Iterable[str]
    ) -> pd.DataFrame:
        extracted_data = self.extract_data_from_pages(pages, doc_ids)
        return pd.DataFrame(extracted_data, columns=ORDER)

    def extract_data(self, page: str, doc_id: str) -> dict[str, Any]:
        soup = BeautifulSoup(page, "html.parser")  # 'page' contains the HTML
        extracted_data = (
            {"id": doc_id}
            | self._extract_number_and_type(soup)
            | self._extract_table(soup)
            | self._extract_text_sections(soup)
        )
        assert set(extracted_data.keys()) - set(ORDER) == set()
        extracted_data = {k: extracted_data[k] for k in ORDER if k in extracted_data}
        return extracted_data

    def _extract_number_and_type(self, soup: BeautifulSoup) -> dict[str, Any]:
        number_and_type = soup.find("span", class_="war_header").get_text(strip=True)
        if number_and_type.startswith("- ") and " - " not in number_and_type:
            number_and_type = " " + number_and_type
        if len(number_and_type.split(" - ")) != 2:
            print(number_and_type)
        number, judgement_type = number_and_type.split(" - ")
        number, judgement_type = number.strip(), judgement_type.strip()
        if not number:
            number = None
        if not judgement_type:
            judgement_type = None
        return {FIELD_MAP["Sygnatura"]: number, FIELD_MAP["Rodzaj orzeczenia"]: judgement_type}

    def _extract_text_sections(self, soup: BeautifulSoup) -> dict[str, Any]:
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
                elif "<br/>" in value_text or label_text in (
                    "Hasła tematyczne",
                    "Symbol z opisem",
                    "Sędziowie",
                    "Treść wyniku",
                ):
                    extracted_data |= self._extract_fields_with_br(label_text, value_text)
                elif "Data orzeczenia" in label_text:
                    extracted_data |= self._extract_date_finality(value)
                else:
                    extracted_data[FIELD_MAP[label_text]] = value_text
        return extracted_data

    def _extract_text_data_with_br_tags(self, value: str) -> list[str]:
        return [item.strip() for item in value.split("<br/>") if item.strip()]

    # Function to preserve paragraphs with \n\n
    def _extract_text_preserve_paragraphs(self, html_element: Tag) -> str:
        paragraphs = html_element.find_all(["p", "br"])
        text_parts = []
        for paragraph in paragraphs:
            text = paragraph.get_text(strip=True)
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)

    def _extract_regulations(self, value: Tag) -> list[dict[str, Any]]:
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
                current["Link"] = chunk[0]["href"] if isinstance(chunk[0], Tag) else None
        if current:
            regulations.append(current)
        return regulations

    def _extract_related(self, value: Tag) -> list[dict[str, Any]]:
        related = []
        for a in value.find_all("a"):
            number_type_date = a.get_text(strip=True)
            if number_type_date.startswith("- ") and " - " not in number_type_date:
                number_type_date = " " + number_type_date
            assert " z " in number_type_date
            number, judgement_type, date = re.match(
                r"(.*) - (.*) z (.*)", number_type_date
            ).groups()
            link = a["href"]
            if not number:
                number = None
            if not judgement_type:
                judgement_type = None
            if not date:
                date = None
            related.append(
                {
                    FIELD_MAP["Sygnatura"]: number,
                    FIELD_MAP["Rodzaj orzeczenia"]: judgement_type,
                    FIELD_MAP["Data orzeczenia"]: date,
                    "Link": link,
                }
            )
        return related

    def _extract_fields_with_br(self, label_text: str, value_text: str) -> dict[str, Any]:
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
        date = dict()
        date_value = value.find_all("td")[0].get_text(strip=True)
        judgement_type = value.find_all("td")[1].get_text(strip=True)
        if len(judgement_type) == 0:
            judgement_type = None
        date[FIELD_MAP["Data orzeczenia"]] = date_value
        date[FIELD_MAP["Prawomocność"]] = judgement_type
        return date
