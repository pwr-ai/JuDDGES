from functools import cached_property
from typing import Any

import requests
import xmltodict
from loguru import logger
from requests import HTTPError


class PolishCourtAPI:
    def __init__(self) -> None:
        self.url = "https://apiorzeczenia.wroclaw.sa.gov.pl/ncourt-api"

    @cached_property
    def schema(self) -> dict[str, list[str]]:
        return {
            "judgement": [
                "_id",
                "signature",
                "date",
                "publicationDate",
                "lastUpdate",
                "courtId",
                "departmentId",
                "type",
                "excerpt",
            ],
            "content": ["content"],
            "details": [
                "chairman",
                "judges",
                "themePhrases",
                "references",
                "legalBases",
                "recorder",
                "decision",
                "reviser",
                "publisher",
            ],
        }

    def get_number_of_judgements(self, params: dict[str, Any] | None = None) -> int:
        if params is None:
            params = {}
        elif "limit" in params.keys():
            logger.warning("Setting limit to query the number of judgements has no effect!")

        params = {**params, "limit": 0}
        endpoint = f"{self.url}/judgements"
        res = requests.get(endpoint, params=params)
        res.raise_for_status()
        total_judgements = xmltodict.parse(res.content.decode("utf-8"))["judgements"]["@total"]

        return int(total_judgements)

    def get_judgements(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = f"{self.url}/judgements"
        res = requests.get(endpoint, params=params)
        res.raise_for_status()
        judgements = xmltodict.parse(res.content.decode("utf-8"))["judgements"]["judgement"]

        assert isinstance(judgements, list)

        return judgements

    def get_content(self, id: str) -> dict[str, Any]:
        params = {"id": id}
        endpoint = f"{self.url}/judgement/content"
        res = requests.get(endpoint, params=params)

        try:
            res.raise_for_status()
        except HTTPError as err:
            if err.response.status_code == 404:
                raise DataNotFoundError(f"Not found content for document: {id}")
            raise

        content = res.content.decode("utf-8")

        return {"content": content}

    def get_cleaned_details(self, id: str) -> dict[str, Any]:
        """Downloads details without repeating fields retrieved in get_judgements."""
        details = self.get_details(id)
        return {k: v for k, v in details.items() if k in self.schema["details"]}

    def get_details(self, id: str) -> dict[str, Any]:
        params = {"id": id}
        endpoint = f"{self.url}/judgement/details"
        res = requests.get(endpoint, params=params)
        res.raise_for_status()

        # for details, API returns XML with error info instead of 404 status code
        data = xmltodict.parse(res.content.decode("utf-8"))
        try:
            details = data["judgement"]
        except KeyError:
            if "error" in data.keys():
                raise DataNotFoundError(f"Not found details for document: {id}")
            raise
        else:
            assert isinstance(details, dict)
            details = self.parse_details(data)
            return details

    def parse_details(self, details: dict[str, Any]) -> dict[str, Any]:
        cols_to_unnest = [
            ("judges", "judge"),
            ("themePhrases", "themePhrase"),
            ("references", "reference"),
            ("legalBases", "legalBasis"),
        ]
        for feature, nested_key in cols_to_unnest:
            if details.get(feature) is None:
                continue
            details[feature] = self._unnest_dict(details.get(feature), nested_key)

        return details

    def _unnest_dict(
        self,
        ndict: dict[str, list[str] | str | None],
        key: str,
    ) -> list[str] | None:
        if len(ndict) != 1:
            raise ValueError("To unnest dict should contain exactly one element")

        if isinstance(ndict[key], list):
            return ndict[key]
        return [ndict[key]]


class DataNotFoundError(Exception):
    pass
