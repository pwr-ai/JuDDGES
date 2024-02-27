from typing import Any

import requests
import xmltodict


class PolishCourtAPI:

    def __init__(self) -> None:
        self.url = "https://apiorzeczenia.wroclaw.sa.gov.pl/ncourt-api"

    def get_number_of_judgements(self, params: dict[str, Any] | None = None) -> int:
        if params is None:
            params = {}

        params |= {"limit": 0}
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

        return judgements

    def get_content(self, id: str) -> str:
        params = {"id": id}
        endpoint = f"{self.url}/judgement/content"
        res = requests.get(endpoint, params=params)
        res.raise_for_status()
        content = res.content.decode("utf-8")

        return content
