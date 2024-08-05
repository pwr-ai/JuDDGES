import re

import mechanicalsoup
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter


class NSAScraper:
    def __init__(self, requests_adapters: dict[str, HTTPAdapter] | None = None):
        self.requests_adapters = requests_adapters

    def search_documents(self, start_date, end_date):
        browser = mechanicalsoup.StatefulBrowser(
            user_agent="MechanicalSoup", requests_adapters=self.requests_adapters
        )
        self.browser_open(browser, "https://orzeczenia.nsa.gov.pl/cbo")
        browser.select_form()
        # browser["symbole"] = "648"
        browser["odDaty"] = start_date
        browser["doDaty"] = end_date
        browser.submit_selected()
        if self.any_documents_found(browser):
            documents = self.retrieve_documents(browser)
            return documents
        else:
            return None

    def any_documents_found(self, browser: mechanicalsoup.StatefulBrowser) -> bool:
        warning_text = "Nie znaleziono orzeczeń spełniających podany warunek!"
        return warning_text not in browser.page.text

    def retrieve_documents(
        self, browser: mechanicalsoup.StatefulBrowser
    ) -> dict[int, list[str] | None]:
        page_links = browser.links(url_regex="^/cbo/find\?p=")
        if not page_links:
            last_page = 1
        else:
            last_page_link = page_links[-2]
            last_page = int(last_page_link.text)

        documents: dict[int, list[str] | None] = {}
        for page_id in range(1, last_page + 1):
            self.browser_open(browser, f"https://orzeczenia.nsa.gov.pl/cbo/find?p={page_id}")

            if browser.url.endswith(f"{page_id}"):
                page_documents = self.find_documents_on_page(browser.page)
                assert (
                    0 < len(page_documents) <= 10
                ), f"Page {page_id} has {len(page_documents)} documents"
                documents[page_id] = page_documents
            else:
                documents[page_id] = None
        return documents

    def find_documents_on_page(self, page: BeautifulSoup) -> list[str]:
        all_links = page.find_all("a", href=True)
        pattern = re.compile(r"^/doc/[A-Za-z0-9]+$")

        filtered_links = []

        for link in all_links:
            href = link["href"]
            if pattern.match(href):
                # Check if the link is within a <span class="powiazane">
                if not link.find_parent("span", class_="powiazane"):
                    filtered_links.append(href)

        return filtered_links

    def browser_open(self, browser: mechanicalsoup.StatefulBrowser, url: str) -> None:
        response = browser.open(url)
        if response.status_code != 200:
            raise Exception(f"Failed to open the website. Status code: {response.status_code}")
