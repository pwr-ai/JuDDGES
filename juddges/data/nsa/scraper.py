import random
import re
import time

import mechanicalsoup
from bs4 import BeautifulSoup
from loguru import logger
from requests import HTTPError, RequestException
from requests.adapters import HTTPAdapter
from retry import retry
from urllib3 import Retry


class IncorrectNumberOfDocumentsFound(Exception):
    pass


class IncorrectPage(Exception):
    pass


class NoNumberOfDocumentsFound(Exception):
    pass


class NSAScraper:
    def __init__(
        self, user_agent: str, proxy_config: dict[str, str] | None = None, wait: bool = True
    ) -> None:
        self.wait = wait
        self.browser = mechanicalsoup.StatefulBrowser(
            user_agent=user_agent,
            requests_adapters={
                "https://": HTTPAdapter(
                    max_retries=Retry(
                        total=10,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504, 403, 429],
                    )
                ),
                "http://": HTTPAdapter(
                    max_retries=Retry(
                        total=10,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504, 403, 429],
                    )
                ),
            },
        )

        if proxy_config:
            self.browser.session.proxies = proxy_config

    def close(self) -> None:
        self.browser.close()

    @retry(
        tries=5,
        exceptions=(RequestException, HTTPError, IncorrectNumberOfDocumentsFound, IncorrectPage),
    )
    def search_documents_for_date(self, date: str) -> dict[int, list[str] | None] | None:
        final_num_documents = self.get_num_docs_for_date(date)
        self._browser_open("https://orzeczenia.nsa.gov.pl/cbo")
        self.browser.select_form()
        self.browser["odDaty"] = date
        self.browser["doDaty"] = date
        self._browser_submit_selected()
        if self._any_documents_found(self.browser):
            documents = self._retrieve_documents()
            num_documents = sum(map(lambda x: len(x) if x else 0, documents.values()))
            if num_documents != final_num_documents:
                raise IncorrectNumberOfDocumentsFound(
                    f"Found {num_documents} documents on {len(documents)} pages. Expected {final_num_documents} documents."
                )
            logger.info(f"Found {num_documents} documents on {len(documents)} pages.")
            return documents
        else:
            logger.info("No documents found")
            return None

    @retry(
        tries=5,
        exceptions=(RequestException, HTTPError, IncorrectNumberOfDocumentsFound, IncorrectPage),
    )
    def get_page_for_doc(self, doc_id: str) -> str:
        self._browser_open(f"https://orzeczenia.nsa.gov.pl/{doc_id}")
        return self.browser.page.prettify()

    @retry(
        tries=5,
        exceptions=(
            RequestException,
            HTTPError,
            IncorrectNumberOfDocumentsFound,
            IncorrectPage,
            NoNumberOfDocumentsFound,
        ),
    )
    def get_num_docs_for_date(self, date: str) -> int:
        self._browser_open("https://orzeczenia.nsa.gov.pl/cbo")
        self.browser.select_form()
        self.browser["odDaty"] = date
        self.browser["doDaty"] = date
        self._browser_submit_selected()
        if self._any_documents_found(self.browser):
            num_documents = self._extract_document_count(self.browser.page.prettify())
            return num_documents
        else:
            return 0

    def _browser_open(self, url: str) -> None:
        response = self.browser.open(url, verify=False, timeout=60)
        self._post_call(response)

    def _browser_submit_selected(self) -> None:
        response = self.browser.submit_selected(verify=False, timeout=60)
        self._post_call(response)

    def _post_call(self, response) -> None:
        response.raise_for_status()
        if self.wait:
            # wait random from normal distribution
            time_to_wait = random.normalvariate(1, 0.5)
            time.sleep(time_to_wait if time_to_wait > 0 else 0)
        if not self._correct_page():
            raise IncorrectPage(f"Incorrect page: {self.browser.page.text}")

    def _correct_page(self) -> bool:
        title = "Centralna Baza Orzeczeń Sądów Administracyjnych"
        return title in self.browser.page.text

    def _any_documents_found(self, browser: mechanicalsoup.StatefulBrowser) -> bool:
        warning_text = "Nie znaleziono orzeczeń spełniających podany warunek!"
        return warning_text not in browser.page.text

    def _retrieve_documents(self) -> dict[int, list[str] | None]:
        page_links = self.browser.links(url_regex="^/cbo/find\?p=")
        if not page_links:
            last_page = 1
        else:
            last_page_link = page_links[-2]
            last_page = int(last_page_link.text)

        documents: dict[int, list[str] | None] = {}
        for page_id in range(1, last_page + 1):
            documents[page_id] = self._retrieve_documents_from_page(page_id)
        return documents

    @retry(
        tries=5,
        exceptions=(RequestException, HTTPError, IncorrectNumberOfDocumentsFound, IncorrectPage),
    )
    def _retrieve_documents_from_page(self, page_id: int) -> list[str] | None:
        self._browser_open(f"https://orzeczenia.nsa.gov.pl/cbo/find?p={page_id}")
        if self.browser.url.endswith(f"{page_id}"):
            page_documents = self._find_documents_on_page(self.browser.page)
            if not (0 < len(page_documents) <= 10):
                raise IncorrectNumberOfDocumentsFound(
                    f"Page {page_id} has {len(page_documents)} documents. Page URL: {self.browser.url}."
                )
            documents = page_documents
        else:
            documents = None
        return documents

    def _find_documents_on_page(self, page: BeautifulSoup) -> list[str]:
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

    def _extract_document_count(self, text: str) -> int:
        pattern = r"Znaleziono\s+(\d+)\s+orzeczeń"
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
        else:
            raise NoNumberOfDocumentsFound(f"No document count found in text: {text}")
