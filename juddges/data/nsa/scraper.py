import re
from itertools import repeat

import mechanicalsoup
import html
import urllib.parse

import mpire
from mechanicalsoup import LinkNotFoundError
from tqdm import tqdm, trange


def find_documents_on_page(page):
    all_links = page.find_all('a', href=True)
    pattern = re.compile(r'^/doc/[A-Za-z0-9]+$')

    filtered_links = []

    for link in all_links:
        href = link['href']
        if pattern.match(href):
            # Check if the link is within a <span class="powiazane">
            if not link.find_parent('span', class_='powiazane'):
                filtered_links.append(href)

    return filtered_links


def retrieve_documents(browser, worker_id: int, num_workers: int):
    documents = []
    last_page_link = browser.links(url_regex='^/cbo/find\?p=')[-2]
    last_page = int(last_page_link.text)

    for page_id in trange(worker_id, last_page + 1, num_workers, desc=f"Worker {worker_id}"):
        print(browser.open(f"https://orzeczenia.nsa.gov.pl/cbo/find?p={page_id}"))

        if browser.url.endswith(f"{page_id}"):
            page_documents = find_documents_on_page(browser.page)
            assert 0 < len(page_documents) <= 10, f"Page {page_id} has {len(page_documents)} documents"
            documents.extend(page_documents)
        else:
            print(f"Page {page_id} not found")
    return documents

def work(worker_id: int, num_workers: int):
    browser = mechanicalsoup.StatefulBrowser(user_agent="MechanicalSoup")
    browser.open("https://orzeczenia.nsa.gov.pl/cbo")

    browser.select_form()
    browser["symbole"] = "648"
    browser.submit_selected()

    return retrieve_documents(browser, worker_id, num_workers)


with mpire.WorkerPool(2, pass_worker_id=True) as pool:
    documents = pool.map(work, [2]*2)
    print(documents[:2])


