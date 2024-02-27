import typer
from dotenv import load_dotenv
from mpire import WorkerPool
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from requests import HTTPError
from rich import print
from tenacity import retry, wait_random_exponential, retry_if_exception_type, stop_after_attempt

from juddges.data.pl_court_api import PolishCourtAPI

N_JOBS = 10
BATCH_SIZE = 1_000

load_dotenv("secrets.env", verbose=True)


def main(
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    batch_size: int = typer.Option(BATCH_SIZE),
    n_jobs: int = typer.Option(N_JOBS),
):
    client = MongoClient(mongo_uri, server_api=ServerApi("1"))
    collection = client["juddges"]["judgements"]
    client.admin.command("ping")

    query = {"content": {"$exists": False}}
    num_docs_without_content = collection.count_documents(query)
    print(f"There are {num_docs_without_content} documents without content")

    cursor = collection.find(query, batch_size=batch_size)

    docs_to_update = (doc["_id"] for doc in cursor)
    download_content = ContentDownloader(mongo_uri)
    with WorkerPool(n_jobs=n_jobs) as pool:
        pool.map(
            download_content,
            docs_to_update,
            progress_bar=True,
            iterable_len=num_docs_without_content,
        )


class ContentDownloader:
    def __init__(self, mongo_uri: str):
        self.mongo_uri = mongo_uri

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        retry=retry_if_exception_type(HTTPError),
        stop=stop_after_attempt(3),
    )
    def __call__(self, doc_id: str):
        client = MongoClient(self.mongo_uri)
        collection = client["juddges"]["judgements"]

        api = PolishCourtAPI()

        try:
            content = api.get_content(doc_id)
        except HTTPError as err:
            if err.response.status_code == 404:
                content = None
            else:
                raise

        collection.update_one({"_id": doc_id}, {"$set": {"content": content}})


if __name__ == "__main__":
    typer.run(main)
