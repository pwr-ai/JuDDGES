import pandas as pd
import pymongo
import typer
import urllib3
from tqdm import tqdm

from juddges.settings import NSA_DATA_PATH

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_URI = "mongodb://localhost:27017/"


def main(
    db_uri: str = typer.Option(DB_URI),
) -> None:
    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    docs_col = db["document_pages"]
    errors_col = db["document_pages_errors"]

    NSA_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = NSA_DATA_PATH / "pages.json"
    data = pd.DataFrame(tqdm(docs_col.find()))
    data["_id"] = data["_id"].astype(str)
    data.to_json(output_path, orient="records", indent=4)

    output_path = NSA_DATA_PATH / "errors.json"
    data = pd.DataFrame(tqdm(errors_col.find()))
    data["_id"] = data["_id"].astype(str)
    data.to_json(output_path, orient="records", indent=4)


typer.run(main)
