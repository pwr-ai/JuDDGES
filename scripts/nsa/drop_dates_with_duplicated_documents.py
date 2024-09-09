import pandas as pd
import pymongo
import typer

DB_URI = "mongodb://localhost:27017/"


def main(
    db_uri: str = typer.Option(DB_URI),
) -> None:
    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    dates_col = db["dates"]

    df = pd.DataFrame(dates_col.find().sort("date"))
    df = df.explode("document_ids")
    df = df[~df["document_ids"].isna()]
    duplicated = df[df.duplicated(subset="document_ids", keep=False)]
    dates_to_drop = duplicated["date"].unique()

    for date in dates_to_drop:
        dates_col.delete_many({"date": date})


if __name__ == "__main__":
    typer.run(main)
