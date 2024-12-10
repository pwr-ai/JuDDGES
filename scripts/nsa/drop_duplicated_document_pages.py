import pymongo
import pandas as pd
import typer
from tqdm import tqdm


def main(
    db_uri: str = typer.Option(..., envvar="DB_URI"),
) -> None:
    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    pages_col = db["document_pages"]

    print("Checking for duplicates. This may take a while...")
    cursor = pages_col.aggregate(
        [{"$group": {"_id": "$page", "count": {"$sum": 1}}}, {"$match": {"count": {"$gt": 1}}}]
    )
    duplicates = pd.DataFrame(list(cursor))

    if len(duplicates) == 0:
        print("No duplicates found.")
        return

    # ask user for confirmation
    print(f"Found {len(duplicates)} duplicates.")
    if not typer.confirm("Do you want to delete them?"):
        print("Aborted.")
        return

    # delete duplicates
    print("Deleting duplicates...")
    for page in tqdm(duplicates["_id"]):
        pages_col.delete_many({"page": page})

    print("Done.")


typer.run(main)
