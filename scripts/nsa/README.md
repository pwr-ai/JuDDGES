# NSA Data Scraping & Extraction Scripts

This directory contains a suite of Python scripts designed for scraping, processing, and storing
document data from the NSA archive.
Below is an overview of the purpose, requirements, and usage of each script.

## Requirements

1. **MongoDB**  
   The scripts rely on a MongoDB instance to store and manage the scraped data.
   The scripts will use the `nsa` database under the provided URI.
   You can specify the URI of the MongoDB instance using the `--db-uri` argument in the scripts or
   by setting the `DB_URI` environment variable.

2. **Proxy**  
   A working proxy is required for certain scripts to download data reliably.

3. **Data Storage Path**  
   Configure `NSA_DATA_PATH` in the `juddges.settings` module to specify the directory for storing
   scraped and processed data.

---

## Script Descriptions and Order of Execution

### 1. **`scrap_documents_list.py`**

- **Purpose:** Scrapes a list of documents and from the NSA website for a specified date range.
- **Usage:**
  ```bash
  python scrap_documents_list.py [OPTIONS]
  ```
- **Arguments:**

  | Argument         | Description                                    | Default                    |
       |------------------|------------------------------------------------|----------------------------|
  | `--proxy-address`| Proxy address for scraping (required).         | None                       |
  | `--db-uri`       | MongoDB URI.                                   | None                       |
  | `--start-date`   | Start date for scraping (YYYY-MM-DD).          | `1981-01-01`               |
  | `--end-date`     | End date for scraping (YYYY-MM-DD).            | Yesterday’s date in Poland |
  | `--n-jobs`       | Number of parallel workers.                    | `30`                       |
- **Output:** Saves the scraped document list in MongoDB (`dates` collection) and as
  `documents.json` in `data/datasets/nsa`.

---

### 2. (optional) **`drop_dates_with_duplicated_documents.py`**

- **Purpose:** Removes duplicate document entries from the `dates` collection in MongoDB.
  If documents are duplicated, the script removes whole dates for which duplicates are found.
- **Note:** You should to rerun `scrap_documents_list.py` after running this script to update
  `documents.json`.
- **Usage:**
  ```bash
  python drop_dates_with_duplicated_documents.py [OPTIONS]
  ```
- **Arguments:**

  | Argument         | Description                    | Default |
       |------------------|--------------------------------|---------|
  | `--db-uri`       | MongoDB URI.                   | None    |
- **Output:** Cleans up the `dates` collection by deleting dates with duplicate document entries.

---

### 3. **`download_document_pages.py`**

- **Purpose:** Downloads document pages (raw HTML) using IDs retrieved from the `documents.json`
  file.
- **Usage:**
  ```bash
  python download_document_pages.py [OPTIONS]
  ```
- **Arguments:**

  | Argument         | Description                                    | Default |
       |------------------|------------------------------------------------|---------|
  | `--proxy-address`| Proxy address for scraping (required).         | None    |
  | `--db-uri`       | MongoDB URI.                                   | None    |
  | `--n-jobs`       | Number of parallel workers.                    | `25`    |
- **Output:** Stores downloaded pages in the `document_pages` collection in MongoDB. Errors are
  stored in the `document_pages_errors` collection.

---

### 4. (optional) **`drop_duplicated_document_pages.py`**

- **Purpose:** Identifies and removes duplicate pages from the `document_pages` collection in
  MongoDB.
- **Note:** You should to rerun `download_document_pages.py` after running this script to update the
  collection.
- **Usage:**
  ```bash
  python drop_duplicated_document_pages.py [OPTIONS]
  ```
- **Arguments:**

  | Argument         | Description                    | Default |
       |------------------|--------------------------------|---------|
  | `--db-uri`       | MongoDB URI.                   | None    |
- **Output:** Cleans up the `document_pages` collection by deleting duplicate pages.

---

### 5. **`save_pages_from_db_to_file.py`**

- **Purpose:** Exports document pages and errors from MongoDB to Parquet files for further
  processing.
- **Usage:**
  ```bash
  python save_pages_from_db_to_file.py [OPTIONS]
  ```
- **Arguments:**

  | Argument         | Description                    | Default |
       |------------------|--------------------------------|---------|
  | `--db-uri`       | MongoDB URI.                   | None    |
- **Output:** Saves pages to `pages/pages_chunk_*.parquet` in `data/datasets/nsa`.

---

### 6. **`extract_data_from_pages.py`**

- **Purpose:** Extracts structured data from downloaded document pages.
- **Usage:**
     ```bash
     python extract_data_from_pages.py [OPTIONS]
     ```
- **Arguments:**

  | Argument         | Description                                    | Default                |
       |------------------|------------------------------------------------|------------------------|
  | `--n-jobs`       | Number of parallel workers.                    | `10`                  |
- **Output:** Saves processed data in Parquet files within `NSA_DATA_PATH/dataset`.
