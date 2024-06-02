import requests
from bs4 import BeautifulSoup
import pandas as pd
from multiprocessing import Pool
import os
import time
from tqdm import tqdm

# Define the base URL
base_url = "https://caselaw.nationalarchives.gov.uk/judgments/advanced_search?query=&court=ewca%2Fcrim&order=date&per_page=50&page="
num_pages = 124
output_folder = "dump"
csv_file = 'judgments.csv'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)


# Scrape data from a single page
def scrape_page(page_number):
    url = base_url + str(page_number)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []

    for li in soup.select('ul.judgment-listing__list > li'):
        title_tag = li.find('a')
        date_tag = li.find('time')

        if title_tag and date_tag:
            href = title_tag['href']
            title = title_tag.text.strip()
            date = date_tag.text.strip()
            link = "https://caselaw.nationalarchives.gov.uk" + href
            results.append((title, link, date))

    return results


# Download XML files
def download_xml(data):
    title, link, date, sno = data
    date_formatted = pd.to_datetime(date).strftime('%Y_%m_%d')
    xml_url = link + "/data.xml"
    file_name = f"{date_formatted}-{sno}.xml"
    file_path = os.path.join(output_folder, file_name)

    response = requests.get(xml_url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

    time.sleep(1)  # Pause to avoid blocking IP address


# Initialize CSV file
if not os.path.exists(csv_file):
    pd.DataFrame(columns=['Title', 'Link', 'Date', 'SNo']).to_csv(csv_file, index=False)

# Scrape all pages and process data incrementally
sno = 1
for page in tqdm(range(1, num_pages + 1), desc="Scraping pages"):
    results = scrape_page(page)

    # Add serial number to each result
    results_with_sno = [(title, link, date, sno + i) for i, (title, link, date) in enumerate(results)]
    sno += len(results)

    # Save results to CSV incrementally
    df = pd.DataFrame(results_with_sno, columns=['Title', 'Link', 'Date', 'SNo'])
    df.to_csv(csv_file, mode='a', header=False, index=False)

    # Download XML files
    with Pool() as pool:
        pool.map(download_xml, results_with_sno)

print("Scraping and downloading completed successfully!")
