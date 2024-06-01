import os
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

def extract_information_from_xml(xml_content):
    soup = BeautifulSoup(xml_content, 'lxml')

    # Extract required fields
    _id = soup.find('uk:hash').text
    signature = soup.find('neutralcitation').text if soup.find('neutralcitation') else None
    hearing_date = soup.find('hearingdate').text if soup.find('hearingdate') else None
    date = hearing_date.strip() if hearing_date else None
    publication_date = soup.find('frbrwork').find('frbrdate')['date']
    court_type = soup.find('courttype').text if soup.find('courttype') else None

    # Get the excerpt
    header_text = soup.header.get_text(separator=' ', strip=True)
    excerpt = header_text[:500]

    # Get the full content of the judgment body as XML string
    judgment_body = soup.find('judgmentbody')
    content = str(judgment_body) if judgment_body else None

    # Get the judges list
    judges = [judge.get_text() for judge in soup.find_all('judge')]

    # Get case numbers
    case_numbers = [case_number.get_text() for case_number in soup.find_all('p', class_='CoverText') if
                    'Case Nos:' in case_number.text]
    case_numbers = [num.strip() for sublist in case_numbers for num in sublist.replace('Case Nos:', '').split()]

    return {
        "_id": _id,
        "signature": signature,
        "date": date,
        "publicationDate": publication_date,
        "type": court_type,
        "excerpt": excerpt,
        "content": content,
        "judges": judges,
        "caseNumbers": case_numbers
    }


def process_directory(directory_path, output_file):
    with open(output_file, 'w') as jsonl_file:
        xml_files = [f for f in os.listdir(directory_path) if f.endswith('.xml')]
        for filename in tqdm(xml_files, desc="Processing XML files"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as xml_file:
                xml_content = xml_file.read()
                judgment_data = extract_information_from_xml(xml_content)
                jsonl_file.write(json.dumps(judgment_data) + '\n')

directory_path = '/home/stirunag/work/github/ML4-legal-documents/judgements_xml/dump/'
output_file = '/home/stirunag/work/github/ML4-legal-documents/judgements_xml/englad_wales_data.jsonl'

process_directory(directory_path, output_file)
