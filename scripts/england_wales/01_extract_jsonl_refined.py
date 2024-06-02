import os
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from multiprocessing import Pool

def extract_appeal_type(text):
    patterns = [
        (r'appeal\s+against\s+\S+\s+sentence\s+or\s+\S+\s+conviction', 'conviction_sentence'),
        (r'appeal\s+against\s+\S+\s+conviction\s+or\s+\S+\s+sentence', 'conviction_sentence'),
        (r'appeal\s+against\s+\S+\s+conviction', 'conviction'),
        (r'appeal\s+against\s+\S+\s+sentence', 'sentence')
    ]

    for pattern, appeal_type in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return appeal_type
    return None


def extract_appeal_outcome(text):
    outcome_patterns = {
        'granted': r'appeal\s+is\s+granted',
        'dismissed': r'appeal\s+is\s+dismissed',
        'refused': r'appeal\s+is\s+refused',
        'allowed': r'appeal\s+is\s+allowed'
    }

    for outcome, pattern in outcome_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return outcome
    return None

def extract_and_clean_judges(paragraphs):
    judges = []
    for para in paragraphs:
        text = para.get_text(strip=True)
        if re.search(r'\bJustice\b|\bJudge\b|\bSIR\b|\bHonour\b|\bHHJ\b', text, re.IGNORECASE):
            # Remove text within parentheses
            cleaned_text = re.sub(r'\([^)]*\)', '', text).strip()
            # Remove dashes and any text following them
            cleaned_text = re.sub(r'-.*', '', cleaned_text).strip()
            # Check for specific keywords and ensure it's not empty or unwanted text
            if cleaned_text and 'Royal Courts of Justice' not in cleaned_text and cleaned_text != "THE LORD CHIEF JUSTICE OF ENGLAND AND WALES":
                judges.append(cleaned_text)
    return judges

def categorize_court(court_name):
    if 'SUPREME_COURT' in court_name:
        return 'supreme_court'
    elif "HIGH_COURT" in court_name and "ADMINISTRATIVE_COURT" in court_name:
        return 'high_court_administrative_court'
    elif 'HIGH_COURT' in court_name and 'DIVISIONAL_COURT' in court_name:
        return 'high_court_division_court'
    elif 'HIGH_COURT' in court_name:
        return 'high_court'
    elif 'CIVIL_AND_CRIMINAL' in court_name:
        return 'civil_criminal_court'
    elif 'MARTIAL' in court_name:
        return 'martial_court'
    elif 'DIVISIONAL_COURT' in court_name:
        return 'division_court'
    else:
        return 'crown_court'

def extract_information_from_xml(xml_content, file_name):
    soup = BeautifulSoup(xml_content, 'xml')  # Using 'xml' parser for handling namespaces

    # Extract required fields
    _id = soup.find('uk:hash').text if soup.find('uk:hash') else None
    citation = soup.find('uk:cite').text if soup.find('uk:cite') else None
    signature = citation.split('] ')[1] if citation else None  # Removing the year part
    if signature:
        signature = signature.replace(' ', '_')
    hearing_date = soup.find('hearingdate').text if soup.find('hearingdate') else None
    date = hearing_date.strip() if hearing_date else None
    publication_date = soup.find('FRBRdate', {'name': 'judgment'})['date'] if soup.find('FRBRdate',
                                                                                        {'name': 'judgment'}) else None

    court_type_tags = soup.find_all('courtType')
    # Use a set to collect unique court types
    unique_court_types = set(
        re.sub(r'\([^)]*\)', '', tag.get_text(strip=True)).replace(' ', '_') for tag in court_type_tags)

    # Join the unique court types
    court_type_ = "_".join(unique_court_types)
    court_type_ = re.sub(r'_+', '_', court_type_).strip('_')

    # Categorize the combined court types
    court_type = categorize_court(court_type_)

    # Get the excerpt
    header_text = soup.header.get_text(separator=' ', strip=True) if soup.header else ""
    excerpt = header_text[:500]

    # Get the full content of the header and judgment body as text
    header_content = soup.header.get_text(separator='\n', strip=True) if soup.header else ""
    judgment_body_content = soup.find('judgmentBody').get_text(separator='\n', strip=True) if soup.find(
        'judgmentBody') else ""
    content = header_content + "\n" + judgment_body_content

    # Get the judges list
    # Get the judges list from TLCPerson elements
    judges = [judge['showAs'] for judge in soup.find_all('TLCPerson') if 'showAs' in judge.attrs and re.search(r'\bJustice\b|\bJudge\b|\bSIR\b|\bHonour\b|\bHHJ\b', judge['showAs'], re.IGNORECASE)]
    # Filter judges using regex criteria
    judges = [judge for judge in judges if
              re.search(r'\bJustice\b|\bJudge\b|\bSIR\b|\bHonour\b|\bHHJ\b', judge, re.IGNORECASE)]

    # If no judges found, get text from <judge> elements
    if not judges:
        judges = [judge.get_text(strip=True) for judge in soup.find_all('judge')]

    # If no judges found, use regex to extract them from header content
    if not judges and soup.header:
        # Extract all <p> tags
        paragraphs = soup.header.find_all('p')
        judges = extract_and_clean_judges(paragraphs)

    # If still no judges found, look for text in <p> tags with style="text-align:center"
    if not judges:
        centered_paragraphs = soup.find_all('p', style=lambda x: x and 'text-align:center' in x)
        judges.extend(extract_and_clean_judges(centered_paragraphs))

    # If still no judges found, look for text in <p> tags with style="text-align:right"
    if not judges:
        right_aligned_paragraphs = soup.find_all('p', style=lambda x: x and 'text-align:right' in x)
        judges.extend(extract_and_clean_judges(right_aligned_paragraphs))

    # Filter judges using regex criteria
    judges = [judge for judge in judges if
              re.search(r'\bJustice\b|\bJudge\b|\bSIR\b|\bHonour\b|\bHHJ\b', judge, re.IGNORECASE)]

    # Extract URIs
    xml_uri = soup.find('FRBRManifestation').find('FRBRuri')['value'] if soup.find('FRBRManifestation') and soup.find('FRBRManifestation').find('FRBRuri') else None
    uri = soup.find('FRBRWork').find('FRBRuri')['value'] if soup.find('FRBRWork') and soup.find('FRBRWork').find('FRBRuri') else None

    # Extract legislation texts
    legislation_tags = soup.find_all('ref', {'uk:type': 'legislation'})
    legislation_texts = set(tag.get_text() for tag in legislation_tags)
    legislation_list = list(legislation_texts)  # Convert set to list to remove duplicates

    # Extract case references
    case_tags = soup.find_all('ref', {'uk:type': 'case'})
    case_references = set(tag.get_text() for tag in case_tags)
    case_references_list = list(case_references)  # Convert set to list to remove duplicates

    # Extract case numbers
    case_numbers = set()
    docket_number_tags = soup.find_all('docketNumber')
    for tag in docket_number_tags:
        case_numbers.add(tag.get_text())

    # Extract case numbers from <p class="CoverText"> tags containing "Case No:"
    cover_text_tags = soup.find_all('p', class_='CoverText')
    case_no_pattern = re.compile(r'Case No:\s*(.*)')
    for tag in cover_text_tags:
        match = case_no_pattern.search(tag.get_text())
        if match:
            case_numbers.update([num.strip() for num in match.group(1).split(',')])

    # If no case numbers found, look for text in <p> tags with style="text-align:right"
    if not case_numbers:
        right_aligned_paragraphs = soup.find_all('p', style=lambda x: x and 'text-align:right' in x)
        case_no_pattern = re.compile(r'\b\d{4}/\d{4}/\w+\b|\d{6}')
        for tag in right_aligned_paragraphs:
            matches = case_no_pattern.findall(tag.get_text())
            case_numbers.update(matches)

    # Extract appeal type
    appeal_type = extract_appeal_type(content)

    # Extract appeal outcome
    appeal_outcome = extract_appeal_outcome(content)

    def null_if_empty(value):
        return value if value else None

    return {
        "_id": null_if_empty(_id),
        "citation": null_if_empty(citation),
        "signature": null_if_empty(signature),
        "date": null_if_empty(date),
        "publicationDate": null_if_empty(publication_date),
        "type": null_if_empty(court_type),
        "excerpt": null_if_empty(excerpt),
        "content": null_if_empty(content),
        "judges": null_if_empty(judges),
        "caseNumbers": null_if_empty(list(case_numbers)),
        "citation_references": null_if_empty(case_references_list),
        "legislation": null_if_empty(legislation_list),
        "file_name": null_if_empty(file_name),
        "appeal_type": null_if_empty(appeal_type),
        "appeal_outcome": null_if_empty(appeal_outcome),
        "xml_uri": null_if_empty(xml_uri),
        "uri": null_if_empty(uri)
    }

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as xml_file:
        xml_content = xml_file.read()
        file_name = os.path.basename(file_path)
        return extract_information_from_xml(xml_content, file_name)

def process_directory(directory_path, output_file):
    xml_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.xml')]

    with Pool() as pool, open(output_file, 'w') as jsonl_file:
        for judgment_data in tqdm(pool.imap(process_file, xml_files), total=len(xml_files),
                                  desc="Processing XML files"):
            jsonl_file.write(json.dumps(judgment_data) + '\n')

directory_path = '/home/stirunag/work/github/ML4-legal-documents/judgements_xml/dump/'
output_file = '/home/stirunag/work/github/ML4-legal-documents/judgements_xml/england_wales_data_refined_7.jsonl'

process_directory(directory_path, output_file)
