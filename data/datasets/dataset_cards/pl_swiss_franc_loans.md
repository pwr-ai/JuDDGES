---
language:
- pl
license: cc-by-4.0
tags:
- information-extraction
- legal
- swiss-franc-loans
- polish-court
annotations_creators:
- expert-generated, machine-generated
pretty_name: Polish Court Swiss Franc Loans
size_categories:
- 1K<n<10K
source_datasets:
- JuDDGES/pl-court-raw
task_categories:
- text2text-generation
task_ids:
- abstractive-qa
configs:
- config_name: default
  data_files:
  - split: train
    path: train.json
  - split: test
    path: test.json
  - split: annotated
    path: annotated.json

dataset_info:
  features:
    - name: context
      dtype: string
    - name: output
      dtype: string
  config_name: default
  splits:
  - name: train
    num_bytes: 341276419
    num_examples: 4933
  - name: test
    num_bytes: 46032333
    num_examples: 666
  - name: annotated
    num_bytes: 46032799
    num_examples: 666
  download_size: 517600048
  dataset_size: 433341551
---

# Swiss Franc Loans Judgments - Information Extraction

Dataset for training and evaluating Large Language Models (LLMs) for information extraction in domain of Polish court judgments ragarding Swiss Franc loans cases.

## Dataset Details

### Dataset Description

- **Curated by:** Łukasz Augustyniak, Jakub Binkowski, Albert Sawczyn
- **Funded by:** CHIST ERA call ORD “Open & Re-usable Research Data & Software” (Judicial Decision Data Gathering, Encoding and Sharing/ No ANR-23-CHRO0001)
- **Language(s) (NLP):** Polish
- **License:** [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

The instruction dataset for Polish was created using a semi-automatic process. First, legal experts (lawyers) designed extraction schemas and coding guidelines for the key legal attributes to be extracted from judgments. Using these schemas, we leveraged large language models (LLMs) to automatically extract candidate values for each attribute from the raw judgment texts. To ensure high quality, a subset of these automatically processed examples was sampled and reviewed by lawyers, who validated the extraction accuracy and provided corrections where necessary. This process allowed us to efficiently scale annotation while maintaining expert-level quality for the final dataset. The resulting instruction dataset thus combines the scalability of LLM-based extraction with the reliability of human validation.

### Dataset Sources

Dataset is primarily based on the [JuDDGES/pl-court-raw](https://huggingface.co/datasets/JuDDGES/pl-court-raw) dataset, which is curated from the [Polish Court Portal](https://orzeczenia.ms.gov.pl/).

## Uses

### Direct Use

The dataset should be used for training and evaluating Large Language Models (LLMs) for information extraction in domain of Polish court judgments ragarding Swiss Franc loans.

### Out-of-Scope Use

The datasets should not be used for legal research analysis as it might not fully present the current state of the law practice.

## Dataset Structure

Dataset is partitioned into 3 splits, as described in the following table.

| Split     | Annotation                                      | #samples |
|-----------|------------------------------------------------|----------|
| train     | automatic (`gpt-4.1-mini-2025-04-14`)          | 4933     |
| test      | automatic (`gpt-4.1-2025-04-14`)               | 666      |
| annotated | manual (with pre-annotations from `test` split) | 666      |

Each split has exactly two columns:

- `context`: the full text of the court judgment
- `output`: the JSON object with the extracted information

The output JSON object has the following fields:

| Field Name | English Translation | Description |
|------------|-------------------|-------------|
| sprawa_frankowiczow | Swiss Franc Loan Case | Whether the case concerns a Swiss Franc (CHF) loan |
| apelacja | Court of Appeal | Designation of the court of appeal where the case is being heard |
| data_wyroku | Judgment Date | Date of the judgment |
| typ_sadu | Court Type | Type of court hearing the case |
| instancja_sadu | Court Instance | Whether the court is of first instance or appellate |
| podstawa_prawna | Legal Basis | Legal basis for the claim |
| podstawa_prawna_podana | Legal Basis Provided | Whether the plaintiff provided a legal basis |
| rodzaj_roszczenia | Type of Claim | Type of claim |
| modyfikacje_powodztwa | Claim Modifications | Whether there were modifications to the claim |
| typ_modyfikacji | Type of Modification | Type of claim modification |
| status_kredytobiorcy | Borrower Status | Status of the borrower |
| wspoluczestnictwo_powodowe | Plaintiff Co-participation | Whether there is co-participation on the plaintiff's side |
| wspoluczestnictwo_pozwanego | Defendant Co-participation | Whether there is co-participation on the defendant's side |
| typ_wspoluczestnictwa | Type of Co-participation | Type of co-participation |
| strony_umowy | Contract Parties | Whether the plaintiff was a party to the contract or a legal successor |
| wczesniejsze_skargi_do_rzecznika | Previous Complaints to Ombudsman | Whether there were previous complaints to the financial ombudsman |
| umowa_kredytowa | Loan Agreement | Whether the loan agreement was concluded directly with the bank or through an intermediary |
| klauzula_niedozwolona | Prohibited Clause | Existence of a prohibited clause in the contract |
| wpisana_do_rejestru_uokik | Registered with UOKiK | Whether the clause is registered in the UOKiK register of prohibited clauses |
| waluta_splaty | Payment Currency | Currency of payment |
| aneks_do_umowy | Contract Annex | Whether there was an annex to the contract |
| data_aneksu | Annex Date | Date of the annex |
| przedmiot_aneksu | Annex Subject | What the annex concerned |
| status_splaty_kredytu | Loan Repayment Status | Whether the loan was repaid |
| rozstrzygniecie_sadu | Court Decision | Decision |
| typ_rozstrzygniecia | Type of Decision | Type of decision |
| sesja_sadowa | Court Session | Whether the judgment was issued during a hearing or in camera |
| dowody | Evidence | What evidence was presented |
| oswiadczenie_niewaznosci | Invalidity Declaration | Whether the plaintiff's declaration about the effects of contract invalidity was received |
| odwolanie_do_sn | Reference to Supreme Court | Whether reference was made to Supreme Court jurisprudence |
| odwolanie_do_tsue | Reference to CJEU | Whether reference was made to Court of Justice of the EU jurisprudence |
| teoria_prawna | Legal Theory | Legal theory on which the judgment was based |
| zarzut_zatrzymania | Retention Objection | Whether the retention objection was considered |
| zarzut_potracenia | Set-off Objection | Whether the set-off objection was considered |
| odsetki_ustawowe | Statutory Interest | Whether statutory interest was considered |
| data_rozpoczecia_odsetek | Interest Start Date | Date from which interest began to accrue |
| koszty_postepowania | Court Costs | Whether court costs were awarded |
| beneficjent_kosztow | Cost Beneficiary | Which party was awarded court costs |
| zabezpieczenie_udzielone | Security Granted | Whether security was granted |
| rodzaj_zabezpieczenia | Type of Security | Type of security |
| zabezpieczenie_pierwsza_instancja | First Instance Security | Whether security was granted by the court of first instance |
| czas_trwania_sprawy | Case Duration | Time from filing the lawsuit to judgment |
| wynik_sprawy | Case Outcome | Assessment of whether the bank or borrower won the case |
| szczegoly_wyniku_sprawy | Case Outcome Details | Details regarding the case outcome |

## Dataset Creation

### Curation Rationale

The dataset was prepared to evaluate the ability of LLMs to extract information from court judgments in the particular use case of Swiss Franc loans.

### Source Data

The original source of data is the [JuDDGES/pl-court-raw](https://huggingface.co/datasets/JuDDGES/pl-court-raw) dataset which were curated from the [Polish Court Portal](https://orzeczenia.ms.gov.pl/).

#### Data Collection and Processing

1. Selecting the cases (query to [Weaviate](https://weaviate.io/) vector database with judgments embeddings)
2. Automated extraction of information from the cases using `gpt-4.1-mini-2025-04-14` and `gpt-4.1-2025-04-14` models
3. Reviewing the extracted information by a legal expert
4. Final preprocessing of the data (filtering items with context exceeding 64000 tokens)

#### Who are the source data producers?

The judgments were originally created by the judges of the Polish courts and published via the [Polish Court Portal](https://orzeczenia.ms.gov.pl/), which we also curated into the [JuDDGES/pl-court-raw](https://huggingface.co/datasets/JuDDGES/pl-court-raw) dataset.

### Annotations

The data contains two types of annotations:

- automatic annotations generated by an LLM
- manual annotations created by a human legal expert

#### Annotation process

To obtain the `train` split, which is intended to be used for training the LLM, we used `gpt-4.1-mini-2025-04-14` model to annotate the data. We asked the model to extract the information from the `context` according to the provided schema which contains the description of the expected output JSON object, especially the type and meaning of each field (see Table with field descriptions). To obtain the `test` split, we used `gpt-4.1-2025-04-14` model to annotate the data in the same way as for the `train` split. We decided to use larger model to obtain higher quality test data and potentially better pre-annotations for the `annotated` split. Finally, to obtain the `annotated` split, we used the `test` split and manually annotated it with the help of a legal expert.

#### Who are the annotators?

Professional legal expert.

#### Personal and Sensitive Information

The `context` column contains the full text of the judgment. While court employees perform anonymization of the judgments before publishing them, several personal information, like name of the judges, is still present in the data, as it permitted by law. Therefore, the data adhers to GDPR regulations.

## Bias, Risks, and Limitations

The curation process might not find all cases present at the time of the curation which are related to Swiss Franc loans.

### Recommendations

The dataset should not be used for legal research analysis as it might not fully present the current state of the law practice.

## Citation

**BibTeX:**

```
TBA
```

**APA:**

```
TBA
```

## Dataset Card Contact

- [lukasz.augustyniak@pwr.edu.pl](mailto:lukasz.augustyniak@pwr.edu.pl)
- [jakub.binkowski@pwr.edu.pl](mailto:jakub.binkowski@pwr.edu.pl)
