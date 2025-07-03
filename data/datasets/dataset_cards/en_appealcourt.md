---
language:
- en
license: other
license_name: open-government-license-v3
license_link: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
tags:
- legal
- appeal court
- judgments
- information-extraction
annotations_creators: 
- expert-generated, machine-generated
pretty_name: English Appeal Court Judgments
size_categories:
- n<1K
source_datasets:
- JuDDGES/en-court-raw
task_categories:
- text2text-generation
task_ids:
- abstractive-qa
configs:
- config_name: default
  data_files:
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
  - name: test
    num_bytes: 15615422
    num_examples: 573
  - name: annotated
    num_bytes: 15525317
    num_examples: 573
  download_size: 31806589
  dataset_size: 31140739
---
 

# English Appeal Court Judgments

Dataset for training and evaluating large language models (LLMs) for information extraction in the domain of publicly available England and Wales Court of Appeal (Criminal Division) judgments. The coding schema includes 43 annotated codes across several categories of judgments, including:

1. Sampling judgments/cases

2. Extracting basic information on court hearings

3. Extracting basic information on offence, trial, and sentence

4. Extracting basic information on appeals

## Dataset Details

### Dataset Description

The instruction dataset for England and Wales was created using a manual process. Legal experts first defined extraction schemas and coding guidelines in a workshop. We then built an HTML parser to download all 6,154 Criminal Division Court of Appeal judgments (as XML, up to 15 May 2024) under the Crown Copyright and Open Government licences. Yearly word-length statistics (min/25th/median/75th/max) revealed a bi-modal distribution, a few extreme outliers (>40 k words), and an incomplete 2024 tranche.

For curator training, judgments were stratified into five token-length bins, with five cases sampled per bin (110 total; four later removed), yielding 106 candidates. Twenty were used for inter-rater (20 distinct coders) and intra-rater (10 recoded) reliability checks. Once reliability thresholds (>80%) were achieved, two further stratified batches of 800 judgments each were assigned to Curator 1 (531 annotations) and Curator 2 (96 annotations). After careful cleaning, 573 annotated judgments were selected for the final dataset.

Throughout, curators followed strict guidelines on text-span boundaries, code-type selection, comment conventions, and spreadsheet documentation, all anchored to a shared coding scheme.

- **Curated by:** Łukasz Augustyniak, Jakub Binkowski, Albert Sawczyn, Prof. Mandeep Dhami, Prof. David Windridge
- **Funded by:** CHIST ERA call ORD "Open & Re-usable Research Data & Software" (Judicial Decision Data Gathering, Encoding and Sharing/ No ANR-23-CHRO0001)
- **Language(s) (NLP):** English
- **License:** [More Information Needed]

### Dataset Sources

The dataset is primarily based on the [JuDDGES/en-court-raw dataset](https://huggingface.co/datasets/JuDDGES/en-court-raw), which is curated from publicly available judgments from the [Court of Appeal (Criminal Division)](https://caselaw.nationalarchives.gov.uk/search?query=&from_date_0=&from_date_1=&from_date_2=&to_date_0=&to_date_1=&to_date_2=&court=ewca%2Fcrim&party=&judge=).

## Uses

### Direct Use

The dataset should be used for evaluating Large Language Models (LLMs) for information extraction in domain of England and Wales Court of Appeal (Criminal Division) judgments.

### Out-of-Scope Use

The datasets should not be used for legal research analysis as it might not fully present the current state of the law practice.

## Dataset Structure

Dataset is partitioned into 3 splits, as described in the following table.

| Split     | Annotation                                      | #samples |
|-----------|------------------------------------------------|----------|
| test      | automatic (`gpt-4.1-2025-04-14`)               | 573      |
| annotated | manual (without pre-annotations)               | 573      |

Each split has exactly two columns:

- `context`: the full text of the court judgment
- `output`: the JSON object with the extracted information

The output JSON object has the following fields:

| Field Name | Description |
|------------|-------------|
| ConvCourtName | Name(s) of the court where the defendant was convicted or pleaded guilty |
| ConvictPleaDate | Date(s) on which the defendant was convicted or pleaded guilty |
| ConvictOffence | Offence(s) of which the defendant was convicted |
| AcquitOffence | Offence(s) of which the defendant was acquitted |
| ConfessPleadGuilty | Did the defendant confess or plead guilty? |
| PleaPoint | Stage at which the plea was entered |
| RemandDecision | Remand decision post-conviction |
| RemandCustodyTime | Duration in days of any remand in custody |
| SentCourtName | Name(s) of the court where the defendant was sentenced |
| Sentence | Sentence(s) imposed |
| SentServe | How sentences run |
| WhatAncillary | Ancillary orders applied by the court |
| OffSex | Gender(s) of the defendant(s) |
| OffAgeOffence | Age of defendant at offence |
| OffJobOffence | Employment status at offence |
| OffHomeOffence | Accommodation status at offence |
| OffMentalOffence | Learning/developmental or mental-health issues noted |
| OffIntoxOffence | Intoxication status |
| OffVicRelation | Relationship defendant→victim |
| VictimType | Type of victim |
| VicNum | Number of victims or ratio |
| VicSex | Gender(s) of victim(s) |
| VicAgeOffence | Age of victim(s) at offence |
| VicJobOffence | Employment status of victim(s) |
| VicHomeOffence | Accommodation status of victim(s) |
| VicMentalOffence | Learning/developmental or mental-health issues for victim(s) |
| VicIntoxOffence | Victim's intoxication status |
| ProsEvidTypeTrial | Evidence types by prosecution |
| DefEvidTypeTrial | Evidence types by defence |
| PreSentReport | Risk level from pre-sentence report |
| AggFactSent | Aggravating factors at sentencing |
| MitFactSent | Mitigating factors at sentencing |
| VicImpactStatement | Was a victim impact statement provided? |
| Appellant | Who brings the appeal |
| CoDefAccNum | Number of co-defendants/co-accused |
| AppealAgainst | Ground(s) for appeal |
| AppealGround | Specific legal grounds of appeal |
| SentGuideWhich | Sentencing guidelines or statutes cited |
| AppealOutcome | Outcome of the appeal |
| ReasonQuashConv | Reasons for quashing conviction |
| ReasonSentExcessNotLenient | Reasons why sentence was unduly excessive |
| ReasonSentLenientNotExcess | Reasons why sentence was unduly lenient |
| ReasonDismiss | Reasons for dismissal of the appeal |

## Dataset Creation

The dataset was created strictly adhering to the Annotation Guidelines for Curators:

### Source Data

The dataset sourced from [JuDDGES/en-court-raw dataset](https://huggingface.co/datasets/JuDDGES/en-court-raw), which is curated from publicly available judgments from the [Court of Appeal (Criminal Division)](https://caselaw.nationalarchives.gov.uk/search?query=&from_date_0=&from_date_1=&from_date_2=&to_date_0=&to_date_1=&to_date_2=&court=ewca%2Fcrim&party=&judge=).


#### Data Collection and Processing

The dataset consists of publicly available judgments from the Court of Appeal (Criminal Division) for England and Wales. Legal experts first defined extraction schemas and coding guidelines during a dedicated workshop. An HTML parser was subsequently developed to automatically download all 6,154 judgments (XML format) published by 15 May 2024 under the Crown Copyright and Open Government licences.

Yearly word-length statistics (minimum, 25th percentile, median, 75th percentile, maximum) indicated a bi-modal distribution, the presence of extreme outliers (judgments exceeding 40,000 words), and incomplete data for 2024. Judgment lengths showed considerable variability across years:
- 2003–2007: trend toward shorter judgments.
- 2008–2011: increasing lengths, peaking at 4,618 words in 2011.
- 2012–2016: consistently high median lengths.
- 2017–2024: fluctuations, notable peaks in 2019 and 2021, and declines in 2023 and 2024.

Significant outliers in judgment length occurred in 2015 and 2020, surpassing 70,000 and 115,000 words, respectively.


#### Who are the source data producers?

The original source of data is the from [JuDDGES/en-court-raw dataset](https://huggingface.co/datasets/JuDDGES/en-court-raw), which is curated from publicly available judgments from the [Court of Appeal (Criminal Division)](https://caselaw.nationalarchives.gov.uk/search?query=&from_date_0=&from_date_1=&from_date_2=&to_date_0=&to_date_1=&to_date_2=&court=ewca%2Fcrim&party=&judge=).

These judgments are official court documents published by the Court of Appeal (Criminal Division) in England and Wales.


### Annotations

Initially 20 documents have been given to two curators as part of training as well for checking the consistency of the annotations. Once a reliable annotion evaluation was reached the curators we assigned with additional 800 judgements to curate.

Annotaions have been curated using (Hypothes.is)[https://hypothes.is] Chrome plugin.

**Annotation Training Sample Selection:**  
- Judgments were stratified into five token-length bins (min, 25th percentile, median, 75th percentile, max).  
- 5 judgments were sampled per bin, resulting in 110 cases (106 after removing four for legal reasons).  
- 20 were selected for reliability checks:  
  - Inter-rater: 20 coders annotated the same 20 documents  
  - Intra-rater: 10 of those documents were re-coded by the same curators  
- Once >80% agreement was achieved (Krippendorff’s α), two additional batches of 800 cases were assigned:  
  - Curator 1: 531 annotations  
  - Curator 2: 96 annotations  
- Final cleaned dataset includes 573 annotated judgments.

#### Annotation process

Annotations were created using the Hypothes.is Chrome extension.

**Steps:**
1. Install Hypothes.is from the Chrome Web Store.  
2. Open the court judgment URL in the browser.  
3. Use the extension to highlight and annotate relevant text spans.  
4. Add tags as per the shared tagging scheme.  
5. Add relevant comments (or a dash \"–\" if none).  
6. Submit to the correct annotation group.

**Annotation Guidelines:**
- All annotations must remain within a single sentence.
- Spans must match predefined code types exactly.
- Comments must be relevant to the hypothesis.  
- Missing or inapplicable fields should be marked as “N/A,” “–,” or “Don’t Know.”

#### Who are the annotators?

The annotators were post-graduate level students from Middlesex University under the supervison of Prof. Mandeep Dhami and Prof. David Windrige.

#### Personal and Sensitive Information

The context column contains the full text of the judgment. While court employees perform anonymization of the judgments before publishing them, several personal information, like name of the judges, offender names (aged more than 18) are still present in the data, as it permitted by law.

## Bias, Risks, and Limitations

The curation process might not find all cases present at the time of the curation which has a cut of date of May 2024.


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
- [Prof. Mandeep Dhami](mailto:m.dhami@mdx.ac.uk)
- [Prof. David Windridge](mailto:d.windridge@mdx.ac.uk)
