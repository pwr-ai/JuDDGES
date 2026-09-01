# Croissant RAI Specification

## Version 1.0

Published: 2024/03/06

[http://mlcommons.org/croissant/RAI/1.0](http://mlcommons.org/croissant/RAI/1.0)

### Authors:

- Mubashara Akhtar* (King's College London)
- Nitisha Jain* (King's College London)
- Joan Giner-Miguelez (Universitat Oberta de Catalunya)
- Omar Benjelloun (Google)
- Elena Simperl (King's College London & ODI)
- Lora Aroyo (Google)
- Rajat Shinde (NASA-IMPACT)
- Michael Kuchnik (Meta)

## Introduction & overview

As AI advances rapidly, there is growing recognition that we must "explore, understand, manage, and assess its economic, social, and environmental impacts." Existing responsible AI approaches range from manual assessments and documentation to system architectures and algorithms supporting developers.

The Croissant format supports both approaches:

1. It provides a machine-readable way to capture and publish metadata about AI datasets, making documentation easier to publish, share, discover, and reuse.
2. It records at a granular level how datasets were created, processed, and enriched throughout their lifecycle—a process meant to be automated by integrating Croissant with popular AI development environments.

Dataset documentation is a main instrument for operationalizing responsible AI. This specification describes the responsible AI (RAI) aspects of Croissant, which were defined through a multi-step vocabulary engineering process:

1. Define use cases for the RAI-Croissant extension
2. Compare and contrast existing dataset documentation vocabularies
3. Specify scope of RAI Croissant extension via competency questions
4. Define the RAI conceptualization on top of Croissant
5. Implement the conceptual model on top of Croissant
6. Evaluate the implementation through example annotations

A Croissant RAI dataset description consists of properties aligned to use cases. An initial set of use cases was defined during the vocabulary engineering process. The list remains open to future community extensions.

### Initial use cases

- The data life cycle
- Data labeling
- Participatory data
- AI safety and fairness evaluation
- Traceability
- Regulatory Compliance
- Inclusion

## Prerequisites

RAI properties build on the [schema.org/Dataset](http://schema.org/Dataset) vocabulary.

The Croissant RAI vocabulary is defined in its owned namespace, identified by the IRI: `http://mlcommons.org/croissant/RAI/`

This namespace IRI is abbreviated using the prefix `rai`.

The presented vocabulary relies on the following namespaces:

| Prefix | IRI | Description |
|--------|-----|-------------|
| sc | http://schema.org/ | The schema.org namespace |
| cr | http://mlcommons.org/croissant/ | MLCommons Croissant namespace |

The Croissant RAI specification is versioned, with the version included in the URI: `http://mlcommons.org/croissant/RAI/1.0`

Croissant datasets must declare conformance to this specification by including:

```json
"dct:conformsTo" : "http://mlcommons.org/croissant/RAI/1.0"
```

While the Croissant RAI specification is versioned, the Croissant RAI namespace is not, so constructs will maintain stable URIs even when the specification version changes.

## Alignment with existing approaches to ML dataset documentation

The RAI vocabulary was built through careful analysis of existing ML dataset documentation toolkits such as Kaggle and HuggingFace, focusing on RAI-related properties. By identifying properties not in the core Croissant vocabulary, the RAI vocabulary extends Croissant. For example, properties like `dataSocialImpact`, `dataBiases`, and `dataLimitations` were inspired by HuggingFace documentation, while properties like `dataCollection` were mapped from Kaggle documentation.

The following ML dataset documentation toolkits were evaluated:

| Toolkit | Reference |
|---------|-----------|
| Dataset cards | https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md |
| Kaggle metadata | https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata |
| Data nutrition labels | https://datanutrition.org/ |
| Data cards | https://sites.research.google/datacardsplaybook/ |
| Croissant core vocabulary | https://github.com/mlcommons/croissant/blob/main/docs/croissant-spec.md |
| Crowdworksheets | https://arxiv.org/abs/2206.08931 |
| Fairness datasets vocabulary | https://fairnessdatasets.dei.unipd.it/schema/ |
| DescribeML | https://www.sciencedirect.com/science/article/pii/S2590118423000199 |

## Overview: Croissant RAI properties and use cases

The following table provides an overview of Croissant RAI vocabulary and maps them to use cases. The current release includes properties relevant to five of the initial use cases. Future releases will expand on each use case, particularly use cases 5 and 7.

| RAI use case | Croissant RAI properties | Croissant properties | Schema.org properties |
|--------------|--------------------------|----------------------|------------------------|
| Use case 1: The data life cycle | rai:dataLimitations<br>rai:dataCollection<br>rai:useCases<br>rai:dataReleaseMaintenance | cr:distribution<br>cr:isLiveDataset<br>cr:citeAs | sc:creator<br>sc:publisher<br>sc:datePublished<br>sc:dateCreated<br>sc:dateModified<br>sc:version<br>sc:license<br>sc:maintainer |
| Use case 2: Data labeling | rai:annotationPlatform<br>rai:annotationsPerItem<br>rai:annotatorDemographics<br>rai:machineAnnotationTools | cr:distribution<br>cr:isLiveDataset<br>cr:Label | |
| Use case 3: Participatory data | rai:annotationPlatform<br>rai:annotatorDemographics | | sc:participant<br>sc:contributor |
| Use case 4: AI safety and fairness | rai:dataLimitations<br>rai:dataBiases<br>rai:useCases<br>rai:personalSensitiveInformation | | sc:diversityPolicy<br>sc:ethicsPolicy<br>sc:inLanguage |
| Use case 5: Traceability | | | |
| Use case 6: Regulatory compliance | rai:personalSensitiveInformation<br>rai:useCases<br>rai:dataReleaseMaintenance<br>rai:dataManipulationProtocol | | |
| Use case 7: Inclusion | | | |

## Use cases

This section provides an overview of various use cases served by the vocabulary. We distinguish between metadata properties at the dataset level, similar to existing data cards, and at the record level (e.g., extracting information from records as record-level annotations), which are needed for fairness or safety evaluation requiring a granular view of the dataset lifecycle. Records are atomic units of datasets: sentences, conversations, images, videos, etc. Record-level annotations will be aggregated at dataset-level to indicate descriptions such as coverage of concepts, topics, and level of adversariality for safety, and extracting context-specific bias insights.

### Use case 1: The data life cycle (level: dataset)

Key stages of the dataset lifecycle include "motivation, composition, collection process, preprocessing/cleaning/labeling, uses, distribution, and maintenance." Documenting RAI-related properties encourages creators to reflect on processes and improves user understanding.

Information generated throughout the cycle addresses different aspects for responsible data usage:

1. Who created the dataset and for which purpose
2. When the dataset was created
3. Which data sources were used
4. Dataset versioning information with timestamps for each version
5. Data composition, noise, redundancies, and privacy-critical information
6. Data processing (including crowdsourcing information—see use case 2)
7. Intended data uses
8. Dataset maintenance plans

Documenting provenance and lineage of datasets derived from revision, modification, or extension of existing datasets is also relevant.

This use case will be covered by the core vocabulary. Its main purpose is identifying additional properties not yet in Croissant.

### Use case 2: Data labeling (level: dataset or record)

Dataset-level metadata will be aggregated from record-level annotations. These can be achieved through human input, including labels and annotations created via labeling services and crowdsourcing platforms (specifying which platform, how many human labels per record, and annotator demographics if available), or through machine annotations (concept extraction, NER, and tool characteristics enabling replication or extension).

Information about the labeling process helps "understand how the data was created, the sample the labels apply to," making assessment, repetition, replication, and reproduction easier. This increases resulting data reliability.

### Use case 3: Participatory data

Some ML datasets result from well-understood but poorly documented processes. Others emerge from community or collaborative work involving many entities with limited coordination. Examples include citizen science datasets from participatory sensing; Wikidata, created by ~23k editors; or datasets using crowdsourcing platforms.

Documenting participatory elements helps understand biases and limitations, and makes processes easier to monitor, assess, repeat, replicate, and reproduce.

### Use case 4: AI safety and fairness evaluation

Safety and bias information involves "understanding the potential risks and fairness aspects associated with data usage" to prevent unintended, potentially harmful consequences from model training or evaluation. Identifying features for known and intended dataset uses (adversarial datasets for safety evaluation, counterfactual annotations for fairness evaluation) and usage restrictions is necessary.

Accounting for personal and sensitive information can help mitigate risks and support responsible use. Such information is typically gathered at item-level and aggregated at dataset-level in scorecards or nutrition labels.

### Use case 5: Traceability

Data transparency and traceability are critical for responsible AI, especially in high-stakes applications like healthcare and finance. Dataset documentation enhances AI system transparency by allowing understanding of which features or data properties most strongly contributed to model predictions. Knowing feature importance helps explain AI system reasoning and enhances overall explainability.

### Use case 6: Regulatory compliance

Compliance officers and legal teams need data-related information to "assess the dataset's fit to privacy and current regulation laws." Regulations like the European AI Act require documentation about data used to train ML applications. The RAI extension allows structured annotation of this information.

Relevant considerations include:

- **Sensitive and personally identifiable information**: Description of data types like personally identifiable information, sensitive data, or categories subject to GDPR Art. 5 privacy regulations (`rai:personalSensitiveInformation`)
- **Data purposes and limitations**: Information about intended data use and collection purposes (`rai:dataUseCases`), and potential generalization limits and warnings (`rai:dataLimitations`)
- **Data collection processes**: Explanation of how data was collected using fields like `rai:dataCollection`, `rai:dataCollectionType`, and `rai:dataCollectionTimeFrame`
- **Data annotation processes**: Information about annotation processes (`rai:annotationProtocol`), platforms used (`rai:dataAnnotationPlatform`), and validation methods applied (`rai:dataAnnotationAnalysis`)
- Data retention policies: Duration data will be stored considering legal requirements and data protection laws
- Data access control: Information about data access, privilege levels, and implemented control measures
- Data anonymization or pseudonymization: Details about applied techniques, if applicable
- Synthetic data: Methods used for generation, if applicable
- Data sharing agreements: Information about agreements or contracts when sharing data with third parties, including privacy and security provisions
- Data governance and security measures: Documentation of policies and procedures ensuring data security, protection, access control, and breach response

This information is relevant across research, business, and public sector fields.

### Use case 7: Inclusion

Representation of cultural and social demographics of humans is often missing in dataset creation, labeling, and annotation. Documentation of these properties—such as representation of people with disabilities—promotes dataset inclusivity and diversity, enabling wider adoption and accessibility. Lacking representativeness can affect ML system performance, potentially resulting in biased classifiers.

This includes profiling humans involved in dataset creation (active and passive actors) by defining demographic information (`rai:annotatorsDemographics`) and, if datasets represent or gather from people, target collection process demographics (`dataCollectionDemographics`).

## RAI property information

| Property | ExpectedType | Use Case | Cardinality | Description |
|----------|--------------|----------|-------------|-------------|
| rai:dataCollection | sc:Text | Data life cycle | ONE | Description of the data collection process |
| rai:dataCollectionType | sc:Text | Data life cycle | MANY | Define the data collection type. Recommended values: Surveys, Secondary Data analysis, Physical data collection, Direct measurement, Document analysis, Manual Human Curator, Software Collection, Experiments, Web Scraping, Web API, Focus groups, Self-reporting, Customer feedback data, User-generated content data, Passive Data Collection, Others |
| rai:dataCollectionMissingData | sc:Text | Data life cycle | ONE | Description of missing data in structured/unstructured form |
| rai:dataCollectionRawData | sc:Text | Data life cycle | ONE | Description of the raw data, i.e. source of the data |
| rai:dataCollectionTimeframe | sc:DateTime | Data life cycle | MANY | Timeframe in terms of start and end date of the collection process |
| rai:dataImputationProtocol | sc:Text | Compliance | ONE | Description of data imputation process if applicable |
| rai:dataManipulationProtocol | sc:Text | Compliance | ONE | Description of data manipulation process if applicable |
| rai:dataPreprocessingProtocol | sc:Text | Data life cycle | MANY | Description of the steps that were required to bring collected data to a state that can be processed by an ML model/algorithm, e.g. filtering out incomplete entries etc. |
| rai:dataAnnotationProtocol | sc:Text | Data labeling | ONE | Description of annotations (labels, ratings) produced, including how these were created or authored — Annotation Workforce Type, Annotation Characteristic(s), Annotation Description(s), Annotation Task(s), Annotation Distribution(s) |
| rai:dataAnnotationPlatform | sc:Text | Data labeling | MANY | Platform, tool, or library used to collect annotations by human annotators |
| rai:dataAnnotationAnalysis | sc:Text | Data labeling | MANY | Considerations related to the process of converting the "raw" annotations into the labels that are ultimately packaged in a dataset — Uncertainty or disagreement between annotations on each instance as a signal in the dataset, analysis of systematic disagreements between annotators of different socio-demographic group, how the final dataset annotations will relate to individual annotator responses |
| rai:dataReleaseMaintenancePlan | sc:Text | Compliance | MANY | Versioning information in terms of the updating timeframe, the maintainers, and the deprecation policies. |
| rai:personalSensitiveInformation | sc:Text | Compliance | MANY | Sensitive Human Attribute(s) — Gender, Socio-economic status, Geography, Language, Age, Culture, Experience or Seniority, Others (please specify) |
| rai:dataSocialImpact | sc:Text | AI safety and fairness evaluation | ONE | Discussion of social impact, if applicable |
| rai:dataBiases | sc:Text | AI safety and fairness evaluation | MANY | Description of biases in dataset, if applicable |
| rai:dataLimitations | sc:Text | AI safety and fairness evaluation | MANY | Known limitations — Data generalization limits (e.g. related to data distribution, data quality issues, or data sources) and non-recommended uses. |
| rai:dataUseCases | sc:Text | AI safety and fairness evaluation | MANY | Dataset Use(s) — Training, Testing, Validation, Development or Production Use, Fine Tuning, Others (please specify), Usage Guidelines. Recommended uses. |
| rai:annotationsPerItem | sc:Text | Data labeling | ONE | Number of human labels per dataset item |
| rai:annotatorDemographics | sc:Text | Data labeling | MANY | List of demographics specifications about the annotators |
| rai:machineAnnotationTools | sc:Text | Data labeling | MANY | List of software used for data annotation (e.g. concept extraction, NER, and additional characteristics of the tools used for annotation to allow for replication or extension) |

## Examples

Below are examples illustrating how properties related to data collection can be defined. For properties where both `sc:Text` and `sc:ItemList` are allowed, information may be entered in the most suitable format. The attributes in the examples are applicable at the dataset level.

### RAI properties for Geospatial AI-ready dataset

Geospatial AI (GeoAI) refers to "the integration of artificial intelligence techniques with geospatial data, enabling advanced location-based analysis, mapping, and decision-making." GeoAI is powered by data captured by sensors on spaceborne, airborne, and ground platforms along with in-situ sensors. This leads to spatio-temporal heterogeneity in geospatial datasets, enabling multiple applications including weather prediction, earth observation, urban planning, and agricultural crop yield prediction.

Responsible AI emphasizes ethical, transparent, and accountable AI development and deployment, ensuring fair and unbiased outcomes. Geospatial Responsible AI involves ethical considerations in geospatial data acquisition and utilization, addressing potential biases, environmental impact, and privacy concerns, while emphasizing transparency and fairness.

Two examples showcase RAI properties' significance for GeoAI use cases:

1. **Importance of location**: Location or spatial properties are critical for AI-ready dataset credibility for GeoAI. AI predictions and estimations can change with locational accuracy changes. For example, crop yield prediction requires precise training label annotation from agricultural farms. Developing robust, accurate AI models requires precise annotation. Most annotations are approximated due to privacy concerns. Using labeled datasets with AI models can lead to inaccurate predictions. RAI properties like annotator demographics and data preprocessing and manipulation details increase confidence in AI modeling.
2. **Importance of Sampling Strategy and biases**: Due to large training data volumes, sampling is necessary, especially for petabyte-scale datasets. Conventional sampling aims to reduce training data size by masking redundant samples. Uninformed sampling strategies can introduce training data biases causing AI-model inaccuracies. Training datasets with imbalanced class information exemplify such biases. RAI properties describing data biases and limitations enhance pre-training awareness, enabling adoption of better representation techniques.
3. **GeoAI Training Data life cycle**: The temporal specificity of many GeoAI applications renders training data obsolete once designated time windows elapse, limiting continued dataset relevance and effectiveness. This is prominent in disaster monitoring, assessment, and seasonal agricultural crop yield estimation. For such use cases, data lifecycle RAI properties describing collection processes, missing data descriptions, and collection timeframes play key roles in improving AI model applicability.

Below is an example of RAI properties in a Geospatial AI-ready dataset — HLS Burn Scar Scenes dataset — in Croissant format. This openly available dataset on [Hugging Face](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) contains Harmonized Landsat and Sentinel-2 imagery of burn scars and associated masks for 2018-2021 over the contiguous United States.

```json
{
  "@context": {
    "@language": "en",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "sc": "https://schema.org/"
  },
  "@type": "schema.org/Dataset",
  "name": "Name of the dataset",
  "dct:conformsTo": "http://mlcommons.org/croissant/RAI/1.0",
  "rai:dataCollection": "After co-locating the shapefile and HLS scene, the 512x512 chip was formed by taking a window with the burn scar in the center. Burn scars near the edges of HLS tiles are offset from the center. Images were manually filtered for cloud cover and missing data to provide as clean a scene as possible, and burn scar presence was also manually verified.",
  "rai:dataCollectionType": "The dataset comprises 804 512x512 scenes. Each scene contain six bands, and masks have one band.",
  "rai:dataCollectionRawData": "Imagery is from V1.4 of Harmonized Landsat and Sentinel-2 (HLS). A full description and access to HLS may be found at https://hls.gsfc.nasa.gov/. The labels were from shapefiles maintained by the Monitoring Trends in Burn Severity (MTBS) group. The masks may be found at: https://mtbs.gov/",
  "rai:dataUseCases": [
    "The dataset can be used for training, validation, testing and fine-tuning."
  ],
  "cr:citeAs": "@software{HLS_Foundation_2023,author={Phillips, Christopher and Roy, Sujit and Ankur, Kumar and Ramachandran, Rahul},doi={10.57967/hf/0956},month=aug,title=,url={https://huggingface.co/ibm-nasa-geospatial/hls_burn_scars},year={2023}"
}
```

### RAI properties for DICES-350

The DICES (Diversity In Conversational AI Evaluation for Safety) dataset addresses the need for nuanced safety evaluation in language modeling, emphasizing diverse perspectives in model training and assessment. Key features include rater diversity, where opinion differences are viewed as diversity rather than bias. The dataset ensures balanced demographic representation among raters and assesses safety across five categories, offering detailed evaluations of harm, bias, misinformation, politics, and safety policy violations. With two annotated conversation sets (DICES-990 and DICES-350), each containing numerous annotations per conversation, DICES enables robust statistical analysis regarding annotator demographic diversity. It facilitates developing metrics for evaluating conversational AI systems' safety and diversity, allowing inter-rater reliability comparisons between demographic subgroups.

The DICES dataset and analyses are available at [https://github.com/google-research-datasets/dices-dataset/](https://github.com/google-research-datasets/dices-dataset/). Below we define RAI properties for the DICES-350 dataset.

```json
{
  "@context": {
    "@language": "en",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "sc": "https://schema.org/"
  },
  "@type": "schema.org/Dataset",
  "name": "DICES-350",
  "dct:conformsTo": "http://mlcommons.org/croissant/RAI/1.0",
  "rai:dataCollection": "The input data was sampled from an 8K multi-turn conversation corpus (comprising 48K turns in total) generated by human agents interacting with a generative AI-chatbot. The human agents were instructed to generate adversarial multi-turn conversations, where they attempt to provoke the chatbot to respond with an undesirable or unsafe answer. All conversations were of maximum five turns and varied in terms of their level of adversariality (i.e., degree of harm) and topics. A subset of the conversations (DICES-350) were annotated with gold safety labels (from trust and safety experts) and all conversations with platinum safety labels (from a diverse rater crowd)",
  "rai:dataCollectionType": "DICES-350 consists of 350 adversarial multi-turn conversations, annotated by a pool of annotators along 16 safety criteria.",
  "rai:dataCollectionRawData": "The input data for this data collection was sampled from an 8K multi-turn conversation corpus (comprising 48K turns in total) generated by human agents interacting with a generative AI-chatbot.",
  "rai:dataAnnotationProtocol": "The annotation task included the following six sets of questions: Q1: addresses the whole conversation and asks the raters to assess the legibility of the conversation - is it (1) in English, (2) comprehensible, (3) on a topic that the rater is familiar with or (4) none of the above. Q2: eight sub-question checks whether the conversation contains any harmful content, i.e., whether it could directly facilitate serious and immediate harm to individuals, groups or animals. Q3: four sub-question checks whether the conversation contains any unfair bias, i.e., whether it incites hatred against an individual or group. Q4: one multiple choice question checks whether the conversation contains any misinformation, i.e., whether it contains theories that are demonstrably false or outdated. Q5: one multiple choice question checks whether the conversation contains any political affiliations, i.e., engaging in or downplaying any controversial topic. Q6: two sub-question checks whether the conversation contains any policy violations for polarising topics and endorsements, i.e., proactively bring up or reactively promoting brands, celebrities, products, etc. All questions included an option 'other' to ensure coverage of possible safety reasons outside this re-defined space. Rating options for Q2 to Q6 were: \"No\" (i.e., safe), \"Yes\" (i.e., unsafe) and \"Unsure\". In effect, a 'Yes' answer to any of the questions Q2 to Q6 should be seen as an explanation of why a conversation is considered unsafe.",
  "rai:dataAnnotationPlatform": "Crowdworker annotators with task specific UI",
  "rai:dataAnnotationAnalysis": "Initial recruitment of 123 raters for the DICES-350 dataset, after all annotation tasks were completed, a quality assessment was performed on the raters and 19 raters were filtered out due to low quality work (e.g., raters who spent suspiciously little time in comparison to the other raters to complete the task and raters who rated all conversations with the same label), results reported with remaining 104 raters. In order to understand better the conversations in terms of their topics and adversariality type and level, all conversations in DICES-350 were also rated by in-house experts to assess their degree of harm. All conversations in DICES-350 have gold ratings, i.e. they were annotated for safety by a trust and safety expert. Further, aggregated ratings were generated from all granular safety ratings. They include a single aggregated overall safety rating ('Q_overall'), and aggregated ratings for the three safety categories that the 16 more granular safety ratings correspond to: 'Harmful content' ('Q2_harmful_content_overall'), 'Unfair bias' ('Q3_bias_overall') and 'Safety policy violations' ('Q6_policy_guidelines_overall').",
  "rai:dataUseCases": "The dataset is to be used as a shared resource and benchmark that respects diverse perspectives during safety evaluation of conversational AI systems. It can be used to develop metrics to examine and evaluate conversational AI systems in terms of both safety and diversity.",
  "rai:dataBiases": "Dataset includes multiple sub-ratings which specify the type of safety concern, such as type of hate speech and the type of bias or misinformation, for each conversation. A limitation of the dataset is the selection of demographic characteristics. The number of demographic categories was limited to four (race/ethnicity, gender and age group). Within these demographic axes, the number of subgroups was further limited (i.e., two locales, five main ethnicity groups, three age groups and two genders), this constrained the insights from systematic differences between different groupings of raters.",
  "rai:annotationsPerItem": "350 conversations were rated along 16 safety criteria, i.e., 104 unique ratings per conversation.",
  "rai:annotatorDemographics": "DICES-350 was annotated by a pool of 104 raters. The rater breakdown for this pool is: 57 women and 47 men; 27 gen X+, 28 millennial, and 49 gen z; and 21 Asian, 23 Black/African American, 22 Latine/x, 13 multiracial and 25 white. All raters signed a consent form agreeing for the detailed demographics to be collected for this task."
}
```

### RAI properties for the BigScience Roots Corpus dataset

[The bigscience roots corpus: A 1.6 tb composite multilingual dataset](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ce9e92e3de2372a4b93353eb7f3dc0bd-Abstract-Datasets_and_Benchmarks.html)

As large language models continue increasing in size, there is growing demand for extensive, high-quality text datasets, particularly in multilingual contexts. The BigScience workshop emerged as a year-long international and interdisciplinary effort investigating and training large language models with strong emphasis on ethical considerations, potential harm, and governance issues. The BigScience roots corpus was instrumental in training the 176-billion-parameter BigScience Large Open-science Open-access Multilingual (BLOOM) language model. The dataset aims to provide both data and processing tools facilitating large-scale monolingual and multilingual modeling projects and stimulating further research on this multilingual corpus.

```json
{
  "@context": {
    "@language": "en",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "sc": "https://schema.org/"
  },
  "@type": "schema.org/Dataset",
  "name": "BigScience Root Corpus",
  "dct:conformsTo": "http://mlcommons.org/croissant/RAI/1.0",
  "rai:dataCollection": "The first part of the corpus, accounting for 62% of the final dataset size (in bytes), is made up of a collection of monolingual and multilingual language resources that were selected and documented collaboratively through various efforts of the BigScience Data Sourcing working group. The 38% remaining is get from the OSCAR version 21.09, based on the Common Crawl snapshot of February.",
  "rai:dataCollectionType": [
    "Web Scraping",
    "Secondary Data Analysis",
    "Manual Human Curation",
    "Software Collection"
  ],
  "rai:dataUseCases": [
    "A comprehensive and multilingual corpus designed to support the training of large language models (LLMs)",
    "It may also be of particular interest for research aimed at improving the linguistic and cultural inclusiveness of language technologies"
  ],
  "rai:dataLimitations": [
    "Crawled content also over-represents pornographic text across languages, especially in the form of spam ads. Finally, it contains personal information that may constitute a privacy risk. The present section outlines our approach to mitigating those issues",
    "The preprocessing removes some categories of PII but is still far from exhaustive, and the nature of crawled datasets makes it next to impossible to identify individual contributors and ask for their consent.",
    "The reliance on medium to large sources of digitized content still over-represents privileged voices and language varieties."
  ],
  "rai:dataBiases": "Dataset includes multiple sub-ratings which specify the type of safety concern, such as type of hate speech and the type of bias or misinformation, for each conversation. A limitation of the dataset is the selection of demographic characteristics. The number of demographic categories was limited to four (race/ethnicity, gender and age group). Within these demographic axes, the number of subgroups was further limited (i.e., two locales, five main ethnicity groups, three age groups and two genders), this constrained the insights from systematic differences between different groupings of raters.",
  "rai:personalSensitiveInformation": "We used a rule-based approach leveraging regular expressions (Appendix C). The elements redacted were instances of KEY (numeric & alphanumeric identifiers such as phone numbers, credit card numbers, hexadecimal hashes and the like, while skipping instances of years and simple numbers), EMAIL (email addresses), USER (a social media handle) and IP_ADDRESS (an IPv4 or IPv6 address).",
  "rai:dataSocialImpact": "The authors emphasized that the BigScience Research Workshop, under which the dataset was developed, was conceived as a collaborative and value-driven endeavor from the beginning. This approach significantly influenced the project's decisions, leading to numerous discussions aimed at aligning the project's core values with those of the data contributors, as well as considering the social impact on individuals directly and indirectly impacted by the project. These discussions and the project's governance strategy highlighted the importance of: Centre human selection of the data, suggesting a conscientious approach to choosing what data to include in the corpus based on ethical considerations and the potential social impact. Data release and governance strategies that would responsibly manage the distribution and use of the data. Although the document does not explicitly list specific potential social impacts, the emphasis on value-driven efforts, ethical considerations, and the human-centered approach to data selection suggests a keen awareness and proactive stance on mitigating negative impacts while enhancing positive social outcomes through responsible data collection and usage practices.",
  "rai:dataManipulationProtocol": [
    "Pseudocode to recreate the text structure from the HTML code. The HTML code of a web page provides information about the structure of the text. The final structure of a web page is, however, the one produced by the rendering engine of the web browser and any CSS instructions. The latter two elements, which can vary enormously from one situation to another, always use the tag types for their rendering rules. Therefore, we have used a 20 fairly simple heuristic on tag types to reconstruct the structure of the text extracted from an HTML code. To reconstruct the text, the HTML DOM, which can be represented as a tree is traversed with a depth-first search algorithm. The text is initially empty and each time a new node with textual content is reached its content is concatenated according to the rules presented in the Algorithm 1 of the accompanying paper.",
    "Data cleaning and filtering: documents were filtered with: Too high character repetition or word repetition as a measure of repetitive content. Too high ratios of special characters to remove page code or crawling artifacts. Insufficient ratios of closed class words to filter out SEO pages. Too high ratios of flagged words to filter out pornographic spam. We asked contributors to tailor the word list in their language to this criterion (as opposed to generic terms related to sexuality) and to err on the side of high precision. Too high perplexity values to filter out non-natural language. Insufficient number of words, as LLM training requires extensive context sizes.",
    "Deduplication: we applied substring deduplication (Lee et al., 2022) based on Suffix Array (Manber and Myers, 1993) as a complementary method that clusters documents sharing a long substring, for documents with more than 6000 characters. We found on average 21.67% (10.61% to 32.30%) of the data (in bytes) being duplicated."
  ]
}
```

### RAI properties for the BigCode - The Stack dataset

[The Stack](https://huggingface.co/datasets/bigcode/the-stack) contains over 6TB of permissively-licensed source code files covering 358 programming languages. The dataset was created as part of the BigCode Project, an open scientific collaboration working on responsible Large Language Model development for Code (Code LLMs). The Stack serves as a pre-training dataset for Code LLMs — code-generating AI systems enabling program synthesis from natural language descriptions and code snippets.

```json
{
  "@context": {
    "@language": "en",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "sc": "https://schema.org/"
  },
  "@type": "schema.org/Dataset",
  "name": "BigScience - The Stack",
  "dct:conformsTo": "http://mlcommons.org/croissant-RAI/1.0",
  "rai:dataCollection": "The collection process is composed of the collection of 220.92M active GitHub repository names from the event archives published between January 1st, 2015 and March 31st, 2022 on GHArchive. Only 137.36M of these repositories were public and accessible on GitHub – others were not accessible as they had been deleted by their owners. 51.76B files were downloaded from the public repositories on GitHub between November 2021 and June 2022. 5.28B files were unique. The uncompressed size of all stored files is 92.36TB",
  "rai:dataCollectionType": "Web Scraping",
  "rai:dataCollectionRaw": "Files containing code data.",
  "rai:dataCollectionTimeFrameStart": {
    "@value": "2015-01-01T00:00:00",
    "dataType": "sc:Date"
  },
  "rai:dataCollectionTimeFrameEnd": {
    "@value": "2022-12-31T00:00:00",
    "dataType": "sc:Date"
  },
  "rai:dataUseCases": [
    "The Stack is a pre-training dataset for creating code LLMs. Code LLMs can be used for a wide variety of downstream tasks such as code completion from natural language descriptions (HumanEval, MBPP), documentation generation for individual functions (CodeSearchNet), and auto-completion of code snippets (HumanEval-Infilling)."
  ],
  "rai:dataLimitations": [
    "One of the current limitations of The Stack is that scraped HTML for websites may not be compliant with Web Content Accessibility Guidelines (WCAG). This could have an impact on HTML-generated code that may introduce web accessibility issues.",
    "The training dataset could contain malicious code and/or the model could be used to generate malware or ransomware.",
    "Despite datasets containing personal information, researchers should only use public, non-personal information in support of conducting and publishing their open-access research. Personal information should not be used for spamming purposes, including sending unsolicited emails or selling of personal information."
  ],
  "rai:dataBiases": [
    "Widely adopted programming languages like C and Javascript are overrepresented compared to niche programming languages like Julia and Scala. Some programming languages such as SQL, Batchfile, TypeScript are less likely to be permissively licensed (4% vs the average 10%). This may result in a biased representation of those languages. Permissively licensed files also tend to be longer",
    "Roughly 40 natural languages are present in docstrings and comments with English being the most prevalent. In python files, it makes up ~96% of the dataset",
    "The code collected from GitHub does not contain demographic information or proxy information about the demographics. However, it is not without risks, as the comments within the code may contain harmful or offensive language, which could be learned by the models."
  ],
  "rai:personalSensitiveInformation": [
    "The released dataset may contain sensitive information such as emails, IP addresses, and API/ssh keys that have previously been published to public repositories on GitHub. Deduplication has helped to reduce the amount of sensitive data that may exist. The PII pipeline for this dataset is still a work in progress. Researchers who wish to contribute to the anonymization pipeline of the project can apply to join here: https://www.bigcode-project.org/docs/about/join/."
  ],
  "rai:dataSocialImpact": "The Stack is released with the aim to increase access, reproducibility, and transparency of code LLMs in the research community. We expect code LLMs to enable people from diverse backgrounds to write higher quality code and develop low-code applications. Mission-critical software could become easier to maintain as professional developers are guided by code-generating systems on how to write more robust and efficient code. While the social impact is intended to be positive, the increased accessibility of code LLMs comes with certain risks such as over-reliance on the generated code and long-term effects on the software development job market.",
  "rai:dataPreprocessingProtocol": [
    "Near-deduplication was implemented in the pre-processing pipeline on top of exact deduplication. To find near-duplicates, MinHash with 256 permutations of all documents was computed in linear time. Locality Sensitive Hashing was used to find the clusters of duplicates. Jaccard Similarities were computed inside these clusters to remove any false positives and with a similarity threshold of 0.85. Roughly 40% of permissively licensed files were (near-)duplicates.",
    "Non detected licenses: GHArchive contained the license information for approximately 12% of the collected repositories. For the remaining repositories, go-license-detector was run to detect the most likely SPDX license identifier. The detector did not detect a license for ~81% of the repositories, in which case the repository was excluded from the dataset."
  ]
}
```

## References

[1] Surbhi Mittal, Kartik Thakral, Richa Singh, Mayank Vatsa, Tamar Glaser, Cristian Canton-Ferrer, Tal Hassner: On Responsible Machine Learning Datasets with Fairness, Privacy, and Regulatory Norms. CoRR abs/2310.15848 (2023)

[2] Christopher Phillips and Sujit Roy and Kumar Ankur, Rahul Ramachandran: HLS Foundation Burnscars Dataset. https://doi.org/10.57967/hf/0956 (Aug 2023)

[3] Lora Aroyo, Alex S. Taylor, Mark Díaz, Christopher Homan, Alicia Parrish, Gregory Serapio-García, Vinodkumar Prabhakaran, Ding Wang: DICES Dataset: Diversity in Conversational AI Evaluation for Safety. NeurIPS 2023

[4] Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, Jörg Frohberg, Mario Sasko, Quentin Lhoest, Angelina McMillan-Major, Gérard Dupont, Stella Biderman, Anna Rogers, Loubna Ben Allal, Francesco De Toni, Giada Pistilli, Olivier Nguyen, Somaieh Nikpoor, Maraim Masoud, Pierre Colombo, Javier de la Rosa, Paulo Villegas, Tristan Thrush, Shayne Longpre, Sebastian Nagel, Leon Weber, Manuel Muñoz, Jian Zhu, Daniel van Strien, Zaid Alyafeai, Khalid Almubarak, Minh Chien Vu, Itziar Gonzalez-Dios, Aitor Soroa, Kyle Lo, Manan Dey, Pedro Ortiz Suarez, Aaron Gokaslan, Shamik Bose, David Ifeoluwa Adelani, Long Phan, Hieu Tran, Ian Yu, Suhas Pai, Jenny Chim, Violette Lepercq, Suzana Ilic, Margaret Mitchell, Sasha Luccioni, Yacine Jernite: The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset. CoRR abs/2303.03915 (2023)

[5] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna M. Wallach, Hal Daumé III, Kate Crawford: Datasheets for datasets. Commun. ACM 64(12): 86-92 (2021)

---

Source: https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html
