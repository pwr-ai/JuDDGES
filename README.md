# JuDDGES


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

> JuDDGES stands for Judicial Decision Data Gathering, Encoding, and
> Sharing

The JuDDGES project aims to revolutionize the accessibility and analysis
of judicial decisions across varied legal systems using advanced Natural
Language Processing and Human-In-The-Loop technologies. It focuses on
criminal court records from jurisdictions with diverse legal
constitutions, including Poland and England & Wales. By overcoming
barriers related to resources, language, data, and format inhomogeneity,
the project facilitates the development and testing of theories on
judicial decision-making and informs judicial policy and practice. Open
software and tools produced by the project will enable extensive,
flexible meta-annotation of legal texts, benefiting researchers and
public legal institutions alike. This initiative not only advances
empirical legal research by adopting Open Science principles but also
creates the most comprehensive legal research repository in Europe,
fostering cross-disciplinary and cross-jurisdictional collaboration.

![baner](https://raw.githubusercontent.com/pwr-ai/JuDDGES/bffb1d75ba7c78f101fc94bd9086499886b2c128/nbs/images/baner.png)

## Usage

### Installation

- to install necessary dependencies use available `Makefile`, you can
  use `python>=3.10`: `make install`
- if you want to run evaluation and fine-tuning with `unsloth`, use the
  following command with `python=3.10` inside conda environment:
  `make install_unsloth`

### Dataset creation

The specific details of dataset creation are available in
[scripts/README.md](scripts/README.md).

### Fine tuning

To run evaluation or fine-tuning, run proper stages declared
[`dvc.yaml`](dvc.yaml) (see [DVC docs for
details](https://dvc.org/doc/user-guide))

## Project details

The JuDDGES project encompasses several Work Packages (WPs) designed to
cover all aspects of its objectives, from project management to the open
science practices and engaging early career researchers. Below is an
overview of the project’s WPs based on the provided information:

### WP1: Project Management

**Duration**: 24 Months

**Main Aim**: To ensure the project’s successful completion on time and
within budget. This includes administrative management, scientific and
technological management, quality innovation and risk management,
ethical and legal consideration, and facilitating open science.

### WP2: Gathering and Human Encoding of Judicial Decision Data

**Duration**: 22 Months

**Main Aim**: To establish the data foundation for developing and
testing the project’s tools. This involves collating/gathering legal
case records and judgments, developing a coding scheme, training human
coders, making human-coded data available for WP3, facilitating
human-in-loop coding for WP3, and enabling WP4 to make data open and
reusable beyond the project team.

### WP3: NLP and HITL Machine Learning Methodological Development

**Duration**: 24 Months

**Main Aim**: To create a bridge between machine learning (led by WUST
and MUHEC) and Open Science facilitation (by ELICO), focusing on the
development and deployment of annotation methodologies. This includes
baseline information extraction, intelligent inference methods for legal
corpus data, and constructing an annotation tool through active learning
and human-in-the-loop annotation methods.

### WP4: Open Science Practices & Engaging Early Career Researchers

**Duration**: 12 Months

**Main Aim**: To implement the Open Science policy of the call and
engage with relevant early career researchers (ECRs). Objectives include
providing open access to publication data and software,
disseminating/exploiting project results, and promoting the project and
its findings.

Each WP includes specific tasks aimed at achieving its goals, involving
collaboration among project partners and contributing to the overarching
aim of the JuDDGES project​​.

## Acknowledgements

The universities involved in the JuDDGES project are:

1.  Wroclaw University of Science and Technology (Poland)
2.  Middlesex University London (UK)
3.  University of Lyon 1 (France)​​.
