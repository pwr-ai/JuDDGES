AGITATION_SCHEMA = """
case_id: string, description: "Unique identifier of the case", example: "SO/Wa/123/2023"
court_type: enum [Sąd Okręgowy, Sąd Apelacyjny], description: "Type of court handling the case", example: "Sąd Okręgowy"
court_location: string, description: "City or region where the court is located", example: "Warszawa"
case_date: date (ISO 8601), description: "Date of the judgment", example: "2023-10-15"
docket_number: string, description: "Court case docket number", example: "I Ns 123/23"
is_agitation_case: boolean, description: "Whether the judgment is related to electoral agitation", example: true
art_111_cited: boolean, description: "Whether Article 111 § 1 of the Electoral Code is cited as legal basis", example: true
legal_basis_description: string, description: "Description of the legal basis as mentioned in the judgment", example: "Art. 111 § 1 Kodeksu wyborczego"
additional_legal_basis: list of string, description: "Other legal provisions cited in the judgment", example: ["Art. 111 § 2 Kodeksu wyborczego", "Art. 113 Kodeksu wyborczego"]
applicant_type: enum [Kandydat, Komitet Wyborczy, Pełnomocnik Komitetu, Wyborca, Inny], description: "Type of applicant filing the case", example: "Komitet Wyborczy"
applicant_name: string, description: "Name of the applicant", example: "Jan Kowalski"
applicant_political_affiliation: string, description: "Political party or committee affiliation of the applicant", example: "Partia X"
respondent_name: string, description: "Name of the respondent", example: "Komitet Wyborczy Partii X"
respondent_type: enum [Kandydat, Komitet Wyborczy, Media, Osoba prywatna, Inny], description: "Type of respondent in the case", example: "Komitet Wyborczy"
respondent_political_affiliation: string, description: "Political party or committee affiliation of the respondent", example: "Partia Y"
election_type: enum [Parlamentarne, Prezydenckie, Samorządowe, Europejskie, Inne], description: "Type of election the case is related to", example: "Parlamentarne"
election_date: date (ISO 8601), description: "Date of the election", example: "2023-10-15"
electoral_material_present: boolean, description: "Whether electoral material is mentioned in the case", example: true
electoral_material_type: enum [Ulotka, Plakat, Billboard, Materiał cyfrowy, Wypowiedź ustna, Inny], description: "Type of electoral material in question", example: "Materiał cyfrowy"
electoral_material_description: string, description: "Description of the electoral material as mentioned in the judgment", example: "Post na portalu Facebook zawierający nieprawdziwe informacje o kandydacie"
digital_material: boolean, description: "Whether the electoral material is in digital form", example: true
digital_material_type: enum [Blog, Mem, Post, Wpis na portalu, Wpis na stronie internetowej, Artykuł na stronie internetowej, Grafika, Komentarz, Nagranie wideo, Film, Zapis dźwięku, Inny], description: "Type of digital material", example: "Post"
digital_platform: enum [Facebook, Instagram, Twitter, TikTok, YouTube, Strona internetowa, Blog, Forum, Inny], description: "Platform where the digital material was published", example: "Facebook"
digital_material_url: string, description: "URL of the digital material if provided", example: "https://facebook.com/post/123456"
digital_material_reach: integer, description: "Estimated reach or views of the digital material if mentioned", example: 5000
publication_date: date (ISO 8601), description: "Date when the material was published", example: "2023-09-20"
case_outcome: enum [Uwzględniono wniosek, Oddalono wniosek, Umorzono postępowanie, Inny], description: "Outcome of the case", example: "Uwzględniono wniosek"
rectification_ordered: boolean, description: "Whether rectification was ordered", example: true
rectification_type: string, description: "Type of rectification ordered", example: "Usunięcie postu i opublikowanie sprostowania"
rectification_deadline: integer, description: "Deadline in hours for implementing the rectification", example: 48
financial_penalty: boolean, description: "Whether financial penalty was imposed", example: false
financial_penalty_amount: float, description: "Amount of financial penalty in PLN", example: 5000.0
agitation_present: boolean, description: "Whether electoral agitation is mentioned in the case", example: true
agitation_description: string, description: "Description of the electoral agitation as mentioned in the judgment", example: "Nakłanianie do głosowania na konkretnego kandydata poprzez rozpowszechnianie nieprawdziwych informacji o kontrkandydacie"
false_information_claim: boolean, description: "Whether false information was claimed to be spread", example: true
false_information_description: string, description: "Description of the allegedly false information", example: "Nieprawdziwe informacje o wykształceniu kandydata"
false_information_category: enum [Wykształcenie, Doświadczenie zawodowe, Majątek, Poglądy polityczne, Życie prywatne, Działalność przestępcza, Inny], description: "Category of false information", example: "Wykształcenie"
expert_testimony: boolean, description: "Whether expert testimony was used in the case", example: false
expert_field: string, description: "Field of expertise of the expert witness", example: "Językoznawstwo"
case_duration: integer, description: "Duration of the case in days from filing to judgment", example: 14
appeal_filed: boolean, description: "Whether an appeal was filed", example: false
appeal_outcome: enum [Utrzymano wyrok, Zmieniono wyrok, Uchylono wyrok, W toku, Nie dotyczy], description: "Outcome of the appeal if applicable", example: "Nie dotyczy"
precedent_cited: boolean, description: "Whether previous judgments were cited as precedent", example: true
precedent_details: string, description: "Details of precedents cited", example: "Wyrok Sądu Najwyższego z dnia 10.05.2020, sygn. III SZP 2/19"
""".strip()
