from langchain.output_parsers.json import parse_json_markdown
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_openai import ChatOpenAI

SCHEMA_PROMPT_TEMPLATE = """
Act as a assistant that prepares schema for information extraction

Based on the user input prepare schema containing variables with their short description and type.
Be precise about variable names, format names using snake_case.
If user asks irrelevant question always return empty JSON.
As example:
User: I want extract age, gender, and plea from the judgement
Agent:
    age: integer
    gender: male or female
    plea: string

====
{SCHEMA_TEXT}
====

Format response as JSON:
"""

EXTRACTION_PROMPT_TEMPLATE = """Act as a highly skilled legal analyst specializing in extracting structured information from court judgments.

Your task is to carefully analyze the provided judgment text and extract specific information according to the schema provided.

Key instructions:
- Language: Extract information in {LANGUAGE}, maintaining the original language of the judgment
- Accuracy: Only extract information that is explicitly stated in the text
- Empty fields: Use empty string "" when information cannot be found
- Consistency: Ensure extracted values match the specified data types and enums
- Context: Consider the full context when extracting information
- Validation: Double-check that extracted values are supported by the text
- Objectivity: Extract factual information without interpretation

For boolean fields:
- Only mark as true when explicitly confirmed in the text
- Default to false when information is unclear or not mentioned

For enum fields:
- Only use values from the provided options
- Use empty string if none of the options match exactly

For date fields:
- Use ISO 8601 format (YYYY-MM-DD)
- Extract complete dates when available
- Leave empty if date is partial or ambiguous

Schema for extraction:
{SCHEMA}

Judgment text to analyze:
====
{TEXT}
====

Format response as JSON, ensuring all schema fields are included:
"""

EXAMPLE_SCHEMA = """verdict_date: date as ISO 8601
verdict: string, text representing verdict of the judgement
verdict_summary: string, short summary of the verdict
verdict_id: string
court: string
parties: string
appeal_against: string
first_trial: boolean
drug_offence: boolean
child_offence: boolean
offence_seriousness: boolean
verdict_tags: List[string]"""

SWISS_FRANC_LOAN_SCHEMA = """apelacja: string, description: "Określenie apelacji, w której znajduje się sąd rozpoznający sprawę", example: "Apelacja warszawska"
typ_sadu: enum [Sąd Rejonowy, Sąd Okręgowy, Sąd Frankowy], description: "Typ sądu rozpoznającego sprawę", example: "Sąd Rejonowy"
instancja_sadu: enum [Sąd I instancji, Sąd odwoławczy], description: "Czy sąd jest I instancji czy odwoławczy", example: "Sąd I instancji"
podstawa_prawna: string, description: "Podstawa prawna roszczenia", example: "Art. 385(1) Kodeksu cywilnego"
podstawa_prawna_podana: boolean, description: "Czy powód podał podstawę prawną?", example: true
rodzaj_roszczenia: enum [O ustalenie istnienia/nieistnienia stosunku prawnego, O ukształtowanie stosunku prawnego, O zapłatę, Roszczenia dodatkowe], description: "Rodzaj roszczenia", example: "O zapłatę"
modyfikacje_powodztwa: boolean, description: "Czy były modyfikacje powództwa", example: true
typ_modyfikacji: enum [Rodzaj roszczenia, Kwoty roszczenia], description: "Typ modyfikacji powództwa", example: "Kwoty roszczenia"
status_kredytobiorcy: enum [Konsument, Przedsiębiorca], description: "Status kredytobiorcy", example: "Konsument"
wspoluczestnictwo_powodowe: boolean, description: "Czy współuczestnictwo po stronie powodowej", example: false
typ_wspoluczestnictwa: enum [Małżeństwo, Konkubinat, Inni członkowie rodziny, Spadkobiercy], description: "Rodzaj współuczestnictwa", example: "Małżeństwo"
rola_pozwanego: enum [Strona umowy, Następca prawny], description: "Czy pozwany faktycznie był stroną umowy czy następcą prawnym", example: "Strona umowy"
wczesniejsze_skargi_do_rzecznika: boolean, description: "Czy były uprzednie skargi do rzecznika finansowego", example: false
umowa_kredytowa: enum [Zawarta bezpośrednio w banku, Z udziałem pośrednika], description: "Czy umowa kredytowa zawierana bezpośrednio w banku czy z pośrednikiem", example: "Zawarta bezpośrednio w banku"
klauzula_niedozwolona: boolean, description: "Istnienie klauzuli niedozwolonej w umowie", example: true
wpisana_do_rejestru_uokik: boolean, description: "Czy klauzula wpisana do rejestru klauzul niedozwolonych UOKiK", example: true
waluta_splaty: enum [PLN, CHF], description: "Waluta spłaty", example: "CHF"
aneks_do_umowy: boolean, description: "Czy był aneks do umowy", example: true
data_aneksu: date (ISO 8601), description: "Data aneksu", example: "2023-01-15"
przedmiot_aneksu: enum [Zmiana waluty spłaty, Inne kwestie], description: "Czego dotyczył aneks", example: "Zmiana waluty spłaty"
status_splaty_kredytu: boolean, description: "Czy kredyt był spłacony, w tym w trakcie procesu", example: true
data_wyroku: date (ISO 8601), description: "Data wydania wyroku", example: "2024-11-30"
rozstrzygniecie_sadu: string, description: "Rozstrzygnięcie", example: "Oddalenie powództwa"
typ_rozstrzygniecia: enum [Uwzględnienie powództwa w całości, Uwzględnienie powództwa w części, Oddalenie powództwa, Oddalenie apelacji, Zmiana wyroku, Przekazanie do ponownego rozpoznania], description: "Typ rozstrzygnięcia", example: "Oddalenie powództwa"
sesja_sadowa: enum [Rozprawa, Posiedzenie niejawne], description: "Czy wyrok wydano na rozprawie czy na posiedzeniu niejawnym", example: "Rozprawa"
dowody: list of enum [Przesłuchanie stron, Przesłuchanie świadków, Dowód z opinii biegłego], description: "Jakie dowody zostały przeprowadzone", example: ["Przesłuchanie stron", "Dowód z opinii biegłego"]
oswiadczenie_niewaznosci: boolean, description: "Czy odbierano oświadczenie powoda o skutkach nieważności umowy", example: false
odwolanie_do_sn: boolean, description: "Czy odwołano się do orzecznictwa SN", example: true
odwolanie_do_tsue: boolean, description: "Czy odwołano się do orzecznictwa TSUE", example: true
teoria_prawna: enum [Teoria dwóch kondykcji, Teoria salda], description: "Teoria prawna, na której oparto wyrok", example: "Teoria dwóch kondykcji"
zarzut_zatrzymania_lub_potracenia: boolean, description: "Czy uwzględniono zarzut zatrzymania lub potrącenia", example: false
odsetki_ustawowe: boolean, description: "Czy uwzględniono odsetki ustawowe", example: true
data_rozpoczecia_odsetek: enum [Od dnia wezwania do zapłaty, Od daty wytoczenia powództwa, Od daty wydania wyroku, Od innej daty], description: "Data rozpoczęcia naliczania odsetek", example: "Od daty wydania wyroku"
koszty_postepowania: boolean, description: "Czy zasądzono zwrot kosztów postępowania", example: true
beneficjent_kosztow: string, description: "Na rzecz której strony zasądzono zwrot kosztów", example: "Pozwany"
zabezpieczenie_udzielone: boolean, description: "Czy udzielono zabezpieczenia", example: true
rodzaj_zabezpieczenia: string, description: "Rodzaj zabezpieczenia", example: "Wstrzymanie egzekucji"
zabezpieczenie_pierwsza_instancja: boolean, description: "Czy zabezpieczenia udzielił sąd I instancji", example: true
czas_trwania_sprawy: string, description: "Czas rozpoznania sprawy – od złożenia pozwu do wydania wyroku", example: "2 lata 3 miesiące
wynik_sprawy: enum [Wygrana kredytobiorcy, Wygrana banku, Częściowe uwzględnienie roszczeń obu stron], description: "Ocena, czy bank czy kredytobiorca wygrał sprawę", example: "Wygrana kredytobiorcy"
szczegoły_wyniku_sprawy: string, description: "Szczegóły dotyczące wyniku sprawy", example: "Kredytobiorca wygrał, umowa uznana za nieważną"""


def prepare_information_extraction_chain_from_user_prompt() -> RunnableSequence:
    schema_chain = prepare_schema_chain()
    inputs = {
        "SCHEMA": schema_chain,
        "TEXT": RunnablePassthrough(),
        "LANGUAGE": RunnablePassthrough(),
    }
    return inputs | RunnableLambda(route)


def prepare_information_extraction_chain(
    model_name: str = "gpt-4-0125-preview",
    log_to_mlflow: bool = False,
) -> RunnableSequence:
    model = ChatOpenAI(model=model_name, temperature=0)
    human_message_template = HumanMessagePromptTemplate.from_template(
        EXTRACTION_PROMPT_TEMPLATE
    )
    _prompt = ChatPromptTemplate(
        messages=[human_message_template],
        input_variables=["TEXT", "LANGUAGE", "SCHEMA"],
    )

    if log_to_mlflow:
        import mlflow

        mlflow.log_dict(_prompt.save_to_json(), "prompt.json")

    return _prompt | model | (lambda x: parse_json_markdown(x.content))


def prepare_schema_chain(model_name: str = "gpt-3.5-turbo") -> RunnableSequence:
    model = ChatOpenAI(model=model_name, temperature=0)
    human_message_template = HumanMessagePromptTemplate.from_template(
        SCHEMA_PROMPT_TEMPLATE
    )
    _prompt = ChatPromptTemplate(
        messages=[human_message_template],
        input_variables=["TEXT", "LANGUAGE", "SCHEMA"],
    )

    return _prompt | model | parse_schema


def parse_schema(ai_message: AIMessage) -> str:
    response_schema = parse_json_markdown(ai_message.content)
    return "\n".join(f"{key}: {val}" for key, val in response_schema.items())


def route(response_schema: str) -> dict[str, str]:
    if response_schema["SCHEMA"]:
        return prepare_information_extraction_chain()

    raise ValueError(
        "Cannot determine schema for the given input prompt. Please try different query."
    )
