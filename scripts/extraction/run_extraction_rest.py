"""Run Gemini extraction using Weaviate REST API directly.

This script bypasses the Weaviate Python client GRPC issues by using REST API directly.
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import requests
import weaviate.util
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyMd5Cache
from langfuse.langchain import CallbackHandler
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from sqlalchemy import create_engine

from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema
from juddges.extraction.extraction_storage import ExtractionStorage
from juddges.settings import ROOT_PATH

# Load environment variables
load_dotenv(ROOT_PATH / ".env", override=True)

console = Console()


def create_comprehensive_schema() -> ExtractionSchema:
    """Create comprehensive extraction schema based on Weaviate properties and Polish legal system.

    Schema includes:
    - 2 NEW fields: factual_state (stan faktyczny) and legal_state (stan prawny)
    - Updated enumerations with complete Polish legal terminology
    - Concrete examples for each field
    - Expanded abbreviations (VAT = podatek od towarów i usług, etc.)
    """
    return ExtractionSchema(
        fields={
            # ============================================================================
            # CORE DOCUMENT IDENTIFICATION (4 fields)
            # ============================================================================
            "document_number": """string, official case/document reference number (sygnatura sprawy).
Examples:
- "I ACa 123/23" (sąd apelacyjny)
- "III SA/Wa 1234/23" (wojewódzki sąd administracyjny)
- "0114-KDIP2-1.4010.123.2023.1.AB" (indywidualna interpretacja podatkowa)
- "I CSK 456/22" (Sąd Najwyższy)""",
            "document_type": """string, type of legal document.
ENUM (exact values only):
- "judgment" (wyrok sądu)
- "tax_interpretation" (interpretacja podatkowa)
- "legal_act" (akt prawny)

Examples:
- judgment: wyrok sądu cywilnego, administracyjnego, karnego
- tax_interpretation: indywidualna lub ogólna interpretacja podatkowa
- legal_act: ustawa, rozporządzenie""",
            "title": """string, document title or generated descriptive title.
Should be concise (max 200 characters) but informative.

Examples:
- "Wyrok w sprawie o zapłatę z tytułu umowy kredytu frankowego"
- "Interpretacja indywidualna w zakresie VAT - usługi transgraniczne"
- "Wyrok NSA - odliczenie VAT od nabycia samochodu osobowego"
- "Uchwała SN - klauzule abuzywne w umowach kredytowych"

If no title in document, generate based on:
(1) document type, (2) court/authority, (3) main legal issue""",
            "date_issued": """date ISO 8601 (YYYY-MM-DD), date when document was officially issued.

Examples:
- "2024-01-15" (wyrok z dnia 15 stycznia 2024 r.)
- "2023-12-05" (interpretacja wydana 5 grudnia 2023 r.)

Look for phrases: "z dnia", "wydany dnia", "orzekł dnia"
Use YYYY-MM-DD format always, leave empty if date not found""",
            # ============================================================================
            # HIGH-PRIORITY AUGMENTATION FIELDS (3 fields)
            # ============================================================================
            "summary": """string, concise 3-5 sentence summary covering ALL of:
(1) Document type and issuing body/court
(2) Main legal issue or dispute subject
(3) Key facts of the case
(4) Decision/outcome
(5) Primary legal basis

Example (judgment):
"Wyrok Sądu Apelacyjnego w Warszawie z dnia 15.01.2024 r. w sprawie kredytu frankowego.
Powód domagał się unieważnienia umowy kredytu indeksowanego do CHF z uwagi na klauzule abuzywne.
Bank stosował spread walutowy do przeliczania rat kredytu, co prowadziło do znacznego wzrostu zadłużenia.
Sąd uwzględnił powództwo w całości i stwierdził nieważność umowy kredytu.
Podstawa prawna: art. 385¹ § 1 Kodeksu cywilnego (klauzule abuzywne)."

Example (tax interpretation):
"Indywidualna interpretacja podatkowa Dyrektora KIS z dnia 05.12.2023 r. w zakresie VAT.
Wnioskodawca świadczy usługi IT dla kontrahentów z UE i poza UE.
Pytanie dotyczyło miejsca świadczenia usług i obowiązku rozliczenia VAT w Polsce.
Stanowisko wnioskodawcy zostało uznane za prawidłowe - usługi podlegają opodatkowaniu w kraju nabywcy.
Podstawa prawna: art. 28b ustawy o VAT oraz art. 44 Dyrektywy 2006/112/WE."

Must be factual, comprehensive, in Polish""",
            "thesis": """string, main legal principle, rule, or legal conclusion established by the document (1-3 sentences).
This is the key legal finding - the "teza prawna".

Examples:
- "Postanowienia umowy kredytu indeksowanego do waluty obcej, które nie określają jednoznacznie sposobu
  ustalania kursu waluty, stanowią klauzule niedozwolone w rozumieniu art. 385¹ § 1 k.c."
- "Usługi IT świadczone przez polskiego podatnika na rzecz kontrahenta unijnego podlegają opodatkowaniu
  w kraju siedziby nabywcy zgodnie z art. 28b ustawy o VAT."
- "Koszty nabycia samochodu osobowego mogą stanowić koszty uzyskania przychodów tylko w części
  odpowiadającej wykorzystaniu pojazdu do działalności gospodarczej."

Should state the legal rule/principle clearly and concisely, in Polish""",
            "keywords": """List[string], 5-15 relevant legal keywords covering:
- Legal domains (prawo cywilne, VAT, CIT, etc.)
- Legal institutions (Sąd Najwyższy, WSA, KIS)
- Specific legal concepts (klauzule abuzywne, odliczenie VAT, koszty uzyskania przychodów)

Examples:
["prawo cywilne", "kredyt frankowy", "klauzule abuzywne", "art. 385¹ KC", "spread walutowy", "Sąd Apelacyjny"]
["VAT", "usługi transgraniczne", "miejsce świadczenia", "podatnik UE", "art. 28b ustawy o VAT"]
["CIT", "koszty uzyskania przychodów", "samochód służbowy", "ewidencja przebiegu pojazdu", "art. 15 ustawy o CIT"]

All keywords in Polish""",
            # ============================================================================
            # NEW FIELDS: FACTUAL AND LEGAL CONTEXT (2 fields)
            # ============================================================================
            "factual_state": """string, description of factual circumstances forming the basis for case/interpretation (STAN FAKTYCZNY).

In JUDGMENTS: Established facts, events leading to dispute, evidence findings, chronology of events.
In TAX INTERPRETATIONS: Taxpayer's situation, business activities, transaction description, concrete scenario.

Should be:
- Objective, fact-based narrative (no legal assessment)
- Chronological presentation of events
- Complete (all relevant facts)
- In original document language (Polish)
- Multiple sentences forming coherent narrative

Example (judgment - kredyt frankowy):
"Powód Jan Kowalski zawarł w dniu 15.03.2020 r. z pozwanym Bankiem ABC S.A. umowę kredytu hipotecznego
na kwotę 300.000 PLN indeksowaną do waluty CHF. W dniu podpisania umowy kurs CHF wynosił 3,50 PLN za 1 CHF.
Bank stosował własną tabelę kursów walut do przeliczania rat kredytu. W trakcie trwania umowy kurs CHF
wzrósł do 4,80 PLN, co spowodowało znaczne zwiększenie zadłużenia powoda. Powód twierdzi, że nie został
poinformowany o ryzyku kursowym i nie rozumiał mechanizmu indeksacji. Bank przedstawił protokół z rozmowy
przedkontraktowej oraz dokumenty potwierdzające przekazanie informacji o ryzyku walutowym."

Example (tax interpretation - VAT):
"Wnioskodawca prowadzi działalność gospodarczą w zakresie usług programowania i tworzenia oprogramowania.
W ramach działalności świadczy usługi dla kontrahentów z Niemiec (60% przychodów) i Polski (40% przychodów).
Usługi obejmują tworzenie dedykowanego oprogramowania według specyfikacji klienta oraz jego wdrożenie.
Wnioskodawca rozważa zmianę formy opodatkowania ze skali podatkowej na podatek liniowy 19%.
Przychody w 2023 r. wyniosły 450.000 PLN, z czego eksport usług stanowił 270.000 PLN."

Look for sections: "Stan faktyczny", "Ustalenia faktyczne", "Okoliczności faktyczne sprawy"
Extract complete narrative maintaining chronology and all relevant quantitative data""",
            "legal_state": """string, legal framework and provisions forming the basis for reasoning/decision (STAN PRAWNY).

In JUDGMENTS: Applicable statutes (with specific articles), regulations, case law, legal principles used by court.
In TAX INTERPRETATIONS: Tax law provisions, relevant regulations, legal basis for interpretation.

Should include:
- Specific legal acts with articles (Art. X ust. Y pkt Z)
- Hierarchical order: Constitution → statutes → regulations → case law
- Full names of legal acts
- Court decisions referenced (with signature numbers)
- In original document language (Polish)

Example (judgment - kredyt frankowy):
"Podstawą rozstrzygnięcia są następujące przepisy:
- Art. 385¹ § 1 Kodeksu cywilnego z dnia 23 kwietnia 1964 r. (Dz.U. 2023 poz. 1610) - klauzule abuzywne
- Art. 58 § 1 k.c. - nieważność czynności prawnej sprzecznej z ustawą lub mającej na celu obejście ustawy
- Art. 69 ust. 1 ustawy z dnia 29 sierpnia 1997 r. Prawo bankowe (Dz.U. 2023 poz. 2488) - umowa kredytu
- Wyrok Trybunału Sprawiedliwości UE z dnia 3 października 2019 r., C-260/18 (Dziubak)
- Uchwała Sądu Najwyższego z dnia 7 maja 2021 r., III CZP 6/21

Zgodnie z art. 385¹ § 1 k.c., postanowienia umowy zawieranej z konsumentem nieuzgodnione indywidualnie,
które kształtują prawa i obowiązki konsumenta w sposób sprzeczny z dobrymi obyczajami, rażąco naruszając
jego interesy, nie wiążą konsumenta."

Example (tax interpretation - VAT):
"Ocena stanowiska wnioskodawcy wymaga uwzględnienia następujących przepisów:
- Art. 5 ust. 1 pkt 1 ustawy z dnia 11 marca 2004 r. o podatku od towarów i usług (Dz.U. 2023 poz. 1570)
  - opodatkowaniu podlega odpłatne świadczenie usług
- Art. 28b ust. 1 ustawy o VAT - miejsce świadczenia usług na rzecz podatnika
- Art. 28b ust. 2 ustawy o VAT - zasada kraju przeznaczenia
- Art. 44 Dyrektywy Rady 2006/112/WE z dnia 28 listopada 2006 r. w sprawie wspólnego systemu VAT
- § 8 ust. 1 rozporządzenia Ministra Finansów z dnia 26 sierpnia 2003 r. w sprawie zwrotu podatku

Według art. 28b ust. 1 ustawy o VAT, miejscem świadczenia usług w przypadku świadczenia usług na rzecz
podatnika jest miejsce, w którym podatnik będący usługobiorcą posiada siedzibę działalności gospodarczej."

Look for sections: "Stan prawny", "Podstawa prawna", "Przepisy prawne", "Uzasadnienie prawne"
Extract all cited legal acts maintaining hierarchical structure""",
            # ============================================================================
            # OUTCOME (1 field with UPDATED enumeration)
            # ============================================================================
            "outcome": """JSON object describing the decision and its effects: {
                "decision_type": "enum (EXACT values based on document_type)",
                "decision_summary": "string, brief 2-3 sentence summary of decision",
                "awarded_amounts": [{"type": "string", "amount": number, "currency": "string", "recipient": "string"}],
                "legal_effect": "string, practical legal consequence"
            }

DECISION_TYPE enumeration (document_type = "judgment"):
- "uwzgledniono_w_calosci" (granted in full - pozew/skarga uwzględniona w całości)
- "uwzgledniono_w_czesci" (granted in part - pozew/skarga uwzględniona częściowo)
- "oddalono" (dismissed - pozew/skarga oddalona jako bezzasadna)
- "umorzono" (discontinued - postępowanie umorzone)
- "uchylono" (overturned/annulled - wyrok uchylony)
- "uchylono_i_przekazano" (remanded - uchylono i przekazano do ponownego rozpoznania)
- "zmieniono" (modified - wyrok zmieniony przez sąd wyższej instancji)
- "utrzymano_w_mocy" (upheld - wyrok utrzymany w mocy)
- "odrzucono" (rejected on procedural grounds - pozew/skarga odrzucona formalnie)

DECISION_TYPE enumeration (document_type = "tax_interpretation"):
- "stanowisko_prawidlowe" (position correct - stanowisko wnioskodawcy prawidłowe)
- "stanowisko_nieprawidlowe" (position incorrect - stanowisko wnioskodawcy nieprawidłowe)
- "stanowisko_czesciowo_prawidlowe" (position partially correct - częściowo prawidłowe)

Example (judgment):
{
  "decision_type": "uwzgledniono_w_calosci",
  "decision_summary": "Sąd uwzględnił powództwo w całości i stwierdził nieważność umowy kredytu.
                       Zasądził od pozwanego na rzecz powoda kwotę 45.000 PLN tytułem zwrotu
                       nienależnie pobranych świadczeń oraz koszty postępowania.",
  "awarded_amounts": [
    {"type": "zwrot_swiadczen", "amount": 45000, "currency": "PLN", "recipient": "powód Jan Kowalski"},
    {"type": "koszty_procesu", "amount": 5170, "currency": "PLN", "recipient": "powód Jan Kowalski"}
  ],
  "legal_effect": "Umowa kredytu jest nieważna od początku. Strony zobowiązane do zwrotu wzajemnie
                   otrzymanych świadczeń. Bank zwraca pobrane raty, powód zwraca kapitał kredytu."
}

Example (tax interpretation):
{
  "decision_type": "stanowisko_prawidlowe",
  "decision_summary": "Dyrektor KIS uznał stanowisko wnioskodawcy za prawidłowe. Usługi IT świadczone
                       na rzecz podatników z UE podlegają opodatkowaniu w kraju nabywcy.",
  "awarded_amounts": [],
  "legal_effect": "Wnioskodawca nie jest zobowiązany do rozliczania VAT w Polsce od usług świadczonych
                   dla kontrahentów unijnych. Zastosowanie procedury odwrotnego obciążenia."
}""",
            # ============================================================================
            # LEGAL CONTENT (3 fields with UPDATED enumerations)
            # ============================================================================
            "legal_references": """JSON array of legal citations: [{
                "type": "enum (EXACT values - typ aktu prawnego)",
                "title": "string, full title of legal source",
                "article": "string, specific article/section with all subdivisions",
                "jurisdiction": "string, jurisdiction",
                "citation": "string, full official citation as in document",
                "context": "string, how reference is used in reasoning"
            }]

TYPE enumeration (8 types of Polish legal acts):
- "konstytucja" (Konstytucja RP - Constitution)
- "ustawa" (Ustawa - Statute/Act, including kodeksy/codes)
- "rozporzadzenie" (Rozporządzenie - Regulation/Decree)
- "akt_prawa_miejscowego" (Akty prawa miejscowego - Local law acts)
- "umowa_miedzynarodowa" (Umowa międzynarodowa - International treaty)
- "przepis_ue" (Przepisy UE - EU regulations/directives)
- "orzecznictwo" (Orzecznictwo - Case law: wyroki SN, NSA, TK, TSUE, uchwały)
- "statut" (Statut - Organizational statute)

Examples:
[
  {
    "type": "konstytucja",
    "title": "Konstytucja Rzeczypospolitej Polskiej z dnia 2 kwietnia 1997 r.",
    "article": "Art. 64 ust. 1",
    "jurisdiction": "Poland",
    "citation": "Art. 64 ust. 1 Konstytucji RP (Dz.U. 1997 nr 78 poz. 483)",
    "context": "Podstawa konstytucyjna prawa własności jako podstawa roszczeń powoda"
  },
  {
    "type": "ustawa",
    "title": "Ustawa z dnia 23 kwietnia 1964 r. - Kodeks cywilny",
    "article": "Art. 385¹ § 1",
    "jurisdiction": "Poland",
    "citation": "Art. 385¹ § 1 ustawy z dnia 23 kwietnia 1964 r. - Kodeks cywilny (Dz.U. 2023 poz. 1610)",
    "context": "Klauzule niedozwolone (abuzywne) jako podstawa nieważności postanowień umowy"
  },
  {
    "type": "rozporzadzenie",
    "title": "Rozporządzenie Ministra Finansów z dnia 26 sierpnia 2003 r. w sprawie...",
    "article": "§ 8 ust. 1 pkt 2",
    "jurisdiction": "Poland",
    "citation": "§ 8 ust. 1 pkt 2 rozporządzenia Ministra Finansów z dnia 26.08.2003 r. (Dz.U. 2003 nr 155 poz. 1516)",
    "context": "Określenie warunków zwrotu podatku VAT"
  },
  {
    "type": "przepis_ue",
    "title": "Dyrektywa Rady 2006/112/WE z dnia 28 listopada 2006 r. w sprawie wspólnego systemu VAT",
    "article": "Art. 44",
    "jurisdiction": "EU",
    "citation": "Art. 44 Dyrektywy 2006/112/WE (Dz.Urz. UE L 347 z 11.12.2006)",
    "context": "Harmonizacja przepisów VAT - miejsce świadczenia usług"
  },
  {
    "type": "orzecznictwo",
    "title": "Wyrok Sądu Najwyższego z dnia 7 maja 2021 r.",
    "article": "",
    "jurisdiction": "Poland",
    "citation": "Uchwała SN z dnia 7 maja 2021 r., sygn. III CZP 6/21",
    "context": "Uchwała wyjaśniająca zagadnienie prawne dotyczące kredytów frankowych - precedens"
  }
]

Extract ALL legal references with complete citations""",
            "legal_concepts": """JSON array of legal concepts used in document: [{
                "concept_name": "string, name of legal concept in Polish",
                "legal_area": "string, area of law",
                "definition_context": "string, how concept is defined/used in document",
                "relevance": "enum (primary/secondary/mentioned)"
            }]

RELEVANCE enumeration:
- "primary" (główne pojęcie - central to case/interpretation)
- "secondary" (pojęcie pomocnicze - supporting concept)
- "mentioned" (wymienione - referenced but not central)

Examples:
[
  {
    "concept_name": "klauzule abuzywne (klauzule niedozwolone)",
    "legal_area": "prawo cywilne - ochrona konsumentów",
    "definition_context": "Postanowienia umowy, które kształtują prawa i obowiązki konsumenta w sposób
                           sprzeczny z dobrymi obyczajami, rażąco naruszając jego interesy (art. 385¹ KC)",
    "relevance": "primary"
  },
  {
    "concept_name": "spread walutowy",
    "legal_area": "prawo bankowe",
    "definition_context": "Różnica między kursem kupna i sprzedaży waluty stosowana przez bank przy
                           przeliczaniu rat kredytu indeksowanego",
    "relevance": "primary"
  },
  {
    "concept_name": "miejsce świadczenia usług",
    "legal_area": "VAT - podatek od towarów i usług",
    "definition_context": "Miejsce, w którym usługa podlega opodatkowaniu VAT, określone zgodnie z
                           art. 28b ustawy o VAT (zasada kraju siedziby nabywcy dla usług B2B)",
    "relevance": "primary"
  },
  {
    "concept_name": "koszty uzyskania przychodów",
    "legal_area": "CIT - podatek dochodowy od osób prawnych",
    "definition_context": "Koszty poniesione w celu osiągnięcia przychodów lub zachowania albo
                           zabezpieczenia źródła przychodów (art. 15 ust. 1 ustawy o CIT)",
    "relevance": "primary"
  }
]

Extract 3-10 key legal concepts, focusing on those central to reasoning""",
            "parties": """JSON array of parties involved in case/interpretation: [{
                "party_type": "enum (EXACT values - typ strony postępowania)",
                "party_name": "string, name (may be anonymized as 'J.K.' or 'Spółka X')",
                "party_category": "enum (kategoria podmiotu)",
                "representation": "string, legal representative if mentioned"
            }]

PARTY_TYPE enumeration (13 types across all proceedings):
Civil proceedings:
- "powod" (powód - plaintiff in civil case)
- "pozwany" (pozwany - defendant in civil case)
- "interwenient_uboczny" (interwenient uboczny - third-party intervenor)
- "uczestnik" (uczestnik postępowania - participant in non-contentious proceedings)
- "wnioskodawca" (wnioskodawca - petitioner/applicant)

Administrative proceedings:
- "skarzacy" (skarżący - complainant/appellant in administrative court)
- "organ_administracji" (organ administracji - administrative body/authority)

Criminal proceedings:
- "oskarżony" (oskarżony - accused/defendant in criminal case)
- "pokrzywdzony" (pokrzywdzony - injured party/victim)
- "oskarżyciel_publiczny" (oskarżyciel publiczny - public prosecutor)
- "oskarżyciel_posilkowy" (oskarżyciel posiłkowy - auxiliary prosecutor)

Tax interpretations:
- "podatnik" (podatnik - taxpayer requesting interpretation)
- "organ_podatkowy" (organ podatkowy - tax authority issuing interpretation)

PARTY_CATEGORY enumeration (6 types):
- "osoba_fizyczna" (osoba fizyczna - natural person/individual)
- "osoba_prawna" (osoba prawna - legal person: company, foundation, association)
- "jednostka_organizacyjna_bez_osobowosci" (j.o. bez osobowości prawnej - organizational unit without legal personality: partnership)
- "organ_panstwowy" (organ państwowy - state authority: ministry, government agency)
- "organ_samorzadowy" (organ samorządowy - local government authority: gmina, powiat)
- "przedsiebiorca" (przedsiębiorca - entrepreneur/business entity)

Example (judgment - civil case):
[
  {
    "party_type": "powod",
    "party_name": "Jan Kowalski",
    "party_category": "osoba_fizyczna",
    "representation": "adw. Anna Nowak, Kancelaria Adwokacka w Warszawie"
  },
  {
    "party_type": "pozwany",
    "party_name": "Bank ABC Spółka Akcyjna z siedzibą w Warszawie",
    "party_category": "osoba_prawna",
    "representation": "r.pr. Tomasz Wiśniewski, Departament Prawny Banku ABC S.A."
  }
]

Example (tax interpretation):
[
  {
    "party_type": "podatnik",
    "party_name": "XYZ Sp. z o.o.",
    "party_category": "osoba_prawna",
    "representation": ""
  },
  {
    "party_type": "organ_podatkowy",
    "party_name": "Dyrektor Krajowej Informacji Skarbowej",
    "party_category": "organ_panstwowy",
    "representation": ""
  }
]

Example (administrative case):
[
  {
    "party_type": "skarzacy",
    "party_name": "Maria Nowak",
    "party_category": "osoba_fizyczna",
    "representation": "pełnomocnik Jan Kowalski"
  },
  {
    "party_type": "organ_administracji",
    "party_name": "Naczelnik Urzędu Skarbowego w Warszawie",
    "party_category": "organ_panstwowy",
    "representation": ""
  }
]""",
            # ============================================================================
            # STRUCTURED CONTENT (1 field)
            # ============================================================================
            "legal_analysis": """JSON object with structured legal reasoning: {
                "facts_summary": "string, key factual findings (2-3 sentences)",
                "legal_issues": ["string, main legal questions posed (lista pytań prawnych)"],
                "reasoning": "string, court's/authority's legal reasoning (uzasadnienie - 3-5 sentences)",
                "conclusion": "string, final legal conclusion (konkluzja - 1-2 sentences)"
            }

Example (judgment):
{
  "facts_summary": "Powód zawarł z pozwanym umowę kredytu indeksowanego do CHF. Bank stosował własną
                    tabelę kursów do przeliczania rat, co prowadziło do wzrostu zadłużenia.
                    Powód nie został należycie poinformowany o ryzyku kursowym.",
  "legal_issues": [
    "Czy postanowienia umowy dotyczące mechanizmu indeksacji stanowią klauzule abuzywne?",
    "Czy możliwe jest utrzymanie umowy po wyeliminowaniu klauzul niedozwolonych?",
    "Jakie są skutki stwierdzenia nieważności umowy dla obu stron?"
  ],
  "reasoning": "Sąd uznał, że postanowienia umowy dotyczące przeliczania rat kredytu według kursu z tabeli
                banku stanowią klauzule niedozwolone w rozumieniu art. 385¹ § 1 k.c., gdyż kształtują
                prawa i obowiązki konsumenta w sposób sprzeczny z dobrymi obyczajami, rażąco naruszając
                jego interesy. Klauzule te nie zostały uzgodnione indywidualnie, a bank posiadał pełną
                swobodę w ustalaniu kursów walut. Eliminacja klauzul abuzywnych prowadzi do niemożności
                utrzymania umowy, gdyż mechanizm indeksacji stanowił jej istotny element. Zgodnie z
                wyrokiem TSUE C-260/18, w takiej sytuacji umowa jest nieważna od początku.",
  "conclusion": "Umowa kredytu jest nieważna od początku. Strony zobowiązane są do zwrotu wzajemnie
                 otrzymanych świadczeń zgodnie z przepisami o bezpodstawnym wzbogaceniu."
}

Example (tax interpretation):
{
  "facts_summary": "Wnioskodawca świadczy usługi IT na rzecz podatników z krajów UE. Pytanie dotyczy
                    miejsca opodatkowania tych usług VAT.",
  "legal_issues": [
    "Czy usługi IT świadczone dla podatników unijnych podlegają opodatkowaniu w Polsce?",
    "Jakie zasady określają miejsce świadczenia usług w transakcjach B2B?"
  ],
  "reasoning": "Zgodnie z art. 28b ust. 1 ustawy o VAT, miejscem świadczenia usług na rzecz podatnika
                jest miejsce, w którym podatnik będący usługobiorcą posiada siedzibę działalności
                gospodarczej. Przepis ten implementuje art. 44 Dyrektywy 2006/112/WE. W przypadku usług
                IT świadczonych przez polskiego podatnika na rzecz podatników z siedzibą w innych krajach
                UE, miejscem opodatkowania jest kraj siedziby nabywcy usługi.",
  "conclusion": "Stanowisko wnioskodawcy jest prawidłowe. Usługi IT świadczone na rzecz podatników UE
                 podlegają opodatkowaniu w kraju nabywcy, a wnioskodawca nie rozlicza VAT w Polsce."
}""",
            # ============================================================================
            # DOCUMENT-TYPE SPECIFIC FIELDS (2 fields with UPDATED enumerations)
            # ============================================================================
            "judgment_specific": """JSON object (ONLY if document_type = "judgment"): {
                "court_name": "string, full official court name",
                "court_type": "enum (EXACT values - typ sądu)",
                "department_name": "string, court department/division (wydział)",
                "judges": [{"name": "string", "role": "enum (presiding/member/reporting)"}],
                "legal_bases": ["string, specific legal basis articles for decision"],
                "judgment_type": "enum (typ orzeczenia)"
            }

COURT_TYPE enumeration (8 types - Polish court system):
Common courts (sądy powszechne):
- "rejonowy" (Sąd Rejonowy - District Court - lowest level, first instance)
- "okregowy" (Sąd Okręgowy - Regional Court - medium level, first/second instance)
- "apelacyjny" (Sąd Apelacyjny - Court of Appeal - second instance)
- "najwyzszy" (Sąd Najwyższy - Supreme Court - cassation, final instance)

Administrative courts (sądownictwo administracyjne):
- "wojewodzki_administracyjny" (WSA - Wojewódzki Sąd Administracyjny - Provincial Administrative Court)
- "naczelny_administracyjny" (NSA - Naczelny Sąd Administracyjny - Supreme Administrative Court)

Special courts (sądy szczególne):
- "trybunal_konstytucyjny" (Trybunał Konstytucyjny - Constitutional Tribunal)
- "trybunal_stanu" (Trybunał Stanu - State Tribunal)

JUDGMENT_TYPE enumeration:
- "wyrok" (wyrok - judgment, final decision on merits)
- "postanowienie" (postanowienie - order, procedural decision)
- "zarzadzenie" (zarządzenie - administrative order)
- "uchwala" (uchwała - resolution, especially from Supreme Court)

JUDGE ROLE enumeration:
- "presiding" (przewodniczący - presiding judge)
- "reporting" (sprawozdawca - reporting judge, prepares case)
- "member" (członek składu - panel member)

Example:
{
  "court_name": "Sąd Apelacyjny w Warszawie, I Wydział Cywilny",
  "court_type": "apelacyjny",
  "department_name": "I Wydział Cywilny",
  "judges": [
    {"name": "SSA Jan Kowalski", "role": "presiding"},
    {"name": "SSA Anna Nowak", "role": "reporting"},
    {"name": "SSA Tomasz Wiśniewski", "role": "member"}
  ],
  "legal_bases": [
    "art. 385¹ § 1 Kodeksu cywilnego",
    "art. 58 § 1 k.c.",
    "art. 69 ust. 1 Prawa bankowego"
  ],
  "judgment_type": "wyrok"
}

Leave null if document_type is not "judgment".""",
            "tax_interpretation_specific": """JSON object (ONLY if document_type = "tax_interpretation"): {
                "interpretation_type": "enum (typ interpretacji)",
                "tax_authority": "string, full name of issuing tax authority",
                "tax_matter": "string, specific tax question/issue (2-3 sentences)",
                "tax_type": "enum (EXACT values - typ podatku)",
                "related_taxes": ["enum, other taxes mentioned"]
            }

INTERPRETATION_TYPE enumeration:
- "indywidualna" (indywidualna - individual interpretation for specific taxpayer)
- "ogolna" (ogólna - general interpretation for all taxpayers)

TAX_TYPE enumeration (15 types - complete Polish tax system):
Most common (95% of interpretations):
- "VAT" (podatek od towarów i usług - Value Added Tax)
- "CIT" (podatek dochodowy od osób prawnych - Corporate Income Tax)
- "PIT" (podatek dochodowy od osób fizycznych - Personal Income Tax)
- "PCC" (podatek od czynności cywilnoprawnych - Tax on Civil Law Transactions)

Other taxes:
- "akcyza" (podatek akcyzowy - Excise Tax on alcohol, tobacco, fuel, cars)
- "podatek_od_spadkow_i_darowizn" (podatek od spadków i darowizn - Inheritance and Gift Tax)
- "podatek_od_nieruchomosci" (podatek od nieruchomości - Real Estate Tax)
- "podatek_rolny" (podatek rolny - Agricultural Tax)
- "podatek_lesny" (podatek leśny - Forestry Tax)
- "oplata_skarbowa" (opłata skarbowa - Treasury Fee)
- "podatek_od_gier" (podatek od gier - Gambling Tax)
- "podatek_od_srodkow_transportowych" (podatek od środków transportowych - Motor Vehicle Tax)
- "skladki_zus" (składki ZUS - Social Security Contributions)
- "podatek_od_wydobycia_kopalin" (podatek od wydobycia kopalin - Mineral Extraction Tax)
- "oplata_paliwowa" (opłata paliwowa - Fuel Charge)

Example:
{
  "interpretation_type": "indywidualna",
  "tax_authority": "Dyrektor Krajowej Informacji Skarbowej",
  "tax_matter": "Pytanie dotyczy możliwości zastosowania stawki 0% VAT do usług IT świadczonych na rzecz
                 podatników z siedzibą w innych krajach UE oraz określenia miejsca świadczenia usług
                 zgodnie z art. 28b ustawy o VAT.",
  "tax_type": "VAT",
  "related_taxes": []
}

Leave null if document_type is not "tax_interpretation".""",
        },
        instructions="""
INSTRUKCJE EKSTRAKCJI - Polish Legal Documents Extraction

OGÓLNE ZASADY:
- Ekstraktuj informacje WYŁĄCZNIE z tekstu dokumentu - nie domyślaj się
- Używaj POLSKIEJ terminologii dla wszystkich enumeracji
- Wszystkie enumeracje: używaj TYLKO wartości z listy (nie inne wartości!)
- Daty: format ISO 8601 (YYYY-MM-DD) ZAWSZE
- Puste pola proste (string): użyj pustego stringa ""
- Puste pola złożone (array): użyj pustej tablicy []
- Puste pola obiektowe: użyj null
- Język: polski (maintain original document language)

ENUMERACJE - TYLKO DOZWOLONE WARTOŚCI:
decision_type (wyroki):
  uwzgledniono_w_calosci | uwzgledniono_w_czesci | oddalono | umorzono |
  uchylono | uchylono_i_przekazano | zmieniono | utrzymano_w_mocy | odrzucono

decision_type (interpretacje):
  stanowisko_prawidlowe | stanowisko_nieprawidlowe | stanowisko_czesciowo_prawidlowe

legal_references.type:
  konstytucja | ustawa | rozporzadzenie | akt_prawa_miejscowego |
  umowa_miedzynarodowa | przepis_ue | orzecznictwo | statut

court_type:
  rejonowy | okregowy | apelacyjny | najwyzszy | wojewodzki_administracyjny |
  naczelny_administracyjny | trybunal_konstytucyjny | trybunal_stanu

party_type:
  powod | pozwany | interwenient_uboczny | uczestnik | wnioskodawca |
  skarzacy | organ_administracji | oskarżony | pokrzywdzony |
  oskarżyciel_publiczny | oskarżyciel_posilkowy | podatnik | organ_podatkowy

party_category:
  osoba_fizyczna | osoba_prawna | jednostka_organizacyjna_bez_osobowosci |
  organ_panstwowy | organ_samorzadowy | przedsiebiorca

tax_type:
  VAT | CIT | PIT | PCC | akcyza | podatek_od_spadkow_i_darowizn |
  podatek_od_nieruchomosci | podatek_rolny | podatek_lesny | oplata_skarbowa |
  podatek_od_gier | podatek_od_srodkow_transportowych | skladki_zus |
  podatek_od_wydobycia_kopalin | oplata_paliwowa

NOWE POLA - SZCZEGÓLNA UWAGA:
factual_state: Ekstraktuj pełną narrację o stanie faktycznym - wszystkie istotne
               okoliczności, chronologia, liczby, kwoty. Szukaj sekcji "Stan faktyczny",
               "Ustalenia faktyczne", "Okoliczności faktyczne".

legal_state: Ekstraktuj wszystkie przepisy prawne z numerami artykułów - hierarchicznie
             (Konstytucja → ustawy → rozporządzenia → orzecznictwo). Szukaj sekcji
             "Stan prawny", "Podstawa prawna", "Przepisy prawne".

CYTOWANIA PRAWNE:
- Ekstraktuj z pełnymi numerami: "Art. 385¹ § 1 k.c." (nie skracaj)
- Uwzględniaj wszystkie poziomy: artykuł, ustęp, punkt, litera
- Dla orzeczeń: podaj sygnaturę (np. "III CZP 6/21")
- Dla UE: podaj numer dyrektywy/rozporządzenia

WALIDACJA PRZED ZWRÓCENIEM:
1. Sprawdź czy wszystkie enumeracje używają TYLKO dozwolonych wartości
2. Sprawdź czy daty są w formacie YYYY-MM-DD
3. Sprawdź czy wszystkie JSON są poprawnie sformatowane
4. Sprawdź czy factual_state i legal_state nie są puste (chyba że brak w dokumencie)
5. Sprawdź czy legal_references.type używa polskiej terminologii

ZWRÓĆ TYLKO JSON - bez dodatkowego tekstu, markdown, czy objaśnień.
        """,
        language="polish",
    )


def fetch_documents_rest(
    weaviate_host: str,
    weaviate_port: int,
    api_key: str,
    sample_size: int = 50,
    chunk_size: int = 1000,
    search_query: str = None,
    document_type_filter: str = None,
) -> List[Dict[str, Any]]:
    """Fetch documents using Weaviate REST API directly with pagination.

    Args:
        weaviate_host: Weaviate host
        weaviate_port: Weaviate port
        api_key: Weaviate API key
        sample_size: Number of documents to sample
        chunk_size: Number of documents to fetch per request (max 10000)
        search_query: Optional search query for hybrid/semantic search
        document_type_filter: Optional filter by document_type (e.g., "judgment", "tax_interpretation")

    Returns:
        List of document properties
    """
    base_url = f"http://{weaviate_host}:{weaviate_port}"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Fetch documents in chunks with pagination
    # Note: Weaviate has a hard limit of offset < 10000, so max we can fetch is ~10000 docs
    target_size = sample_size * 5  # Fetch more to filter for valid full_text
    max_offset = 10000  # Weaviate offset limit
    target_size = min(target_size, max_offset)  # Cap at Weaviate limit
    chunk_size = min(chunk_size, 1000)  # Reasonable chunk size

    # Build filter info for logging
    filter_info = []
    if search_query:
        filter_info.append(f"search='{search_query}'")
    if document_type_filter:
        filter_info.append(f"type={document_type_filter}")

    filter_str = f" with filters: {', '.join(filter_info)}" if filter_info else ""

    logger.info(
        f"Fetching up to {target_size} documents from {base_url} in chunks of {chunk_size} (REST API){filter_str}..."
    )
    logger.info(
        f"Note: Weaviate offset limit is {max_offset}, so maximum fetchable documents is {max_offset}"
    )

    all_documents = []
    offset = 0

    while len(all_documents) < target_size and offset < max_offset:
        # Calculate how many more documents we need
        remaining = min(target_size - len(all_documents), max_offset - offset)
        current_limit = min(chunk_size, remaining)

        # Build GraphQL query with optional search and filter
        # Construct the where clause if needed
        where_clause = ""
        if document_type_filter:
            where_clause = f"""
                where: {{
                    path: ["document_type"],
                    operator: Equal,
                    valueText: "{document_type_filter}"
                }}
            """

        # Construct hybrid search or regular query
        if search_query:
            # Use hybrid search (combines semantic + keyword search)
            query_method = f"""
                hybrid: {{
                    query: "{search_query}",
                    alpha: 0.5
                }}
                {where_clause}
                limit: {current_limit}
                offset: {offset}
            """
        else:
            # Regular query with optional filter
            query_method = f"""
                {where_clause}
                limit: {current_limit}
                offset: {offset}
            """

        # Full GraphQL query
        query = {
            "query": """
            {
                Get {
                    LegalDocuments(%s) {
                        document_id
                        document_type
                        full_text
                        language
                        document_number
                    }
                }
            }
            """
            % query_method
        }

        try:
            logger.info(f"Fetching chunk: offset={offset}, limit={current_limit}")
            response = requests.post(
                f"{base_url}/v1/graphql",
                headers=headers,
                json=query,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                # Check if it's the offset limit error
                error_msg = str(data["errors"])
                if "query maximum results exceeded" in error_msg or offset >= max_offset:
                    logger.warning(
                        f"Reached Weaviate offset limit at {offset}. Using {len(all_documents)} documents."
                    )
                    break
                logger.error(f"GraphQL errors: {data['errors']}")
                raise Exception(f"GraphQL query failed: {data['errors']}")

            documents = data.get("data", {}).get("Get", {}).get("LegalDocuments", [])

            if not documents:
                logger.info(f"No more documents available at offset {offset}")
                break

            all_documents.extend(documents)
            logger.info(f"Fetched {len(documents)} documents (total: {len(all_documents)})")

            # Move to next chunk
            offset += len(documents)

            # Break if we got fewer documents than requested (end of data)
            if len(documents) < current_limit:
                logger.info("Reached end of available documents")
                break

        except Exception as e:
            # If we hit the offset limit but have documents, continue
            if "query maximum results exceeded" in str(e) and len(all_documents) > 0:
                logger.warning(
                    f"Hit Weaviate offset limit at {offset}. Continuing with {len(all_documents)} documents."
                )
                break
            logger.error(f"Failed to fetch chunk at offset {offset}: {e}")
            raise

    logger.info(f"Fetched {len(all_documents)} total documents from Weaviate")

    # Filter documents with non-empty full_text
    valid_docs = []
    for doc in all_documents:
        full_text = doc.get("full_text", "")
        if full_text and len(full_text.strip()) > 100:  # At least 100 chars
            valid_docs.append(doc)

    logger.info(f"Found {len(valid_docs)} documents with valid full_text")

    if not valid_docs:
        logger.warning("No documents with valid full_text found!")
        return []

    # Random sample
    sample = random.sample(valid_docs, min(sample_size, len(valid_docs)))

    logger.info(f"Sampled {len(sample)} documents for extraction")

    return sample


def _process_batch(
    batch_docs: List[Dict[str, Any]],
    chain: GeminiExtractionChain,
    schema: ExtractionSchema,
    langfuse_handler,
    batch_idx: int,
    storage: Optional[ExtractionStorage] = None,
    run_id: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Process a single batch of documents (used for parallel processing).

    Args:
        batch_docs: Documents in this batch
        chain: Extraction chain
        schema: Extraction schema
        langfuse_handler: Langfuse handler
        batch_idx: Batch index for logging

    Returns:
        List of extraction results
    """
    batch_results = []

    # Prepare batch data
    batch_texts = []
    batch_metadata = []

    for doc in batch_docs:
        document_id = doc.get("document_id", "unknown")
        full_text = doc.get("full_text", "")
        doc_type_str = doc.get("document_type", "judgment")

        # Map document type
        if "interpret" in doc_type_str.lower():
            doc_type = DocumentType.TAX_INTERPRETATION
        else:
            doc_type = DocumentType.JUDGMENT

        batch_texts.append(full_text)
        batch_metadata.append(
            {
                "document_id": document_id,
                "document_type": doc_type,
                "doc_type_str": doc_type_str,
                "full_text_length": len(full_text),
                "language": doc.get("language", "unknown"),
            }
        )

    # Get document type for batch
    batch_doc_type = batch_metadata[0]["document_type"]

    try:
        # Run batch extraction
        logger.info(f"[Batch {batch_idx}] Processing {len(batch_texts)} documents...")
        extracted_batch = chain.batch_extract(
            document_type=batch_doc_type,
            texts=batch_texts,
            schema=schema,
            langfuse_handler=langfuse_handler,
            max_text_length=150000,
        )

        # Process batch results
        for extracted, metadata, doc in zip(extracted_batch, batch_metadata, batch_docs):
            result = {
                "document_id": metadata["document_id"],
                "document_number": doc.get("document_number"),
                "document_type": metadata["doc_type_str"],
                "extraction_status": "success",
                "extracted_data": extracted,
                "full_text": doc.get("full_text", ""),
                "full_text_length": metadata["full_text_length"],
                "source_language": metadata["language"],
            }
            batch_results.append(result)

            # Save to database if storage provided
            if storage and run_id:
                try:
                    storage.save_extraction_result(
                        run_id=run_id,
                        document_id=metadata["document_id"],
                        document_number=doc.get("document_number"),
                        document_type=metadata["doc_type_str"],
                        full_text=doc.get("full_text", ""),
                        extraction_status="success",
                        extracted_data=extracted,
                        source_language=metadata["language"],
                    )
                except Exception as db_error:
                    logger.warning(f"Failed to save to database: {db_error}")

            logger.info(
                f"[Batch {batch_idx}] ✓ Extracted {metadata['document_id']} ({len(extracted)} fields)"
            )

    except Exception as e:
        # If batch fails, fall back to individual processing
        logger.warning(
            f"[Batch {batch_idx}] Batch extraction failed: {e}, falling back to individual processing"
        )

        for text, metadata, doc in zip(batch_texts, batch_metadata, batch_docs):
            try:
                extracted = chain.extract(
                    document_type=metadata["document_type"],
                    text=text,
                    schema=schema,
                    langfuse_handler=langfuse_handler,
                    max_text_length=150000,
                )

                result = {
                    "document_id": metadata["document_id"],
                    "document_number": doc.get("document_number"),
                    "document_type": metadata["doc_type_str"],
                    "extraction_status": "success",
                    "extracted_data": extracted,
                    "full_text": doc.get("full_text", ""),
                    "full_text_length": metadata["full_text_length"],
                    "source_language": metadata["language"],
                }
                batch_results.append(result)

                # Save to database if storage provided
                if storage and run_id:
                    try:
                        storage.save_extraction_result(
                            run_id=run_id,
                            document_id=metadata["document_id"],
                            document_number=doc.get("document_number"),
                            document_type=metadata["doc_type_str"],
                            full_text=doc.get("full_text", ""),
                            extraction_status="success",
                            extracted_data=extracted,
                            source_language=metadata["language"],
                        )
                    except Exception as db_error:
                        logger.warning(f"Failed to save to database: {db_error}")

                logger.info(
                    f"[Batch {batch_idx}] ✓ Extracted {metadata['document_id']} ({len(extracted)} fields)"
                )

            except Exception as e2:
                logger.error(
                    f"[Batch {batch_idx}] ✗ Failed to extract {metadata['document_id']}: {e2}"
                )
                result = {
                    "document_id": metadata["document_id"],
                    "document_number": doc.get("document_number"),
                    "document_type": metadata["doc_type_str"],
                    "extraction_status": "failed",
                    "error": str(e2),
                    "full_text": doc.get("full_text", ""),
                    "full_text_length": metadata["full_text_length"],
                }
                batch_results.append(result)

                # Save failed result to database if storage provided
                if storage and run_id:
                    try:
                        storage.save_extraction_result(
                            run_id=run_id,
                            document_id=metadata["document_id"],
                            document_number=doc.get("document_number"),
                            document_type=metadata["doc_type_str"],
                            full_text=doc.get("full_text", ""),
                            extraction_status="failed",
                            error_message=str(e2),
                            error_type=type(e2).__name__,
                        )
                    except Exception as db_error:
                        logger.warning(f"Failed to save error to database: {db_error}")

    return batch_results


def run_extraction(
    documents: List[Dict[str, Any]],
    chain: GeminiExtractionChain,
    schema: ExtractionSchema,
    langfuse_handler=None,
    batch_size: int = 10,
    max_workers: int = 1,
    storage: Optional[ExtractionStorage] = None,
    run_id: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run extraction on sampled documents using parallel batch processing.

    Args:
        documents: List of document properties
        chain: Extraction chain
        schema: Extraction schema
        langfuse_handler: Optional Langfuse callback handler for observability
        batch_size: Number of documents to process in each batch (default: 10)
        max_workers: Number of parallel threads for batch processing (default: 1, max: 5)
        storage: Optional ExtractionStorage for database persistence
        run_id: Optional run_id for database storage

    Returns:
        List of extraction results with metadata
    """
    # Create batches
    batches = []
    for batch_start in range(0, len(documents), batch_size):
        batch_docs = documents[batch_start : batch_start + batch_size]
        batches.append((batch_start // batch_size, batch_docs))

    logger.info(
        f"Created {len(batches)} batches of size {batch_size} with {max_workers} parallel workers"
    )

    results = []
    results_lock = Lock()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting documents...", total=len(documents))

        if max_workers == 1:
            # Sequential processing (no threading)
            for batch_idx, batch_docs in batches:
                batch_results = _process_batch(
                    batch_docs, chain, schema, langfuse_handler, batch_idx, storage, run_id
                )
                with results_lock:
                    results.extend(batch_results)
                    progress.update(task, advance=len(batch_results))
        else:
            # Parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(
                        _process_batch,
                        batch_docs,
                        chain,
                        schema,
                        langfuse_handler,
                        batch_idx,
                        storage,
                        run_id,
                    ): batch_idx
                    for batch_idx, batch_docs in batches
                }

                # Process completed batches
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        with results_lock:
                            results.extend(batch_results)
                            progress.update(task, advance=len(batch_results))
                    except Exception as e:
                        logger.error(f"[Batch {batch_idx}] Thread execution failed: {e}")

    return results


def save_results(
    documents: List[Dict[str, Any]],
    extraction_results: List[Dict[str, Any]],
    output_dir: Path,
):
    """Save full_text and extraction results to separate files.

    Args:
        documents: Original documents with full_text
        extraction_results: Extraction results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full_text documents
    full_text_file = output_dir / "sample_documents_full_text.jsonl"
    with open(full_text_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(documents)} full_text documents to {full_text_file}")

    # Save extraction results (single-line JSON for proper JSONL format)
    extracted_file = output_dir / "sample_documents_extracted.jsonl"
    with open(extracted_file, "w", encoding="utf-8") as f:
        for result in extraction_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(extraction_results)} extraction results to {extracted_file}")

    # Save summary statistics
    summary_file = output_dir / "extraction_summary.json"

    successful = sum(1 for r in extraction_results if r.get("extraction_status") == "success")
    failed = len(extraction_results) - successful

    # Analyze field coverage
    field_coverage = {}
    for result in extraction_results:
        if result.get("extraction_status") == "success":
            extracted_data = result.get("extracted_data") or {}
            if not isinstance(extracted_data, dict):
                continue
            for field, value in extracted_data.items():
                if field not in field_coverage:
                    field_coverage[field] = {"populated": 0, "empty": 0}

                # Check if field is populated
                if value:
                    if isinstance(value, str) and value.strip():
                        field_coverage[field]["populated"] += 1
                    elif isinstance(value, list) and value:
                        field_coverage[field]["populated"] += 1
                    elif isinstance(value, dict) and value:
                        field_coverage[field]["populated"] += 1
                    else:
                        field_coverage[field]["empty"] += 1
                else:
                    field_coverage[field]["empty"] += 1

    summary = {
        "total_documents": len(documents),
        "successful_extractions": successful,
        "failed_extractions": failed,
        "success_rate": f"{(successful / len(extraction_results) * 100):.1f}%",
        "field_coverage": field_coverage,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved extraction summary to {summary_file}")

    # Print summary to console
    console.print("\n[bold cyan]Extraction Summary[/bold cyan]")
    console.print(f"Total documents: {len(documents)}")
    console.print(f"[green]Successful: {successful}[/green]")
    console.print(f"[red]Failed: {failed}[/red]")
    console.print(f"Success rate: {summary['success_rate']}")

    console.print("\n[bold cyan]Field Coverage[/bold cyan]")
    for field, stats in sorted(field_coverage.items()):
        total = stats["populated"] + stats["empty"]
        coverage = (stats["populated"] / total * 100) if total > 0 else 0
        console.print(f"  {field}: {coverage:.1f}% ({stats['populated']}/{total})")


# ============================================================================
# WEAVIATE INGESTION FUNCTIONS
# ============================================================================

# Field mapping from extraction schema to Weaviate properties
# All properties already exist in Weaviate - no need for "extracted_" prefix!
# NOTE: document_number, document_type, and date_issued are excluded as they're always already valid
EXTRACTION_TO_WEAVIATE_MAPPING = {
    # Direct TEXT mappings (existing properties in Weaviate)
    "title": "title",
    "summary": "summary",
    "thesis": "thesis",  # Use existing property
    # TEXT_ARRAY (existing property - native array support)
    "keywords": "keywords",
    # NEW properties (just added to schema)
    "factual_state": "factual_state",
    "legal_state": "legal_state",
    # TEXT (JSON) properties (existing - need JSON serialization)
    "legal_references": "legal_references",
    "legal_concepts": "legal_concepts",
    "parties": "parties",
    "outcome": "outcome",
    "legal_analysis": "legal_analysis",
    "judgment_specific": "judgment_specific",
    "tax_interpretation_specific": "tax_interpretation_specific",
}


def build_update_payload(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform extracted data to Weaviate property update payload.

    Handles:
    - Direct TEXT fields (no transformation)
    - TEXT_ARRAY fields (keywords - no transformation)
    - TEXT (JSON) fields (need JSON serialization for lists/objects)

    Args:
        extracted_data: Dictionary with extracted fields from LLM

    Returns:
        Dictionary with Weaviate properties ready for PATCH request
    """
    payload = {}

    # Fields that need JSON serialization (stored as TEXT in Weaviate)
    json_fields = {
        "legal_references",
        "legal_concepts",
        "parties",
        "outcome",
        "legal_analysis",
        "judgment_specific",
        "tax_interpretation_specific",
    }

    for extracted_field, weaviate_property in EXTRACTION_TO_WEAVIATE_MAPPING.items():
        value = extracted_data.get(extracted_field)

        # Skip empty/null values
        if value is None or value == "":
            continue

        # Handle list fields
        if isinstance(value, list):
            # Filter out empty strings
            cleaned_list = [v for v in value if v and str(v).strip()]
            if not cleaned_list:
                continue

            # keywords is TEXT_ARRAY - use directly
            # Other lists need JSON serialization
            if extracted_field == "keywords":
                payload[weaviate_property] = cleaned_list
            elif extracted_field in json_fields:
                payload[weaviate_property] = json.dumps(cleaned_list, ensure_ascii=False)
            else:
                # Default: direct assignment
                payload[weaviate_property] = cleaned_list

        # Handle object/dict fields (judgment_specific, tax_interpretation_specific)
        elif isinstance(value, dict):
            # Only include if dict has meaningful content
            if not value or all(v is None or v == "" for v in value.values()):
                continue

            if extracted_field in json_fields:
                payload[weaviate_property] = json.dumps(value, ensure_ascii=False)
            else:
                # Default: direct assignment (shouldn't happen)
                payload[weaviate_property] = value

        # Handle string fields
        elif isinstance(value, str):
            # Special case: keywords field might be a JSON string or comma-separated string
            if extracted_field == "keywords":
                # Try to parse as JSON array first
                if value.startswith("[") and value.endswith("]"):
                    try:
                        parsed_list = json.loads(value)
                        if isinstance(parsed_list, list):
                            cleaned_list = [v for v in parsed_list if v and str(v).strip()]
                            if cleaned_list:
                                payload[weaviate_property] = cleaned_list
                            continue
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(
                            f"Failed to parse keywords as JSON list: {value[:100]}"
                        )

                # If not JSON, try comma-separated string
                if "," in value or value.strip():  # Single keyword or comma-separated
                    # Split by comma and clean up
                    keywords_list = [k.strip() for k in value.split(",") if k.strip()]
                    if keywords_list:
                        payload[weaviate_property] = keywords_list
                        continue

            if extracted_field in json_fields:
                # String fields that need JSON wrapping (outcome, legal_analysis)
                payload[weaviate_property] = json.dumps(value, ensure_ascii=False)
            else:
                # Direct TEXT fields (most common)
                payload[weaviate_property] = value

        else:
            # Default: direct assignment for other types
            payload[weaviate_property] = value

    return payload


def ingest_batch_via_batch_api(
    batch_data: List[Dict[str, Any]],
    base_url: str,
    headers: Dict[str, str],
    overwrite_existing: bool = False,
) -> tuple[int, int, List[Dict[str, str]]]:
    """
    Ingest a batch of documents using Weaviate's batch API.

    This is more efficient than individual PATCH requests and less likely to block queries.

    Args:
        batch_data: List of dicts with 'uuid', 'payload', and 'document_id'
        base_url: Weaviate base URL
        headers: Request headers
        overwrite_existing: Whether to overwrite existing values

    Returns:
        Tuple of (successful_updates, failed_updates, errors)
    """
    successful = 0
    failed = 0
    errors = []

    # Build batch objects for update
    batch_objects = []

    for item in batch_data:
        uuid = item["uuid"]
        payload = item["payload"]
        document_id = item["document_id"]

        if not payload:
            # No fields to update
            successful += 1
            continue

        # Fetch existing data if not overwriting
        if not overwrite_existing:
            try:
                get_url = f"{base_url}/v1/objects/LegalDocuments/{uuid}"
                get_response = requests.get(get_url, headers=headers, timeout=10)

                if get_response.ok:
                    existing_data = get_response.json().get("properties", {})

                    # Filter out non-empty fields
                    filtered_payload = {}
                    for field, value in payload.items():
                        existing_value = existing_data.get(field)
                        is_empty = (
                            existing_value is None
                            or existing_value == ""
                            or (isinstance(existing_value, list) and not existing_value)
                        )
                        if is_empty:
                            filtered_payload[field] = value

                    payload = filtered_payload

                    if not payload:
                        # All fields already populated
                        successful += 1
                        continue
            except Exception as e:
                logger.warning(f"Error fetching existing data for {document_id}: {e}")
                # Continue with full payload

        batch_objects.append({
            "id": uuid,
            "class": "LegalDocuments",
            "properties": payload,
        })

    if not batch_objects:
        # All documents skipped (no updates needed)
        return successful, failed, errors

    # Send batch request
    batch_url = f"{base_url}/v1/batch/objects"

    try:
        response = requests.post(
            batch_url,
            headers=headers,
            json={"objects": batch_objects, "action": "MERGE"},  # MERGE updates only specified fields
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()

        # Process batch results
        if isinstance(result, list):
            for item_result in result:
                if item_result.get("result", {}).get("status") == "SUCCESS":
                    successful += 1
                else:
                    failed += 1
                    error_msg = item_result.get("result", {}).get("errors", {})
                    errors.append({
                        "document_id": item_result.get("id", "unknown"),
                        "error": str(error_msg),
                    })
        else:
            # Assume all successful if no error
            successful += len(batch_objects)

    except requests.exceptions.HTTPError as e:
        # Batch failed - mark all as failed
        failed += len(batch_objects)
        for item in batch_data:
            errors.append({
                "document_id": item["document_id"],
                "error": f"Batch API error: {e}",
                "status_code": e.response.status_code if e.response else None,
            })
        logger.error(f"Batch API request failed: {e}")

    except Exception as e:
        failed += len(batch_objects)
        for item in batch_data:
            errors.append({
                "document_id": item["document_id"],
                "error": f"Batch processing error: {e}",
            })
        logger.error(f"Batch processing failed: {e}")

    return successful, failed, errors


def ingest_extracted_to_weaviate(
    extraction_results: List[Dict[str, Any]],
    weaviate_host: str,
    weaviate_port: int,
    api_key: str,
    batch_size: int = 50,
    skip_on_error: bool = True,
    delay_between_batches: float = 0.5,
    overwrite_existing: bool = False,
    use_batch_api: bool = True,
) -> Dict[str, Any]:
    """
    Ingest extracted data back into Weaviate, updating existing documents.

    This function:
    1. Filters for successfully extracted documents
    2. Builds update payloads mapping extracted fields to Weaviate properties
    3. Checks existing document values before updating (unless overwrite_existing=True)
    4. Uses PATCH requests to update documents in batches
    5. Tracks success/failure statistics
    6. Generates detailed error reports

    Args:
        extraction_results: List of extraction results from run_extraction()
        weaviate_host: Weaviate server host
        weaviate_port: Weaviate server port
        api_key: Weaviate API key
        batch_size: Number of documents to update per batch (default: 50)
        skip_on_error: Continue on individual document errors (default: True)
        delay_between_batches: Seconds to wait between batches (default: 0.5)
        overwrite_existing: If False (default), only update empty/null fields in Weaviate.
                           If True, overwrite existing non-empty values.
        use_batch_api: If True (default), use Weaviate's batch API for better performance.
                       If False, use individual PATCH requests (legacy behavior).

    Returns:
        Dictionary with ingestion statistics:
        {
            "total_documents": int,
            "successful_updates": int,
            "failed_updates": int,
            "skipped_documents": int,
            "duration_seconds": float,
            "timestamp": str,
            "errors": List[Dict[str, str]]
        }
    """
    start_time = time.time()

    # Filter for successful extractions only
    successful_results = [r for r in extraction_results if r.get("extraction_status") == "success"]

    total_documents = len(extraction_results)
    skipped_documents = total_documents - len(successful_results)

    console.print(f"\n[cyan]Ingestion Plan:[/cyan]")
    console.print(f"  • Total extraction results: {total_documents}")
    console.print(f"  • Successful extractions: {len(successful_results)}")
    console.print(f"  • Skipped (failed extractions): {skipped_documents}")
    console.print(f"  • Batch size: {batch_size}")
    console.print(
        f"  • Batches to process: {(len(successful_results) + batch_size - 1) // batch_size}"
    )

    if not successful_results:
        logger.warning("No successful extractions to ingest")
        return {
            "total_documents": total_documents,
            "successful_updates": 0,
            "failed_updates": 0,
            "skipped_documents": skipped_documents,
            "duration_seconds": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "errors": [],
        }

    # Setup for batch processing
    base_url = f"http://{weaviate_host}:{weaviate_port}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    successful_updates = 0
    failed_updates = 0
    errors = []

    # Process in batches with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
    ) as progress:
        task = progress.add_task("Ingesting to Weaviate...", total=len(successful_results))

        # Process in batches
        for batch_idx, batch_start in enumerate(range(0, len(successful_results), batch_size)):
            batch = successful_results[batch_start : batch_start + batch_size]

            logger.info(
                f"Processing batch {batch_idx + 1}/{(len(successful_results) + batch_size - 1) // batch_size}"
            )

            # Use batch API for more efficient updates
            if use_batch_api:
                batch_data = []

                for result in batch:
                    document_id = result.get("document_id", "")
                    extracted_data = result.get("extracted_data", {})

                    if not document_id or not extracted_data:
                        logger.warning(f"Skipping result with missing document_id or extracted_data")
                        failed_updates += 1
                        progress.update(task, advance=1)
                        continue

                    # Convert document_id to Weaviate UUID
                    weaviate_uuid = weaviate.util.generate_uuid5(document_id)

                    # Build update payload
                    try:
                        update_payload = build_update_payload(extracted_data)

                        if not update_payload:
                            logger.debug(f"No non-empty fields to update for {document_id}")
                            successful_updates += 1
                            progress.update(task, advance=1)
                            continue

                        batch_data.append({
                            "uuid": weaviate_uuid,
                            "payload": update_payload,
                            "document_id": document_id,
                        })

                    except Exception as e:
                        logger.error(f"Error building payload for {document_id}: {e}")
                        failed_updates += 1
                        errors.append({
                            "document_id": document_id,
                            "error": f"Payload build error: {e}",
                        })
                        progress.update(task, advance=1)

                # Process batch via batch API
                if batch_data:
                    batch_successful, batch_failed, batch_errors = ingest_batch_via_batch_api(
                        batch_data=batch_data,
                        base_url=base_url,
                        headers=headers,
                        overwrite_existing=overwrite_existing,
                    )

                    successful_updates += batch_successful
                    failed_updates += batch_failed
                    errors.extend(batch_errors)

                    progress.update(task, advance=len(batch_data))

                    logger.info(
                        f"Batch {batch_idx + 1}: {batch_successful} successful, {batch_failed} failed"
                    )

            else:
                # Legacy: individual PATCH requests
                for result in batch:
                    document_id = result.get("document_id", "")
                    extracted_data = result.get("extracted_data", {})

                    if not document_id or not extracted_data:
                        logger.warning(f"Skipping result with missing document_id or extracted_data")
                        failed_updates += 1
                        progress.update(task, advance=1)
                        continue

                    # Convert document_id to Weaviate UUID using deterministic UUID generation
                    # This ensures we use the same UUID that was used when ingesting the document
                    weaviate_uuid = weaviate.util.generate_uuid5(document_id)

                    # Build update payload
                    try:
                        update_payload = build_update_payload(extracted_data)

                        if not update_payload:
                            logger.debug(f"No non-empty fields to update for {document_id}")
                            successful_updates += 1  # Consider this success (nothing to update)
                            progress.update(task, advance=1)
                            continue

                        # Fetch existing document properties if overwrite_existing is False
                        if not overwrite_existing:
                            get_url = f"{base_url}/v1/objects/LegalDocuments/{weaviate_uuid}"
                            try:
                                get_response = requests.get(get_url, headers=headers, timeout=10)
                                if get_response.ok:
                                    existing_data = get_response.json().get("properties", {})

                                    # Filter out fields that already have non-empty values
                                    filtered_payload = {}
                                    for field, value in update_payload.items():
                                        existing_value = existing_data.get(field)

                                        # Check if existing value is empty/null
                                        is_empty = (
                                            existing_value is None
                                            or existing_value == ""
                                            or (isinstance(existing_value, list) and not existing_value)
                                        )

                                        # Only include field if existing value is empty
                                        if is_empty:
                                            filtered_payload[field] = value
                                        else:
                                            logger.debug(
                                                f"Skipping field '{field}' for {document_id} - already has value"
                                            )

                                    update_payload = filtered_payload

                                    # Skip update if no fields remain after filtering
                                    if not update_payload:
                                        logger.debug(
                                            f"No empty fields to update for {document_id} (all fields already populated)"
                                        )
                                        successful_updates += 1
                                        progress.update(task, advance=1)
                                        continue
                                else:
                                    logger.warning(
                                        f"Could not fetch existing data for {document_id}: {get_response.status_code}"
                                    )
                                    # Continue with full payload if GET fails
                            except Exception as get_error:
                                logger.warning(
                                    f"Error fetching existing data for {document_id}: {get_error}, continuing with full update"
                                )
                                # Continue with full payload if GET fails

                        # PATCH request to update document
                        url = f"{base_url}/v1/objects/LegalDocuments/{weaviate_uuid}"

                        response = requests.patch(
                            url=url,
                            headers=headers,
                            json={"properties": update_payload},
                            timeout=30,
                        )

                        response.raise_for_status()
                        successful_updates += 1

                        logger.debug(f"✓ Updated {document_id} with {len(update_payload)} properties")

                    except requests.exceptions.HTTPError as e:
                        error_info = {
                            "document_id": document_id,
                            "weaviate_uuid": weaviate_uuid,
                            "error": str(e),
                            "status_code": e.response.status_code if e.response else None,
                            "response": e.response.text if e.response else None,
                        }
                        errors.append(error_info)
                        failed_updates += 1

                        # Enhanced logging for 422 errors to see validation details
                        if e.response and e.response.status_code == 422:
                            logger.warning(
                                f"✗ Failed to update {document_id} with 422 validation error. "
                                f"Response: {e.response.text[:500]}"
                            )
                        else:
                            logger.warning(f"✗ Failed to update {document_id}: {e}")

                        if not skip_on_error:
                            raise

                    except Exception as e:
                        error_info = {
                            "document_id": document_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                        errors.append(error_info)
                        failed_updates += 1

                        logger.warning(f"✗ Error processing {document_id}: {e}")

                        if not skip_on_error:
                            raise

                    progress.update(task, advance=1)

            # Small delay between batches to avoid overwhelming Weaviate
            if batch_idx < (len(successful_results) + batch_size - 1) // batch_size - 1:
                time.sleep(delay_between_batches)

    # Calculate statistics
    duration = time.time() - start_time

    stats = {
        "total_documents": total_documents,
        "successful_updates": successful_updates,
        "failed_updates": failed_updates,
        "skipped_documents": skipped_documents,
        "duration_seconds": round(duration, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "errors": errors,
    }

    return stats


def display_ingestion_results(stats: Dict[str, Any], console: Console):
    """Display ingestion results in a formatted way."""

    console.print("\n" + "=" * 60)
    console.print("[bold green]✓ Weaviate Ingestion Complete![/bold green]")
    console.print("=" * 60)

    console.print(f"\n[cyan]Statistics:[/cyan]")
    console.print(f"  • Total documents: {stats['total_documents']}")
    console.print(f"  • Successful updates: [green]{stats['successful_updates']}[/green]")
    console.print(f"  • Failed updates: [red]{stats['failed_updates']}[/red]")
    console.print(
        f"  • Skipped (failed extractions): [yellow]{stats['skipped_documents']}[/yellow]"
    )
    console.print(f"  • Duration: {stats['duration_seconds']:.1f} seconds")

    if stats["successful_updates"] > 0:
        success_rate = (
            stats["successful_updates"] / (stats["successful_updates"] + stats["failed_updates"])
        ) * 100
        console.print(f"  • Success rate: [green]{success_rate:.1f}%[/green]")

    if stats["errors"]:
        console.print(f"\n[red]Errors ({len(stats['errors'])}):[/red]")
        for error in stats["errors"][:5]:  # Show first 5 errors
            console.print(f"  • {error['document_id']}: {error.get('error', 'Unknown error')}")
        if len(stats["errors"]) > 5:
            console.print(f"  ... and {len(stats['errors']) - 5} more errors")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Gemini extraction using Weaviate REST API")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of documents to sample",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        choices=[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            # "gemini-2.0-flash-exp",
            # "gemini-1.5-pro",
            # "gemini-1.5-flash",
        ],
        help="Gemini model to use (via Vertex AI)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extraction_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=".cache/extraction_sample.db",
        help="Path to SQLite cache",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--weaviate-host",
        type=str,
        default=None,
        help="Weaviate host (defaults to env var WEAVIATE_HOST)",
    )
    parser.add_argument(
        "--weaviate-port",
        type=int,
        default=None,
        help="Weaviate port (defaults to env var WEAVIATE_PORT)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of documents to process in each batch (default: 5)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel threads for batch processing (default: 5, max: 10)",
    )
    parser.add_argument(
        "--search-query",
        type=str,
        default=None,
        help="Optional search query for hybrid/semantic search (e.g., 'kredyt frankowy', 'VAT')",
    )
    parser.add_argument(
        "--document-type",
        type=str,
        default=None,
        choices=["judgment", "tax_interpretation"],
        help="Optional filter by document type (judgment or tax_interpretation)",
    )
    parser.add_argument(
        "--ingest-to-weaviate",
        action="store_true",
        help="Ingest extracted data back to Weaviate after extraction (updates existing documents)",
    )
    parser.add_argument(
        "--ingest-batch-size",
        type=int,
        default=50,
        help="Number of documents to ingest per batch when ingesting to Weaviate (default: 50)",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing non-empty values in Weaviate (default: only update empty fields)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Get Weaviate connection details
    weaviate_host = args.weaviate_host or os.getenv("WEAVIATE_HOST", "localhost")
    weaviate_port = args.weaviate_port or int(os.getenv("WEAVIATE_PORT", "8084"))
    api_key = os.getenv("WEAVIATE_API_KEY", "")

    # Get GCP project
    vertex_project = os.getenv("VERTEX_PROJECT", "insbay-b32351")
    vertex_location = os.getenv("VERTEX_LOCATION", "us-central1")

    # Build filter display
    filter_display = []
    if args.search_query:
        filter_display.append(f"Search: '{args.search_query}'")
    if args.document_type:
        filter_display.append(f"Type: {args.document_type}")
    filter_str = "\n" + "\n".join(filter_display) if filter_display else ""

    console.print(
        f"\n[bold cyan]Gemini Extraction - Vertex AI Mode[/bold cyan]\n"
        f"Weaviate: {weaviate_host}:{weaviate_port}\n"
        f"GCP Project: {vertex_project}\n"
        f"GCP Location: {vertex_location}\n"
        f"Sample size: {args.sample_size}\n"
        f"Batch size: {args.batch_size}\n"
        f"Max workers: {args.max_workers} {'(parallel)' if args.max_workers > 1 else '(sequential)'}\n"
        f"Model: {args.model}\n"
        f"Output: {args.output_dir}\n"
        f"Random seed: {args.seed}{filter_str}\n"
    )

    # Initialize LangChain PostgreSQL Cache using SQLAlchemyMd5Cache
    # This stores MD5 hashes instead of full prompts, avoiding index size limits
    postgres_cache_url = os.getenv("POSTGRES_CACHE_URL")
    if postgres_cache_url:
        try:
            logger.info(f"Initializing LangChain PostgreSQL cache (MD5) at {postgres_cache_url}...")
            engine = create_engine(postgres_cache_url)

            # Use SQLAlchemyMd5Cache instead of SQLAlchemyCache
            # This stores MD5 hashes of prompts to avoid index size limits with large prompts
            set_llm_cache(SQLAlchemyMd5Cache(engine=engine))

            console.print(
                f"[green]✓[/green] LangChain PostgreSQL cache (MD5) enabled ({postgres_cache_url})"
            )
            logger.info("LangChain PostgreSQL MD5 cache initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain cache: {e}")
            console.print(f"[yellow]Warning: LangChain cache initialization failed: {e}[/yellow]")
    else:
        console.print("[yellow]LangChain cache disabled (POSTGRES_CACHE_URL not set)[/yellow]")

    # Initialize extraction chain (Vertex AI uses application default credentials)
    logger.info("Initializing Vertex AI Gemini extraction chain...")
    chain = GeminiExtractionChain(
        model_name=args.model,
        project=vertex_project,
        location=vertex_location,
        cache_path=args.cache_path,
        temperature=0.0,
    )

    # Create schema
    schema = create_comprehensive_schema()
    logger.info(f"Created extraction schema with {len(schema.fields)} fields")

    # Initialize extraction storage
    storage = None
    run_id = None
    try:
        storage = ExtractionStorage()
        logger.info("Initialized extraction storage (PostgreSQL)")
        console.print("[green]✓[/green] Extraction storage enabled (PostgreSQL)")
    except Exception as e:
        logger.warning(f"Failed to initialize extraction storage: {e}")
        console.print(f"[yellow]Warning: Extraction storage disabled - {e}[/yellow]")

    # Initialize Langfuse (enable by default if keys are available)
    langfuse_handler = None
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            langfuse_handler = CallbackHandler()
            console.print(
                f"[green]✓[/green] Langfuse tracing enabled "
                f"(host: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')})"
            )
            logger.info("Langfuse tracing initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            console.print(f"[yellow]Warning: Langfuse initialization failed: {e}[/yellow]")
    else:
        console.print("[yellow]Langfuse tracing disabled (keys not set)[/yellow]")

    # Fetch documents using REST API
    logger.info("Fetching documents from Weaviate...")
    documents = fetch_documents_rest(
        weaviate_host=weaviate_host,
        weaviate_port=weaviate_port,
        api_key=api_key,
        sample_size=args.sample_size,
        search_query=args.search_query,
        document_type_filter=args.document_type,
    )

    if not documents:
        console.print("[red]No documents found for extraction![/red]")
        return

    # Create extraction run in database
    start_time = time.time()
    if storage:
        try:
            run_id = storage.create_extraction_run(
                model_name=args.model,
                sample_size=args.sample_size,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                weaviate_host=weaviate_host,
                weaviate_port=weaviate_port,
                search_query=args.search_query,
                document_type_filter=args.document_type,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                temperature=0.0,
                prompt_template=schema.instructions,  # Save full instructions
                extraction_schema=schema.fields,  # Save complete schema
                random_seed=args.seed,
                notes=f"Extraction run with {args.sample_size} documents",
            )
            logger.info(f"Created extraction run: {run_id}")
            console.print(f"[cyan]Extraction run ID:[/cyan] {run_id}")
        except Exception as e:
            logger.error(f"Failed to create extraction run: {e}")
            console.print(f"[red]Failed to create extraction run: {e}[/red]")

    # Run extraction with parallel batch processing
    logger.info(
        f"Running extraction on {len(documents)} documents "
        f"(batch size: {args.batch_size}, workers: {args.max_workers})..."
    )
    extraction_results = run_extraction(
        documents,
        chain,
        schema,
        langfuse_handler,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        storage=storage,
        run_id=run_id,
    )

    # Save results
    output_dir = Path(args.output_dir)
    save_results(documents, extraction_results, output_dir)

    # Complete extraction run in database
    if storage and run_id:
        try:
            duration = time.time() - start_time
            successful = sum(
                1 for r in extraction_results if r.get("extraction_status") == "success"
            )
            failed = len(extraction_results) - successful

            storage.complete_extraction_run(
                run_id=run_id,
                total_documents=len(extraction_results),
                successful_extractions=successful,
                failed_extractions=failed,
                duration_seconds=duration,
            )

            # Save field coverage
            field_coverage = {}
            for result in extraction_results:
                if result.get("extraction_status") == "success":
                    extracted_data = result.get("extracted_data") or {}
                    if isinstance(extracted_data, dict):
                        for field, value in extracted_data.items():
                            if field not in field_coverage:
                                field_coverage[field] = {"populated": 0, "empty": 0}

                            # Check if field is populated
                            if value:
                                if isinstance(value, str) and value.strip():
                                    field_coverage[field]["populated"] += 1
                                elif isinstance(value, list) and value:
                                    field_coverage[field]["populated"] += 1
                                elif isinstance(value, dict) and value:
                                    field_coverage[field]["populated"] += 1
                                else:
                                    field_coverage[field]["empty"] += 1
                            else:
                                field_coverage[field]["empty"] += 1

            if field_coverage:
                storage.save_field_coverage(run_id, field_coverage)

            logger.info(f"Completed extraction run: {run_id}")
            console.print(f"[green]✓[/green] Extraction run completed and saved to database")
        except Exception as e:
            logger.error(f"Failed to complete extraction run: {e}")
            console.print(f"[red]Failed to complete extraction run: {e}[/red]")

    # Optional: Ingest extracted data back to Weaviate
    if args.ingest_to_weaviate:
        console.print("\n[bold blue]Starting Weaviate ingestion...[/bold blue]")

        try:
            ingestion_stats = ingest_extracted_to_weaviate(
                extraction_results=extraction_results,
                weaviate_host=weaviate_host,
                weaviate_port=weaviate_port,
                api_key=api_key,
                batch_size=args.ingest_batch_size,
                skip_on_error=True,
                overwrite_existing=args.overwrite_existing,
            )

            # Display results
            display_ingestion_results(ingestion_stats, console)

            # Log ingestion to database
            if storage and run_id:
                try:
                    storage.log_ingestion(
                        run_id=run_id,
                        batch_size=args.ingest_batch_size,
                        overwrite_existing=args.overwrite_existing,
                        total_documents=ingestion_stats["total_documents"],
                        successful_updates=ingestion_stats["successful_updates"],
                        failed_updates=ingestion_stats["failed_updates"],
                        skipped_documents=ingestion_stats["skipped_documents"],
                        duration_seconds=ingestion_stats["duration_seconds"],
                        errors=ingestion_stats.get("errors", []),
                        status="completed",
                    )
                    logger.info(f"Logged ingestion for run: {run_id}")
                except Exception as e:
                    logger.error(f"Failed to log ingestion: {e}")

            # Save ingestion report
            ingestion_report_path = output_dir / "ingestion_report.json"
            with open(ingestion_report_path, "w", encoding="utf-8") as f:
                json.dump(ingestion_stats, f, ensure_ascii=False, indent=2)

            console.print(f"\n[cyan]Ingestion report saved to:[/cyan] {ingestion_report_path}")

        except Exception as e:
            console.print(f"\n[red]✗ Ingestion failed: {e}[/red]")
            logger.exception("Ingestion error")
            raise

    console.print(f"\n[bold green]✓ Extraction complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
