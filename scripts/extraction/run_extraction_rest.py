"""Run Gemini extraction using Weaviate REST API directly.

This script bypasses the Weaviate Python client GRPC issues by using REST API directly.
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any

from dotenv import load_dotenv
import requests
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from langfuse.langchain import CallbackHandler

from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema
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
) -> List[Dict[str, Any]]:
    """Fetch documents using Weaviate REST API directly with pagination.

    Args:
        weaviate_host: Weaviate host
        weaviate_port: Weaviate port
        api_key: Weaviate API key
        sample_size: Number of documents to sample
        chunk_size: Number of documents to fetch per request (max 10000)

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

    logger.info(
        f"Fetching up to {target_size} documents from {base_url} in chunks of {chunk_size} (REST API)..."
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

        # GraphQL query with offset and limit
        query = {
            "query": """
            {
                Get {
                    LegalDocuments(limit: %d, offset: %d) {
                        document_id
                        document_type
                        full_text
                        language
                        document_number
                    }
                }
            }
            """
            % (current_limit, offset)
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
        for extracted, metadata in zip(extracted_batch, batch_metadata):
            result = {
                "document_id": metadata["document_id"],
                "document_type": metadata["doc_type_str"],
                "extraction_status": "success",
                "extracted_data": extracted,
                "full_text_length": metadata["full_text_length"],
                "source_language": metadata["language"],
            }
            batch_results.append(result)
            logger.info(
                f"[Batch {batch_idx}] ✓ Extracted {metadata['document_id']} ({len(extracted)} fields)"
            )

    except Exception as e:
        # If batch fails, fall back to individual processing
        logger.warning(
            f"[Batch {batch_idx}] Batch extraction failed: {e}, falling back to individual processing"
        )

        for text, metadata in zip(batch_texts, batch_metadata):
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
                    "document_type": metadata["doc_type_str"],
                    "extraction_status": "success",
                    "extracted_data": extracted,
                    "full_text_length": metadata["full_text_length"],
                    "source_language": metadata["language"],
                }
                batch_results.append(result)
                logger.info(
                    f"[Batch {batch_idx}] ✓ Extracted {metadata['document_id']} ({len(extracted)} fields)"
                )

            except Exception as e2:
                logger.error(
                    f"[Batch {batch_idx}] ✗ Failed to extract {metadata['document_id']}: {e2}"
                )
                batch_results.append(
                    {
                        "document_id": metadata["document_id"],
                        "document_type": metadata["doc_type_str"],
                        "extraction_status": "failed",
                        "error": str(e2),
                        "full_text_length": metadata["full_text_length"],
                    }
                )

    return batch_results


def run_extraction(
    documents: List[Dict[str, Any]],
    chain: GeminiExtractionChain,
    schema: ExtractionSchema,
    langfuse_handler=None,
    batch_size: int = 10,
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Run extraction on sampled documents using parallel batch processing.

    Args:
        documents: List of document properties
        chain: Extraction chain
        schema: Extraction schema
        langfuse_handler: Optional Langfuse callback handler for observability
        batch_size: Number of documents to process in each batch (default: 10)
        max_workers: Number of parallel threads for batch processing (default: 1, max: 5)

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
                    batch_docs, chain, schema, langfuse_handler, batch_idx
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
                        _process_batch, batch_docs, chain, schema, langfuse_handler, batch_idx
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

    # Save extraction results
    extracted_file = output_dir / "sample_documents_extracted.jsonl"
    with open(extracted_file, "w", encoding="utf-8") as f:
        for result in extraction_results:
            f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    logger.info(f"Saved {len(extraction_results)} extraction results to {extracted_file}")

    # Save summary statistics
    summary_file = output_dir / "extraction_summary.json"

    successful = sum(1 for r in extraction_results if r.get("extraction_status") == "success")
    failed = len(extraction_results) - successful

    # Analyze field coverage
    field_coverage = {}
    for result in extraction_results:
        if result.get("extraction_status") == "success":
            extracted_data = result.get("extracted_data", {})
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
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
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
        default=10,
        help="Number of documents to process in each batch (default: 10)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel threads for batch processing (default: 1, recommended: 3-5)",
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
        f"Random seed: {args.seed}\n"
    )

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
    )

    if not documents:
        console.print("[red]No documents found for extraction![/red]")
        return

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
    )

    # Save results
    output_dir = Path(args.output_dir)
    save_results(documents, extraction_results, output_dir)

    console.print(f"\n[bold green]✓ Extraction complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
