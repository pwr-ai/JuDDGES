"""Legal document extraction schemas for Polish legal system.

This module provides schema definitions for extracting structured information
from Polish legal documents (judgments and tax interpretations).
"""

from juddges.extraction.gemini_chain import ExtractionSchema


def create_polish_legal_schema() -> ExtractionSchema:
    """Create comprehensive extraction schema for Polish legal documents.

    Schema includes:
    - Core document identification (4 fields)
    - High-priority augmentation fields (3 fields)
    - Factual and legal context (2 NEW fields)
    - Outcome (1 field with enumerations)
    - Legal content (3 fields with references, concepts, parties)
    - Structured legal analysis (1 field)
    - Document-type specific fields (2 fields)

    Total: 16 fields covering all aspects of Polish legal documents.

    Returns:
        ExtractionSchema with complete field definitions and instructions
    """
    return ExtractionSchema(
        fields={
            # ====================================================================
            # CORE DOCUMENT IDENTIFICATION (4 fields)
            # ====================================================================
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

            # ====================================================================
            # HIGH-PRIORITY AUGMENTATION FIELDS (3 fields)
            # ====================================================================
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

            # ====================================================================
            # NEW FIELDS: FACTUAL AND LEGAL CONTEXT (2 fields)
            # ====================================================================
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

            # ====================================================================
            # OUTCOME (1 field with enumeration)
            # ====================================================================
            "outcome": """JSON object describing the decision and its effects: {
                "decision_type": "enum (EXACT values based on document_type)",
                "decision_summary": "string, brief 2-3 sentence summary of decision",
                "awarded_amounts": [{"type": "string", "amount": number, "currency": "string", "recipient": "string"}],
                "legal_effect": "string, practical legal consequence"
            }

DECISION_TYPE enumeration (document_type = "judgment"):
- "uwzgledniono_w_calosci" (granted in full)
- "uwzgledniono_w_czesci" (granted in part)
- "oddalono" (dismissed)
- "umorzono" (discontinued)
- "uchylono" (overturned/annulled)
- "uchylono_i_przekazano" (remanded)
- "zmieniono" (modified)
- "utrzymano_w_mocy" (upheld)
- "odrzucono" (rejected on procedural grounds)

DECISION_TYPE enumeration (document_type = "tax_interpretation"):
- "stanowisko_prawidlowe" (position correct)
- "stanowisko_nieprawidlowe" (position incorrect)
- "stanowisko_czesciowo_prawidlowe" (position partially correct)

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

            # ====================================================================
            # LEGAL CONTENT (3 fields)
            # ====================================================================
            "legal_references": """JSON array of legal citations (extract 5-15 most important)""",
            "legal_concepts": """JSON array of legal concepts (extract 3-10 key concepts)""",
            "parties": """JSON array of parties involved (all parties with roles)""",

            # ====================================================================
            # STRUCTURED CONTENT (1 field)
            # ====================================================================
            "legal_analysis": """JSON object with structured legal reasoning""",

            # ====================================================================
            # DOCUMENT-TYPE SPECIFIC FIELDS (2 fields)
            # ====================================================================
            "judgment_specific": """JSON object (ONLY if document_type = "judgment")""",
            "tax_interpretation_specific": """JSON object (ONLY if document_type = "tax_interpretation")""",
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
