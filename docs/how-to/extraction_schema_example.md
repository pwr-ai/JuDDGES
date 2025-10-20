# Gemini Extraction Schema - Example Output

## Summary

This document provides comprehensive extraction schema for legal documents with example outputs demonstrating the structure and content of extracted information from Polish legal documents.

## Schema Overview

Based on Weaviate schema analysis and field coverage statistics, the extraction schema focuses on high-priority fields with low coverage that can be augmented using Gemini LLM.

### Prioritized Fields for Extraction

| Priority | Field | Current Coverage | Target | Description |
|----------|-------|------------------|--------|-------------|
| **1** | `summary` | 0-10% | 100% | Concise 3-5 sentence document summary |
| **2** | `thesis` | 0-10% | 100% | Main legal principle/holding |
| **3** | `keywords` | 0-15% | 100% | 5-15 relevant legal keywords |
| **4** | `outcome` | 0-20% | 90% | Decision outcome with amounts and effects |
| **5** | `legal_concepts` | 0-20% | 85% | Legal concepts discussed |
| **6** | `legal_references` | 20-40% | 95% | Complete legal citations |
| **7** | `parties` | 30-50% | 90% | Party information with roles |
| **8** | `legal_analysis` | 0-10% | 80% | Structured reasoning |
| **9** | `structured_content` | 0-10% | 75% | Document sections |
| **10** | `judgment_specific` | 40-60% | 95% | Court metadata |

---

## Example 1: Court Judgment (Bank Fee Case)

### Input: Document `full_text`

```text
WYROK
W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ

Dnia 15 stycznia 2024 r.

Sąd Okręgowy w Warszawie, V Wydział Cywilny
w składzie:

Przewodniczący: SSO Anna Kowalska
Sędziowie: SSO Jan Nowak, SSR del. Piotr Wiśniewski
Protokolant: Małgorzata Zielińska

po rozpoznaniu w dniu 10 stycznia 2024 r. w Warszawie
na rozprawie
sprawy z powództwa Jana Kowalskiego
przeciwko Bankowi XYZ S.A. z siedzibą w Warszawie
o zapłatę
sygn. akt V C 1234/23

ORZEKA:

I. Zasądza od pozwanego Banku XYZ S.A. na rzecz powoda Jana Kowalskiego
kwotę 50.000 zł (pięćdziesiąt tysięcy złotych) wraz z odsetkami ustawowymi
za opóźnienie od dnia 1 stycznia 2023 r. do dnia zapłaty.

II. Zasądza od pozwanego na rzecz powoda kwotę 5.000 zł tytułem zwrotu
kosztów procesu.

III. Nakazuje ściągnąć od pozwanego na rzecz Skarbu Państwa kwotę 2.500 zł
tytułem opłaty sądowej, od której zapłaty powód był zwolniony.

UZASADNIENIE

Powód Jana Kowalski wniósł o zasądzenie od pozwanego Banku kwoty 50.000 zł
tytułem zwrotu nienależnie pobranych opłat za prowadzenie rachunku bankowego
w latach 2020-2022, wraz z odsetkami oraz kosztami procesu.

W uzasadnieniu powód wskazał, że Bank pobierał opłaty za prowadzenie rachunku
w wysokości przekraczającej stawki określone w umowie rachunku bankowego z dnia
10 maja 2015 r. Powód zaznaczył, że wielokrotnie składał reklamacje, które Bank
odrzucał bez merytorycznego uzasadnienia.

Pozwany Bank w odpowiedzi na pozew wniósł o oddalenie powództwa w całości,
twierdząc, że opłaty pobierane były zgodnie z obowiązującym Taryfą prowizji
i opłat bankowych, która była integralną częścią umowy.

Sąd ustalił następujący stan faktyczny:

W dniu 10 maja 2015 r. powód zawarł z pozwanym umowę o prowadzenie rachunku
bankowego. Zgodnie z § 5 umowy, Bank miał prawo do pobierania opłat zgodnie
z Taryfą prowizji i opłat bankowych obowiązującą w dacie zawarcia umowy.

Analiza dokumentów wykazała, że Bank w latach 2020-2022 pobierał opłaty
miesięczne w wysokości 25 zł, podczas gdy zgodnie z Taryfą z dnia zawarcia
umowy opłata powinna wynosić 10 zł miesięcznie.

Bank wprowadził nową Taryfę w 2019 r., jednak nie uzyskał na to zgody powoda
ani nie poinformował go skutecznie o zmianie, co czyni te postanowienia
niewiążącymi dla powoda zgodnie z art. 384 k.c.

Sąd zważył, co następuje:

Powództwo zasługuje na uwzględnienie w całości.

Bank pobierał opłaty niezgodne z umową, co stanowiło bezpodstawne wzbogacenie
w rozumieniu art. 405 k.c. Bank wzbogacił się o kwotę 50.000 zł kosztem powoda,
bez podstawy prawnej, gdyż postanowienia nowej Taryfy nie wiązały powoda z uwagi
na brak skutecznego wprowadzenia zmian do umowy.

Zgodnie z art. 410 k.c. w związku z art. 405 k.c., osoba wzbogacona obowiązana
jest do zwrotu korzyści w naturze, a gdy to nie jest możliwe, do zwrotu jej
wartości.

Sąd uznał również za zasadne żądanie odsetek ustawowych za opóźnienie od dnia
1 stycznia 2023 r., jako że powód wezwał Bank do zwrotu nienależnie pobranych
kwot pismem z dnia 15 grudnia 2022 r., wyznaczając 14-dniowy termin.

O kosztach procesu orzeczono na podstawie art. 98 k.p.c., zgodnie z zasadą
odpowiedzialności za wynik procesu. Pozwany Bank przegrał sprawę w całości,
wobec czego zobowiązany jest do zwrotu powodowi kosztów procesu w kwocie
5.000 zł (zastępstwo procesowe 3.600 zł, opłata skarbowa od pełnomocnictwa
17 zł, pozostałe koszty 1.383 zł).

SSO Anna Kowalska    SSO Jan Nowak    SSR del. Piotr Wiśniewski
```

---

### Output: Extracted Information

```json
{
  "document_id": "V_C_1234_23_SO_Warszawa_2024",
  "document_number": "V C 1234/23",
  "document_type": "judgment",
  "title": "Wyrok Sądu Okręgowego w Warszawie w sprawie o zwrot nienależnie pobranych opłat bankowych",
  "date_issued": "2024-01-15",

  "summary": "Wyrok Sądu Okręgowego w Warszawie V Wydział Cywilny z dnia 15.01.2024 r., sygn. V C 1234/23. Sprawa dotyczyła roszczenia konsumenta o zwrot nienależnie pobranych opłat bankowych za prowadzenie rachunku w latach 2020-2022. Powód Jan Kowalski wniósł o zapłatę 50.000 zł od Banku XYZ S.A., twierdząc, że bank pobierał opłaty przekraczające stawki określone w umowie. Sąd ustalił, że Bank pobierał opłaty niezgodne z postanowieniami umowy (25 zł zamiast 10 zł miesięcznie), co stanowiło bezpodstawne wzbogacenie. Powództwo uwzględniono w całości na podstawie art. 410 k.c. w związku z art. 405 k.c. oraz art. 384 k.c.",

  "thesis": "Pobieranie przez bank opłat za czynności bankowe w wysokości przekraczającej stawki określone w umowie rachunku bankowego, bez skutecznego wprowadzenia zmian do umowy i uzyskania zgody klienta, stanowi bezpodstawne wzbogacenie w rozumieniu art. 405 k.c. i obliguje do zwrotu nienależnie pobranych kwot na podstawie art. 410 k.c.",

  "keywords": [
    "prawo cywilne",
    "bezpodstawne wzbogacenie",
    "umowa bankowa",
    "rachunek bankowy",
    "opłaty bankowe",
    "art. 405 k.c.",
    "art. 410 k.c.",
    "art. 384 k.c.",
    "ochrona konsumenta",
    "zmiana umowy",
    "taryfa bankowa",
    "roszczenia konsumenckie"
  ],

  "outcome": {
    "decision_type": "uwzględniono",
    "decision_summary": "Sąd uwzględnił powództwo w całości, zasądzając od pozwanego Banku XYZ S.A. na rzecz powoda Jana Kowalskiego kwotę 50.000 zł wraz z odsetkami ustawowymi od 01.01.2023 r. oraz kwotę 5.000 zł tytułem zwrotu kosztów procesu.",
    "awarded_amounts": [
      {
        "type": "kwota główna",
        "amount": 50000.00,
        "currency": "PLN",
        "recipient": "Jan Kowalski (powód)"
      },
      {
        "type": "odsetki ustawowe za opóźnienie",
        "amount": null,
        "currency": "PLN",
        "recipient": "Jan Kowalski (powód)",
        "note": "od 01.01.2023 do dnia zapłaty"
      },
      {
        "type": "koszty procesu",
        "amount": 5000.00,
        "currency": "PLN",
        "recipient": "Jan Kowalski (powód)"
      },
      {
        "type": "opłata sądowa",
        "amount": 2500.00,
        "currency": "PLN",
        "recipient": "Skarb Państwa"
      }
    ],
    "legal_effect": "Bank zobowiązany do zwrotu nienależnie pobranych opłat wraz z odsetkami oraz kosztami procesu. Orzeczenie potwierdza nieważność postanowień umowy zmienionych jednostronnie przez Bank bez zgody klienta."
  },

  "legal_references": [
    {
      "type": "statute",
      "title": "Kodeks cywilny",
      "article": "art. 405",
      "jurisdiction": "Poland",
      "citation": "art. 405 k.c.",
      "context": "podstawa prawna roszczenia o zwrot bezpodstawnego wzbogacenia"
    },
    {
      "type": "statute",
      "title": "Kodeks cywilny",
      "article": "art. 410",
      "jurisdiction": "Poland",
      "citation": "art. 410 k.c.",
      "context": "określenie sposobu zwrotu korzyści - zwrot wartości"
    },
    {
      "type": "statute",
      "title": "Kodeks cywilny",
      "article": "art. 384",
      "jurisdiction": "Poland",
      "citation": "art. 384 k.c.",
      "context": "skuteczność wprowadzenia zmian do wzorca umowy"
    },
    {
      "type": "statute",
      "title": "Kodeks postępowania cywilnego",
      "article": "art. 98",
      "jurisdiction": "Poland",
      "citation": "art. 98 k.p.c.",
      "context": "podstawa orzeczenia o kosztach procesu"
    }
  ],

  "legal_concepts": [
    {
      "concept_name": "bezpodstawne wzbogacenie",
      "legal_area": "prawo cywilne",
      "definition_context": "Korzyść uzyskana przez Bank bez podstawy prawnej kosztem klienta poprzez pobieranie zawyżonych opłat. Zastosowanie art. 405 k.c.",
      "relevance": "primary"
    },
    {
      "concept_name": "skuteczność zmiany umowy",
      "legal_area": "prawo cywilne",
      "definition_context": "Wymóg uzyskania zgody klienta na zmianę postanowień umowy, w szczególności Taryfy opłat bankowych. Art. 384 k.c.",
      "relevance": "primary"
    },
    {
      "concept_name": "ochrona konsumenta",
      "legal_area": "prawo cywilne",
      "definition_context": "Ochrona klienta-konsumenta przed jednostronnymi zmianami umowy dokonywanymi przez profesjonalistę (Bank) bez właściwego poinformowania.",
      "relevance": "secondary"
    }
  ],

  "parties": [
    {
      "party_type": "plaintiff",
      "party_name": "Jan Kowalski",
      "party_category": "natural_person",
      "representation": "pełnomocnik procesowy (adwokat/radca prawny)",
      "identification": ""
    },
    {
      "party_type": "defendant",
      "party_name": "Bank XYZ S.A.",
      "party_category": "company",
      "representation": "pełnomocnik procesowy",
      "identification": "z siedzibą w Warszawie"
    }
  ],

  "legal_analysis": {
    "facts_summary": "W okresie 2020-2022 Bank XYZ S.A. pobierał od powoda Jana Kowalskiego opłaty za prowadzenie rachunku bankowego w wysokości 25 zł miesięcznie, podczas gdy umowa z 2015 r. przewidywała opłatę 10 zł. Bank wprowadził nową Taryfę w 2019 r. bez skutecznego poinformowania powoda i uzyskania jego zgody. Łączna suma zawyżonych opłat wyniosła 50.000 zł.",
    "legal_issues": [
      "Czy Bank miał prawo do pobierania opłat według nowej Taryfy wprowadzonej jednostronnie?",
      "Czy pobrane opłaty stanowią bezpodstawne wzbogacenie Banku?",
      "Od jakiego momentu należą się powodowi odsetki za opóźnienie?"
    ],
    "reasoning": "Sąd uznał, że Bank nie wprowadził skutecznie zmian do umowy zgodnie z art. 384 k.c., gdyż nie uzyskał zgody powoda na nową Taryfę ani nie poinformował go właściwie o zmianie. W konsekwencji postanowienia nowej Taryfy nie wiązały powoda, a pobierane przez Bank opłaty przekraczające stawki określone w pierwotnej umowie stanowiły bezpodstawne wzbogacenie w rozumieniu art. 405 k.c. Bank wzbogacił się kosztem powoda bez podstawy prawnej, co rodzi obowiązek zwrotu na podstawie art. 410 k.c.",
    "conclusion": "Powództwo zasługuje na uwzględnienie w całości. Bank obowiązany jest do zwrotu nienależnie pobranych kwot wraz z odsetkami ustawowymi za opóźnienie od dnia następującego po upływie terminu wyznaczonego w wezwaniu do zapłaty (1 stycznia 2023 r.) oraz do zwrotu kosztów procesu zgodnie z zasadą odpowiedzialności za wynik sprawy (art. 98 k.p.c.)."
  },

  "judgment_specific": {
    "court_name": "Sąd Okręgowy w Warszawie",
    "court_type": "regional",
    "department_name": "V Wydział Cywilny",
    "case_number": "V C 1234/23",
    "hearing_date": "2024-01-10",
    "judgment_type": "wyrok",
    "proceeding_type": "civil",
    "judges": [
      {
        "name": "SSO Anna Kowalska",
        "role": "presiding"
      },
      {
        "name": "SSO Jan Nowak",
        "role": "panel_member"
      },
      {
        "name": "SSR del. Piotr Wiśniewski",
        "role": "panel_member"
      }
    ],
    "legal_bases": [
      "art. 405 k.c.",
      "art. 410 k.c.",
      "art. 384 k.c.",
      "art. 98 k.p.c."
    ],
    "is_final": null
  },

  "structured_content": {
    "sections": [
      {
        "section_type": "header",
        "section_title": "Sentencja",
        "content_summary": "Wyrok wydany w imieniu Rzeczypospolitej Polskiej, dane składu orzekającego i stron",
        "key_points": [
          "Data wyroku: 15 stycznia 2024",
          "Sąd Okręgowy w Warszawie, V Wydział Cywilny",
          "Skład: SSO Anna Kowalska (przewodnicząca), SSO Jan Nowak, SSR del. Piotr Wiśniewski"
        ]
      },
      {
        "section_type": "decision",
        "section_title": "Orzeczenie",
        "content_summary": "Zasądzenie kwoty głównej 50.000 zł z odsetkami, kosztów procesu 5.000 zł oraz opłaty sądowej 2.500 zł",
        "key_points": [
          "Uwzględnienie powództwa w całości",
          "Zasądzenie 50.000 zł wraz z odsetkami od 01.01.2023",
          "Zasądzenie kosztów procesu 5.000 zł",
          "Opłata sądowa 2.500 zł na rzecz Skarbu Państwa"
        ]
      },
      {
        "section_type": "facts",
        "section_title": "Stan faktyczny",
        "content_summary": "Ustalenie, że Bank pobierał zawyżone opłaty (25 zł zamiast 10 zł) przez 3 lata bez skutecznego wprowadzenia zmian do umowy",
        "key_points": [
          "Umowa z 10.05.2015 przewidywała opłatę 10 zł miesięcznie",
          "Bank wprowadził nową Taryfę w 2019 r. z opłatą 25 zł",
          "Brak zgody powoda na zmianę",
          "Reklamacje powoda odrzucane bez uzasadnienia"
        ]
      },
      {
        "section_type": "reasoning",
        "section_title": "Uzasadnienie prawne",
        "content_summary": "Analiza prawna wskazująca na bezpodstawne wzbogacenie Banku oraz brak skuteczności jednostronnej zmiany umowy",
        "key_points": [
          "Zastosowanie art. 405 k.c. - bezpodstawne wzbogacenie",
          "Zastosowanie art. 410 k.c. - obowiązek zwrotu",
          "Zastosowanie art. 384 k.c. - brak skuteczności zmiany umowy",
          "Odsetki od dnia następującego po upływie terminu z wezwania"
        ]
      }
    ],
    "document_structure": {
      "has_dissent": false,
      "has_appendices": false,
      "page_count": null
    }
  }
}
```

---

## Example 2: Tax Interpretation

### Input: Document `full_text`

```text
INTERPRETACJA INDYWIDUALNA

Dyrektor Krajowej Informacji Skarbowej
Warszawa, dnia 20 lutego 2024 r.

Numer: 0114-KDIP2-2.4011.123.2024.1.AB

INTERPRETACJA INDYWIDUALNA
dotycząca podatku dochodowego od osób fizycznych w zakresie skutków podatkowych
umorzenia udziałów

Szanowny Panie,

Na podstawie art. 14b § 1 i § 6 ustawy z dnia 29 sierpnia 1997 r. Ordynacja podatkowa
(tekst jedn.: Dz. U. z 2023 r., poz. 2383 z późn. zm.) - dalej: Ordynacja podatkowa,
Dyrektor Krajowej Informacji Skarbowej stwierdza, że stanowisko Wnioskodawcy,
przedstawione we wniosku z dnia 15 stycznia 2024 r. (data wpływu 18 stycznia 2024 r.),
o wydanie interpretacji przepisów prawa podatkowego dotyczącej podatku dochodowego
od osób fizycznych w zakresie skutków podatkowych umorzenia udziałów - jest prawidłowe.

UZASADNIENIE

W dniu 18 stycznia 2024 r. wpłynął do tutejszego Organu ww. wniosek o wydanie
interpretacji indywidualnej dotyczącej podatku dochodowego od osób fizycznych.

STAN FAKTYCZNY

Wnioskodawca jest właścicielem 100% udziałów w spółce z ograniczoną odpowiedzialnością
(dalej: Spółka). W związku z restrukturyzacją działalności gospodarczej Wnioskodawca
planuje umorzenie posiadanych udziałów w trybie umorzenia dobrowolnego, za wynagrodzeniem,
zgodnie z art. 199 Kodeksu spółek handlowych.

Udziały zostały nabyte przez Wnioskodawcę w 2015 r. za kwotę 100.000 zł. Wartość
nominalna umarzanych udziałów wynosi 100.000 zł. Wynagrodzenie za umorzenie udziałów
ustalone zostanie na poziomie 500.000 zł, co odpowiada wartości rynkowej przedsiębiorstwa
Spółki pomniejszonej o zobowiązania.

Wnioskodawca wskazał, że wynagrodzenie przekracza wartość nominalną udziałów oraz cenę
ich nabycia. Wnioskodawca przebywa na terytorium Polski ponad 183 dni w roku podatkowym
i podlega w Polsce nieograniczonemu obowiązkowi podatkowemu.

PYTANIE

Czy w przedstawionym stanie faktycznym dochód uzyskany przez Wnioskodawcę w związku
z umorzeniem udziałów za wynagrodzeniem będzie opodatkowany podatkiem dochodowym od
osób fizycznych jako dochód z kapitałów pieniężnych, o którym mowa w art. 17 ust. 1
pkt 6 lit. a ustawy o podatku dochodowym od osób fizycznych, według stawki 19%,
zgodnie z art. 30a ust. 1 pkt 4 tej ustawy?

STANOWISKO WNIOSKODAWCY

Zdaniem Wnioskodawcy, dochód uzyskany w wyniku umorzenia udziałów za wynagrodzeniem
stanowi dochód z kapitałów pieniężnych, o którym mowa w art. 17 ust. 1 pkt 6 lit. a
ustawy o PIT, i podlega opodatkowaniu w sposób określony w art. 30a ust. 1 pkt 4 tej
ustawy, tj. według stawki 19%. Podstawą opodatkowania jest różnica między wynagrodzeniem
otrzymanym z tytułu umorzenia udziałów a wydatkami poniesionymi na nabycie tych udziałów.

OCENA STANOWISKA

Stanowisko Wnioskodawcy jest prawidłowe.

UZASADNIENIE INTERPRETACJI ORGANU

[...szczegółowa analiza prawna...]

Reasumując, dochód z tytułu umorzenia udziałów za wynagrodzeniem stanowi dochód z
kapitałów pieniężnych w rozumieniu art. 17 ust. 1 pkt 6 lit. a ustawy o PIT i podlega
opodatkowaniu 19% podatkiem na zasadach określonych w art. 30a ust. 1 pkt 4 tej ustawy.
Podstawę opodatkowania stanowi różnica między wynagrodzeniem otrzymanym za umorzenie
udziałów (500.000 zł) a kosztami uzyskania przychodów w postaci wydatków na nabycie
tych udziałów (100.000 zł), tj. kwota 400.000 zł.

Dyrektor Krajowej Informacji Skarbowej
/-/ Anna Nowak
```

---

### Output: Extracted Information (Tax Interpretation)

```json
{
  "document_number": "0114-KDIP2-2.4011.123.2024.1.AB",
  "document_type": "tax_interpretation",
  "title": "Interpretacja indywidualna dotycząca skutków podatkowych umorzenia udziałów w spółce z o.o.",
  "date_issued": "2024-02-20",

  "summary": "Interpretacja indywidualna Dyrektora Krajowej Informacji Skarbowej z dnia 20.02.2024 r. dotycząca podatku dochodowego od osób fizycznych w zakresie skutków podatkowych umorzenia udziałów za wynagrodzeniem. Wnioskodawca planuje umorzenie 100% udziałów w spółce z o.o. nabytych za 100.000 zł za wynagrodzeniem 500.000 zł. Organ uznał stanowisko wnioskodawcy za prawidłowe, potwierdzając, że dochód z umorzenia udziałów stanowi dochód z kapitałów pieniężnych (art. 17 ust. 1 pkt 6 lit. a ustawy o PIT) opodatkowany stawką 19% (art. 30a ust. 1 pkt 4). Podstawą opodatkowania jest różnica między wynagrodzeniem a kosztami nabycia, tj. 400.000 zł.",

  "thesis": "Dochód z tytułu umorzenia udziałów za wynagrodzeniem w trybie art. 199 KSH stanowi dochód z kapitałów pieniężnych w rozumieniu art. 17 ust. 1 pkt 6 lit. a ustawy o podatku dochodowym od osób fizycznych i podlega opodatkowaniu według stawki 19% na podstawie art. 30a ust. 1 pkt 4 tej ustawy. Podstawę opodatkowania stanowi różnica między wynagrodzeniem za umorzenie a kosztami nabycia udziałów.",

  "keywords": [
    "podatek dochodowy od osób fizycznych",
    "PIT",
    "umorzenie udziałów",
    "dochód z kapitałów pieniężnych",
    "art. 17 ust. 1 pkt 6 lit. a ustawy o PIT",
    "art. 30a ust. 1 pkt 4 ustawy o PIT",
    "stawka 19%",
    "koszty uzyskania przychodów",
    "spółka z o.o.",
    "art. 199 KSH",
    "interpretacja indywidualna"
  ],

  "outcome": {
    "decision_type": "stanowisko pozytywne",
    "decision_summary": "Organ uznał stanowisko wnioskodawcy za prawidłowe. Dochód z umorzenia udziałów podlega opodatkowaniu jako dochód z kapitałów pieniężnych według stawki 19%.",
    "awarded_amounts": [],
    "legal_effect": "Zastosowanie stawki podatkowej 19% do dochodu w wysokości 400.000 zł (różnica między wynagrodzeniem 500.000 zł a kosztami nabycia 100.000 zł). Kwalifikacja dochodu jako dochód z kapitałów pieniężnych zgodnie z art. 17 ust. 1 pkt 6 lit. a ustawy o PIT."
  },

  "legal_references": [
    {
      "type": "statute",
      "title": "Ustawa o podatku dochodowym od osób fizycznych",
      "article": "art. 17 ust. 1 pkt 6 lit. a",
      "jurisdiction": "Poland",
      "citation": "art. 17 ust. 1 pkt 6 lit. a ustawy o PIT",
      "context": "definicja dochodu z kapitałów pieniężnych - umorzenie udziałów"
    },
    {
      "type": "statute",
      "title": "Ustawa o podatku dochodowym od osób fizycznych",
      "article": "art. 30a ust. 1 pkt 4",
      "jurisdiction": "Poland",
      "citation": "art. 30a ust. 1 pkt 4 ustawy o PIT",
      "context": "sposób opodatkowania dochodów z kapitałów pieniężnych - stawka 19%"
    },
    {
      "type": "statute",
      "title": "Kodeks spółek handlowych",
      "article": "art. 199",
      "jurisdiction": "Poland",
      "citation": "art. 199 KSH",
      "context": "tryb umorzenia dobrowolnego udziałów za wynagrodzeniem"
    },
    {
      "type": "statute",
      "title": "Ordynacja podatkowa",
      "article": "art. 14b § 1 i § 6",
      "jurisdiction": "Poland",
      "citation": "art. 14b § 1 i § 6 Ordynacji podatkowej",
      "context": "podstawa prawna wydania interpretacji indywidualnej"
    }
  ],

  "legal_concepts": [
    {
      "concept_name": "dochód z kapitałów pieniężnych",
      "legal_area": "prawo podatkowe",
      "definition_context": "Dochód uzyskany z tytułu umorzenia udziałów za wynagrodzeniem kwalifikowany jako dochód z kapitałów pieniężnych na podstawie art. 17 ust. 1 pkt 6 lit. a ustawy o PIT",
      "relevance": "primary"
    },
    {
      "concept_name": "umorzenie udziałów za wynagrodzeniem",
      "legal_area": "prawo handlowe",
      "definition_context": "Tryb umorzenia dobrowolnego udziałów w spółce z o.o. za wynagrodzeniem zgodnie z art. 199 KSH",
      "relevance": "primary"
    },
    {
      "concept_name": "koszty uzyskania przychodów",
      "legal_area": "prawo podatkowe",
      "definition_context": "Wydatki poniesione na nabycie umarzanych udziałów (100.000 zł) stanowiące koszty uzyskania przychodów przy ustalaniu podstawy opodatkowania",
      "relevance": "primary"
    }
  ],

  "parties": [
    {
      "party_type": "applicant",
      "party_name": "[anonimizowane - wnioskodawca]",
      "party_category": "natural_person",
      "representation": "",
      "identification": "właściciel 100% udziałów w spółce z o.o., rezydent podatkowy Polski"
    }
  ],

  "tax_interpretation_specific": {
    "interpretation_type": "individual",
    "tax_authority": "Dyrektor Krajowej Informacji Skarbowej",
    "authority_level": "national",
    "applicant_type": "natural_person",
    "tax_matter": "Skutki podatkowe umorzenia udziałów za wynagrodzeniem w trybie art. 199 KSH - kwalifikacja dochodu i sposób opodatkowania",
    "tax_type": "PIT",
    "fiscal_year": "2024",
    "interpretation_status": "binding",
    "validity_period": {
      "start_date": "2024-02-20",
      "end_date": null
    }
  }
}
```

---

## Integration Guide

### Using the Schema with Gemini Extraction

```python
from juddges.extraction import GeminiExtractionChain, ExtractionSchema, DocumentType

# Create schema
schema = ExtractionSchema(
    fields={
        "document_number": "string, official document reference number",
        "summary": "string, concise 3-5 sentence summary",
        "thesis": "string, main legal principle",
        "keywords": "List[string], 5-15 relevant keywords",
        "outcome": "JSON object with decision details",
        "legal_references": "JSON array of legal citations",
        # ... more fields
    },
    instructions="Extract factual information maintaining original language...",
    language="polish"
)

# Initialize chain
chain = GeminiExtractionChain(model_name="gemini-2.5-flash")

# Extract
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=document_full_text,
    schema=schema
)
```

---

## Benefits of This Schema

1. **Comprehensive Coverage**: Covers 14 high-priority fields with low current coverage
2. **Weaviate-Aligned**: Field names match exact Weaviate property names
3. **Structured Data**: Complex fields use JSON for structured storage
4. **Language-Aware**: Maintains original language (Polish/English)
5. **LLM-Optimized**: Designed for extraction by Gemini 2.5 models
6. **Production-Ready**: Includes validation, error handling, and caching

---

## Next Steps

1. **Run extraction on sample dataset** (50-100 documents)
2. **Validate extraction quality** (manual review of 10% sample)
3. **Update Weaviate documents** with extracted information
4. **Measure coverage improvement** (before/after statistics)
5. **Iterate schema** based on extraction quality feedback

For full implementation details, see:

- [Extraction Schema Documentation](./gemini_extraction_schema.md)
- [Extraction Script](../../scripts/extraction/run_extraction_sample.py)
- [Weaviate Schema](../../juddges/data/documents_weaviate_db.py)
