# Schemat ekstrakcji: Interpretacje podatkowe

> Wszystkie pola mają jawnie określony typ danych i metodę agregacji.
> Obiekty JSON rozbite na flat kolumny. Free text pola uzupełnione o enum klasyfikatory.
> Gotowy do implementacji w HuggingFace Datasets / Parquet / PostgreSQL.

---

## Profil agregowalności

```
73 pola łącznie:
  ENUM .............. 22 pól (30%)  → GROUP BY, COUNT, rozkłady
  BOOL .............. 14 pól (19%)  → proporcje, korelacje
  INT/FLOAT .......... 8 pól (11%)  → avg, median, histogram
  SCORE (1-5) ........ 4 pola ( 5%)  → rozkłady, korelacje
  DATE ............... 2 pola ( 3%)  → szeregi czasowe
  ENUM_ARRAY ......... 3 pola ( 4%)  → multi-label frequency
  KEYWORDS ........... 4 pola ( 5%)  → frequency, TF-IDF
  ID_REF ............. 5 pól ( 7%)  → citation networks
  FREE_TEXT .......... 11 pól (15%)  → embedding, NLP

  Natychmiastowa agregacja (bez NLP): 78%
```

---

## A. IDENTYFIKACJA (5 pól)

| # | Pole | Typ danych | Agregacja | Opis | Wartości dozwolone |
|---|------|-----------|-----------|------|--------------------|
| 1 | `document_number` | ID | — | Sygnatura, np. `0114-KDIP2-1.4010.123.2023.1.AB` | — |
| 2 | `date_issued` | DATE | szeregi czasowe, trendy | Data wydania interpretacji (ISO 8601) | — |
| 3 | `date_received` | DATE | szeregi czasowe | Data wpływu wniosku (ISO 8601), jeśli podana | — |
| 4 | `document_type` | ENUM | GROUP BY | Typ dokumentu | `interpretacja_indywidualna` / `interpretacja_ogolna` / `objasnienia_podatkowe` |
| 5 | `issuing_authority` | ENUM | GROUP BY | Organ wydający | `dyrektor_kis` / `minister_finansow` / `szef_kas` |

---

## B. KLASYFIKACJA MERYTORYCZNA — ENUM (17 pól)

Pola natychmiastowo agregowalne: pie chart, bar chart, cross-tabulation, chi-square.

| # | Pole | Typ danych | Wartości dozwolone | Opis |
|---|------|-----------|--------------------|----- |
| 6 | `tax_type` | ENUM | `PIT` / `CIT` / `VAT` / `PCC` / `akcyza` / `podatek_od_nieruchomosci` / `ordynacja_podatkowa` / `clo` / `podatek_od_spadkow` / `inne` | Rodzaj podatku |
| 7 | `tax_subtype` | ENUM | **PIT:** `skala` / `liniowy` / `ryczalt` / `karta_podatkowa` / `kapitalowy` / `inne_pit`; **CIT:** `ogolny` / `estonski` / `minimalny` / `WHT` / `inne_cit`; **VAT:** `krajowy` / `wdt_wnt` / `import_export` / `odliczenie` / `zwolnienie` / `stawka` / `mpp` / `inne_vat` | Podtyp podatkowy — uszczegóławia `tax_type` |
| 8 | `decision_type` | ENUM | `stanowisko_prawidlowe` / `stanowisko_nieprawidlowe` / `stanowisko_czesciowo_prawidlowe` | Rozstrzygnięcie |
| 9 | `applicant_type` | ENUM | `osoba_fizyczna` / `spolka_kapitalowa` / `spolka_osobowa` / `spolka_cywilna` / `fundacja` / `stowarzyszenie` / `jst` / `grupa_vat` / `spoldzielnia` / `jednoosobowa_dzialalnosc` / `inne` | Typ wnioskodawcy |
| 10 | `applicant_industry` | ENUM | `IT_software` / `e_commerce` / `finanse_bankowosc` / `nieruchomosci` / `budownictwo` / `produkcja` / `transport_logistyka` / `handel_detaliczny` / `handel_hurtowy` / `uslugi_profesjonalne` / `zdrowie_farmacja` / `energia` / `rolnictwo` / `gastronomia_hotelarstwo` / `media_reklama` / `edukacja` / `telekomunikacja` / `gornictwo` / `inne` / `brak_danych` | Branża wnioskodawcy |
| 11 | `transaction_type` | ENUM | `krajowa` / `wewnatrzwspolnotowa` / `eksport` / `import` / `transgraniczna_pozaunijna` / `mieszana` | Typ transakcji |
| 12 | `trigger_event` | ENUM | `nowa_transakcja` / `zmiana_przepisow` / `planowanie_podatkowe` / `reorganizacja` / `kontrola_podatkowa` / `spor_z_organem` / `poczatek_dzialalnosci` / `zmiana_formy_opodatkowania` / `transakcja_miedzynarodowa` / `otrzymanie_interpretacji_negatywnej` / `inne` | Co wywołało potrzebę uzyskania interpretacji |
| 13 | `case_subject_category` | ENUM | `koszty_uzyskania_przychodow` / `przychod_moment_powstania` / `zwolnienie_podatkowe` / `stawka_podatku` / `odliczenie_vat` / `miejsce_swiadczenia` / `podstawa_opodatkowania` / `obowiazek_podatkowy` / `ceny_transferowe` / `amortyzacja` / `reprezentacja_reklama` / `daniny_publiczne` / `zabezpieczenie_spoleczne` / `dzialalnosc_nierejestrowana` / `kryptoaktywa` / `ip_box` / `ulga_badawczo_rozwojowa` / `estonski_cit` / `ryczalt_najem` / `split_payment` / `biala_lista` / `inne` | Główny temat merytoryczny interpretacji |
| 14 | `tax_planning_aggressiveness` | ENUM | `konserwatywne` / `umiarkowane` / `agresywne` / `graniczne` | Poziom agresywności planowania |
| 15 | `fiscal_impact_direction` | ENUM | `korzystna_dla_podatnika` / `korzystna_dla_fiskusa` / `neutralna` | Kierunek wpływu fiskalnego |
| 16 | `position_type` | ENUM | `szersza_wykladnia` / `wezsza_wykladnia` / `zwolnienie` / `wylaczenie` / `inna_kwalifikacja` / `inna_stawka` / `inna_podstawa` / `brak_obowiazku` / `inne` | Typ stanowiska wnioskodawcy (czego oczekuje) |
| 17 | `ambiguity_type` | ENUM | `leksykalna` / `strukturalna` / `celowosciowa` / `kolizja_norm` / `luka_prawna` / `brak_definicji` / `sprzeczne_orzecznictwo` / `brak` | Typ wieloznaczności przepisu powodującego wątpliwość |
| 18 | `overruling_risk_level` | ENUM | `niskie` / `srednie` / `wysokie` | Ryzyko uchylenia przez sąd |
| 19 | `evolution_direction` | ENUM | `liberalizacja` / `zaostrzenie` / `stabilna` / `brak_danych` | Kierunek ewolucji linii interpretacyjnej |
| 20 | `conflict_resolution_method` | ENUM | `lex_specialis` / `lex_posterior` / `hierarchiczny` / `wykladnia_prounijna` / `wykladnia_prokonstytucyjna` / `brak_konfliktu` | Jak rozwiązano konflikt norm (jeśli występował) |
| 21 | `primary_legal_act` | ENUM | `ustawa_pit` / `ustawa_cit` / `ustawa_vat` / `ordynacja_podatkowa` / `ustawa_pcc` / `ustawa_akcyzowa` / `ustawa_rachunkowosci` / `kodeks_cywilny` / `kodeks_spolek_handlowych` / `prawo_celne` / `dyrektywa_vat_2006_112` / `dyrektywa_2011_96` / `inne` | Główny akt prawny będący przedmiotem interpretacji |
| 22 | `applicant_size` | ENUM | `mikro` / `maly` / `sredni` / `duzy` / `brak_danych` | Wielkość przedsiębiorstwa wnioskodawcy (jeśli wynika z opisu) |

---

## C. KLASYFIKACJA — BOOL (14 pól)

Agregacja: proporcje (% true), korelacje z innymi polami, trendy czasowe.

| # | Pole | Opis |
|---|------|------|
| 23 | `is_prospective` | Czy dotyczy planowanej (true) czy zrealizowanej (false) transakcji |
| 24 | `eu_law_dimension` | Czy interpretacja odwołuje się do prawa UE |
| 25 | `cross_border_element` | Czy występuje element transgraniczny |
| 26 | `triggered_by_law_change` | Czy interpretacja dotyczy nowych/zmienionych przepisów |
| 27 | `has_business_purpose` | Czy transakcja ma realny cel gospodarczy |
| 28 | `form_over_substance_risk` | Czy zachodzi ryzyko formy nad treścią |
| 29 | `consistent_with_prior` | Czy spójna z wcześniejszymi interpretacjami na ten temat |
| 30 | `likely_court_challenge` | Czy sprawa prawdopodobnie trafi do sądu |
| 31 | `needs_further_interpretation` | Czy wymaga dalszych wyjaśnień |
| 32 | `regulatory_gap_identified` | Czy zidentyfikowano lukę regulacyjną |
| 33 | `cites_case_law` | Czy organ powołuje się na orzecznictwo sądowe |
| 34 | `cites_other_interpretations` | Czy organ powołuje się na inne interpretacje |
| 35 | `gaar_relevant` | Czy klauzula GAAR mogła mieć zastosowanie |
| 36 | `tp_relevant` | Czy dotyczy cen transferowych |

---

## D. METRYKI NUMERYCZNE (8 pól)

Agregacja: avg, median, percentyle, histogram, korelacje, regresja.

| # | Pole | Typ | Opis | Zakres |
|---|------|-----|------|--------|
| 37 | `question_count` | INT | Liczba pytań we wniosku | 1-20 |
| 38 | `processing_time_days` | INT | Czas od wniosku do wydania (dni) | 1-365 |
| 39 | `legal_references_count` | INT | Liczba cytowanych przepisów | 0-100 |
| 40 | `text_length_chars` | INT | Długość pełnego tekstu interpretacji | 1000-500000 |
| 41 | `argumentation_chain_length` | INT | Liczba kroków w łańcuchu rozumowania organu | 1-20 |
| 42 | `disagreement_points_count` | INT | Liczba punktów rozbieżności wnioskodawca vs organ | 0-15 |
| 43 | `reasoning_gaps_count` | INT | Liczba luk w argumentacji organu | 0-10 |
| 44 | `implicit_assumptions_count` | INT | Liczba nieuzasadnionych założeń | 0-10 |

---

## E. SKALE (SCORE 1-5) (4 pola)

Agregacja: avg, median, rozkład, korelacje, ranking.

| # | Pole | Opis | Skala |
|---|------|------|-------|
| 45 | `legal_certainty_score` | Przewidywalność wyniku (jasność przepisu × spójność orzecznictwa × jednoznaczność stanu faktycznego) | 1=skrajnie niepewne, 5=oczywiste |
| 46 | `compliance_complexity_score` | Złożoność zastosowania się do interpretacji w praktyce | 1=proste, 5=bardzo skomplikowane |
| 47 | `argumentation_quality_score` | Jakość argumentacji organu (kompletność, logiczność, brak luk) | 1=słaba, 5=wzorcowa |
| 48 | `economic_significance_score` | Znaczenie ekonomiczne rozstrzygnięcia | 1=marginalne, 5=systemowe |

---

## F. MULTI-LABEL ENUM (3 pola)

Agregacja: częstości poszczególnych wartości, co-occurrence matrix, multi-label classification.

| # | Pole | Typ | Wartości dozwolone | Opis |
|---|------|-----|--------------------|----- |
| 49 | `interpretation_methods` | ENUM_ARRAY | `jezykowa` / `celowosciowa` / `systemowa` / `historyczna` / `prounijna` / `prokonstytucyjna` / `funkcjonalna` / `autonomiczna` | Metody wykładni zastosowane przez organ |
| 50 | `anti_avoidance_clauses` | ENUM_ARRAY | `GAAR` / `SAAR` / `CFC` / `TP` / `WHT_beneficial_owner` / `thin_cap` / `exit_tax` / `brak` | Klauzule antyabuzywne mające zastosowanie |
| 51 | `documentation_requirements` | ENUM_ARRAY | `ewidencja_vat` / `jpk` / `dokumentacja_tp` / `ewidencja_przebiegu` / `oswiadczenie` / `certyfikat_rezydencji` / `faktura` / `umowa` / `opinia_bieglego` / `wycena` / `brak_specjalnych` | Wymagania dokumentacyjne wynikające z interpretacji |

---

## G. KEYWORDS — listy z otwartego słownika (4 pola)

Agregacja: word frequency, TF-IDF, topic modeling, co-occurrence.

| # | Pole | Typ | Opis |
|---|------|-----|----- |
| 52 | `keywords` | KEYWORDS (5-15) | Słowa kluczowe w j. polskim |
| 53 | `legal_concepts` | KEYWORDS (3-10) | Pojęcia prawne (np. "miejsce świadczenia usług") |
| 54 | `foreign_jurisdictions` | KEYWORDS (0-10) | Kraje, których prawo podatkowe jest istotne |
| 55 | `substance_indicators` | KEYWORDS (0-10) | Wskaźniki substancji ekonomicznej (np. "pracownicy", "biuro", "zarząd") |

---

## H. REFERENCJE — identyfikatory do budowania grafów (5 pól)

Agregacja: citation networks, PageRank, frequency ranking, co-citation analysis.

| # | Pole | Typ | Format elementu | Opis |
|---|------|-----|-----------------|------|
| 56 | `legal_references` | ID_REF_ARRAY | `{act, article, paragraph}` | Cytowane przepisy krajowe |
| 57 | `eu_directives_cited` | ID_REF_ARRAY | `{directive_number, article}` | Cytowane dyrektywy/rozporządzenia UE |
| 58 | `tax_treaties_cited` | ID_REF_ARRAY | `{country, article}` | Cytowane umowy o unikaniu podwójnego opodatkowania |
| 59 | `cited_case_law_signatures` | ID_REF_ARRAY | `{court, signature, date}` | Sygnatury cytowanych orzeczeń |
| 60 | `cited_interpretations` | ID_REF_ARRAY | `{signature}` | Sygnatury cytowanych innych interpretacji |

---

## I. TEKST OTWARTY (11 pól)

Agregacja wymaga NLP: embeddings, clustering, topic modeling, NER, similarity.
Każde pole ma companion enum (sekcja B) do natychmiastowej agregacji.

| # | Pole | Companion enum | Opis |
|---|------|----------------|------|
| 61 | `title` | (`case_subject_category`) | Tytuł opisowy (max 200 znaków) |
| 62 | `summary` | (`case_subject_category`, `decision_type`) | Streszczenie 3-5 zdań |
| 63 | `thesis` | (`case_subject_category`, `primary_legal_act`) | Teza interpretacyjna (1-3 zdań) |
| 64 | `factual_state` | (`applicant_industry`, `applicant_type`, `transaction_type`) | Pełna narracja o stanie faktycznym |
| 65 | `legal_state` | (`primary_legal_act`, `tax_type`) | Podstawa prawna z artykułami |
| 66 | `applicant_position` | (`position_type`) | Stanowisko wnioskodawcy |
| 67 | `authority_position` | (`decision_type`) | Stanowisko organu |
| 68 | `decision_summary` | (`decision_type`, `fiscal_impact_direction`) | Skrót rozstrzygnięcia (2-3 zdań) |
| 69 | `legal_effect` | (`fiscal_impact_direction`) | Praktyczny skutek prawny |
| 70 | `ambiguous_provision` | (`ambiguity_type`, `primary_legal_act`) | Konkretny przepis powodujący wątpliwość |
| 71 | `questions_text` | (`question_count`, `case_subject_category`) | Pełna treść pytań wnioskodawcy |

---

## J. POLA STRUKTURALNE — pozostające jako JSON (2 pola)

Zbyt złożone na flat kolumny, ale z wyciągniętymi agregatami (sekcje C, D, E).

| # | Pole | Wyciągnięte agregaty | Opis |
|---|------|---------------------|------|
| 72 | `argumentation_graph` | → `argumentation_chain_length`, `reasoning_gaps_count`, `argumentation_quality_score` | Pełny graf argumentacji: `{premises: [], intermediate_conclusions: [], final_conclusion, chain: []}` |
| 73 | `quantitative_thresholds` | → `economic_significance_score` | `[{threshold_type, value, currency, context}]` — progi, limity, stawki |

---

## Kontrolowane słowniki — pełne definicje

### `case_subject_category` — taksonomia tematyczna (21 kategorii)

| Wartość | Zakres tematyczny | Typowe pytania |
|---------|-------------------|----------------|
| `koszty_uzyskania_przychodow` | Art. 15/22 CIT/PIT | Czy wydatek stanowi KUP? |
| `przychod_moment_powstania` | Art. 12/14 CIT/PIT | Kiedy powstaje przychód? |
| `zwolnienie_podatkowe` | Art. 17/21 CIT/PIT, art. 43 VAT | Czy przysługuje zwolnienie? |
| `stawka_podatku` | Art. 41/stawki VAT | Jaka stawka ma zastosowanie? |
| `odliczenie_vat` | Art. 86-91 VAT | Czy przysługuje odliczenie VAT naliczonego? |
| `miejsce_swiadczenia` | Art. 28a-28o VAT | Gdzie opodatkować usługę/dostawę? |
| `podstawa_opodatkowania` | Art. 29a VAT, art. 12 CIT | Co wchodzi do podstawy opodatkowania? |
| `obowiazek_podatkowy` | Art. 19a VAT | Kiedy powstaje obowiązek podatkowy? |
| `ceny_transferowe` | Art. 11a-11t CIT | Ceny transferowe, dokumentacja TP |
| `amortyzacja` | Art. 16a-16m CIT | Zasady amortyzacji środków trwałych |
| `reprezentacja_reklama` | Art. 16 ust. 1 pkt 28 CIT | Kwalifikacja wydatków na reprezentację |
| `daniny_publiczne` | Ordynacja podatkowa | Kwestie proceduralne, przedawnienie |
| `zabezpieczenie_spoleczne` | Ustawa o SUS | Składki ZUS, oskładkowanie |
| `dzialalnosc_nierejestrowana` | Art. 5 Prawo przedsiębiorców | Opodatkowanie drobnej sprzedaży |
| `kryptoaktywa` | Art. 17 ust. 1f PIT | Opodatkowanie kryptowalut |
| `ip_box` | Art. 24d CIT / 30ca PIT | Preferencja IP Box 5% |
| `ulga_badawczo_rozwojowa` | Art. 18d CIT | Ulga B+R |
| `estonski_cit` | Art. 28c-28t CIT | Ryczałt od dochodów spółek |
| `ryczalt_najem` | Art. 6 ust. 1a ryczałtu | Ryczałt od najmu prywatnego |
| `split_payment` | Art. 108a VAT | Mechanizm podzielonej płatności |
| `inne` | — | Pozostałe tematy |

### `applicant_industry` — taksonomia branżowa (19 kategorii)

Oparta na PKD 2007 (sekcje), uproszczona do granularności przydatnej w analizie podatkowej:

| Wartość | Sekcje PKD | Przykłady podmiotów |
|---------|------------|---------------------|
| `IT_software` | J (62, 63) | Software house, SaaS, freelancer programista |
| `e_commerce` | G (47.91) | Sklep internetowy, marketplace, dropshipping |
| `finanse_bankowosc` | K (64-66) | Bank, ubezpieczyciel, fundusz, fintech |
| `nieruchomosci` | L (68) | Deweloper, zarządca, pośrednik, najem |
| `budownictwo` | F (41-43) | Firma budowlana, instalacyjna, remontowa |
| `produkcja` | C (10-33) | Producent, przetwórca, montażownia |
| `transport_logistyka` | H (49-53) | Przewoźnik, spedytor, magazyn, kurier |
| `handel_detaliczny` | G (47) | Sklep stacjonarny, sieć handlowa |
| `handel_hurtowy` | G (46) | Dystrybutor, hurtownia |
| `uslugi_profesjonalne` | M (69-75) | Kancelaria, doradztwo, architektura, audyt |
| `zdrowie_farmacja` | Q (86), C (21) | Szpital, przychodnia, apteka, producent leków |
| `energia` | D (35), B (06) | Elektrownia, OZE, obrót energią |
| `rolnictwo` | A (01-03) | Gospodarstwo rolne, rybołówstwo |
| `gastronomia_hotelarstwo` | I (55-56) | Hotel, restauracja, catering |
| `media_reklama` | J (58-60), M (73) | Wydawnictwo, radio/TV, agencja reklamowa |
| `edukacja` | P (85) | Szkoła, uczelnia, centrum szkoleniowe |
| `telekomunikacja` | J (61) | Operator telekomunikacyjny, ISP |
| `gornictwo` | B (05-09) | Kopalnia, wydobycie |
| `inne` | — | Pozostałe |

### `trigger_event` — taksonomia zdarzeń wyzwalających (11 kategorii)

| Wartość | Typowy kontekst |
|---------|----------------|
| `nowa_transakcja` | Wnioskodawca planuje lub realizuje nowy typ transakcji |
| `zmiana_przepisow` | Weszła w życie nowelizacja ustawy podatkowej |
| `planowanie_podatkowe` | Optymalizacja struktury podatkowej |
| `reorganizacja` | Połączenie, podział, przekształcenie spółki |
| `kontrola_podatkowa` | W trakcie lub po kontroli/postępowaniu |
| `spor_z_organem` | Rozbieżne stanowisko US/UCS |
| `poczatek_dzialalnosci` | Zakładanie nowej firmy, pierwsze rozliczenie |
| `zmiana_formy_opodatkowania` | Przejście np. ze skali na liniowy, na estoński CIT |
| `transakcja_miedzynarodowa` | Nowy kontrahent zagraniczny, ekspansja |
| `otrzymanie_interpretacji_negatywnej` | Zaskarżenie wcześniejszej negatywnej interpretacji |
| `inne` | Pozostałe |

---

## Instrukcje ekstrakcji

### Zasady ogólne

- Ekstraktuj WYŁĄCZNIE z tekstu dokumentu
- Język: polski
- Daty: ISO 8601 (YYYY-MM-DD)
- Puste pola: `""` (string), `[]` (array), `null` (object), `0` (int), `false` (bool)
- ENUM: TYLKO wartości z listy, `inne`/`brak_danych` gdy żadna nie pasuje

### Reguły mapowania na enum

**`tax_type`** → szukaj w tytule, słowach kluczowych, cytowanych ustawach:

- "podatku dochodowego od osób fizycznych" → `PIT`
- "podatku dochodowego od osób prawnych" → `CIT`
- "podatku od towarów i usług" → `VAT`
- "podatku od czynności cywilnoprawnych" → `PCC`

**`applicant_industry`** → dedukuj ze stanu faktycznego:

- "prowadzi działalność w zakresie tworzenia oprogramowania" → `IT_software`
- "sprzedaje towary za pośrednictwem platformy internetowej" → `e_commerce`
- "jest spółką deweloperską" → `nieruchomosci`

**`trigger_event`** → dedukuj z kontekstu wniosku:

- "w związku ze zmianą ustawy od dnia..." → `zmiana_przepisow`
- "zamierza zawrzeć umowę..." → `nowa_transakcja`
- "w trakcie prowadzonej kontroli..." → `kontrola_podatkowa`

**`case_subject_category`** → klasyfikuj na podstawie głównego pytania wnioskodawcy:

- "czy wydatek na X stanowi koszt uzyskania przychodu" → `koszty_uzyskania_przychodow`
- "czy usługa podlega opodatkowaniu VAT w Polsce" → `miejsce_swiadczenia`
- "czy przysługuje prawo do odliczenia VAT naliczonego" → `odliczenie_vat`

**`tax_planning_aggressiveness`** → oceniaj na podstawie dystansu między literalnym a proponowanym rozumieniem:

- typowe pytanie o prawidłowe zastosowanie → `konserwatywne`
- niestandardowa kwalifikacja w ramach przepisu → `umiarkowane`
- wyraźnie korzystna wykładnia sprzeczna z celem przepisu → `agresywne`
- na granicy obejścia prawa → `graniczne`

### Walidacja

1. Każde pole ENUM używa TYLKO wartości z listy
2. `tax_subtype` jest spójny z `tax_type`
3. Pola BOOL: wyłącznie `true`/`false`, nie `null`
4. Pola INT: nieujemne, w zdefiniowanym zakresie
5. `interpretation_methods` nie może być puste — minimum `jezykowa`
6. `question_count` musi odpowiadać faktycznej liczbie pytań w `questions_text`
7. `date_issued` ≥ `date_received` (jeśli oba dostępne)
